#include "astrosimgpu/astrocyte.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "astrosimgpu/astrocyte_kernel.hpp"

namespace astrosimgpu {

namespace {

/// Draw one per-cell parameter the way the reference model does: Gaussian
/// around the nominal value with std = var * |mean|, resampled until it lands
/// inside [lower * mean, upper * mean]. Bounds swap for negative means so the
/// interval stays ordered.
real draw_scaled(real mean, const RandomizeSpec& spec, CounterRng& rng) {
    if (!spec.enabled || spec.var <= 0.0) {
        return mean;
    }
    const real lo = mean < 0.0 ? mean * spec.upper : mean * spec.lower;
    const real hi = mean < 0.0 ? mean * spec.lower : mean * spec.upper;
    return rng.normal_redraw(mean, spec.var * std::abs(mean), lo, hi);
}

}  // namespace

void AstrocytePopulation::build(index_t count, const AstrocyteParams& base,
                                const RandomizeSpec& randomize, const InputParams& input,
                                CounterRng& rng) {
    p_ = base;
    input_ = input;

    Ca_.assign(count, base.Ca_init);
    IP3_.assign(count, base.IP3_init);
    h_.assign(count, base.h_init);
    ip3_input_.assign(count, 0.0);

    Ca_tot_.resize(count);
    IP3_0_.resize(count);
    tau_IP3_.resize(count);
    delta_IP3_.resize(count);

    for (index_t i = 0; i < count; ++i) {
        Ca_tot_[i] = draw_scaled(base.Ca_tot, randomize, rng);
        IP3_0_[i] = draw_scaled(base.IP3_0, randomize, rng);
        tau_IP3_[i] = draw_scaled(base.tau_IP3, randomize, rng);
        delta_IP3_[i] = draw_scaled(base.delta_IP3, randomize, rng);
        // IP3 itself is not randomised in the reference model; it starts at
        // the nominal baseline rather than the cell's own IP3_0.
        IP3_[i] = base.IP3_init;
    }
}

namespace {
#if defined(ASTROSIMGPU_KOKKOS)
/// Copy a host vector into a freshly allocated device View.
///
/// The label must be a std::string. Kokkos reads a raw const char* as a
/// pointer to existing memory, for constructing an unmanaged View, and rejects
/// it here with a static assertion about pointer-to-memory allocation.
using HostSpan = Kokkos::View<real*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

Kokkos::View<real*> to_device(const std::string& name, const vec<real>& host) {
    Kokkos::View<real*> d(Kokkos::view_alloc(name, Kokkos::WithoutInitializing), host.size());
    auto mirror = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < host.size(); ++i) {
        mirror(i) = host[i];
    }
    Kokkos::deep_copy(d, mirror);
    return d;
}

/// Copy a device View back into a host vector.
void from_device(const Kokkos::View<real*>& d, vec<real>& host) {
    auto mirror = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(mirror, d);
    for (std::size_t i = 0; i < host.size(); ++i) {
        host[i] = mirror(i);
    }
}
#endif
}  // namespace

void AstrocytePopulation::device_begin() {
#if defined(ASTROSIMGPU_CUDA)
    if (size() > 0) {
        cuda_ = cuda_astro_create(size(), Ca_.data(), IP3_.data(), h_.data(), Ca_tot_.data(),
                                  IP3_0_.data(), tau_IP3_.data(), delta_IP3_.data());
    }
    return;
#endif
#if defined(ASTROSIMGPU_KOKKOS)
    if (size() == 0) {
        return;
    }
    d_Ca_ = to_device("Ca", Ca_);
    d_IP3_ = to_device("IP3", IP3_);
    d_h_ = to_device("h", h_);
    d_ip3_input_ = to_device("ip3_input", ip3_input_);
    d_Ca_tot_ = to_device("Ca_tot", Ca_tot_);
    d_IP3_0_ = to_device("IP3_0", IP3_0_);
    d_tau_IP3_ = to_device("tau_IP3", tau_IP3_);
    d_delta_IP3_ = to_device("delta_IP3", delta_IP3_);
    device_ready_ = true;
    return;
#endif
#ifdef ASTROSIMGPU_OFFLOAD
    const std::int64_t n = size();
    if (n == 0) {
        return;
    }
    real* Ca = Ca_.data();
    real* IP3 = IP3_.data();
    real* hv = h_.data();
    real* ip3_in = ip3_input_.data();
    const real* Ca_tot = Ca_tot_.data();
    const real* IP3_0 = IP3_0_.data();
    const real* tau_IP3 = tau_IP3_.data();
    const real* delta_IP3 = delta_IP3_.data();
#pragma omp target enter data map(to                                                     \
                                  : Ca [0:n], IP3 [0:n], hv [0:n], ip3_in [0:n],         \
                                    Ca_tot [0:n], IP3_0 [0:n], tau_IP3 [0:n],            \
                                    delta_IP3 [0:n])
#endif
}

void AstrocytePopulation::device_end() {
#if defined(ASTROSIMGPU_CUDA)
    if (cuda_ != nullptr) {
        cuda_astro_destroy(cuda_, Ca_.data(), IP3_.data(), h_.data());
        cuda_ = nullptr;
    }
    return;
#endif
#if defined(ASTROSIMGPU_KOKKOS)
    if (device_ready_) {
        from_device(d_Ca_, Ca_);
        from_device(d_IP3_, IP3_);
        from_device(d_h_, h_);
        device_ready_ = false;
    }
    return;
#endif
#ifdef ASTROSIMGPU_OFFLOAD
    const std::int64_t n = size();
    if (n == 0) {
        return;
    }
    real* Ca = Ca_.data();
    real* IP3 = IP3_.data();
    real* hv = h_.data();
    real* ip3_in = ip3_input_.data();
    const real* Ca_tot = Ca_tot_.data();
    const real* IP3_0 = IP3_0_.data();
    const real* tau_IP3 = tau_IP3_.data();
    const real* delta_IP3 = delta_IP3_.data();
#pragma omp target exit data map(from                                                    \
                                 : Ca [0:n], IP3 [0:n], hv [0:n])                        \
    map(release                                                                          \
        : ip3_in [0:n], Ca_tot [0:n], IP3_0 [0:n], tau_IP3 [0:n], delta_IP3 [0:n])
#endif
}

void AstrocytePopulation::clear_inputs(const vec<index_t>& cells) {
    for (const index_t a : cells) {
        ip3_input_[a] = 0.0;
    }
}

void AstrocytePopulation::device_push_input() {
#if defined(ASTROSIMGPU_CUDA)
    if (cuda_ != nullptr) {
        cuda_astro_push_input(cuda_, ip3_input_.data());
    }
    return;
#endif
#if defined(ASTROSIMGPU_KOKKOS)
    if (device_ready_) {
        // An unmanaged View over the vector's own memory, so the copy goes
        // straight to the device. Staging through a mirror costs an extra
        // pass over the array on the host every step, which at a million
        // cells is more than the transfer.
        HostSpan h(ip3_input_.data(), ip3_input_.size());
        Kokkos::deep_copy(d_ip3_input_, h);
    }
    return;
#endif
#ifdef ASTROSIMGPU_OFFLOAD
    const std::int64_t n = size();
    if (n == 0) {
        return;
    }
    real* ip3_in = ip3_input_.data();
#pragma omp target update to(ip3_in [0:n])
#endif
}

void AstrocytePopulation::device_pull_calcium() {
#if defined(ASTROSIMGPU_CUDA)
    if (cuda_ != nullptr) {
        cuda_astro_pull_calcium(cuda_, Ca_.data());
    }
    return;
#endif
#if defined(ASTROSIMGPU_KOKKOS)
    if (device_ready_) {
        HostSpan h(Ca_.data(), Ca_.size());
        Kokkos::deep_copy(h, d_Ca_);
    }
    return;
#endif
#ifdef ASTROSIMGPU_OFFLOAD
    const std::int64_t n = size();
    if (n == 0) {
        return;
    }
    real* Ca = Ca_.data();
#pragma omp target update from(Ca [0:n])
#endif
}

AstroConstants AstrocytePopulation::constants() const {
    AstroConstants c{};
    c.Kd_IP3_1 = p_.Kd_IP3_1;
    c.Kd_IP3_2 = p_.Kd_IP3_2;
    c.Kd_act = p_.Kd_act;
    c.Kd_inh = p_.Kd_inh;
    c.Km_SERCA = p_.Km_SERCA;
    c.k_IP3R = p_.k_IP3R;
    c.rate_IP3R = p_.rate_IP3R;
    c.rate_L = p_.rate_L;
    c.rate_SERCA = p_.rate_SERCA;
    c.ratio_ER_cyt = p_.ratio_ER_cyt;
    return c;
}

void AstrocytePopulation::update(const TimeGrid& time, std::int64_t step, std::uint64_t seed) {
    const auto n = static_cast<std::int64_t>(size());
    const real h_step = time.h();
    const int substeps = time.substeps;
    const AstroConstants c = constants();

    // The injected current is held constant across noise_dt, so all cells
    // change value at the same instants even though each draws its own.
    const int steps_per_noise =
        std::max(1, static_cast<int>(std::llround(input_.noise_dt / time.dt)));
    const auto noise_index = static_cast<std::uint64_t>(step / steps_per_noise);
    const std::uint64_t noise_seed = seed ^ 0x9E3779B97F4A7C15ULL;
    const real noise_std = input_.gauss_noise_std;
    const bool independent = input_.independent_noise;
    const real shared_noise =
        (noise_std > 0.0 && !independent) ? noise_std * rng_normal(noise_seed, noise_index) : 0.0;

    // Raw pointers rather than the vectors themselves: the offloaded loop maps
    // arrays, and using the same pointers on the host keeps the two loop
    // bodies textually identical apart from the directive above them.
    real* __restrict Ca = Ca_.data();
    real* __restrict IP3 = IP3_.data();
    real* __restrict hv = h_.data();
    real* __restrict ip3_in = ip3_input_.data();
    const real* __restrict Ca_tot = Ca_tot_.data();
    const real* __restrict IP3_0 = IP3_0_.data();
    const real* __restrict tau_IP3 = tau_IP3_.data();
    const real* __restrict delta_IP3 = delta_IP3_.data();

#if defined(ASTROSIMGPU_CUDA)
    if (cuda_ != nullptr) {
        cuda_astro_update(cuda_, c, h_step, substeps, noise_std, independent, shared_noise,
                          noise_seed, noise_index);
        return;
    }
#endif

#if defined(ASTROSIMGPU_KOKKOS)
    // Same body as the other two backends, expressed as a Kokkos lambda. The
    // per-cell function is shared, so only the dispatch differs and the three
    // can be compared without the arithmetic being a variable.
    if (device_ready_) {
        auto Ca_d = d_Ca_;
        auto IP3_d = d_IP3_;
        auto h_d = d_h_;
        auto ip3_d = d_ip3_input_;
        auto Ca_tot_d = d_Ca_tot_;
        auto IP3_0_d = d_IP3_0_;
        auto tau_d = d_tau_IP3_;
        auto delta_d = d_delta_IP3_;
        Kokkos::parallel_for(
            "astrocyte_update", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const std::int64_t i) {
                const real noise =
                    (noise_std > 0.0 && independent)
                        ? noise_std * rng_normal(noise_seed, noise_index * 1000003ULL +
                                                                 static_cast<std::uint64_t>(i))
                        : shared_noise;
                real ca = Ca_d(i);
                real ip3 = IP3_d(i);
                real hh = h_d(i);
                astro_advance(c, Ca_tot_d(i), IP3_0_d(i), tau_d(i), delta_d(i), ip3_d(i), noise,
                              h_step, substeps, ca, ip3, hh);
                Ca_d(i) = ca;
                IP3_d(i) = ip3;
                h_d(i) = hh;
                ip3_d(i) = 0.0;
            });
        Kokkos::fence();
        for (std::int64_t i = 0; i < n; ++i) {
            ip3_in[i] = 0.0;
        }
        return;
    }
#endif

#ifdef ASTROSIMGPU_OFFLOAD
#pragma omp target teams distribute parallel for                                        \
    map(tofrom : Ca[0 : n], IP3[0 : n], hv[0 : n], ip3_in[0 : n])                       \
    map(to : Ca_tot[0 : n], IP3_0[0 : n], tau_IP3[0 : n], delta_IP3[0 : n], c)
#else
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < n; ++i) {
        const real noise = (noise_std > 0.0 && independent)
                               ? noise_std * rng_normal(noise_seed,
                                                        noise_index * 1000003ULL +
                                                            static_cast<std::uint64_t>(i))
                               : shared_noise;

        real ca = Ca[i];
        real ip3 = IP3[i];
        real h = hv[i];

        astro_advance(c, Ca_tot[i], IP3_0[i], tau_IP3[i], delta_IP3[i], ip3_in[i], noise, h_step,
                      substeps, ca, ip3, h);

        Ca[i] = ca;
        IP3[i] = ip3;
        hv[i] = h;
        ip3_in[i] = 0.0;
    }

}

real AstrocytePopulation::sic_factor(index_t cell) const {
    // y is the calcium excess over threshold expressed in nM; the state is in
    // uM, hence the factor of 1000. The Heaviside term means no output until
    // the excess exceeds 1 nM, where ln y turns positive.
    const real y = (Ca_[cell] - p_.SIC_th) * 1000.0;
    if (y <= 1.0) {
        return 0.0;
    }
    return p_.SIC_scale * std::log(y);
}

}  // namespace astrosimgpu

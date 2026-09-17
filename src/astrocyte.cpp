#include "astrosimgpu/astrocyte.hpp"

#include <algorithm>
#include <cmath>

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

void AstrocytePopulation::device_begin() {
#if defined(ASTROSIMGPU_CUDA)
    if (size() > 0) {
        cuda_ = cuda_astro_create(size(), Ca_.data(), IP3_.data(), h_.data(), Ca_tot_.data(),
                                  IP3_0_.data(), tau_IP3_.data(), delta_IP3_.data());
    }
#endif
}

void AstrocytePopulation::device_end() {
#if defined(ASTROSIMGPU_CUDA)
    if (cuda_ != nullptr) {
        cuda_astro_destroy(cuda_, Ca_.data(), IP3_.data(), h_.data());
        cuda_ = nullptr;
    }
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
#endif
}

void AstrocytePopulation::device_pull_calcium() {
#if defined(ASTROSIMGPU_CUDA)
    if (cuda_ != nullptr) {
        cuda_astro_pull_calcium(cuda_, Ca_.data());
    }
#endif
}

void AstrocytePopulation::drive_device(std::int64_t step, std::uint64_t seed, real lambda,
                                        real weight) {
#if defined(ASTROSIMGPU_CUDA)
    if (cuda_ != nullptr) {
        cuda_astro_drive_input(cuda_, seed, step, lambda, weight);
    }
#else
    (void)step;
    (void)seed;
    (void)lambda;
    (void)weight;
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

    // Held constant across noise_dt, matching NEST's noise_generator: all
    // cells change at the same instants, each with its own draw.
    const int steps_per_noise =
        std::max(1, static_cast<int>(std::llround(input_.noise_dt / time.dt)));
    const auto noise_index = static_cast<std::uint64_t>(step / steps_per_noise);
    const std::uint64_t noise_seed = seed ^ 0x9E3779B97F4A7C15ULL;
    const real noise_std = input_.gauss_noise_std;
    const bool independent = input_.independent_noise;
    const real shared_noise =
        (noise_std > 0.0 && !independent) ? noise_std * rng_normal(noise_seed, noise_index) : 0.0;

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

#pragma omp parallel for schedule(static)
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
    return astro_sic_factor(Ca_[cell], p_.SIC_th, p_.SIC_scale);
}

}  // namespace astrosimgpu

#include "astrosimgpu/neuron.hpp"

#include <algorithm>
#include <cmath>

namespace astrosimgpu {

namespace {

constexpr real kE = 2.7182818284590452353602874713527;

real draw_scaled(real mean, const RandomizeSpec& spec, CounterRng& rng) {
    if (!spec.enabled || spec.var <= 0.0) {
        return mean;
    }
    const real lo = mean < 0.0 ? mean * spec.upper : mean * spec.lower;
    const real hi = mean < 0.0 ? mean * spec.lower : mean * spec.upper;
    return rng.normal_redraw(mean, spec.var * std::abs(mean), lo, hi);
}

}  // namespace

void NeuronPopulation::build(index_t exc_count, index_t inh_count, const NeuronParams& exc,
                             const NeuronParams& inh, const RandomizeSpec& randomize,
                             const InputParams& in_exc, const InputParams& in_inh,
                             CounterRng& rng) {
    exc_count_ = exc_count;
    inh_count_ = inh_count;
    input_exc_ = in_exc;
    input_inh_ = in_inh;

    const index_t total = exc_count + inh_count;
    p_.resize(total);
    V_.resize(total);
    w_.assign(total, 0.0);
    g_ex_.assign(total, 0.0);
    dg_ex_.assign(total, 0.0);
    g_in_.assign(total, 0.0);
    dg_in_.assign(total, 0.0);
    I_sic_.assign(total, 0.0);
    refractory_steps_.assign(total, 0);
    exc_input_.assign(total, 0.0);
    inh_input_.assign(total, 0.0);
    psc_init_ex_.resize(total);
    psc_init_in_.resize(total);

    for (index_t i = 0; i < total; ++i) {
        const NeuronParams& src = (i < exc_count) ? exc : inh;
        CellParams cp{};
        cp.C_m = src.C_m;
        cp.g_L = src.g_L;
        cp.E_L = src.E_L;
        cp.V_th = src.V_th;
        cp.Delta_T = src.Delta_T;
        cp.a = src.a;
        cp.tau_w = src.tau_w;
        cp.V_peak = src.V_peak;
        cp.t_ref = src.t_ref;
        cp.E_ex = src.E_ex;
        cp.E_in = src.E_in;
        cp.tau_syn_ex = src.tau_syn_ex;
        cp.tau_syn_in = src.tau_syn_in;
        cp.I_e = src.I_e;

        // The use case randomises the reset potential and the spike-triggered
        // adaptation; everything else is shared within a population.
        cp.V_reset = draw_scaled(src.V_reset, randomize, rng);
        cp.b = draw_scaled(src.b, randomize, rng);

        p_[i] = cp;
        V_[i] = src.V_m_init;
        psc_init_ex_[i] = kE / cp.tau_syn_ex;
        psc_init_in_[i] = kE / cp.tau_syn_in;
    }
}

void NeuronPopulation::add_synaptic_input(index_t cell, real weight) {
    if (weight > 0.0) {
        exc_input_[cell] += weight;
    } else if (weight < 0.0) {
        inh_input_[cell] -= weight;
    }
}

void NeuronPopulation::derivatives(index_t cell, real V, real w, real g_ex, real g_in, real I_ext,
                                   real& dV, real& dw) const {
    const CellParams& p = p_[cell];

    // Clamp before the exponential so a neuron that has crossed the peak in
    // the middle of a substep cannot overflow before the reset is applied.
    const real Vc = std::min(V, p.V_peak);

    const real I_spike = p.g_L * p.Delta_T * std::exp((Vc - p.V_th) / p.Delta_T);
    const real I_syn_ex = g_ex * (Vc - p.E_ex);
    const real I_syn_in = g_in * (Vc - p.E_in);

    dV = (-p.g_L * (Vc - p.E_L) + I_spike - I_syn_ex - I_syn_in - w + p.I_e + I_ext) / p.C_m;
    dw = (p.a * (Vc - p.E_L) - w) / p.tau_w;
}

void NeuronPopulation::update(const TimeGrid& time, std::int64_t step, std::uint64_t seed,
                              vec<Spike>& out) {
    const index_t n = size();
    const real h_step = time.h();
    const real dt = time.dt;

    // The injected current is piecewise constant over noise_dt; every target
    // draws its own value, but all of them change at the same instants.
    const int steps_per_noise_exc =
        std::max(1, static_cast<int>(std::llround(input_exc_.noise_dt / dt)));
    const int steps_per_noise_inh =
        std::max(1, static_cast<int>(std::llround(input_inh_.noise_dt / dt)));
    const auto noise_index_exc = static_cast<std::uint64_t>(step / steps_per_noise_exc);
    const auto noise_index_inh = static_cast<std::uint64_t>(step / steps_per_noise_inh);

    real noise_exc = 0.0;
    real noise_inh = 0.0;
    if (input_exc_.gauss_noise_std > 0.0 && !input_exc_.independent_noise) {
        CounterRng r(seed ^ 0xD1B54A32D192ED03ULL, noise_index_exc);
        noise_exc = r.normal(0.0, input_exc_.gauss_noise_std);
    }
    if (input_inh_.gauss_noise_std > 0.0 && !input_inh_.independent_noise) {
        CounterRng r(seed ^ 0xA24BAED4963EE407ULL, noise_index_inh);
        noise_inh = r.normal(0.0, input_inh_.gauss_noise_std);
    }

    // Spikes are collected per thread and merged afterwards, then sorted by
    // source, so the emitted order does not depend on how the loop was
    // scheduled and a run stays reproducible at any thread count.
    vec<Spike> local;
    local.reserve(64);

#pragma omp parallel
    {
        vec<Spike> thread_spikes;
#pragma omp for schedule(static) nowait
        for (std::int64_t idx = 0; idx < static_cast<std::int64_t>(n); ++idx) {
            const auto cell = static_cast<index_t>(idx);
            const CellParams& p = p_[cell];
            const bool excitatory = cell < exc_count_;
            const InputParams& in = excitatory ? input_exc_ : input_inh_;

            // Independent Poisson drive, one realisation per target.
            if (in.poiss_rate > 0.0) {
                CounterRng r(seed ^ 0x2545F4914F6CDD1DULL,
                             static_cast<std::uint64_t>(step) * 1000003ULL + cell);
                const real lambda = in.poiss_rate * dt * 1e-3;  // Hz * ms -> events
                const int events = r.poisson(lambda);
                if (events > 0) {
                    exc_input_[cell] += in.poiss_weight * static_cast<real>(events);
                }
            }

            real I_ext = I_sic_[cell];
            if (in.gauss_noise_std > 0.0) {
                if (in.independent_noise) {
                    const std::uint64_t idx = excitatory ? noise_index_exc : noise_index_inh;
                    CounterRng r(seed ^ 0x9E3779B185EBCA87ULL, idx * 1000033ULL + cell);
                    I_ext += r.normal(0.0, in.gauss_noise_std);
                } else {
                    I_ext += excitatory ? noise_exc : noise_inh;
                }
            }

            // Arriving spikes step the alpha cascade.
            if (exc_input_[cell] != 0.0) {
                dg_ex_[cell] += exc_input_[cell] * psc_init_ex_[cell];
                exc_input_[cell] = 0.0;
            }
            if (inh_input_[cell] != 0.0) {
                dg_in_[cell] += inh_input_[cell] * psc_init_in_[cell];
                inh_input_[cell] = 0.0;
            }

            real V = V_[cell];
            real w = w_[cell];
            real g_ex = g_ex_[cell];
            real dg_ex = dg_ex_[cell];
            real g_in = g_in_[cell];
            real dg_in = dg_in_[cell];
            bool spiked = false;

            for (int s = 0; s < time.substeps; ++s) {
                // The conductance cascade is linear and independent of V, so
                // it is propagated exactly rather than through the RK stages.
                const real ex_decay = std::exp(-h_step / p.tau_syn_ex);
                const real in_decay = std::exp(-h_step / p.tau_syn_in);
                const real g_ex_next = ex_decay * (g_ex + h_step * dg_ex);
                const real dg_ex_next = ex_decay * dg_ex;
                const real g_in_next = in_decay * (g_in + h_step * dg_in);
                const real dg_in_next = in_decay * dg_in;

                if (refractory_steps_[cell] > 0) {
                    V = p.V_reset;
                    // Adaptation keeps evolving while the neuron is clamped.
                    real dV, dw;
                    derivatives(cell, V, w, g_ex, g_in, I_ext, dV, dw);
                    w += h_step * dw;
                } else {
                    const real g_ex_mid = 0.5 * (g_ex + g_ex_next);
                    const real g_in_mid = 0.5 * (g_in + g_in_next);

                    real k1V, k1w, k2V, k2w, k3V, k3w, k4V, k4w;
                    derivatives(cell, V, w, g_ex, g_in, I_ext, k1V, k1w);
                    derivatives(cell, V + 0.5 * h_step * k1V, w + 0.5 * h_step * k1w, g_ex_mid,
                                g_in_mid, I_ext, k2V, k2w);
                    derivatives(cell, V + 0.5 * h_step * k2V, w + 0.5 * h_step * k2w, g_ex_mid,
                                g_in_mid, I_ext, k3V, k3w);
                    derivatives(cell, V + h_step * k3V, w + h_step * k3w, g_ex_next, g_in_next,
                                I_ext, k4V, k4w);

                    V += (h_step / 6.0) * (k1V + 2.0 * k2V + 2.0 * k3V + k4V);
                    w += (h_step / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
                }

                g_ex = g_ex_next;
                dg_ex = dg_ex_next;
                g_in = g_in_next;
                dg_in = dg_in_next;

                if (refractory_steps_[cell] > 0) {
                    --refractory_steps_[cell];
                } else if (V >= p.V_peak) {
                    V = p.V_reset;
                    w += p.b;
                    const int ref = static_cast<int>(p.t_ref / h_step + 0.5);
                    refractory_steps_[cell] = ref;
                    spiked = true;
                }
            }

            V_[cell] = V;
            w_[cell] = w;
            g_ex_[cell] = g_ex;
            dg_ex_[cell] = dg_ex;
            g_in_[cell] = g_in;
            dg_in_[cell] = dg_in;

            if (spiked) {
                thread_spikes.push_back(Spike{step, cell});
            }
        }
#pragma omp critical
        { local.insert(local.end(), thread_spikes.begin(), thread_spikes.end()); }
    }

    std::sort(local.begin(), local.end(),
              [](const Spike& a, const Spike& b) { return a.source < b.source; });
    out.insert(out.end(), local.begin(), local.end());
}

}  // namespace astrosimgpu

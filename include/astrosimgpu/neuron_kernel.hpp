#pragma once

#include <cmath>

#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

// Every backend needs the per-cell functions marked differently. Bodies are
// identical in all cases. Same idiom as astrocyte_kernel.hpp's
// ASTROSIMGPU_FN, kept as an independent definition here (rather than
// including that header) so this file stays narrow -- rng.hpp does the same
// thing under its own name, ASTROSIMGPU_RNG_FN.
#if defined(__CUDACC__)
#define ASTROSIMGPU_FN __host__ __device__ inline
#else
#define ASTROSIMGPU_FN inline
#endif

// Parameters are stored per cell rather than per population: the two
// populations differ in every field, and V_reset/b are randomised. Free of
// member state so it can be handed to a device kernel as a plain array.
struct CellParams {
    real C_m, g_L, E_L, V_th, Delta_T, a, b, tau_w;
    real V_reset, V_peak, t_ref, E_ex, E_in;
    real tau_syn_ex, tau_syn_in, I_e;
};

// Per-cell update, free of member state: no `this`, no std::vector, no
// reference into a class. That is what makes it callable from a device
// kernel. Host and device call the same function so they cannot drift apart.

// Right-hand side of the AdEx system.
ASTROSIMGPU_FN void neuron_derivatives(const CellParams& p, real V, real w, real g_ex, real g_in,
                                       real I_ext, real& dV, real& dw) {
    // Clamp before the exponential so a neuron that has crossed the peak in
    // the middle of a substep cannot overflow before the reset is applied.
    // Written as a ternary rather than std::min: the latter is not reliably
    // device-callable under nvcc, unlike the single-function std::cmath
    // calls below (already relied on by rng_normal/rng_poisson).
    const real Vc = (V < p.V_peak) ? V : p.V_peak;

    const real I_spike = p.g_L * p.Delta_T * std::exp((Vc - p.V_th) / p.Delta_T);
    const real I_syn_ex = g_ex * (Vc - p.E_ex);
    const real I_syn_in = g_in * (Vc - p.E_in);

    dV = (-p.g_L * (Vc - p.E_L) + I_spike - I_syn_ex - I_syn_in - w + p.I_e + I_ext) / p.C_m;
    dw = (p.a * (Vc - p.E_L) - w) / p.tau_w;
}

// One cell, one communication step (substeps RK4 stages inside). exc_input/
// inh_input are this step's already-drawn synaptic accumulators (Poisson
// background and independent noise are drawn by the caller, not here, same
// convention as astro_advance not drawing its own noise) -- taken by value,
// since clearing the host/device accumulator back to zero is the caller's
// responsibility, exactly as astro_advance leaves ip3_input's clearing to
// its callers. Returns true if the cell spiked during this step.
ASTROSIMGPU_FN bool neuron_advance(const CellParams& p, real h_step, int substeps, real exc_input,
                                   real inh_input, real psc_init_ex, real psc_init_in, real I_ext,
                                   real& V, real& w, real& g_ex, real& dg_ex, real& g_in,
                                   real& dg_in, int& refractory_steps) {
    // Arriving spikes step the alpha cascade.
    if (exc_input != 0.0) {
        dg_ex += exc_input * psc_init_ex;
    }
    if (inh_input != 0.0) {
        dg_in += inh_input * psc_init_in;
    }

    bool spiked = false;
    for (int s = 0; s < substeps; ++s) {
        // The conductance cascade is linear and independent of V, so it is
        // propagated exactly rather than through the RK stages.
        const real ex_decay = std::exp(-h_step / p.tau_syn_ex);
        const real in_decay = std::exp(-h_step / p.tau_syn_in);
        const real g_ex_next = ex_decay * (g_ex + h_step * dg_ex);
        const real dg_ex_next = ex_decay * dg_ex;
        const real g_in_next = in_decay * (g_in + h_step * dg_in);
        const real dg_in_next = in_decay * dg_in;

        if (refractory_steps > 0) {
            V = p.V_reset;
            // Adaptation keeps evolving while the neuron is clamped.
            real dV, dw;
            neuron_derivatives(p, V, w, g_ex, g_in, I_ext, dV, dw);
            w += h_step * dw;
        } else {
            const real g_ex_mid = 0.5 * (g_ex + g_ex_next);
            const real g_in_mid = 0.5 * (g_in + g_in_next);

            real k1V, k1w, k2V, k2w, k3V, k3w, k4V, k4w;
            neuron_derivatives(p, V, w, g_ex, g_in, I_ext, k1V, k1w);
            neuron_derivatives(p, V + 0.5 * h_step * k1V, w + 0.5 * h_step * k1w, g_ex_mid,
                               g_in_mid, I_ext, k2V, k2w);
            neuron_derivatives(p, V + 0.5 * h_step * k2V, w + 0.5 * h_step * k2w, g_ex_mid,
                               g_in_mid, I_ext, k3V, k3w);
            neuron_derivatives(p, V + h_step * k3V, w + h_step * k3w, g_ex_next, g_in_next, I_ext,
                               k4V, k4w);

            V += (h_step / 6.0) * (k1V + 2.0 * k2V + 2.0 * k3V + k4V);
            w += (h_step / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
        }

        g_ex = g_ex_next;
        dg_ex = dg_ex_next;
        g_in = g_in_next;
        dg_in = dg_in_next;

        if (refractory_steps > 0) {
            --refractory_steps;
        } else if (V >= p.V_peak) {
            V = p.V_reset;
            w += p.b;
            refractory_steps = static_cast<int>(p.t_ref / h_step + 0.5);
            spiked = true;
        }
    }
    return spiked;
}

}  // namespace astrosimgpu

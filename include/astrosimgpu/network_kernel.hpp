#pragma once

#include <cmath>

#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

// Every backend needs the per-synapse function marked differently. Body is
// identical in all cases. Independent copy of the same macro ladder as
// astrocyte_kernel.hpp/neuron_kernel.hpp/rng.hpp, rather than a cross-include,
// so this file stays narrow.
#if defined(ASTROSIMGPU_KOKKOS)
#define ASTROSIMGPU_FN KOKKOS_INLINE_FUNCTION
#elif defined(__CUDACC__)
#define ASTROSIMGPU_FN __host__ __device__ inline
#else
#define ASTROSIMGPU_FN inline
#endif

// Tsodyks-Markram short-term plasticity, extracted from Network::stp_weight
// so the host delivery path and the CUDA deliver_spikes/deliver_sic kernels
// call the same function and cannot drift apart -- same rationale as
// astro_advance/neuron_advance. Named distinctly from Network::stp_weight
// (not "stp_weight"): an unqualified call to that name from inside
// Network::stp_weight's own body would resolve to the member itself
// (class-scope lookup hides the free function of the same name), not this
// one.
//
// When `enabled` is false, x/u/t_last are never read or written, so callers
// may pass any well-formed reference (e.g. one from an empty backing array
// is never reached).
ASTROSIMGPU_FN real synapse_stp_weight(bool enabled, real weight, real U, real tau_rec,
                                       real tau_fac, real t_now, real& x, real& u,
                                       real& t_last) {
    if (!enabled) {
        return weight;
    }
    const real dt = t_now - t_last;
    const real x_decay = std::exp(-dt / tau_rec);
    const real u_decay = tau_fac < 1e-10 ? 0.0 : std::exp(-dt / tau_fac);

    x = 1.0 + (x - x * u - 1.0) * x_decay;
    u = U + u * (1.0 - U) * u_decay;
    t_last = t_now;

    return weight * x * u;
}

}  // namespace astrosimgpu

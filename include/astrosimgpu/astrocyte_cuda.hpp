#pragma once

#include <cstdint>

#include "astrosimgpu/astrocyte_kernel.hpp"
#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

// Narrow and free of CUDA types, so only astrocyte_cuda.cu goes through nvcc.
// The four movement points match the other device backends exactly; had they
// differed, comparing the backends would measure the transfer pattern.
struct CudaAstro;

CudaAstro* cuda_astro_create(index_t n, const real* Ca, const real* IP3, const real* h,
                             const real* Ca_tot, const real* IP3_0, const real* tau_IP3,
                             const real* delta_IP3);

void cuda_astro_destroy(CudaAstro* state, real* Ca, real* IP3, real* h);

void cuda_astro_push_input(CudaAstro* state, const real* ip3_input);

void cuda_astro_pull_calcium(CudaAstro* state, real* Ca);

// Generates this step's background Poisson drive on the device and adds it
// into the resident input buffer (does not overwrite -- device_push_input's
// contribution must already be there). One thread per astrocyte, same
// (seed, step, cell) -> stream mapping as the host drive_astrocytes loop, so
// results match exactly (see rng_poisson in rng.hpp).
void cuda_astro_drive_input(CudaAstro* state, std::uint64_t seed, std::int64_t step, real lambda,
                            real weight);

void cuda_astro_update(CudaAstro* state, const AstroConstants& c, real h_step, int substeps,
                       real noise_std, bool independent_noise, real shared_noise,
                       std::uint64_t noise_seed, std::uint64_t noise_index);

}  // namespace astrosimgpu

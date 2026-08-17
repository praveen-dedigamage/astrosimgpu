#pragma once

#include <cstdint>

#include "astrosimgpu/astrocyte_kernel.hpp"
#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

/// Interface to the native CUDA backend.
///
/// Deliberately narrow and free of CUDA types, so `astrocyte.cpp` stays plain
/// C++ and only `astrocyte_cuda.cu` is compiled by nvcc. The state handle is
/// opaque for the same reason.
///
/// The four data-movement points match the OpenMP target and Kokkos backends
/// exactly: allocate and copy in when a run begins, push the synaptic input
/// each step, pull calcium back each step, copy out at the end. Had they
/// differed, a comparison between the backends would measure the transfer
/// pattern rather than the dispatch.
struct CudaAstro;

/// Allocate device buffers and copy the initial state across.
CudaAstro* cuda_astro_create(index_t n, const real* Ca, const real* IP3, const real* h,
                             const real* Ca_tot, const real* IP3_0, const real* tau_IP3,
                             const real* delta_IP3);

/// Copy the final state back and release the buffers.
void cuda_astro_destroy(CudaAstro* state, real* Ca, real* IP3, real* h);

/// Send this step's synaptic input to the device.
void cuda_astro_push_input(CudaAstro* state, const real* ip3_input);

/// Retrieve calcium, which the host needs in order to deliver the SIC.
void cuda_astro_pull_calcium(CudaAstro* state, real* Ca);

/// Advance every astrocyte by one communication step.
void cuda_astro_update(CudaAstro* state, const AstroConstants& c, real h_step, int substeps,
                       real noise_std, bool independent_noise, real shared_noise,
                       std::uint64_t noise_seed, std::uint64_t noise_index);

}  // namespace astrosimgpu

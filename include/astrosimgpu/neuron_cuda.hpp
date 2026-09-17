#pragma once

#include <cstdint>

#include "astrosimgpu/neuron_kernel.hpp"
#include "astrosimgpu/parameters.hpp"
#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

// Narrow and free of CUDA types, so only neuron_cuda.cu goes through nvcc.
// The movement points match AstrocytePopulation's CUDA backend: state
// (V, w, g_ex, dg_ex, g_in, dg_in, refractory_steps) and the per-cell
// parameters stay resident for the whole run; only the per-step synaptic
// input (in) and the spike flags (out) move every step.
struct CudaNeuron;

CudaNeuron* cuda_neuron_create(index_t n, index_t exc_count, const CellParams* p, const real* V,
                               const real* w, const real* g_ex, const real* dg_ex,
                               const real* g_in, const real* dg_in, const int* refractory_steps,
                               const real* psc_init_ex, const real* psc_init_in);

void cuda_neuron_destroy(CudaNeuron* state, real* V, real* w, real* g_ex, real* dg_ex,
                         real* g_in, real* dg_in, int* refractory_steps);

// This step's synaptic drive: exc_input/inh_input from apply_arrivals'
// ring-buffer delivery, I_sic from the astrocyte SIC pathway.
void cuda_neuron_push_input(CudaNeuron* state, const real* exc_input, const real* inh_input,
                            const real* I_sic);

// One byte per cell, 1 if that cell spiked this step, 0 otherwise -- always
// written for every cell (see neuron_cuda.cu), not just on a spike, so the
// host can reconstruct spike_buffer_ with a single pass and no stale flags
// from a previous step. A dense flag array rather than a device-side
// compacted list: no atomics in the hot path, matching every other kernel
// in this codebase.
void cuda_neuron_pull_spikes(CudaNeuron* state, unsigned char* out_spiked);

// `step`/`seed` seed the Poisson background draw and the independent-noise
// draw exactly as Network::run's host path does; shared_noise_{exc,inh} and
// noise_index_{exc,inh} are computed once on the host per step (same
// CounterRng calls the host path already makes) and passed in, mirroring
// how AstrocytePopulation::update computes shared_noise on the host before
// calling cuda_astro_update.
void cuda_neuron_update(CudaNeuron* state, real h_step, int substeps, real dt,
                        std::uint64_t seed, std::int64_t step, const InputParams& in_exc,
                        const InputParams& in_inh, real shared_noise_exc,
                        std::uint64_t noise_index_exc, real shared_noise_inh,
                        std::uint64_t noise_index_inh);

}  // namespace astrosimgpu

#pragma once

#include <cstdint>

#include "astrosimgpu/network_kernel.hpp"
#include "astrosimgpu/parameters.hpp"
#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

// Narrow and free of CUDA types, so only network_cuda.cu goes through nvcc.
// Owns the delivery-phase resident state Stage 4 adds: the four
// ConnectionSets' CSR arrays, the four ring buffers, and the two precomputed
// index lists (sic_sources_, astro_input_sinks_) -- everything Network::build()
// already computes once and previously kept host-side. It does NOT own
// neuron or astrocyte state; those stay CudaNeuron's/CudaAstro's, reached
// through the device_* accessors declared in neuron_cuda.hpp/astrocyte_cuda.hpp.
struct CudaDelivery;

CudaDelivery* cuda_delivery_create(
    index_t n_neurons, index_t n_astro, index_t n_exc, int ring_slots,
    // exc_primary_: excitatory neuron -> neuron
    const index_t* exc_row_start, const index_t* exc_target, const real* exc_weight,
    const int* exc_delay_steps, const real* exc_stp_x, const real* exc_stp_u,
    const real* exc_stp_t_last, index_t exc_synapses, index_t exc_stp_synapses,
    // inh_primary_: inhibitory neuron -> neuron
    const index_t* inh_row_start, const index_t* inh_target, const real* inh_weight,
    const int* inh_delay_steps, const real* inh_stp_x, const real* inh_stp_u,
    const real* inh_stp_t_last, index_t inh_synapses, index_t inh_stp_synapses,
    // neuron_astro_: excitatory neuron -> astrocyte
    const index_t* na_row_start, const index_t* na_target, const real* na_weight,
    const int* na_delay_steps, const real* na_stp_x, const real* na_stp_u,
    const real* na_stp_t_last, index_t na_synapses, index_t na_stp_synapses,
    // astro_neuron_: astrocyte -> neuron (SIC), no plasticity -- no stp_* arrays
    const index_t* an_row_start, const index_t* an_target, const real* an_weight,
    const int* an_delay_steps, index_t an_synapses,
    // precomputed once in Network::build(); fixed for the whole run
    const index_t* sic_sources, index_t n_sic_sources, const index_t* astro_input_sinks,
    index_t n_astro_input_sinks);

void cuda_delivery_destroy(CudaDelivery* state);

// One thread per neuron (n_neurons, same fixed grid as neuron_update_kernel).
// `spiked` is cuda_neuron_device_spiked's pointer -- read directly, no host
// round trip and no spike-count-sized launch.
void cuda_delivery_spikes(CudaDelivery* state, std::int64_t step, real t_now,
                          const StpParams& stp, const unsigned char* spiked);

// One thread per entry in sic_sources_ (fixed at build time, not per-step
// data-dependent -- unlike spike count, this list never changes size during
// a run). `Ca` is cuda_astro_device_calcium's pointer, read directly.
void cuda_delivery_sic(CudaDelivery* state, std::int64_t step, real SIC_th, real SIC_scale,
                       const real* Ca);

// Applies the current ring slot's contents into the neuron/astrocyte input
// buffers, writing directly into pointers owned by CudaNeuron/CudaAstro
// (obtained via their device_* accessors) -- CudaDelivery never copies them.
// Replaces cuda_neuron_push_input and astro_'s device_push_input for the
// resident path; those two functions remain in place for every other backend.
void cuda_delivery_apply_arrivals(CudaDelivery* state, std::int64_t step, bool sic_arrives,
                                  real* neuron_exc_input, real* neuron_inh_input,
                                  real* neuron_I_sic, real* astro_ip3_input);

}  // namespace astrosimgpu

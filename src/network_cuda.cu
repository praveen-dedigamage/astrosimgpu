// Native CUDA backend for the delivery phases (Stage 4 of docs/gpu-port.md).
// Kernel bodies call synapse_stp_weight/astro_sic_factor, the same functions
// the host deliver_spikes/deliver_sic/Network::stp_weight/sic_factor call --
// ASTROSIMGPU_FN expands to __host__ __device__ under nvcc, same convention
// as astrocyte_cuda.cu/neuron_cuda.cu. Only the launches, the topology/ring
// residency, and the atomics live here.

#include <cstdio>
#include <cstdlib>

#include "astrosimgpu/astrocyte_kernel.hpp"
#include "astrosimgpu/network_cuda.hpp"

namespace astrosimgpu {

namespace {

void check(cudaError_t err, const char* what, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d during %s: %s\n", file, line, what,
                     cudaGetErrorString(err));
        std::abort();
    }
}

#define CUDA_CHECK(call) check((call), #call, __FILE__, __LINE__)

// Generic version of the real*-only device_copy in astrocyte_cuda.cu/
// neuron_cuda.cu: this file also copies index_t and int CSR arrays. `n` is
// always an explicit element count from the caller, not inferred from
// whether `host` looks null -- std::vector::data() on an empty vector is not
// guaranteed to return nullptr, so a disabled-STP zero-length array must be
// signalled by passing n=0, not by hoping the pointer is null.
template <typename T>
T* device_copy(const T* host, std::size_t n) {
    if (n == 0) {
        return nullptr;
    }
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, host, n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// One thread per neuron -- same fixed grid as neuron_update_kernel. Each
// thread reads its OWN slot of the device-resident spiked array (no host
// round trip, no spike-count-sized launch). Weights are stored per synapse;
// STP state (x, u, t_last) is mutated in place, mirroring the host
// deliver_spikes exactly. stp_x/stp_u/stp_t_last are nullptr when STP is
// disabled -- never dereferenced in that case, since synapse_stp_weight's
// `enabled=false` branch returns before touching them.
__global__ void deliver_spikes_kernel(
    index_t n_neurons, index_t n_exc, index_t n_astro, std::int64_t step, real t_now,
    int ring_slots, StpParams stp, const unsigned char* __restrict__ spiked,
    const index_t* __restrict__ exc_row_start, const index_t* __restrict__ exc_target,
    const real* __restrict__ exc_weight, const int* __restrict__ exc_delay_steps,
    real* __restrict__ exc_stp_x, real* __restrict__ exc_stp_u, real* __restrict__ exc_stp_t_last,
    const index_t* __restrict__ inh_row_start, const index_t* __restrict__ inh_target,
    const real* __restrict__ inh_weight, const int* __restrict__ inh_delay_steps,
    real* __restrict__ inh_stp_x, real* __restrict__ inh_stp_u, real* __restrict__ inh_stp_t_last,
    const index_t* __restrict__ na_row_start, const index_t* __restrict__ na_target,
    const real* __restrict__ na_weight, const int* __restrict__ na_delay_steps,
    real* __restrict__ na_stp_x, real* __restrict__ na_stp_u, real* __restrict__ na_stp_t_last,
    real* __restrict__ ring_exc, real* __restrict__ ring_inh, real* __restrict__ ring_astro) {
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons || spiked[i] == 0) {
        return;
    }

    if (i < n_exc) {
        const index_t src = i;
        for (index_t k = exc_row_start[src]; k < exc_row_start[src + 1]; ++k) {
            const int slot = static_cast<int>((step + exc_delay_steps[k]) % ring_slots);
            real dummy_x = 0.0, dummy_u = 0.0, dummy_t = 0.0;
            const real w = synapse_stp_weight(stp.enabled, exc_weight[k], stp.U, stp.tau_rec,
                                              stp.tau_fac, t_now,
                                              stp.enabled ? exc_stp_x[k] : dummy_x,
                                              stp.enabled ? exc_stp_u[k] : dummy_u,
                                              stp.enabled ? exc_stp_t_last[k] : dummy_t);
            atomicAdd(&ring_exc[static_cast<std::size_t>(slot) * n_neurons + exc_target[k]], w);
        }
        for (index_t k = na_row_start[src]; k < na_row_start[src + 1]; ++k) {
            const int slot = static_cast<int>((step + na_delay_steps[k]) % ring_slots);
            real dummy_x = 0.0, dummy_u = 0.0, dummy_t = 0.0;
            const real w = synapse_stp_weight(stp.enabled, na_weight[k], stp.U, stp.tau_rec,
                                              stp.tau_fac, t_now,
                                              stp.enabled ? na_stp_x[k] : dummy_x,
                                              stp.enabled ? na_stp_u[k] : dummy_u,
                                              stp.enabled ? na_stp_t_last[k] : dummy_t);
            atomicAdd(&ring_astro[static_cast<std::size_t>(slot) * n_astro + na_target[k]], w);
        }
    } else {
        const index_t src = i - n_exc;
        for (index_t k = inh_row_start[src]; k < inh_row_start[src + 1]; ++k) {
            const int slot = static_cast<int>((step + inh_delay_steps[k]) % ring_slots);
            real dummy_x = 0.0, dummy_u = 0.0, dummy_t = 0.0;
            const real w = synapse_stp_weight(stp.enabled, inh_weight[k], stp.U, stp.tau_rec,
                                              stp.tau_fac, t_now,
                                              stp.enabled ? inh_stp_x[k] : dummy_x,
                                              stp.enabled ? inh_stp_u[k] : dummy_u,
                                              stp.enabled ? inh_stp_t_last[k] : dummy_t);
            // Weights are negative; the ring stores magnitudes -- matches
            // the host's `ring_inh_[slot][...] -= w`.
            atomicAdd(&ring_inh[static_cast<std::size_t>(slot) * n_neurons + inh_target[k]], -w);
        }
    }
}

// One thread per entry of sic_sources_ (fixed size, computed once).
__global__ void deliver_sic_kernel(index_t n_sic_sources, std::int64_t step, int ring_slots,
                                   real SIC_th, real SIC_scale, index_t n_neurons,
                                   const index_t* __restrict__ sic_sources,
                                   const real* __restrict__ Ca,
                                   const index_t* __restrict__ an_row_start,
                                   const index_t* __restrict__ an_target,
                                   const real* __restrict__ an_weight,
                                   const int* __restrict__ an_delay_steps,
                                   real* __restrict__ ring_sic) {
    const index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sic_sources) {
        return;
    }
    const index_t a = sic_sources[idx];
    const real factor = astro_sic_factor(Ca[a], SIC_th, SIC_scale);
    if (factor == 0.0) {
        return;
    }
    for (index_t k = an_row_start[a]; k < an_row_start[a + 1]; ++k) {
        const int slot = static_cast<int>((step + an_delay_steps[k]) % ring_slots);
        const real contribution = an_weight[k] * factor;
        atomicAdd(&ring_sic[static_cast<std::size_t>(slot) * n_neurons + an_target[k]],
                  contribution);
    }
}

// One thread per neuron. Writes directly into CudaNeuron's device-resident
// exc_input/inh_input/I_sic -- this is what removes cuda_neuron_push_input
// from the hot path entirely.
__global__ void apply_arrivals_neuron_kernel(index_t n_neurons, int slot, bool sic_arrives,
                                             real* __restrict__ ring_exc,
                                             real* __restrict__ ring_inh,
                                             real* __restrict__ ring_sic,
                                             real* __restrict__ exc_input,
                                             real* __restrict__ inh_input,
                                             real* __restrict__ I_sic) {
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) {
        return;
    }
    real* ex = ring_exc + static_cast<std::size_t>(slot) * n_neurons;
    real* in = ring_inh + static_cast<std::size_t>(slot) * n_neurons;
    real* sic = ring_sic + static_cast<std::size_t>(slot) * n_neurons;
    if (ex[i] != 0.0) {
        exc_input[i] += ex[i];
        ex[i] = 0.0;
    }
    if (in[i] != 0.0) {
        inh_input[i] += in[i];
        in[i] = 0.0;
    }
    if (sic_arrives) {
        I_sic[i] = sic[i];
        sic[i] = 0.0;
    }
}

// One thread per entry of astro_input_sinks_ (fixed, deduplicated at build
// time -- each astrocyte index appears at most once, so no atomic is needed
// here, matching the host loop which is plain serial for the same reason).
__global__ void apply_arrivals_astro_kernel(index_t n_sinks, int slot, index_t n_astro,
                                            const index_t* __restrict__ astro_input_sinks,
                                            real* __restrict__ ring_astro,
                                            real* __restrict__ ip3_input) {
    const index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sinks) {
        return;
    }
    const index_t a = astro_input_sinks[idx];
    real* slot_ptr = ring_astro + static_cast<std::size_t>(slot) * n_astro;
    if (slot_ptr[a] != 0.0) {
        ip3_input[a] += slot_ptr[a];
        slot_ptr[a] = 0.0;
    }
}

}  // namespace

struct CudaDelivery {
    index_t n_neurons = 0, n_astro = 0, n_exc = 0;
    int ring_slots = 1;

    index_t *exc_row_start = nullptr, *exc_target = nullptr;
    real *exc_weight = nullptr, *exc_stp_x = nullptr, *exc_stp_u = nullptr,
         *exc_stp_t_last = nullptr;
    int* exc_delay_steps = nullptr;

    index_t *inh_row_start = nullptr, *inh_target = nullptr;
    real *inh_weight = nullptr, *inh_stp_x = nullptr, *inh_stp_u = nullptr,
         *inh_stp_t_last = nullptr;
    int* inh_delay_steps = nullptr;

    index_t *na_row_start = nullptr, *na_target = nullptr;
    real *na_weight = nullptr, *na_stp_x = nullptr, *na_stp_u = nullptr, *na_stp_t_last = nullptr;
    int* na_delay_steps = nullptr;

    index_t *an_row_start = nullptr, *an_target = nullptr;
    real* an_weight = nullptr;
    int* an_delay_steps = nullptr;

    index_t* sic_sources = nullptr;
    index_t n_sic_sources = 0;
    index_t* astro_input_sinks = nullptr;
    index_t n_astro_input_sinks = 0;

    // [ring_slots * n_neurons] for exc/inh/sic, [ring_slots * n_astro] for astro.
    real *ring_exc = nullptr, *ring_inh = nullptr, *ring_sic = nullptr, *ring_astro = nullptr;
};

CudaDelivery* cuda_delivery_create(
    index_t n_neurons, index_t n_astro, index_t n_exc, int ring_slots,
    const index_t* exc_row_start, const index_t* exc_target, const real* exc_weight,
    const int* exc_delay_steps, const real* exc_stp_x, const real* exc_stp_u,
    const real* exc_stp_t_last, index_t exc_synapses, index_t exc_stp_synapses,
    const index_t* inh_row_start, const index_t* inh_target, const real* inh_weight,
    const int* inh_delay_steps, const real* inh_stp_x, const real* inh_stp_u,
    const real* inh_stp_t_last, index_t inh_synapses, index_t inh_stp_synapses,
    const index_t* na_row_start, const index_t* na_target, const real* na_weight,
    const int* na_delay_steps, const real* na_stp_x, const real* na_stp_u,
    const real* na_stp_t_last, index_t na_synapses, index_t na_stp_synapses,
    const index_t* an_row_start, const index_t* an_target, const real* an_weight,
    const int* an_delay_steps, index_t an_synapses, const index_t* sic_sources,
    index_t n_sic_sources, const index_t* astro_input_sinks, index_t n_astro_input_sinks) {
    auto* s = new CudaDelivery;
    s->n_neurons = n_neurons;
    s->n_astro = n_astro;
    s->n_exc = n_exc;
    s->ring_slots = ring_slots;

    s->exc_row_start = device_copy(exc_row_start, static_cast<std::size_t>(n_exc) + 1);
    s->exc_target = device_copy(exc_target, exc_synapses);
    s->exc_weight = device_copy(exc_weight, exc_synapses);
    s->exc_delay_steps = device_copy(exc_delay_steps, exc_synapses);
    s->exc_stp_x = device_copy(exc_stp_x, exc_stp_synapses);
    s->exc_stp_u = device_copy(exc_stp_u, exc_stp_synapses);
    s->exc_stp_t_last = device_copy(exc_stp_t_last, exc_stp_synapses);

    const index_t n_inh = n_neurons - n_exc;
    s->inh_row_start = device_copy(inh_row_start, static_cast<std::size_t>(n_inh) + 1);
    s->inh_target = device_copy(inh_target, inh_synapses);
    s->inh_weight = device_copy(inh_weight, inh_synapses);
    s->inh_delay_steps = device_copy(inh_delay_steps, inh_synapses);
    s->inh_stp_x = device_copy(inh_stp_x, inh_stp_synapses);
    s->inh_stp_u = device_copy(inh_stp_u, inh_stp_synapses);
    s->inh_stp_t_last = device_copy(inh_stp_t_last, inh_stp_synapses);

    s->na_row_start = device_copy(na_row_start, static_cast<std::size_t>(n_exc) + 1);
    s->na_target = device_copy(na_target, na_synapses);
    s->na_weight = device_copy(na_weight, na_synapses);
    s->na_delay_steps = device_copy(na_delay_steps, na_synapses);
    s->na_stp_x = device_copy(na_stp_x, na_stp_synapses);
    s->na_stp_u = device_copy(na_stp_u, na_stp_synapses);
    s->na_stp_t_last = device_copy(na_stp_t_last, na_stp_synapses);

    s->an_row_start = device_copy(an_row_start, static_cast<std::size_t>(n_astro) + 1);
    s->an_target = device_copy(an_target, an_synapses);
    s->an_weight = device_copy(an_weight, an_synapses);
    s->an_delay_steps = device_copy(an_delay_steps, an_synapses);

    s->sic_sources = device_copy(sic_sources, n_sic_sources);
    s->n_sic_sources = n_sic_sources;
    s->astro_input_sinks = device_copy(astro_input_sinks, n_astro_input_sinks);
    s->n_astro_input_sinks = n_astro_input_sinks;

    const std::size_t neuron_ring_bytes =
        static_cast<std::size_t>(ring_slots) * n_neurons * sizeof(real);
    const std::size_t astro_ring_bytes =
        static_cast<std::size_t>(ring_slots) * n_astro * sizeof(real);
    CUDA_CHECK(cudaMalloc(&s->ring_exc, neuron_ring_bytes));
    CUDA_CHECK(cudaMemset(s->ring_exc, 0, neuron_ring_bytes));
    CUDA_CHECK(cudaMalloc(&s->ring_inh, neuron_ring_bytes));
    CUDA_CHECK(cudaMemset(s->ring_inh, 0, neuron_ring_bytes));
    CUDA_CHECK(cudaMalloc(&s->ring_sic, neuron_ring_bytes));
    CUDA_CHECK(cudaMemset(s->ring_sic, 0, neuron_ring_bytes));
    CUDA_CHECK(cudaMalloc(&s->ring_astro, astro_ring_bytes));
    CUDA_CHECK(cudaMemset(s->ring_astro, 0, astro_ring_bytes));

    return s;
}

void cuda_delivery_destroy(CudaDelivery* s) {
    if (s == nullptr) {
        return;
    }
    for (void* p : {(void*)s->exc_row_start, (void*)s->exc_target, (void*)s->exc_weight,
                    (void*)s->exc_delay_steps, (void*)s->exc_stp_x, (void*)s->exc_stp_u,
                    (void*)s->exc_stp_t_last, (void*)s->inh_row_start, (void*)s->inh_target,
                    (void*)s->inh_weight, (void*)s->inh_delay_steps, (void*)s->inh_stp_x,
                    (void*)s->inh_stp_u, (void*)s->inh_stp_t_last, (void*)s->na_row_start,
                    (void*)s->na_target, (void*)s->na_weight, (void*)s->na_delay_steps,
                    (void*)s->na_stp_x, (void*)s->na_stp_u, (void*)s->na_stp_t_last,
                    (void*)s->an_row_start, (void*)s->an_target, (void*)s->an_weight,
                    (void*)s->an_delay_steps, (void*)s->sic_sources, (void*)s->astro_input_sinks,
                    (void*)s->ring_exc, (void*)s->ring_inh, (void*)s->ring_sic,
                    (void*)s->ring_astro}) {
        if (p != nullptr) {
            CUDA_CHECK(cudaFree(p));
        }
    }
    delete s;
}

void cuda_delivery_spikes(CudaDelivery* s, std::int64_t step, real t_now, const StpParams& stp,
                          const unsigned char* spiked) {
    if (s == nullptr || s->n_neurons == 0) {
        return;
    }
    constexpr int block = 128;
    const int grid = static_cast<int>((s->n_neurons + block - 1) / block);
    deliver_spikes_kernel<<<grid, block>>>(
        s->n_neurons, s->n_exc, s->n_astro, step, t_now, s->ring_slots, stp, spiked,
        s->exc_row_start, s->exc_target, s->exc_weight, s->exc_delay_steps, s->exc_stp_x,
        s->exc_stp_u, s->exc_stp_t_last, s->inh_row_start, s->inh_target, s->inh_weight,
        s->inh_delay_steps, s->inh_stp_x, s->inh_stp_u, s->inh_stp_t_last, s->na_row_start,
        s->na_target, s->na_weight, s->na_delay_steps, s->na_stp_x, s->na_stp_u, s->na_stp_t_last,
        s->ring_exc, s->ring_inh, s->ring_astro);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_delivery_sic(CudaDelivery* s, std::int64_t step, real SIC_th, real SIC_scale,
                       const real* Ca) {
    if (s == nullptr || s->n_sic_sources == 0) {
        return;
    }
    constexpr int block = 128;
    const int grid = static_cast<int>((s->n_sic_sources + block - 1) / block);
    deliver_sic_kernel<<<grid, block>>>(s->n_sic_sources, step, s->ring_slots, SIC_th, SIC_scale,
                                        s->n_neurons, s->sic_sources, Ca, s->an_row_start,
                                        s->an_target, s->an_weight, s->an_delay_steps,
                                        s->ring_sic);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_delivery_apply_arrivals(CudaDelivery* s, std::int64_t step, bool sic_arrives,
                                  real* neuron_exc_input, real* neuron_inh_input,
                                  real* neuron_I_sic, real* astro_ip3_input) {
    if (s == nullptr) {
        return;
    }
    const int slot = static_cast<int>(step % s->ring_slots);
    constexpr int block = 128;
    if (s->n_neurons > 0) {
        const int grid = static_cast<int>((s->n_neurons + block - 1) / block);
        apply_arrivals_neuron_kernel<<<grid, block>>>(s->n_neurons, slot, sic_arrives, s->ring_exc,
                                                       s->ring_inh, s->ring_sic, neuron_exc_input,
                                                       neuron_inh_input, neuron_I_sic);
        CUDA_CHECK(cudaGetLastError());
    }
    if (s->n_astro_input_sinks > 0) {
        const int grid = static_cast<int>((s->n_astro_input_sinks + block - 1) / block);
        apply_arrivals_astro_kernel<<<grid, block>>>(s->n_astro_input_sinks, slot, s->n_astro,
                                                      s->astro_input_sinks, s->ring_astro,
                                                      astro_ip3_input);
        CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace astrosimgpu

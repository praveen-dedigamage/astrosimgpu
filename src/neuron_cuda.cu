// Native CUDA backend. The kernel body is neuron_advance/neuron_derivatives,
// the same functions the host loop calls -- ASTROSIMGPU_FN expands to
// __host__ __device__ under nvcc. Only the launch, RNG draws (which cannot
// be shared with the host loop as a single function without threading seed/
// step/cell through neuron_advance's signature for no benefit, so they stay
// inlined here exactly as astro_update_kernel does for its own noise draw),
// and the memory management live here.

#include <cstdio>
#include <cstdlib>

#include "astrosimgpu/neuron_cuda.hpp"
#include "astrosimgpu/rng.hpp"

namespace astrosimgpu {

namespace {

// Abort rather than continue with wrong results. A silent failure here gives
// a run that completes and reports plausible numbers.
void check(cudaError_t err, const char* what, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d during %s: %s\n", file, line, what,
                     cudaGetErrorString(err));
        std::abort();
    }
}

#define CUDA_CHECK(call) check((call), #call, __FILE__, __LINE__)

// One neuron per thread. State stays in registers across the substeps.
// spiked is written for every cell every step (0 or 1), not just on a
// spike -- deliver_spikes' host reconstruction depends on the previous
// step's flag never lingering.
__global__ void neuron_update_kernel(
    index_t n, index_t exc_count, real h_step, int substeps, real dt, std::uint64_t seed,
    std::int64_t step, InputParams in_exc, InputParams in_inh, real shared_noise_exc,
    std::uint64_t noise_index_exc, real shared_noise_inh, std::uint64_t noise_index_inh,
    const CellParams* __restrict__ p, const real* __restrict__ psc_init_ex,
    const real* __restrict__ psc_init_in, const real* __restrict__ I_sic,
    real* __restrict__ V, real* __restrict__ w, real* __restrict__ g_ex,
    real* __restrict__ dg_ex, real* __restrict__ g_in, real* __restrict__ dg_in,
    int* __restrict__ refractory_steps, real* __restrict__ exc_input,
    real* __restrict__ inh_input, unsigned char* __restrict__ spiked) {
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const bool excitatory = i < exc_count;
    const InputParams& in = excitatory ? in_exc : in_inh;

    // Independent Poisson drive, one realisation per target. Always routed
    // into exc_input, matching Network's host drive_astrocytes-adjacent
    // convention (see NeuronPopulation::update's host loop).
    real exc_in = exc_input[i];
    if (in.poiss_rate > 0.0) {
        const std::uint64_t stream =
            static_cast<std::uint64_t>(step) * 1000003ULL + static_cast<std::uint64_t>(i);
        const real lambda = in.poiss_rate * dt * 1e-3;  // Hz * ms -> events
        const int events = rng_poisson(seed ^ 0x2545F4914F6CDD1DULL, stream, lambda);
        if (events > 0) {
            exc_in += in.poiss_weight * static_cast<real>(events);
        }
    }

    real I_ext = I_sic[i];
    if (in.gauss_noise_std > 0.0) {
        if (in.independent_noise) {
            const std::uint64_t idx = excitatory ? noise_index_exc : noise_index_inh;
            I_ext += in.gauss_noise_std *
                     rng_normal(seed ^ 0x9E3779B185EBCA87ULL,
                                idx * 1000033ULL + static_cast<std::uint64_t>(i));
        } else {
            I_ext += excitatory ? shared_noise_exc : shared_noise_inh;
        }
    }

    real v = V[i], ww = w[i], gex = g_ex[i], dgex = dg_ex[i], gin = g_in[i], dgin = dg_in[i];
    int refr = refractory_steps[i];

    const bool fired = neuron_advance(p[i], h_step, substeps, exc_in, inh_input[i],
                                      psc_init_ex[i], psc_init_in[i], I_ext, v, ww, gex, dgex,
                                      gin, dgin, refr);

    V[i] = v;
    w[i] = ww;
    g_ex[i] = gex;
    dg_ex[i] = dgex;
    g_in[i] = gin;
    dg_in[i] = dgin;
    refractory_steps[i] = refr;
    exc_input[i] = 0.0;
    inh_input[i] = 0.0;
    spiked[i] = fired ? 1 : 0;
}

}  // namespace

struct CudaNeuron {
    index_t n = 0;
    index_t exc_count = 0;

    // Resident for the whole run: state.
    real* V = nullptr;
    real* w = nullptr;
    real* g_ex = nullptr;
    real* dg_ex = nullptr;
    real* g_in = nullptr;
    real* dg_in = nullptr;
    int* refractory_steps = nullptr;

    // Resident for the whole run: per-cell parameters, read-only after create.
    CellParams* p = nullptr;
    real* psc_init_ex = nullptr;
    real* psc_init_in = nullptr;

    // Per-step transfer buffers, host -> device.
    real* exc_input = nullptr;
    real* inh_input = nullptr;
    real* I_sic = nullptr;

    // Per-step transfer buffer, device -> host.
    unsigned char* spiked = nullptr;
};

namespace {

real* device_copy(const real* host, index_t n) {
    real* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, static_cast<std::size_t>(n) * sizeof(real)));
    CUDA_CHECK(cudaMemcpy(d, host, static_cast<std::size_t>(n) * sizeof(real),
                          cudaMemcpyHostToDevice));
    return d;
}

}  // namespace

CudaNeuron* cuda_neuron_create(index_t n, index_t exc_count, const CellParams* p, const real* V,
                               const real* w, const real* g_ex, const real* dg_ex,
                               const real* g_in, const real* dg_in, const int* refractory_steps,
                               const real* psc_init_ex, const real* psc_init_in) {
    auto* s = new CudaNeuron;
    s->n = n;
    s->exc_count = exc_count;
    if (n == 0) {
        return s;
    }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(real);

    s->V = device_copy(V, n);
    s->w = device_copy(w, n);
    s->g_ex = device_copy(g_ex, n);
    s->dg_ex = device_copy(dg_ex, n);
    s->g_in = device_copy(g_in, n);
    s->dg_in = device_copy(dg_in, n);
    s->psc_init_ex = device_copy(psc_init_ex, n);
    s->psc_init_in = device_copy(psc_init_in, n);

    CUDA_CHECK(cudaMalloc(&s->refractory_steps, static_cast<std::size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(s->refractory_steps, refractory_steps,
                          static_cast<std::size_t>(n) * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s->p, static_cast<std::size_t>(n) * sizeof(CellParams)));
    CUDA_CHECK(cudaMemcpy(s->p, p, static_cast<std::size_t>(n) * sizeof(CellParams),
                          cudaMemcpyHostToDevice));

    // Overwritten (input) or freshly written (output) every step; the
    // network is freshly built when device_begin runs, so zero is correct.
    CUDA_CHECK(cudaMalloc(&s->exc_input, bytes));
    CUDA_CHECK(cudaMemset(s->exc_input, 0, bytes));
    CUDA_CHECK(cudaMalloc(&s->inh_input, bytes));
    CUDA_CHECK(cudaMemset(s->inh_input, 0, bytes));
    CUDA_CHECK(cudaMalloc(&s->I_sic, bytes));
    CUDA_CHECK(cudaMemset(s->I_sic, 0, bytes));
    CUDA_CHECK(cudaMalloc(&s->spiked, static_cast<std::size_t>(n) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(s->spiked, 0, static_cast<std::size_t>(n) * sizeof(unsigned char)));

    return s;
}

void cuda_neuron_destroy(CudaNeuron* s, real* V, real* w, real* g_ex, real* dg_ex, real* g_in,
                         real* dg_in, int* refractory_steps) {
    if (s == nullptr) {
        return;
    }
    if (s->n > 0) {
        const std::size_t bytes = static_cast<std::size_t>(s->n) * sizeof(real);
        CUDA_CHECK(cudaMemcpy(V, s->V, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(w, s->w, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(g_ex, s->g_ex, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dg_ex, s->dg_ex, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(g_in, s->g_in, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dg_in, s->dg_in, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(refractory_steps, s->refractory_steps,
                              static_cast<std::size_t>(s->n) * sizeof(int),
                              cudaMemcpyDeviceToHost));
        for (real* ptr : {s->V, s->w, s->g_ex, s->dg_ex, s->g_in, s->dg_in, s->psc_init_ex,
                          s->psc_init_in, s->exc_input, s->inh_input, s->I_sic}) {
            CUDA_CHECK(cudaFree(ptr));
        }
        CUDA_CHECK(cudaFree(s->refractory_steps));
        CUDA_CHECK(cudaFree(s->p));
        CUDA_CHECK(cudaFree(s->spiked));
    }
    delete s;
}

void cuda_neuron_push_input(CudaNeuron* s, const real* exc_input, const real* inh_input,
                            const real* I_sic) {
    if (s == nullptr || s->n == 0) {
        return;
    }
    const std::size_t bytes = static_cast<std::size_t>(s->n) * sizeof(real);
    CUDA_CHECK(cudaMemcpy(s->exc_input, exc_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->inh_input, inh_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->I_sic, I_sic, bytes, cudaMemcpyHostToDevice));
}

void cuda_neuron_pull_spikes(CudaNeuron* s, unsigned char* out_spiked) {
    if (s == nullptr || s->n == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpy(out_spiked, s->spiked,
                          static_cast<std::size_t>(s->n) * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));
}

void cuda_neuron_update(CudaNeuron* s, real h_step, int substeps, real dt, std::uint64_t seed,
                        std::int64_t step, const InputParams& in_exc, const InputParams& in_inh,
                        real shared_noise_exc, std::uint64_t noise_index_exc,
                        real shared_noise_inh, std::uint64_t noise_index_inh) {
    if (s == nullptr || s->n == 0) {
        return;
    }
    // Same block size as astro_update_kernel, for the same reason.
    constexpr int block = 128;
    const int grid = static_cast<int>((s->n + block - 1) / block);
    neuron_update_kernel<<<grid, block>>>(
        s->n, s->exc_count, h_step, substeps, dt, seed, step, in_exc, in_inh, shared_noise_exc,
        noise_index_exc, shared_noise_inh, noise_index_inh, s->p, s->psc_init_ex, s->psc_init_in,
        s->I_sic, s->V, s->w, s->g_ex, s->dg_ex, s->g_in, s->dg_in, s->refractory_steps,
        s->exc_input, s->inh_input, s->spiked);
    CUDA_CHECK(cudaGetLastError());
}

const unsigned char* cuda_neuron_device_spiked(const CudaNeuron* s) {
    return s == nullptr ? nullptr : s->spiked;
}

real* cuda_neuron_device_exc_input(CudaNeuron* s) {
    return s == nullptr ? nullptr : s->exc_input;
}

real* cuda_neuron_device_inh_input(CudaNeuron* s) {
    return s == nullptr ? nullptr : s->inh_input;
}

real* cuda_neuron_device_sic(CudaNeuron* s) {
    return s == nullptr ? nullptr : s->I_sic;
}

}  // namespace astrosimgpu

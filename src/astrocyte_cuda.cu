// Native CUDA backend. The kernel body is astro_advance, the same function the
// other backends call -- ASTROSIMGPU_FN expands to __host__ __device__ under
// nvcc. Only the launch and the memory management live here.

#include <cstdio>
#include <cstdlib>

#include "astrosimgpu/astrocyte_cuda.hpp"

namespace astrosimgpu {

namespace {

// Abort rather than continue with wrong results. A silent failure here gives a
// run that completes and reports plausible numbers.
void check(cudaError_t err, const char* what, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d during %s: %s\n", file, line, what,
                     cudaGetErrorString(err));
        std::abort();
    }
}

#define CUDA_CHECK(call) check((call), #call, __FILE__, __LINE__)

// One astrocyte per thread. State stays in registers across the substeps.
__global__ void astro_update_kernel(index_t n, AstroConstants c, real h_step, int substeps,
                                    real noise_std, bool independent_noise, real shared_noise,
                                    std::uint64_t noise_seed, std::uint64_t noise_index,
                                    real* __restrict__ Ca, real* __restrict__ IP3,
                                    real* __restrict__ h, real* __restrict__ ip3_input,
                                    const real* __restrict__ Ca_tot,
                                    const real* __restrict__ IP3_0,
                                    const real* __restrict__ tau_IP3,
                                    const real* __restrict__ delta_IP3) {
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const real noise =
        (noise_std > 0.0 && independent_noise)
            ? noise_std * rng_normal(noise_seed,
                                     noise_index * 1000003ULL + static_cast<std::uint64_t>(i))
            : shared_noise;

    real ca = Ca[i];
    real ip3 = IP3[i];
    real hh = h[i];

    astro_advance(c, Ca_tot[i], IP3_0[i], tau_IP3[i], delta_IP3[i], ip3_input[i], noise, h_step,
                  substeps, ca, ip3, hh);

    Ca[i] = ca;
    IP3[i] = ip3;
    h[i] = hh;
    ip3_input[i] = 0.0;
}

}  // namespace

struct CudaAstro {
    index_t n = 0;
    real* Ca = nullptr;
    real* IP3 = nullptr;
    real* h = nullptr;
    real* ip3_input = nullptr;
    real* Ca_tot = nullptr;
    real* IP3_0 = nullptr;
    real* tau_IP3 = nullptr;
    real* delta_IP3 = nullptr;
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

CudaAstro* cuda_astro_create(index_t n, const real* Ca, const real* IP3, const real* h,
                             const real* Ca_tot, const real* IP3_0, const real* tau_IP3,
                             const real* delta_IP3) {
    auto* s = new CudaAstro;
    s->n = n;
    if (n == 0) {
        return s;
    }
    s->Ca = device_copy(Ca, n);
    s->IP3 = device_copy(IP3, n);
    s->h = device_copy(h, n);
    s->Ca_tot = device_copy(Ca_tot, n);
    s->IP3_0 = device_copy(IP3_0, n);
    s->tau_IP3 = device_copy(tau_IP3, n);
    s->delta_IP3 = device_copy(delta_IP3, n);
    // Overwritten every step.
    CUDA_CHECK(cudaMalloc(&s->ip3_input, static_cast<std::size_t>(n) * sizeof(real)));
    CUDA_CHECK(cudaMemset(s->ip3_input, 0, static_cast<std::size_t>(n) * sizeof(real)));
    return s;
}

void cuda_astro_destroy(CudaAstro* s, real* Ca, real* IP3, real* h) {
    if (s == nullptr) {
        return;
    }
    if (s->n > 0) {
        const std::size_t bytes = static_cast<std::size_t>(s->n) * sizeof(real);
        CUDA_CHECK(cudaMemcpy(Ca, s->Ca, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(IP3, s->IP3, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h, s->h, bytes, cudaMemcpyDeviceToHost));
        for (real* p : {s->Ca, s->IP3, s->h, s->ip3_input, s->Ca_tot, s->IP3_0, s->tau_IP3,
                        s->delta_IP3}) {
            CUDA_CHECK(cudaFree(p));
        }
    }
    delete s;
}

void cuda_astro_push_input(CudaAstro* s, const real* ip3_input) {
    if (s == nullptr || s->n == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpy(s->ip3_input, ip3_input,
                          static_cast<std::size_t>(s->n) * sizeof(real), cudaMemcpyHostToDevice));
}

void cuda_astro_pull_calcium(CudaAstro* s, real* Ca) {
    if (s == nullptr || s->n == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpy(Ca, s->Ca, static_cast<std::size_t>(s->n) * sizeof(real),
                          cudaMemcpyDeviceToHost));
}

void cuda_astro_update(CudaAstro* s, const AstroConstants& c, real h_step, int substeps,
                       real noise_std, bool independent_noise, real shared_noise,
                       std::uint64_t noise_seed, std::uint64_t noise_index) {
    if (s == nullptr || s->n == 0) {
        return;
    }
    // 128 to match the block size the OpenMP target compiler chose for the
    // same loop, so neither is handed an advantage.
    constexpr int block = 128;
    const int grid = static_cast<int>((s->n + block - 1) / block);
    astro_update_kernel<<<grid, block>>>(s->n, c, h_step, substeps, noise_std, independent_noise,
                                        shared_noise, noise_seed, noise_index, s->Ca, s->IP3,
                                        s->h, s->ip3_input, s->Ca_tot, s->IP3_0, s->tau_IP3,
                                        s->delta_IP3);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace astrosimgpu

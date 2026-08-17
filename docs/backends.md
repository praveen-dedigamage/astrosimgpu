# Device backends

The astrocyte update can be dispatched four ways: on the host, through OpenMP
target offload, through Kokkos, or through native CUDA. All four call the same
per-cell function, so a comparison between them measures the dispatch and
nothing else.

| backend | build | status |
|---|---|---|
| host | `make OPENMP=1 CXX=nvc++` | validated on Roihu |
| OpenMP target | `make OFFLOAD=1 CXX=nvc++` | validated on a GH200 |
| Kokkos | `cmake -DASTROSIMGPU_KOKKOS=ON` | builds and runs correctly on a GH200 |
| native CUDA | `cmake -DASTROSIMGPU_CUDA=ON` | validated on a GH200 |

Native CUDA is the reference the portable routes are measured against. All four
reproduce the regime transition exactly (0.0106 asynchronous, 0.4177 bursting).

## What portability costs

Fitted over the linear region, from two runs that agreed to within a few per
cent:

```
CUDA           20.2 us + 0.133 ns per astrocyte
Kokkos         27.5 us + 0.131 ns per astrocyte
OpenMP target  28.4 us + 0.137 ns per astrocyte
host            5.0 us + 0.699 ns per astrocyte
```

The three device marginal costs agree to within 4 %, and Kokkos is marginally
below CUDA. The abstractions cost 7 to 8 microseconds per kernel launch and
nothing per cell.

So the penalty depends on the population: about 40 % at a thousand astrocytes
where the fixed cost is everything, 4 to 11 % at a million where it is diluted,
and less above that. A single percentage would have been misleading.

The same fixed cost sets where each backend beats the host: roughly 27,000
astrocytes for CUDA, about 40,000 for the portable routes.

Where the remaining 7 microseconds goes is not known. All three are far from the
hardware's floating-point capability, and since three different dispatch
mechanisms agree on the marginal cost, the limit is the kernel rather than the
dispatch.

## Why the later ones were cheap to add

The per-cell calculation already lived in free functions taking plain scalars,
because OpenMP target offload required that. Kokkos and CUDA require the same
thing, differing only in how the function is marked:

```cpp
#if defined(ASTROSIMGPU_KOKKOS)
#define ASTROSIMGPU_FN KOKKOS_INLINE_FUNCTION
#elif defined(__CUDACC__)
#define ASTROSIMGPU_FN __host__ __device__ inline
#else
#define ASTROSIMGPU_FN inline
#endif
```

`__CUDACC__` is defined only while nvcc is compiling a translation unit, so the
decoration appears exactly where a device version is needed and the host build
is untouched.

`astro_advance`, `astro_derivatives` and the three random number functions
carry that decoration. Their bodies are untouched, so all three backends run
identical arithmetic and the existing tests cover it through the host path.

## What differs

| | OpenMP target | Kokkos | native CUDA |
|---|---|---|---|
| function marking | `declare target` around definitions | `KOKKOS_INLINE_FUNCTION` | `__host__ __device__` |
| loop | `#pragma omp target teams distribute parallel for` | `Kokkos::parallel_for` | `__global__` kernel, explicit launch |
| device memory | `map` and `target enter data` | `Kokkos::View`, `deep_copy` | `cudaMalloc`, `cudaMemcpy` |
| portability | NVIDIA and AMD via the compiler | CUDA, HIP, SYCL, OpenMP, Serial | NVIDIA only |
| compiler | any with offload support | any Kokkos supports | nvcc for one file |

Data moves at the same four points in both: allocate and copy on entering a
run, push the synaptic input each step, pull calcium back each step, copy out
at the end. That was deliberate. If the transfer pattern differed the
comparison would measure the pattern rather than the abstraction.

## Building CUDA

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DASTROSIMGPU_CUDA=ON
cmake --build build-cuda -j
```

`CMAKE_CUDA_ARCHITECTURES` defaults to 90, which is Hopper and therefore a
GH200. Only `src/astrocyte_cuda.cu` goes through nvcc; the rest of the
simulator is compiled by the host compiler as usual.

The kernel launches 128 threads per block, matching the geometry the OpenMP
target compiler reported choosing for the same loop, so neither is handed an
advantage in the comparison.

Every CUDA call is checked and aborts on failure. A silent failure would
produce a simulation that runs and reports plausible output, which is the
failure mode this project has already lost a day to.

## Building Kokkos

Kokkos is a CMake package, so the Makefile does not support it.

On Roihu use `nvc++` for the host build too. Lmod swaps `gcc` out when `nvhpc`
loads, so the two compilers are not available at once, and objects from both
link into undefined references to NVHPC runtime symbols.

```bash
cmake -S . -B build-kokkos \
      -DCMAKE_BUILD_TYPE=Release \
      -DASTROSIMGPU_KOKKOS=ON \
      -DKokkos_ROOT=/path/to/kokkos/install
cmake --build build-kokkos -j
```

`ASTROSIMGPU_KOKKOS` and `ASTROSIMGPU_OFFLOAD` are alternatives and CMake
refuses both at once.

Which device Kokkos targets is fixed when Kokkos itself is built, not here. A
Kokkos configured with `Kokkos_ENABLE_CUDA` and `Kokkos_ARCH_HOPPER90` gives a
GH200 build; one configured with `Kokkos_ENABLE_OPENMP` gives a host build from
the same source. Every run reports which it got:

```
astrocyte backend     Kokkos, Cuda
```

## Getting Kokkos on Roihu

Check for an existing installation first:

```bash
module spider kokkos
```

If there is none, building it takes a few minutes:

```bash
git clone --depth 1 https://github.com/kokkos/kokkos.git
cmake -S kokkos -B kokkos-build \
      -DCMAKE_INSTALL_PREFIX=$PWD/kokkos-install \
      -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_HOPPER90=ON \
      -DKokkos_ENABLE_OPENMP=ON
cmake --build kokkos-build -j16 --target install
```

`Kokkos_ARCH_HOPPER90` is the GH200's compute capability. Build it on
`roihu-gpu.csc.fi`, since binaries do not cross between the two login
architectures.

## What the comparison is for

Three questions, in order of how much the answer changes:

1. **Does it produce the same results?** The regime transition is 0.0106 and
   0.4177; both must reproduce them. A backend that computes something else is
   not a data point about performance.
2. **What does the abstraction cost?** The OpenMP target path reaches 0.133 ns
   per astrocyte per step. If Kokkos matches that, portability across vendors
   is available for nothing. If it is substantially slower, the cost of
   portability has a number attached for the first time.
3. **Does either reach the hardware?** Both are estimated at roughly 5 % of
   FP64 peak, so the more interesting possibility is that they are equally far
   off and the limit is the kernel rather than the dispatch.

The measurement is the point. Two implementations that agree tell you the
abstraction is free here; two that disagree tell you what it costs.

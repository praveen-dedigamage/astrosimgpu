# Kokkos backend

The astrocyte update can be dispatched through Kokkos instead of OpenMP target
offload. Both express the same loop over the same per-cell function, so the
difference between them is the abstraction and nothing else.

**This has not been compiled or run.** No Kokkos installation was available
where it was written. Expect the first build to fail; what is done is the
restructuring the first build would otherwise demand.

## Why it was cheap to add

The per-cell calculation already lived in free functions taking plain scalars,
because OpenMP target offload required that. Kokkos requires the same thing,
differing only in how the function is marked:

```cpp
#if defined(ASTROSIMGPU_KOKKOS)
#define ASTROSIMGPU_FN KOKKOS_INLINE_FUNCTION   // __host__ __device__ under CUDA
#else
#define ASTROSIMGPU_FN inline
#endif
```

`astro_advance`, `astro_derivatives` and the three random number functions
carry that decoration. Their bodies are untouched, so all three backends run
identical arithmetic and the existing tests cover it through the host path.

## What differs

| | OpenMP target | Kokkos |
|---|---|---|
| function marking | `declare target` around definitions | `KOKKOS_INLINE_FUNCTION` per function |
| loop | `#pragma omp target teams distribute parallel for` | `Kokkos::parallel_for` with a lambda |
| device memory | `map` clauses and `target enter data` | `Kokkos::View` with `deep_copy` |
| portability | NVIDIA and AMD via the compiler | CUDA, HIP, SYCL, OpenMP, Serial from one source |

Data moves at the same four points in both: allocate and copy on entering a
run, push the synaptic input each step, pull calcium back each step, copy out
at the end. That was deliberate. If the transfer pattern differed the
comparison would measure the pattern rather than the abstraction.

## Building

Kokkos is a CMake package, so the Makefile does not support it.

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

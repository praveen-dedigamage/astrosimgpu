# Running on Roihu

Notes for building and profiling on CSC's Roihu before any device code exists.

**These scripts have not been run on Roihu.** They were written from the CSC
documentation, and module names in particular are left to be discovered rather
than hard-coded. Treat the first build as the thing being tested.

## The one that costs a day

Roihu has two login nodes with different architectures, and binaries do not
cross between them:

| Login node | Architecture | Compile here for |
|---|---|---|
| `roihu-cpu.csc.fi` | x86_64 (AMD Turin) | the CPU partitions |
| `roihu-gpu.csc.fi` | **aarch64 (Grace)** | **the GH200 nodes** |

CSC put it plainly: *"Software compiled on Roihu-CPU nodes only works on
Roihu-CPU nodes."* The filesystem is shared, so a build from the wrong node
appears to succeed and then fails on the GPU partition. Everything here builds
from `roihu-gpu.csc.fi`; `build.sh` refuses to run anywhere else.

## Hardware

Per GPU node: four NVIDIA GH200 superchips. Each pairs an H100 with a 72-core
Grace ARM CPU, so a node has 288 ARM cores and four GPUs. Reserving one GPU
grants up to 72 cores and 217 GiB (95 GiB HBM3 + 122 GiB LPDDR5).

## Partitions

| Partition | Walltime | GPUs | Nodes | For |
|---|---|---|---|---|
| `gputest` | 15 min | 1–4 | 2 | everything below |
| `gpumedium` | 36 h | 1–4 | 4 | longer runs |
| `gpularge` | 36 h | 4/4 | 10 | multi-node |

Request GPUs with `--gres=gpu:gh200:N`. The gres name is lowercase. Slurm
gres names are case-sensitive, and `gpu:GH200:1` matches nothing, so the job is
admitted with zero cores and rejected with a message about the request not
being supported, which does not point at the cause.

Do not request memory on a GPU partition. It is allocated with the GPU and an
explicit `--mem` is overridden.

`gputest` is enough for all the baseline work — the runs are seconds of wall
clock and the 15-minute limit usually means a shorter queue.

## Sequence

```bash
ssh roihu-gpu.csc.fi
git clone <repo> && cd astrosimgpu

module spider nvhpc          # find what is actually installed
export COMPILER_MODULES="nvhpc/<version>"

bash scripts/roihu/build.sh  # builds, runs the 81 checks
```

Then edit `--account=project_XXXXXXX` in `scripts/roihu/baseline.sbatch` and:

```bash
sbatch scripts/roihu/baseline.sbatch
```

## What the baseline is for

Three things, in order of how much they matter:

1. **A denominator measured on the target machine.** The profile in
   `docs/gpu-port.md` was taken on an Apple M-series core. Grace is a different
   core with a different memory system, and every speedup claimed later has to
   be against a Grace number or it means nothing.

2. **The shape of the OpenMP scaling curve.** The per-cell update should scale
   across the 72 cores; the delivery phases should not. Where the curve flattens
   is an early, free answer to the question the GPU port is asking. If the
   update stops scaling well before 72 threads at 500 cells, that is the problem
   size talking — seven cells per thread — not the hardware.

3. **The phase ratio in the right regime.** The published benchmarks run 20,000
   cells, and the share of runtime in the update phase depends strongly on
   both the population and how the network is scaled. The batch script ends with
   a 20,000-cell run because the ratio that justifies the port should be measured
   at the scale the science actually uses.

## Expected trouble

- **Module names.** Not hard-coded anywhere here. Use `module spider`.
- **`-march=native` on Grace.** `ASTROSIMGPU_NATIVE=ON` may need
  `-mcpu=neoverse-v2` instead; CMake checks the flag and skips it if
  unsupported, so a failure here is silent rather than fatal. Check the build
  log.
- **Memory on the 20k run.** 20,000 cells at `p_primary = 0.2` is on the order
  of tens of millions of synapses, and connectivity is built with per-source
  vectors before flattening. If it runs out of memory, that is a real finding
  about the build path and worth recording rather than working around.
- **Filesystem.** Write results to scratch, not to `$HOME`, which is small and
  on a different Lustre filesystem.

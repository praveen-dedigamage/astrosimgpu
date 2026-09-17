# Knowledge reference

Everything needed to pick this project back up: what it simulates, how the
code is put together, what's been verified, what's been measured, and how to
run the tools. Written as a reference to come back to, not a narrative --
jump to the section you need. Companion to `docs/optimization-roadmap.md`,
which is the "what to do next" list; this is the "what you need to know to
do it" list.

## 1. What this simulates

A network of spiking neurons coupled to astrocytes (non-neuronal brain
cells), reimplementing the model from Jiang et al. (2025), PLOS Comput Biol.
Each neuron-to-neuron synapse can recruit an astrocyte as a "third factor":
the astrocyte receives presynaptic spikes, accumulates internal calcium, and
above a threshold sends a slow current back to the postsynaptic neuron. The
question the model answers: does that loop hold the network in sparse,
independent firing, or pull it into synchronized bursting? Full narrative in
the top-level `README.md`.

## 2. Architecture, top to bottom

- **`include/astrosimgpu/astrocyte_kernel.hpp`** -- pure per-cell math
  (`astro_derivatives`, `astro_advance`), free functions with no `this`, no
  `std::vector`. This is what makes the same code run on both host and CUDA
  device -- one implementation, no risk of the backends drifting apart. (It
  also ran on OpenMP target-offload and Kokkos, before those backends were
  removed -- see `docs/backends.md`.)
- **`include/astrosimgpu/astrocyte.hpp` / `neuron.hpp`** -- structure-of-arrays
  population classes (`AstrocytePopulation`, `NeuronPopulation`). Each state
  variable is a separate contiguous array so per-cell updates are unit-stride
  and GPU-coalescable.
- **Backend dispatch** -- compile-time only (`#if defined(ASTROSIMGPU_CUDA)`
  etc). `AstrocytePopulation`/`NeuronPopulation`/`Network` each hold an opaque
  `CudaAstro*`/`CudaNeuron*`/`CudaDelivery*` when built with CUDA;
  `astrocyte_cuda.hpp`/`neuron_cuda.hpp`/`network_cuda.hpp` stay free of CUDA
  types so only `src/cuda_kernels.cu` (every `__global__` kernel, one
  translation unit) needs `nvcc`.
- **`include/astrosimgpu/network.hpp` / `src/network.cpp`** -- `Network` owns
  both populations and four `ConnectionSet`s, builds connectivity once
  (`build()`), then runs the per-step time loop (`run()`).
- **`src/main.cpp`** -- CLI parsing -> config load -> `build()` -> `run()` ->
  post-hoc calcium analysis (`analysis.hpp`) -> formatted summary with
  per-phase timing.

See `docs/code-walkthrough.md` for the reasoning behind decisions not
obvious from the code itself.

## 3. The simulation cycle (one step, in order)

From `Network::run`, `src/network.cpp`:

1. **`apply_arrivals(step)`** -- read this step's ring-buffer slot, hand
   off synaptic current to neurons and IP3 to astrocytes, hand off SIC
   current if a packet is pending for this slot.
2. **`drive_astrocytes(step)`** -- independent Poisson IP3 input per
   astrocyte (serial host loop -- see Section 6, this is the current
   bottleneck).
3. **Astrocyte update** -- push inputs to device, run `astro_.update()`
   (RK4, `substeps` internal stages), pull calcium back, clear consumed
   inputs.
4. **Neuron update** -- integrate all neurons one `dt`, collect spikes.
5. **`deliver_spikes(...)`** -- walk each spike source's connectivity row,
   add weight (scaled by STP if enabled) into the ring buffer at
   `step + delay`.
6. **`deliver_sic(step)`**, every `sic_interval` steps -- push astrocyte
   SIC output into the ring buffer, mark the arrival slot pending.

Two phases overall: `pre_steps` (transient, discarded) then `sim_steps`
(recorded). Communication uses fixed-size ring buffers keyed by
`(step + delay) % ring_slots`, not an event queue.

## 4. Model equations -- verified correct

All cross-checked against the documented equations in `README.md` (which
follow Li & Rinzel 1994 / Nadkarni & Jung 2003 / Brette & Gerstner 2005 /
Tsodyks-Markram 2000) by deriving each from scratch and comparing term by
term. No errors found.

| Equation | Location | Status |
|---|---|---|
| Astrocyte calcium dynamics (Li-Rinzel) | `astrocyte_kernel.hpp:47-68` | Every term matches |
| SIC output (`F_SIC = SIC_scale * H(ln y) * ln y`) | `astrocyte.cpp:331-339` | Matches exactly, including the `y<=1` boundary |
| AdEx neuron (`dV`, `dw`) | `neuron.cpp:85-99` | Matches term for term |
| Alpha-synapse cascade, analytically propagated | `neuron.cpp:186-191` | Derived by hand; peak conductance = weight at exactly `t = tau_syn`, confirmed |
| Tsodyks-Markram STP | `network.cpp:240-258` | Traced against the standard published recurrence symbol-by-symbol; matches, including the `u`-before-`x` ordering dependency. **Note:** the project's own docs flag this as never diffed against NEST line-by-line -- only the aggregate firing rate (0.04% agreement) supports it in bulk. Our derivation is independent confirmation, not a replacement for that check. |
| Tripartite connectivity (Bernoulli `p_primary`, conditional `p_third_if_primary`, pool assignment) | `network.cpp:112-238` | Matches; independently reproduced the README's "~16 duplicate SIC connections" figure from the probability math (`400 x 0.2 x 0.2 = 16`) |

## 5. Validation status

- **86/86 component tests** (`make test`) -- covers alpha-conductance
  normalization, IP3 relaxation, SIC threshold, calcium bounds, RNG,
  reproducibility.
- **Regime transition** -- raising `w_n2a` from 0.20 to 0.31 moves the
  network from sparse/uncorrelated (correlation 0.0012) to
  synchronized (0.3429). This is the model's central qualitative claim and
  it reproduces.
- **Published benchmark** -- `config/paper_sparse.json` reproduces the
  paper's "Sparse" model mean firing rate (4.76 Hz) to 0.04% agreement at
  `substeps=2`.
- **Known unconverged setting:** `substeps=1` (the default) undercounts
  calcium transients by ~20%; use `substeps>=2` for anything quantitative.
  Does not affect the regime transition (0.0106 to 0.0208 across the whole
  range vs. bursting's 0.4177 -- a 20x margin).
- **What has NOT been checked:** calcium transient durations/timing against
  NEST directly; the STP update order against NEST's `tsodyks_synapse`
  line-by-line; whether the exc/inh firing-rate asymmetry (~0.1 Hz vs ~2 Hz)
  matches the reference. See `docs/validation.md` for full detail and the
  four historical errors found (each silent, each plausible, worth reading
  before assuming a "reasonable-looking" number is right).

## 6. Performance profile -- what's actually slow

### The one decision that gates everything (see roadmap Stage 0)

Whether GPU offload matters at all depends on network scaling:

| | fixed connection probability | fixed in-degree |
|---|---|---|
| update phases (offloadable) | ~4% of runtime | ~97.6% |
| delivery phases (host-only) | ~96% | ~2.4% |

Fixed in-degree is biologically motivated (constant synapse count per
neuron) and is the regime where any of the below matters.

### Measured baseline (`docs/gpu-port.md`, `config/use_case.json`, 1 core)

```
update astrocytes       4.32 %
update neurons         92.49 %   <-- dominant phase in realistic configs
spike delivery          0.08 %
SIC gather+deliver      0.60 %
arrival application     1.57 %
```

### What we measured ourselves on Roihu (one GH200, `rg2101`)

CUDA astrocyte-kernel throughput (job 1293182) reproduced the previously
published table to within ~1%:

| astrocytes | measured | published (`docs/gpu-port.md`) |
|---|---|---|
| 1,000 | 20.70 us | 20.6 us |
| 100,000 | 36.55 us | 36.6 us |
| 10,000,000 | 1,257.67 us | 1,272.5 us |

`nsys` phase breakdown at 100k astrocytes (job 1293360) confirmed the
documented bottleneck independently:

```
background input (drive_astrocytes)   50.08 %   <-- bigger than the kernel it feeds
update astrocytes (the CUDA kernel)   24.21 %
SIC gather+deliver                     8.92 %
```

GPU-side detail from the same run: `astro_update_kernel` costs ~9.95 us on
the device per launch (cheap); H2D and D2H transfers are exactly 0.8 MB each
(the full 100,000-astrocyte array), while only ~8,006 astrocytes (8%) are
actually connected -- most of what crosses the bus every step is wasted.

`ncu` (per-kernel detail, `docs/profiling.md`): the astrocyte kernel is
capped at 31.25% theoretical occupancy by 94 registers/thread. Not
spilling, not divergent, not bandwidth-bound (DRAM throughput 25%) --
latency-bound: 42% of cycles stall on the RK4 stage dependency chain.

## 7. GPU backends -- what exists, what doesn't

| Backend | Status | Notes |
|---|---|---|
| Host (default) | Complete | No device code, C++17 only |
| Native CUDA | Complete -- astrocyte, neuron, and delivery all on device | `ASTROSIMGPU_CUDA=ON`; `src/cuda_kernels.cu` |
| OpenMP target offload | Removed | Astrocyte update only while it existed; see `docs/backends.md` |
| Kokkos | Removed | Astrocyte update only while it existed; see `docs/backends.md` |

**Neuron update and spike/SIC delivery are on CUDA too now** (Stages 3 and 4
of `docs/gpu-port.md`, both complete) -- this section predates that work; see
`docs/gpu-port.md` for the current state and measurements.

All device backends move data at exactly four points (`device_begin`,
`device_push_input`, `device_pull_calcium`, `device_end`) so comparing them
measures dispatch overhead, not transfer-pattern differences. State stays
resident on the device for the whole run (not re-mapped every step) --
this was a ~3x win when it was implemented (`docs/gpu-port.md`, "What
residency changed").

## 8. Profiling tools

- **`nsys` (Nsight Systems)** -- whole-program timeline at real speed
  (no replay). Shows CPU API calls, GPU kernel launches, memory transfers,
  and (with `--sample=cpu --backtrace=dwarf`) periodic CPU call stacks on
  the same timeline. Answers "where does the time go, across the whole
  step, in order?" Script: `scripts/roihu/nsys_profile.sbatch`.
- **`ncu` (Nsight Compute)** -- one kernel, deep detail: registers,
  occupancy, warp stall reasons, memory/compute throughput. Replays the
  kernel ~40 times to collect all counter groups (`--set full`), so it's
  slow and shouldn't be used for step-timing, only kernel internals.
  Script: `scripts/roihu/ncu_profile.sbatch`.
- **`perf`** -- standard Linux CPU call-graph profiler, zero code changes:
  `perf record -g -- <binary> ...` then `perf report`. Useful for
  host-function-level attribution without needing nsys's GUI.
- **`PhaseProfile`** (in-code, `include/astrosimgpu/network.hpp`) -- the
  project's own manual instrumentation, printed at the end of every run in
  `run.txt`. Coarse (six phases) but always available, no profiler needed.

`PYTHONNOUSERSITE=1` is required before both `nsys` and `ncu` on Roihu --
otherwise report post-processing fails with `unknown encoding: utf-8-sig`
against Roihu's Python.

## 9. Running on Roihu -- operational facts

- **Two login nodes, different architectures; binaries do not cross:**
  `roihu-cpu.csc.fi` (x86_64) vs. `roihu-gpu.csc.fi` (aarch64/Grace). Always
  build from `roihu-gpu.csc.fi` for anything that runs on the GH200 nodes.
- **Hardware:** each GPU node has four GH200 superchips (H100 + 72-core
  Grace ARM CPU each). One GPU reservation grants 72 cores + 217 GiB.
- **Partitions:** `gputest` (15 min, use for everything in this doc),
  `gpumedium` (36h), `gpularge` (36h, multi-node).
- **Module discovery:** `module spider nvhpc` to find the exact version
  string (not hardcoded anywhere in this repo, since module sets change);
  `module load nvhpc/<version>`; `module list` to confirm.
- **Account setup** (once, put in `~/.bashrc`):
  ```bash
  export SBATCH_ACCOUNT=project_XXXXXXX   # sbatch
  export SLURM_ACCOUNT=project_XXXXXXX    # srun
  ```
- **gres name is case-sensitive:** `gpu:gh200:1`, not `gpu:GH200:1` (the
  latter is silently admitted with zero cores, then rejected).
- **Don't request `--mem` on a GPU partition** -- it's allocated with the
  GPU and an explicit value gets overridden.
- **Write results to scratch, not `$HOME`** (small, different filesystem).

### Build commands, by backend

```bash
# OpenMP host build (what scripts/roihu/build.sh does)
cmake -S . -B build-roihu -DCMAKE_BUILD_TYPE=Release \
    -DASTROSIMGPU_OPENMP=ON -DASTROSIMGPU_NATIVE=ON
cmake --build build-roihu -j16

# Native CUDA
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
    -DASTROSIMGPU_CUDA=ON -DASTROSIMGPU_NATIVE=ON
cmake --build build-cuda -j16
```

OpenMP target offload and Kokkos (both built the astrocyte update only) have
since been removed from the codebase entirely, along with the scripts that
built/tested them (`gpu_offload_test.sbatch`, `three_way.sbatch`,
`build_kokkos.sh`) -- see `docs/backends.md` for the measurements that were
recorded while they existed. Only the host build and native CUDA remain.

### This repo's Roihu scripts (`scripts/roihu/`)

| Script | Purpose |
|---|---|
| `build.sh` | Builds the OpenMP host binary, runs the component checks |
| `baseline.sbatch` | CPU thread-scaling baseline (1 to 72 threads) + 20k-cell run |
| `cuda_only.sbatch` | CUDA backend only: correctness + regime transition + astrocyte-update us/step sweep |
| `kernel_scaling.sbatch` | Astrocyte-kernel throughput vs. population size, host vs. CUDA |
| `nsys_sweep.sbatch` | `nsys` phase/kernel profile looped across the full neuron-scaling config sweep, one job |
| `nsys_profile.sbatch` | Whole-step Nsight Systems timeline for one config, `--stats=true` printed to the job log |
| `ncu_profile.sbatch` | Nsight Compute deep-dive on a kernel (regex-filterable via `KERNEL=`), `--print-summary=per-kernel` to the job log |
| `paper_validation.sbatch` | Reproduces the published "Sparse" benchmark firing rate against the current CUDA build |

## 10. Where to go next

- **What to actually change, in order:** `docs/optimization-roadmap.md`
- **Full profiling narrative and the register/occupancy findings:**
  `docs/profiling.md`
- **The staged GPU port plan this roadmap extends:** `docs/gpu-port.md`
- **Full validation history, including the four historical bugs:**
  `docs/validation.md`
- **Why particular design choices were made:** `docs/code-walkthrough.md`
- **Backend implementation notes:** `docs/backends.md`
- **What's been tried and the results (e.g. `sic_interval` sweep):**
  `docs/experiments.md`

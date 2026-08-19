# GPU port plan

This document records the measured performance profile, what it implies for a
GPU port, and the questions that are still open.

The plan was written before any device code existed, so it could be checked
against measurements rather than the other way round. The measurements have
since been taken and are recorded below; several of them contradict the
expectations the plan started from, and those are marked where they occur.

## Measured baseline

One thread, `config/use_case.json`, 60 s of model time. The network has 100
astrocytes, 400 excitatory and 100 inhibitory neurons, 39,790 excitatory and
9,944 inhibitory synapses, and 8,097 tripartite attachments.

```
                       seconds   % of wall
update astrocytes       1.1254     4.32
update neurons         24.1053    92.49
spike delivery          0.0203      0.08
SIC gather+deliver      0.1551      0.60
arrival application     0.4101      1.57
other (I/O, loop)       0.2463      0.95
total                  26.0625    100.00

real-time factor        0.4344
update : communication  43 : 1
```

To reproduce:

```bash
./build/astrosimgpu --config config/use_case.json -t 60000 -o /tmp/prof
```

The phase names match the decomposition used in the reference benchmarks
(update, spike collocation and delivery, SIC gathering and delivery). The two
profiles can therefore be compared directly.

## What the baseline shows

The profile above was measured on a single x86 core. It is not representative
of the target machine, and it is not representative of a larger network either.

On 72 Grace cores, same configuration:

| phase | 1 core, x86 | 72 Grace cores |
|---|---|---|
| update astrocytes | 4.3 % | 26.4 % |
| update neurons | 92.5 % | 60.5 % |
| everything else | 2.3 % | 13.1 % |

Network size changes it far more. At 1000 astrocytes and 5000 neurons on the
same 72 cores:

| | 100 astro, 500 neurons | 1000 / 5000, fixed p | 1000 / 5000, fixed in-degree |
|---|---|---|---|
| synapses | 50 k | 5.0 M | 500 k |
| both update phases | 86.9 % | 4.0 % | 97.6 % |
| everything else | 13.1 % | 96.0 % | 2.4 % |

So the case for offloading the update is not a fact about the code. It depends
on how the network is scaled:

- **Fixed connection probability.** Each neuron's input grows with the
  population, the network saturates, and the delivery phases take 96 % of the
  run. Offloading the update would address 4 % and cap the speedup near 1.04x.
  This is how the reference benchmarks scale, which is why their delivery
  phases grow and only the update phase weak-scales.
- **Fixed in-degree.** The operating point is preserved and the update phases
  take 97.6 %. Offloading them is worth up to roughly 40x.

Both are defensible. Fixed in-degree is the biologically motivated one, since a
cortical neuron has on the order of 10^4 synapses regardless of brain size.
This question has to be settled before any performance target means anything.

There is one further complication, from `docs/validation.md`: the default
`substeps = 1` is not numerically converged, and quantitative work needs 4 or
more. Raising `substeps` multiplies arithmetic per communication step without
adding communication, so a converged configuration puts a larger share in the
update phase than the default one does.

## Which loops become kernels

Ordered by cost under fixed-in-degree scaling, where the update phases dominate.
Under fixed-probability scaling none of this is worth doing and the delivery
phases are the target instead.

1. `NeuronPopulation::update`. The larger of the two update phases in every
   configuration measured except the smallest. One thread per neuron. State is
   seven doubles (V, w, g_ex, dg_ex, g_in, dg_in, I_sic) plus a refractory
   counter, with sixteen more parameter values. All of it can stay in registers
   for the duration of a step.
2. `AstrocytePopulation::update`. Smaller, and already done. Three state
   variables and four per-cell parameters. It was the right one to write first
   because it is the simplest, not because it is the most valuable.
3. The delivery phases stay on the host for now. They are a few per cent under
   fixed-in-degree scaling and almost everything under fixed-probability
   scaling, so whether they matter is the same question as which scaling the
   science uses.

## What is already prepared

These choices were made with an offload in mind:

- **Structure-of-arrays layout.** Each state variable and per-cell parameter is
  a separate contiguous array. See `include/astrosimgpu/neuron.hpp` and
  `include/astrosimgpu/astrocyte.hpp`. Access is unit stride.
- **No shared writes during the update.** Each cell writes only its own index.
  Emitted spikes are collected per thread and merged after the loop, so there
  are no atomic operations in the hot path.
- **Counter-based random numbers.** `CounterRng` in
  `include/astrosimgpu/rng.hpp` computes every draw from (seed, stream,
  counter) with no carried state. A thread can reproduce its own noise without
  reading shared memory. Replacing it with cuRAND's counter-based generators is
  a substitution rather than a redesign.
- **Fixed-step explicit integration.** RK4 with a configurable substep count.
  There is no adaptive stepping, and the only per-cell branch is the refractory
  check. Every thread performs the same number of stages.
- **Phase-level timing.** A port can be measured against the baseline above.

## Current state of the port

The astrocyte update has been restructured for offload. An OpenMP target
version of the loop exists behind a build flag. No other part of the code has
device code.

The per-cell body was moved out of `AstrocytePopulation` into free functions in
`include/astrosimgpu/astrocyte_kernel.hpp`. These take plain scalars only: no
`this` pointer, no `std::vector`, no reference into a class. The random draws
were moved to free functions in `rng.hpp` for the same reason. A member
function cannot be offloaded, so this restructuring was required before any
pragma could be added.

The host loop and the offloaded loop call the same `astro_advance` function, so
they cannot diverge. A test checks that `rng_normal` matches `CounterRng`
exactly. The arithmetic is therefore covered by the existing tests through the
host path. The only untested part when offload is enabled is the mapping
directive.

```bash
make OFFLOAD=1 OFFLOAD_FLAGS="-mp=gpu -gpu=cc90" CXX=nvc++
```

Offload is off by default. The default build contains no device code. The
restructuring was verified to reproduce the previous results exactly: mean
pairwise correlation 0.0106 in the asynchronous regime and 0.4177 in the
bursting regime, unchanged to four decimal places.

### Measured on Roihu

Built with NVHPC 26.3 on `roihu-gpu.csc.fi` and run on one NVIDIA GH200 120GB.
`-Minfo=mp` reports the target region becoming a GPU kernel, parallelised
across teams and 128 threads, with device routines generated for
`astro_advance`, `astro_derivatives` and the random number functions.

All 86 component checks pass on the device, and the offloaded run reproduces
the host result exactly in every configuration tested. The kernel is correct.

Astrocyte update phase only, per timestep. One job per row, one build per
backend, threads pinned. Host is 72 Grace cores.

| astrocytes | host | OpenMP target | Kokkos | CUDA |
|---|---|---|---|---|
| 1,000 | 5.4 | 27.9 | 27.4 | 20.6 |
| 10,000 | 19.9 | 33.9 | 32.3 | 28.2 |
| 100,000 | 87.7 | 48.6 | 46.1 | 36.6 |
| 1,000,000 | 705.5 | 173.6 | 164.1 | 154.7 |
| 10,000,000 | 7,061.6 | 1,371.6 | 1,335.1 | 1,272.5 |

Fitted over the linear region:

```
CUDA           20.2 us + 0.133 ns per astrocyte
Kokkos         27.5 us + 0.131 ns per astrocyte
OpenMP target  28.4 us + 0.137 ns per astrocyte
host            5.0 us + 0.699 ns per astrocyte
```

The kernel is **5.3 times cheaper per cell** than the host. The three device
marginal costs agree to within 4 %: the abstractions cost 7 to 8 microseconds
per launch and nothing per cell, which is also why CUDA meets the host at
roughly 27,000 astrocytes where the portable routes meet it at about 40,000.

Three statements in earlier versions of this document were wrong and are worth
recording as such:

- The device cost was reported as "nearly flat" between 100 and 1000
  astrocytes, concluding the cost was per launch rather than per byte. The
  flatness was real, the conclusion too strong: there is a fixed cost *and* a
  per-cell cost, and the latter only becomes efficient once the device is
  occupied.
- A crossover of 150,000 was extrapolated from measurements spanning two code
  versions and two host compilers.
- Portability was reported as costing 5 to 12 %, from a single run. It depends
  on population size, because the difference is entirely in the fixed cost.

### What residency changed

Before the state was made resident, the device mapped eight arrays on every
step. At ten million astrocytes that was around 960 MB of traffic per step.

| | before | after |
|---|---|---|
| fixed cost per step | 55 us | 28 us |
| 1,000,000 astrocytes | 522 us | 169 us |
| 10,000,000 astrocytes | 4,218 us | 1,371 us |

The device improved by very close to 3x at both large sizes. That the factor is
the same across a tenfold change in population is what identifies the cause as
per-step transfer rather than anything that scales with the data.

The prediction made before the change was 1,750 us at ten million and a
speedup of 3.4x. The measurement is 1,142 us and 5.6x, so the mapping had been
costing more than the estimate allowed.

### What remains

The fixed cost is three device operations per step: transfer the synaptic input
across, launch the kernel, transfer calcium back. Native CUDA pays 20.2
microseconds for the same three, so roughly 7 of the portable routes' 28 is the
abstraction and the rest is the operations themselves.

Removing them means moving SIC delivery onto the device, which is not a small
change: `deliver_sic` writes into ring buffers indexed by neuron, so the
question becomes what else has to become resident. If it were done, the fixed
cost should fall to roughly the launch alone and the crossover with it, to
somewhere near ten thousand astrocytes.

The neuron update needs the same treatment. It is a larger job because it emits
spikes, so the per-thread spike collection has to be reworked into something a
device can write to.

Profiling the whole step rather than the phase changed the order of all of this.
`docs/profiling.md` records the result: at a million astrocytes the device is
idle 91 % of the time, consecutive kernels are 1,438 us apart, and the largest
single cost in the step is `drive_astrocytes`, a serial host loop over every
astrocyte that costs roughly 720 us against the kernel's 56. The transfers are
also whole arrays, 8,000,000 bytes each way every step, of which under one per
cent is read. The astrocyte phase these tables measure is about a tenth of the
step it belongs to.

## Open questions

These are the points where mentor input would be most useful:

1. **Register pressure against occupancy.** Measured for the astrocyte kernel in
   `docs/profiling.md`: 94 registers per thread, nothing spilled, and occupancy
   capped at 31 % by that allocation alone. The state does stay in registers, and
   the price is that five blocks fit per SM. The question stands for the neuron
   kernel, which carries seven state variables rather than three. Is one thread
   per cell correct there, or should the RK stages be spread across a warp?
2. **Single or double precision.** The Li-Rinzel calcium dynamics are
   moderately stiff, and `Ca_ER = (Ca_tot - Ca) / ratio_ER_cyt` subtracts
   similar quantities. FP64 is the safe default and roughly halves throughput.
   How should FP32 sufficiency be established, rather than judged by comparing
   traces visually?
3. **Where the neuron update spends its time.** The split between the Poisson
   draw and the RK4 integration has not been measured. If the RNG dominates,
   the kernel needs a different structure.
4. **Profiling a kernel with little memory traffic.** Answered in
   `docs/profiling.md`. The kernel is neither bandwidth bound nor divergent: it
   is latency bound, with too few resident warps to cover the dependency chain
   through the RK4 stages. What remains open is how far occupancy can be bought
   back before spilling costs more than it returns.
5. **SIC communication across devices.** Astrocyte-to-neuron coupling is
   continuous and independent of activity. It is transmitted every step whether
   or not anything happened. Calcium changes on a timescale of seconds, while
   `dt` is 0.1 ms. The signal may be transmitted far more often than the
   dynamics require. This has since been tested and is recorded in
   `docs/experiments.md`: an interval of ten leaves both regimes intact, and at a
   hundred a seed that synchronises at every shorter interval drops out of the
   synchronised regime entirely. Ten is available and a hundred is not. What
   remains open is what that interval buys once the coupling crosses a device
   boundary rather than a function call.

## Staged plan

**Stage 1. Astrocyte kernel.** Smallest state and simplest dynamics. The
regression targets already exist: the calcium statistics printed at the end of
a run, and the regime transition recorded in `docs/validation.md`. Correctness
first, performance later.

**Stage 2. Input generation and the per-step transfers.** Inserted after
profiling the whole step rather than the phase, and ahead of the neuron kernel
because it is larger. `drive_astrocytes` runs on one host thread over the entire
population every step; it is per-cell with no communication and belongs on the
device, which also removes the transfer across, since the input would no longer
be assembled on the host. The transfer back is gated on `synapse.sic_interval`
first and removed with SIC generation second. Annotate the phases with NVTX
ranges before any of it, so the effect is visible on the timeline rather than
inferred from counters. `docs/profiling.md` has the measurements.

**Stage 3. Neuron kernel.** The remaining 92 %. Keep the delivery phases on the
host and accept the transfers. Measure the new phase ratio.

**Stage 4. Keep state on the device.** Once both populations are offloaded, the
per-step transfers become the main cost. Move the ring buffers and delivery
phases so that state remains on the device for the whole run.

**Stage 5. Multiple devices.** Partition the network across the superchips in a
node and measure the cost of continuous SIC coupling between them. This
addresses open question 5.

Stages 1 to 3 are supported by the measured profile. Stages 4 and 5 are where
the work produces a result rather than only a faster simulator.

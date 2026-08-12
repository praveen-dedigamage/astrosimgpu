# GPU port plan

This document records the measured performance profile, what it implies for a
GPU port, and the questions that are still open. It was written before any
device code existed so the plan can be checked against the measurements.

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

96.8 % of run time is per-cell state update. That work is parallel across
cells: each cell reads and writes only its own array element, and the
integration does not touch shared memory. Communication is 2.3 % at this
network size on one node.

This gives both the opportunity and its limit. Offloading the update phase
addresses nearly all current cost. It also makes communication the limiting
term. If the update took no time at all, the remaining 2.3 % would cap the
speedup at about 43x for this configuration.

Two limitations of this measurement:

- It is one node with 500 cells. The reference benchmarks use 20,000 cells
  across several nodes and report that the update phase is the only one that
  weak-scales. The delivery phases grow with scale. The 43:1 ratio here is
  therefore the most favourable case.
- Neuron update takes more than twenty times longer than astrocyte update.
  There are five times as many neurons, and each neuron also draws Poisson
  input every step. Whether the RNG or the integration dominates has not been
  measured. The answer affects how the kernel should be written.

## Which loops become kernels

In order of measured cost:

1. `NeuronPopulation::update`, 92 % of run time. One thread per neuron. State
   is seven doubles (V, w, g_ex, dg_ex, g_in, dg_in, I_sic) plus a refractory
   counter. Parameters add sixteen more values. All of it can stay in registers
   for the duration of a step.
2. `AstrocytePopulation::update`, 4 %. One thread per astrocyte, three state
   variables and four per-cell parameters. This is the simplest kernel and the
   right one to write first, as a correctness check.
3. The delivery phases stay on the host for now. They are 2 % of current cost.
   They also contain the scaling question described below, which needs its own
   investigation.

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

**This code has never been compiled by an offloading compiler or run on a GPU.**
It compiles with the macro defined and the pragmas ignored. That establishes
only that the syntax is valid. The first real build should be expected to fail.
What has been done is the restructuring that the first build would otherwise
require.

The neuron update needs the same treatment. It is a larger job because it emits
spikes, so the per-thread spike collection has to be reworked into something a
device can write to.

## Open questions

These are the points where mentor input would be most useful:

1. **Register pressure against occupancy.** RK4 over the neuron's seven state
   variables requires many live values. If the kernel spills to memory, the
   design assumption that state stays in registers fails. Is one thread per
   cell correct, or should the RK stages be spread across a warp?
2. **Single or double precision.** The Li-Rinzel calcium dynamics are
   moderately stiff, and `Ca_ER = (Ca_tot - Ca) / ratio_ER_cyt` subtracts
   similar quantities. FP64 is the safe default and roughly halves throughput.
   How should FP32 sufficiency be established, rather than judged by comparing
   traces visually?
3. **Where the neuron update spends its time.** The split between the Poisson
   draw and the RK4 integration has not been measured. If the RNG dominates,
   the kernel needs a different structure.
4. **Profiling a kernel with little memory traffic.** Once state is loaded, the
   update touches almost no memory. Standard occupancy and bandwidth guidance
   may not apply.
5. **SIC communication across devices.** Astrocyte-to-neuron coupling is
   continuous and independent of activity. It is transmitted every step whether
   or not anything happened. Calcium changes on a timescale of seconds, while
   `dt` is 0.1 ms. The signal may be transmitted far more often than the
   dynamics require. Whether a longer exchange interval leaves the results
   unchanged has not been tested. This question also applies to
   `sic_connection` in NEST.

## Staged plan

**Stage 1. Astrocyte kernel.** Smallest state and simplest dynamics. The
regression targets already exist: the calcium statistics printed at the end of
a run, and the regime transition recorded in `docs/validation.md`. Correctness
first, performance later.

**Stage 2. Neuron kernel.** The remaining 92 %. Keep the delivery phases on the
host and accept the transfers. Measure the new phase ratio.

**Stage 3. Keep state on the device.** Once both populations are offloaded, the
per-step transfers become the main cost. Move the ring buffers and delivery
phases so that state remains on the device for the whole run.

**Stage 4. Multiple devices.** Partition the network across the superchips in a
node and measure the cost of continuous SIC coupling between them. This
addresses open question 5.

Stages 1 and 2 are supported by the measured profile. Stages 3 and 4 are where
the work produces a result rather than only a faster simulator.

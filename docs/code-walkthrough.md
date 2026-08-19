# Code walkthrough

This document explains how the simulator is organised and why it is built the
way it is. It assumes C++ but no background in neuroscience, and no prior
knowledge of this project.

The source is about 3,200 lines. Roughly 900 of those are the biological model
and the rest is supporting infrastructure.

---

## What is being simulated

The brain contains neurons, which signal by firing brief electrical pulses
called spikes, and it contains a similar number of non-neuronal cells. The most
common of these are **astrocytes**.

Astrocytes were long assumed to be purely supportive. There is now evidence
that they participate in signalling. The mechanism modelled here works as
follows.

A **synapse** is a connection between two neurons: a presynaptic neuron that
sends, and a postsynaptic neuron that receives. An astrocyte can sit at a
synapse and take part in it, making the connection three-party rather than two.
This is called **tripartite** connectivity.

When the presynaptic neuron fires, the astrocyte responds. A molecule called
IP3 accumulates inside it, and IP3 opens channels on an internal calcium store.
Calcium floods into the cell body. The release is self-reinforcing, because
calcium opens further channels, so it arrives in slow waves lasting seconds
rather than tracking the input directly.

When that calcium passes a threshold, the astrocyte injects a current into the
**postsynaptic** neuron of the same synapse. This current is called a **slow
inward current**, or SIC.

The loop is therefore closed: neurons drive astrocytes, and astrocytes drive
neurons back on a much slower timescale. The question the simulator exists to
answer is whether this loop changes what the neuronal population does
collectively.

The model comes from Jiang et al. (2025), PLOS Computational Biology 21(9):
e1013503. The original implementation is Python running on the NEST simulator.
This is an independent implementation in C++ with no simulator dependency,
written to make the model easier to move onto GPU hardware.

---

## Reading order

The files below are listed in the order that makes them easiest to follow. Each
section describes what the file contains and explains the design decisions that
the code does not make obvious on its own.

---

## 1. Core types — `include/astrosimgpu/types.hpp`

45 lines, and everything else depends on it.

```cpp
using real = double;
using index_t = std::uint32_t;
template <typename T> using vec = std::vector<T>;
```

`real` is defined once. No model code writes `double` directly, so the entire
simulator can be switched to single precision by editing this one line. Whether
single precision is sufficient for these equations is an open question, and
this arrangement makes it cheap to test.

`TimeGrid` separates two different time scales:

| | meaning | typical value |
|---|---|---|
| `dt` | the **communication step**: how often cells exchange signals | 0.1 ms |
| `substeps` | how many integration steps are taken within one `dt` | 1 |

The integration step is `dt / substeps`. Keeping these separate means the
numerical accuracy can be refined without changing how often cells communicate,
which are otherwise easy to confuse.

---

## 2. Random numbers — `include/astrosimgpu/rng.hpp`

145 lines, and the design is unusual enough to explain.

A conventional random number generator stores an internal state and advances it
with each draw. That creates a problem in parallel code. If several threads
share one generator they interfere with each other, and if each thread has its
own then the results depend on how work happened to be divided between them.

This generator stores nothing. Every value is computed directly from three
numbers:

```cpp
inline real rng_uniform(seed, stream, counter);
```

Cell 47 at timestep 900 derives its input from `(seed, 47, 900)`. It obtains
the same value regardless of which thread performs the calculation, in what
order, or whether the calculation happens on a CPU or a GPU. Reproducibility
is a consequence of the arithmetic rather than something that has to be
maintained.

`CounterRng` is a convenience class that advances the counter automatically.
The free functions exist alongside it because GPU code cannot easily use an
object that lives on the host. A test verifies that both produce identical
values; if they were ever to diverge, a GPU run would quietly stop matching the
CPU run used to validate it.

Normally distributed values use the Box-Muller transform, which produces two at
a time. The second is retained, so two consecutive requests cost one pair of
expensive function calls rather than two.

---

## 3. Parameters — `include/astrosimgpu/parameters.hpp`

Plain data structures, one group per model component: astrocyte, neuron,
synapse, connectivity, input.

Every field carries its default value and a comment giving its units and its
source. This file is the reference for what any parameter means.

Configuration files are JSON and need only list the values they change;
anything omitted keeps the default compiled in here. `src/parameters.cpp`
performs the translation.

Unrecognised keys are rejected rather than ignored. `Json::operator[]` records
every key that is asked for, and `collect_unused` afterwards reports anything
in the file that nobody read. A misspelled parameter therefore stops the run
instead of silently leaving the default in place.

---

## 4. The astrocyte model — `astrocyte.hpp`, `astrocyte.cpp`, `astrocyte_kernel.hpp`

### How cell data is stored

The conventional approach groups each cell's variables together:

```cpp
struct Astrocyte { real Ca, IP3, h, Ca_tot, ...; };
vec<Astrocyte> cells;
```

This code stores each variable in its own array instead:

```cpp
vec<real> Ca_, IP3_, h_;
vec<real> Ca_tot_, IP3_0_, tau_IP3_, delta_IP3_;
```

The reason concerns memory access. When the update loop reads the calcium
concentration of every cell in turn, the second arrangement walks straight
through one continuous block of memory. The first jumps forward by the size of
a whole cell each time, and loads six unneeded values into cache alongside each
one that is wanted.

On a CPU this affects how well the loop vectorises. On a GPU it determines
whether neighbouring threads read neighbouring addresses, which is the
difference between one memory transaction and many. Most other decisions in
this codebase follow from this one.

### The equations

Three coupled differential equations per cell, following Li and Rinzel (1994):

```
d[Ca]/dt  = J_channel - J_pump + J_leak + noise
d[IP3]/dt = ([IP3]_0 - [IP3]) / tau_IP3
dh/dt     = alpha_h (1 - h) - beta_h h
```

`J_channel` is calcium released from the internal store through IP3-controlled
channels. `J_pump` is calcium actively returned to the store. `J_leak` is
passive flow between the two. The variable `h` tracks what fraction of the
channels remain available, since calcium eventually closes them.

Two details are easily mistaken for errors:

**Calcium is clamped to `[0, Ca_tot]` after every integration step.** This is
not a safeguard against the integrator. Total calcium is conserved between the
cell body and its internal store, so the concentration in one is bounded by the
total. The clamp also prevents random fluctuations from producing a negative
concentration, which would be meaningless.

**Synaptic input is applied once at the start of a step, outside the
integration loop.** In this model an arriving spike raises IP3 instantaneously
rather than contributing to a rate of change, so it is a discrete jump rather
than a term in the equations.

### The output

```cpp
const real y = (Ca_[cell] - p_.SIC_th) * 1000.0;
if (y <= 1.0) return 0.0;
return p_.SIC_scale * std::log(y);
```

The factor of 1000 converts micromolar to nanomolar. The current begins only
once calcium exceeds the threshold by 1 nM, because the logarithm is negative
below that point. Otherwise the astrocyte produces nothing.

### Separation for GPU execution

`astrocyte_kernel.hpp` holds the per-cell calculation as free functions taking
ordinary numbers as arguments, with no reference to any object.

This is a requirement rather than a preference. A C++ member function cannot be
compiled for a GPU, because it implicitly accesses an object that exists in
host memory. Extracting the calculation was therefore a precondition for
running it on a device.

The CPU and GPU paths call the same function, so they cannot drift apart as the
code changes.

---

## 5. The neuron model — `neuron.hpp`, `neuron.cpp`

The same per-variable array layout. Seven state variables per neuron.

### The membrane equation

```
C dV/dt = -g_L (V - E_L) + g_L Delta_T exp((V - V_th)/Delta_T)
          - g_ex (V - E_ex) - g_in (V - E_in) - w + I_e + I_SIC
tau_w dw/dt = a (V - E_L) - w
```

`V` is the voltage across the cell membrane. The exponential term produces the
spike: once the voltage passes a threshold it accelerates upwards, and when it
reaches a cutoff the neuron is recorded as having fired. The voltage is then
reset and the adaptation variable `w` is increased, which makes the neuron
briefly harder to excite again. `I_SIC` is the current arriving from connected
astrocytes.

### Synaptic conductances

An arriving spike does not change the voltage directly. It opens ion channels,
and the resulting conductance rises and falls with a characteristic shape:

```
g(t) = W (t/tau) exp(1 - t/tau)
```

This is generated by two linear variables per conductance rather than evaluated
directly. An arriving spike of weight `W` adds `W * e / tau` to the first,
which then drives the second. The constant `e / tau` is what makes the peak
conductance equal exactly `W`, one time constant after arrival.

That constant is easy to get wrong, and getting it wrong scales every synapse
in the network by a fixed factor while leaving everything else looking
plausible. A test checks it specifically.

These two variables are advanced using their exact analytic solution rather
than the numerical integrator, because they are linear and do not depend on the
voltage. Only the voltage and the adaptation variable require numerical
integration.

---

## 6. The network — `network.hpp`, `network.cpp`

The largest file. Three data structures are worth understanding.

### Connection storage

```cpp
struct ConnectionSet {
    vec<index_t> row_start;   // one entry per source, plus one
    vec<index_t> target;
    vec<real> weight;
    vec<int> delay_steps;
};
```

The connections leaving source `s` occupy positions `row_start[s]` up to
`row_start[s+1]` in the other arrays. Delivering one cell's output therefore
reads a continuous stretch of memory. This is a standard sparse matrix format,
usually called compressed row storage.

There are four such sets: excitatory neuron to neuron, inhibitory neuron to
neuron, neuron to astrocyte, and astrocyte to neuron.

### Transmission delays

Signals take time to arrive. Rather than maintaining a sorted queue of pending
events, the simulator uses circular buffers:

```cpp
vec<vec<real>> ring_exc_, ring_inh_, ring_sic_, ring_astro_;
```

Each is indexed by `[slot][cell]`. A signal emitted at step `t` with a delay of
`d` steps is added into slot `(t + d) % number_of_slots`. When the simulation
reaches step `t + d`, that slot is read and cleared.

Nothing is sorted, nothing is allocated during the run, and adding a pending
signal costs one addition. The buffers stay small because the number of slots
only needs to cover the longest delay in the network.

Four separate buffers are needed because excitatory and inhibitory conductances
cannot be combined into a single number, and the astrocytic current is
continuous rather than arriving as discrete events.

### Building tripartite connectivity

`build_primary_connections` proceeds in three stages:

1. **Each postsynaptic neuron is assigned a pool of astrocytes** that it may
   later connect to. The pool is either a contiguous block or a random sample,
   and it is fixed for the run.
2. **Neuron-to-neuron connections are drawn**, each with independent
   probability `p_primary`.
3. **Each connection may then recruit an astrocyte** from the postsynaptic
   neuron's pool, with probability `p_third_if_primary`. Recruitment creates
   two further connections: from the presynaptic neuron to the astrocyte, and
   from the astrocyte back to the postsynaptic neuron.

When a neuron receives several connections that each recruit the same
astrocyte, it becomes connected to that astrocyte several times. These
duplicates are kept deliberately. The original paper states that they represent
a neuron interacting with one astrocyte at several separate synapses, and they
determine the strength of the current the neuron receives. With the default
parameters a neuron is connected to its astrocyte around sixteen times.

### The simulation loop

`Network::run` advances the simulation. Each step performs six operations:

1. `apply_arrivals`, moving signals whose delay has elapsed from the circular
   buffers into the cells
2. `drive_astrocytes`, generating background input for the astrocytes
3. `AstrocytePopulation::update`, integrating the astrocytes
4. `NeuronPopulation::update`, integrating the neurons and collecting spikes
5. `deliver_spikes`, placing emitted spikes into the buffers and adjusting
   weights for short-term synaptic changes
6. `deliver_sic`, placing astrocytic currents into the buffers

The last of these does not run every step. `synapse.sic_interval` sets how many
steps pass between exchanges, and the current holds its previous value in
between. `docs/experiments.md` records how far that can be raised before the
dynamics change.

Each operation is timed separately. The categories match those used in the
benchmarks of the original paper, so profiles from the two implementations can
be compared directly.

Two of these operations use precomputed index lists rather than examining every
cell. Only astrocytes that the connectivity actually reaches can participate,
and which ones those are is fixed when the network is built. Scanning the whole
population instead becomes the dominant cost in large networks where most
astrocytes are unconnected.

Step 2 deserves attention out of proportion to its length. It draws a Poisson
variate for every astrocyte, on one thread, every step, and it carries no
OpenMP directive. At a million astrocytes that makes it the largest single
cost in the simulation: roughly 720 us against 56 us for the kernel it feeds,
measured in `docs/profiling.md`. The loop is per-cell with no communication and
`CounterRng` keeps no state between cells, so it is already in the shape a
parallel region or a device kernel wants.

---

## 7. Analysis — `src/analysis.cpp`

Identifies calcium events by finding where the concentration crosses a
threshold, then measures how similar their timing is across the population
using pairwise correlation.

Events separated by less than a configurable interval are merged into one.
Without this, random fluctuation around the threshold splits a single physical
event into several detected ones, which distorts both the count and the
measured durations.

This file produces the quantity used to judge whether astrocytes are acting
independently or together.

---

## 8. Supporting code

These files contain no model logic and can be left until needed.

| File | Purpose |
|---|---|
| `src/json.cpp` | JSON parser, included in the source tree so the build has no external dependencies |
| `src/recorder.cpp` | Writes results as CSV |
| `src/main.cpp` | Command line handling and report formatting |
| `src/parameters.cpp` | Converts JSON into the parameter structures |

---

## 9. GPU execution

The astrocyte update runs on the host by default. Three device backends exist
behind build flags: OpenMP target offload, Kokkos, and native CUDA. All four
produce identical results.

Three pieces make that possible.

**The calculation is a free function.** `astro_advance` in
`astrocyte_kernel.hpp` takes plain scalars and returns plain scalars, knowing
nothing about the population it belongs to. A macro decorates it for whichever
backend is being compiled:

```cpp
#if defined(ASTROSIMGPU_KOKKOS)
#define ASTROSIMGPU_FN KOKKOS_INLINE_FUNCTION
#elif defined(__CUDACC__)
#define ASTROSIMGPU_FN __host__ __device__ inline
#else
#define ASTROSIMGPU_FN inline
#endif
```

That separation is what allows three device backends without three copies of
the equations. `rng.hpp` uses the same pattern for its own functions.

**One dispatch chooses where the loop runs.** `AstrocytePopulation::update`
selects at compile time. The loop body is the same call to `astro_advance` in
every case; only the construct around it changes:

| Backend | Construct |
|---|---|
| host | `#pragma omp parallel for schedule(static)` |
| OpenMP target | `#pragma omp target teams distribute parallel for` |
| Kokkos | `Kokkos::parallel_for` |
| native CUDA | `astro_update_kernel<<<grid, 128>>>` in `astrocyte_cuda.cu` |

The block size of 128 is not arbitrary: it is what the OpenMP target compiler
chose for the same loop, so neither route is handed an advantage in a
comparison. The CUDA path sits behind an interface in `astrocyte_cuda.hpp`
that names no CUDA types, so everything else still compiles with an ordinary
C++ compiler.

**State stays on the device.** `device_begin` transfers the astrocyte arrays
once at the start of a run and `device_end` brings them back at the end. Two
transfers remain each step: the synaptic input travels out, and calcium
returns because `deliver_sic` reads it on the host.

Residency introduced a bug worth knowing about, because the fix looks
arbitrary otherwise. The kernel zeroes its own copy of the input array after
consuming it. Once the arrays are resident nothing carries that zeroing back,
so the host copy accumulated indefinitely and the correlation went to 1.0000
in both regimes. `clear_inputs` zeroes the host copy instead, and it takes the
list of cells that can be non-zero rather than the whole population, because
scanning a million entries to clear a few thousand costs more per step than
the kernel does.

What this achieves and what it does not are measured in `docs/profiling.md`
and `docs/gpu-port.md`. The short version is that the kernel is not the
problem: at a million astrocytes the device is idle 91 % of the step.

---

## Verifying an understanding of the code

A quick way to confirm that a particular mechanism has been understood is to
change it deliberately and observe which test detects the change.

| Change | Expected result |
|---|---|
| In `neuron.cpp`, change `psc_init` from `e / tau` to `1 / tau` | The conductance test fails, reporting a peak too small by a factor of e |
| Remove the calcium clamp in `astro_advance` | The bounds test fails |
| Set `connectivity.unique_third_out` to `true` in `config/bursting.json` | The network no longer synchronises, because the astrocytic current is divided by the number of duplicate connections |
| Change `real` to `float` in `types.hpp` | Everything still builds. Whether the results remain correct has not been established |

Run `make test` after each.

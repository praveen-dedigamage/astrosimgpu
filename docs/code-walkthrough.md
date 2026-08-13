# Code walkthrough

A reading order, and the reason behind each decision that is not obvious from
the code itself. About 3,200 lines in total, of which roughly 900 are the
model and the rest is infrastructure.

Read the files in the order below. Each section says what the file is for, the
one idea in it worth understanding, and what to ignore.

---

## 1. `include/astrosimgpu/types.hpp` — 45 lines

Start here. It is short and everything else uses it.

```cpp
using real = double;
using index_t = std::uint32_t;
template <typename T> using vec = std::vector<T>;
```

**The idea:** `real` is a single alias. Changing it to `float` switches the
entire simulator to single precision without touching a line of model code.
That is why the models never write `double` anywhere.

`TimeGrid` holds the two time scales that matter:

- `dt` is the **communication step**, 0.1 ms. Spikes are delivered on this
  grid and it is what the reference model calls the resolution.
- `substeps` divides `dt` for integration. `h() = dt / substeps` is the actual
  RK4 step.

The distinction is the whole reason `substeps` exists: you can refine the
numerics without changing how often cells talk to each other.

---

## 2. `include/astrosimgpu/rng.hpp` — 145 lines

**The idea: no random number generator has any state.**

A conventional generator holds an internal state word and advances it on every
draw. Two threads sharing one would race; giving each thread its own means the
results depend on how work was scheduled.

Here every draw is a pure function:

```cpp
inline real rng_uniform(seed, stream, counter);
```

Cell 47 at step 900 computes its own noise from `(seed, 47, 900)` and gets the
same answer no matter which thread runs it, in what order, or whether it runs
on a GPU. Reproducibility comes from the arithmetic rather than from
discipline.

`CounterRng` is a convenience wrapper that increments the counter for you. The
free functions exist because a GPU kernel cannot easily carry an object across
the host boundary. A test pins the two to agree bit for bit — if they ever
diverge, an offloaded run silently stops matching the host run it is validated
against.

`normal()` uses Box-Muller and keeps the second variate, so two calls cost one
pair of transcendentals rather than two.

---

## 3. `include/astrosimgpu/parameters.hpp` — 197 lines

Plain structs, one per group: `AstrocyteParams`, `NeuronParams`, `SynapseParams`,
`ConnectivityParams`, `InputParams`, `ModelConfig`.

Every field has its default from NEST written in. Read this file when you want
to know what a parameter means; the comments carry the units and the source.

Nothing clever here. `src/parameters.cpp` fills these from JSON, and a key
absent from a configuration file keeps its compiled-in default — which is why
`config/quick.json` can be twenty lines.

---

## 4. `include/astrosimgpu/astrocyte.hpp` and `src/astrocyte.cpp` — the science, part one

**The data layout is the thing to understand.**

The obvious design is one struct per cell:

```cpp
struct Astrocyte { real Ca, IP3, h, Ca_tot, ...; };
vec<Astrocyte> cells;              // array of structures
```

This code does the opposite:

```cpp
vec<real> Ca_, IP3_, h_;           // structure of arrays
vec<real> Ca_tot_, IP3_0_, tau_IP3_, delta_IP3_;
```

Why it matters: when the update loop reads every cell's calcium in turn, the
second layout walks one contiguous array. The first walks a stride of nine
doubles and drags six values it does not need into cache with each one. On a
GPU it is the difference between a coalesced load and a scattered one.

Everything else follows from this choice, including how straightforward the
offload turned out to be.

**The model.** Three coupled ODEs per cell, Li-Rinzel:

```
d[Ca]/dt  = J_channel - J_pump + J_leak + noise
d[IP3]/dt = ([IP3]_0 - [IP3]) / tau_IP3
dh/dt     = alpha_h (1 - h) - beta_h h
```

`J_channel` releases calcium from the endoplasmic reticulum through IP3
receptors. `J_pump` is SERCA pumping it back. `J_leak` is passive. The
regenerative part — calcium opening more channels — is why transients come as
slow waves rather than following the input.

Two details that look like bugs and are not:

- **`Ca` is clamped to `[0, Ca_tot]` after each substep.** Total calcium is
  conserved between cytosol and store, so this is a property of the model, not
  a guard against the integrator. It is also what stops additive noise from
  producing a negative concentration.
- **Synaptic input is applied once at the top of the step, not inside the RK4
  loop.** Spikes are delta functions in the IP3 equation, so they are an
  instantaneous jump rather than a term in the derivative.

**`sic_factor(a)`** is the astrocyte's output:

```cpp
const real y = (Ca_[cell] - p_.SIC_th) * 1000.0;
if (y <= 1.0) return 0.0;
return p_.SIC_scale * std::log(y);
```

The `* 1000.0` converts µM to nM. The threshold is 1 nM *above* `SIC_th`,
because `ln y` only turns positive there. Below it the astrocyte is silent.

**`astrocyte_kernel.hpp`** holds `astro_advance` and `astro_derivatives` as free
functions taking plain scalars — no `this`, no `std::vector`, no reference into
a class. A member function cannot be offloaded to a GPU, so this separation was
a prerequisite for the device version, not a style preference. The host loop and
the device loop call the same function, so they cannot drift apart.

---

## 5. `include/astrosimgpu/neuron.hpp` and `src/neuron.cpp` — the science, part two

Same structure-of-arrays layout. Seven state variables per neuron: `V`, `w`,
and two variables for each of the excitatory and inhibitory conductances, plus
the astrocytic current.

**The membrane equation** is adaptive exponential integrate-and-fire:

```
C dV/dt = -g_L (V - E_L) + g_L Delta_T exp((V - V_th)/Delta_T)
          - g_ex (V - E_ex) - g_in (V - E_in) - w + I_e + I_SIC
tau_w dw/dt = a (V - E_L) - w
```

The exponential term is the spike mechanism: once `V` passes `V_th` it runs
away, and when it crosses `V_peak` the neuron is declared to have spiked, `V`
is reset and `w` jumps by `b`.

**The conductance cascade is the part worth understanding.** An alpha function

```
g(t) = W (t/tau) exp(1 - t/tau)
```

is not integrated directly. It is generated by two coupled linear variables:

```cpp
dg_ex -> decays with tau
g_ex  -> driven by dg_ex, decays with tau
```

and a spike of weight `W` adds `W * e / tau` to `dg_ex`. That constant is what
makes the peak conductance come out at exactly `W`, one `tau` after the spike
arrives. A test checks this, because getting it wrong scales every synapse in
the network by a constant and nothing else looks wrong.

**These two variables are propagated exactly, not through RK4.** They are
linear and independent of `V`, so the analytic solution is available and
cheaper. Only `V` and `w` go through the Runge-Kutta stages.

---

## 6. `include/astrosimgpu/network.hpp` and `src/network.cpp` — the machinery

The largest file, and where the three interesting data structures live.

### Connections in compressed-row form

```cpp
struct ConnectionSet {
    vec<index_t> row_start;   // size = sources + 1
    vec<index_t> target;
    vec<real> weight;
    vec<int> delay_steps;
};
```

Source `s` owns targets `target[row_start[s] .. row_start[s+1])`. Delivering
one neuron's spike walks a contiguous run. There are four of these sets:
excitatory neuron to neuron, inhibitory neuron to neuron, neuron to astrocyte,
and astrocyte to neuron.

### Ring buffers instead of an event queue

Delays are handled without sorting anything:

```cpp
vec<vec<real>> ring_exc_, ring_inh_, ring_sic_, ring_astro_;
```

Indexed `[slot][cell]`, where `slot = step % ring_slots_`. A spike emitted at
step `t` with delay `d` steps is added into slot `(t + d) % slots`. When the
loop reaches step `t + d`, that slot is read and cleared.

No priority queue, no sorting, no allocation during the run. The buffers are
small because delays are short — `ring_slots_` is the longest delay in steps
plus one.

There are four rings because excitatory and inhibitory conductances cannot be
summed into one number, and the astrocytic current is continuous rather than
event driven.

### Tripartite connectivity

`build_primary_connections` does three things in order:

1. **Assign each postsynaptic neuron a pool of astrocytes.** Either a
   contiguous block or a random sample without replacement, fixed for the run.
2. **Draw neuron-to-neuron connections** pairwise Bernoulli at `p_primary`.
3. **For each connection drawn, recruit an astrocyte** with probability
   `p_third_if_primary`, from that postsynaptic neuron's pool. Recruiting
   creates two further connections: presynaptic neuron to astrocyte, and
   astrocyte to postsynaptic neuron.

**Duplicates are kept deliberately.** If a neuron receives several connections
that each recruit the same astrocyte, it ends up connected to that astrocyte
several times. The paper is explicit that this represents a neuron interacting
with one astrocyte at several synapses, and it sets the scale of the current
the neuron receives — with the default parameters, about sixteenfold.

### The time loop

`Network::run` is the heart of the program. Each step:

1. `apply_arrivals` — move whatever the ring buffers hold for this step into
   the cells
2. `drive_astrocytes` — background Poisson input
3. `astro_.update` — integrate the astrocytes
4. `neurons_.update` — integrate the neurons, collect spikes
5. `deliver_spikes` — route spikes into ring buffers, applying short-term
   plasticity
6. `deliver_sic` — route astrocytic current into ring buffers

Each phase is timed separately, using the same decomposition as the reference
benchmarks so profiles can be compared.

**Two loops are restricted to what the connectivity reaches.** `sic_sources_`
and `astro_input_sinks_` are built once. Without them these phases scan the
whole population every step looking for the few cells that matter, which at ten
million astrocytes was three quarters of the run.

---

## 7. `src/analysis.cpp` — 222 lines

Detects calcium transients by threshold crossing, merges ones separated by less
than a configurable gap, then computes the mean pairwise Pearson correlation
between transient onset times across the population.

The merging exists because noise around the threshold splits one physical
transient into several detected ones, which biases both the count and the
durations.

This is the file that produces the number the whole model is judged by.

---

## 8. Infrastructure you can skip

- `src/json.cpp`, 335 lines. A JSON parser, in-tree so the build needs no
  dependencies. Read only if it breaks.
- `src/recorder.cpp`. Writes CSV.
- `src/main.cpp`. Argument parsing and report formatting.
- `src/parameters.cpp`. JSON to structs.

---

## 9. The GPU offload

Three pieces, all in `src/astrocyte.cpp` and `astrocyte_kernel.hpp`:

**The kernel functions** take plain scalars, so they can be compiled for the
device. Marked with `#pragma omp declare target`.

**The loop directive** switches between host and device:

```cpp
#ifdef ASTROSIMGPU_OFFLOAD
#pragma omp target teams distribute parallel for map(...)
#else
#pragma omp parallel for schedule(static)
#endif
```

The loop body is identical. Only the directive above it changes.

**Residency.** `device_begin` places the state on the device with
`omp target enter data` for the whole run. OpenMP map is reference counted, so
the per-step `map` clauses then find the arrays present and move nothing. Two
transfers remain each step: input goes to the device, calcium comes back
because `deliver_sic` reads it on the host.

---

## Suggested exercises

The fastest way to own this code is to break it deliberately and watch the
tests catch you.

1. In `neuron.cpp`, change `psc_init` from `e / tau` to `1 / tau`. Run
   `make test`. The alpha conductance check fails and tells you the peak
   conductance is wrong by a factor of e.
2. Remove the calcium clamp in `astro_advance`. The bounds test fails.
3. Set `connectivity.unique_third_out` to `true` in `config/bursting.json`.
   The network stops synchronising, because the SIC is divided by sixteen.
4. Change `real` to `float` in `types.hpp`. Everything builds. Compare the
   regime transition against `docs/validation.md` — this is experiment 6 and
   the answer is not known yet.

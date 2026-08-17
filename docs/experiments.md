# Planned experiments

Ordered by how much each result changes what we do next, not by how
interesting it is. Every one states in advance what outcome would mean what,
so a result cannot be read after the fact to mean whatever we hoped.

Current state, measured on one GH200 with 100 astrocytes and 500 neurons over
800,000 steps:

| | CPU, 72 Grace threads | Offload |
|---|---|---|
| astrocyte update | 2.48 s | 32.98 s |
| per step | 3.1 us | 41 us |

The offloaded run is correct and 13x slower. 41 us per step is not arithmetic
on 100 cells. The working hypothesis is that the `map` clauses inside
`AstrocytePopulation::update` allocate, copy and free eight arrays on every one
of the 800,000 steps, and that this dominates everything else.

**The profile depends on how the network is scaled, and this changes the
conclusion rather than the numbers.** At 1000 astrocytes and 5000 neurons on
Grace:

| | fixed p_primary | fixed in-degree |
|---|---|---|
| synapses | 5,001,457 | 500,189 |
| update astrocytes | 2.9 % | 4.5 % |
| update neurons | 1.1 % | 93.1 % |
| communication | ~96 % | ~2.4 % |
| mean pairwise correlation | 1.0000 | 0.0000 |

Holding the connection probability fixed while scaling the population
multiplies each neuron's input by the same factor. The network saturates, every
astrocyte correlates perfectly with every other, and the delivery phases
dominate. Holding the in-degree fixed keeps the operating point and leaves the
update phase dominant.

The reference benchmarks scale with a fixed connection probability, which is
why their delivery phases grow with scale and only the update phase
weak-scales. Whether offloading the update is worth anything therefore depends
on which scaling the science uses, and that question has to be settled before
the performance question means anything.

Experiments 1 to 4 need no new code. Experiment 5 is the fix that 1 and 2 are
meant to justify or rule out.

---

## 1. Is the cost per launch, or per byte?

**Question.** Does the offloaded astrocyte update cost a fixed amount per step
regardless of how many astrocytes there are, or does it grow with the array
size?

**Method.** Hold the step count and the neuron population fixed. Vary only the
number of astrocytes: 100, 1000, 4000, 16000. Use `pool_type: random`, which
decouples the astrocyte count from the neuron count and keeps the host work
constant. Record `update astrocytes` seconds, divide by step count.

**Outcome.**

- Flat time per step as the population grows: the cost is launch and mapping
  overhead. Keeping data resident on the device is then the entire problem, and
  experiment 5 is the whole answer.
- Time per step growing roughly linearly with the population: transfer volume
  dominates. Residency still helps but the arrays themselves are the issue, and
  what to move and when needs thought.
- Time per step falling as the population grows: the device is being starved at
  small sizes and the problem is simply too small. Then the question becomes
  what network size makes offload worthwhile at all.

**Cost.** Four short runs, about 15 minutes on `gputest`.

**Result: both, and neither answer alone was right.** There is a fixed cost per
step, and a per-cell cost that only becomes efficient once the device is
occupied. Measured on one build after the residency change:

```
CUDA           20.2 us + 0.133 ns per astrocyte
Kokkos         27.5 us + 0.131 ns per astrocyte
OpenMP target  28.4 us + 0.137 ns per astrocyte
host            5.0 us + 0.699 ns per astrocyte
```

The three outcomes listed above did not include this one. The flat region below
a thousand cells is the fixed cost showing through; the efficient region above
ten thousand is the device filling up. CUDA crosses the host at about 27,000
astrocytes, the portable routes at about 40,000.

A later four-way run added the reason for that difference: the three device
marginal costs agree to within 4 %, so the abstractions cost per launch and not
per cell. See docs/backends.md.

---

## 2. Does the host cost scale on Grace?

**Question.** How far does the per-cell update actually scale across the 72
Grace cores, and at what network size?

**Method.** `OMP_NUM_THREADS` in 1, 2, 4, 8, 16, 32, 72, at 100 and at 1000
astrocytes. CPU build only. `scripts/roihu/baseline.sbatch` already does the
thread sweep.

**Why it matters.** The 96.8 % figure in `docs/gpu-port.md` was measured on a
single laptop core. On Grace with 72 threads the astrocyte update is 26 % and
the neuron update 60 %. The profile that motivates the port is therefore
different on the target machine, and the ratio has to be quoted from the
machine we are actually porting for.

**Outcome.** If the astrocyte update stops scaling well before 72 threads at
100 astrocytes, the earlier profile was measuring thread dispatch rather than
work, and any speedup claim against the single-core number is inflated.

**Cost.** One batch job, 15 minutes.

**Result: the host baseline is genuine.** One million astrocytes, 2000 measured
steps:

| threads | update astrocytes | us/step | speedup | efficiency |
|---|---|---|---|---|
| 1 | 99.03 s | 49,515 | 1x | |
| 8 | 12.46 s | 6,232 | 7.94x | 99 % |
| 72 | 1.59 s | 797 | 62.1x | 86 % |

86 % parallel efficiency across 72 Grace cores. Every host figure quoted
elsewhere is a real many-core measurement rather than a serial run.

This check also exposed an arithmetic error in the sweep script. It divided the
phase timings by the total step count including the discarded transient,
whereas the timers only accumulate over the recorded window. Every absolute
figure it produced was therefore low by the ratio between them, a factor of
1.2. The ratios between the two columns were unaffected, since both carried the
same error, so the speedups and the crossover population did not change. The
absolute microsecond values have been restated.

---

## 3. Is the integrator step size adequate?

**Question.** Do the published measures change if the integration step is
refined?

**Method.** CPU only. Run `config/use_case.json` with `substeps` of 1, 2, 4 and
8, everything else fixed. Compare the mean pairwise correlation, the number of
calcium transients, and the mean transient duration.

**Outcome.** If the measures are stable from substeps 1 to 8, the fixed-step
RK4 choice is defensible and `dt = 0.1 ms` is fine. If they drift, the
comparison against the reference implementation is confounded by the
integrator, and that has to be said plainly before any performance claim is
made.

**Cost.** Four CPU runs. No GPU allocation needed.

**This one is worth doing first.** It is cheap, it needs no GPU, and a
negative result would undermine every other number in the repository.

---

## 4. Is the slow inward current over-communicated?

**Question.** Astrocytic calcium changes on a timescale of seconds. The SIC is
transmitted every 0.1 ms step regardless of whether anything happened. Does
exchanging it less often change the dynamics?

**Method.** Add a configuration option that updates the SIC every *k* steps and
holds it constant in between. Run `config/use_case.json` and
`config/bursting.json` with k of 1, 10, 100, 1000. Compare the mean pairwise
correlation and the firing rates against k = 1.

**Outcome.** If the measures are unchanged up to some k, the SIC is being
communicated orders of magnitude more often than the dynamics require, and that
is a property of the model rather than of any particular implementation of it.
If the measures change immediately, the continuous exchange is load-bearing and
its cost is unavoidable, which is worth knowing before designing any
distributed version around it.

**Cost.** A small code change and eight CPU runs. No GPU needed.

**This is the one whose result is about the model rather than the code.** Spike
communication in neuronal network simulators is sparse and event-driven, and
has been optimised for decades. Continuous third-factor coupling has not.

---

## 5. Device-resident state

**Question.** How much of the 41 us per step survives once the arrays stop
being mapped on every step?

**Method.** Allocate the astrocyte state on the device once with
`omp target enter data` at the start of the run, remove the `map` clauses from
the per-step region, and copy back only when recording. Repeat experiment 1.

**Outcome.** This is the change that decides whether the approach is viable.
Anything short of bringing the per-step cost near the host's 3.1 us means the
remaining overhead needs to be identified before the neuron update is offloaded
as well, because the neuron update has more state and would pay more.

**Status: done. The prediction was too conservative.**

| astrocytes | device before | device after | host | speedup |
|---|---|---|---|---|
| 1,000,000 | 522.1 us | 169.2 us | 805.1 us | 4.76x |
| 10,000,000 | 4,217.9 us | 1,370.8 us | 7,666.1 us | 5.59x |

The fixed cost per step fell from 55 to 28 microseconds, and the crossover
population with it, from about 150,000 to about 40,000.

The device improved 3.0x at both sizes. That consistency across a tenfold
change in population identifies the cause as per-step transfer rather than
anything that scales with the data.

The prediction below was 1,750 us and 3.4x at ten million; the measurement is
1,169 us and 5.10x. The mapping was costing more than the estimate allowed for.

One confound: the host baseline in the later run was built with nvc++ rather
than GCC, following a module conflict on the machine. That made the host faster
at one million, 832 to 582 microseconds, so the speedup is quoted against a
stronger baseline than the earlier figures. Against the original GCC baseline
the same device numbers would read 5.83x rather than 4.08x.

Original prediction, kept for the record:

**Implementation.** `AstrocytePopulation::device_begin`
places the state on the device for the whole run with `omp target enter data`.
The map clauses inside the update then find the arrays present and move
nothing, since OpenMP map is reference counted. Two explicit transfers remain
per step: the synaptic input goes to the device, and calcium comes back because
`deliver_sic` reads it on the host.

Predicted effect, at ten million astrocytes. Before: eight arrays mapped per
step, about 960 MB of traffic, roughly 2,100 of the measured 3,515 us. After:
two arrays, about 160 MB, roughly 350 us. That should take the astrocyte update
from 3,515 to about 1,750 us per step and the speedup against the host from
1.71x to about 3.4x.

If the measurement lands well short of that, the remaining cost is not transfer
volume and the next thing to look at is whether the runtime is really honouring
the residency.

**Cost.** Roughly an hour of work, then a repeat of experiment 1.

**Do experiments 1 and 2 first.** If experiment 1 shows the cost growing with
population size rather than staying flat, residency alone will not fix it and
the design needs rethinking rather than patching.

---

## 6. Single or double precision

**Question.** Is FP32 sufficient for the Li-Rinzel calcium dynamics?

**Method.** `real` is a single type alias in `include/astrosimgpu/types.hpp`.
Change it to `float`, rebuild, and compare the calcium statistics and the
regime transition against the FP64 results. Also compare run times.

**Why there is doubt.** The dynamics are moderately stiff, and
`Ca_ER = (Ca_tot - Ca) / ratio_ER_cyt` subtracts quantities of similar
magnitude. Cancellation there would show up as drift over the 800,000 steps of
a run rather than as an obvious error.

**Outcome.** If the regime transition survives (0.0106 against 0.4177,
approximately) and the transient statistics are close, FP32 roughly doubles
throughput on this hardware and is worth taking. If the numbers move, FP64
stays and the performance target is correspondingly lower.

**Cost.** One line changed, two builds, four runs.

---

## Order

1. Experiment 3, integrator convergence. CPU only, and everything else depends
   on the numbers being sound.
2. Experiment 2, host scaling on Grace. Fixes the baseline that all speedups
   are quoted against.
3. Experiment 1, launch overhead against data volume. Diagnoses the 13x.
4. Experiment 5, device residency. The fix, once 1 has justified it.
5. Experiment 6, precision. Independent of the others.
6. Experiment 4, SIC interval. Independent of all the above, needs no GPU, and
   has the most interesting possible outcome.

Experiments 3, 4 and 6 need no GPU allocation at all.

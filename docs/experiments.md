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
communicated orders of magnitude more often than the dynamics require. That is
a property of the model rather than of this implementation, so it applies
equally to `sic_connection` in NEST, where the same signal is sent every step.
If the measures change immediately, the continuous exchange is load-bearing and
the cost is unavoidable, which is itself worth knowing before designing a
distributed version around it.

**Cost.** A small code change and eight CPU runs. No GPU needed.

**This is the one with a result beyond this codebase.** Spike communication in
neuronal network simulators is sparse and event-driven, and has been optimised
for decades. Continuous third-factor coupling has not.

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

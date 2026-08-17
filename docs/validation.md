# Validation

What has been checked, what has not, and how to reproduce both. This is the
regression target any GPU port has to keep hitting.

## Component tests

```bash
make test
```

81 checks over the pieces where an error is quiet rather than loud: the alpha
conductance normalisation (a spike of weight W must produce a peak conductance
of W one `tau_syn` after arrival), IP3 relaxation against its time constant,
the SIC threshold, calcium remaining inside `[0, Ca_tot]` over a long undriven
run, RNG distributions, and reproducibility from a seed.

## Regime transition

The model's central claim is that astrocytic coupling can move a network
between activity regimes. Increasing the neuron-to-astrocyte weight from 0.20
to 0.31, changing nothing else, should carry the network from sparse
asynchronous firing into global synchrony.

```bash
./build/astrosimgpu --config config/use_case.json -t 120000 -o results/async
./build/astrosimgpu --config config/bursting.json -t 120000 -o results/burst
```

Result, 120 s of model time, seed 1:

| | `use_case` (w_n2a 0.20) | `bursting` (w_n2a 0.31) |
|---|---|---|
| excitatory rate | 0.109 Hz | 0.235 Hz |
| inhibitory rate | 1.964 Hz | 2.243 Hz |
| astrocytes with transients | 64 / 100 | 100 / 100 |
| transients detected | 107 | 296 |
| mean pairwise correlation | 0.0012 | **0.3429** |

The transition reproduces: near-zero correlation between astrocytic transient
onsets becomes 0.34, excitatory firing roughly doubles, and every astrocyte in
the population becomes active. This is the qualitative behaviour the model
exists to demonstrate.

## Silent-neuron control

With neuronal input and synaptic weights set to zero, astrocytes are driven
only by their own Poisson input:

```bash
./build/astrosimgpu --config config/no_spiking.json -t 60000 -o results/control
```

Zero spikes, sporadic calcium transients in 10 of 100 astrocytes, mean pairwise
correlation -0.019. Activity-independent calcium is present but uncorrelated,
which is what the control is for: it establishes that the correlation seen in
the bursting regime comes from the network and not from the astrocytes' own
input.

This control is also a useful canary. Three separate implementation errors
were caught by noticing that the default configuration had become
indistinguishable from it.

## Comparison against a published value

The paper reports a mean neuronal firing rate for its "Sparse" benchmark model:
4.76 spikes/s at 20,000 cells, averaged over nine simulations, against 11.95
for the "Synchronous" variant.

`config/paper_sparse.json` reproduces that model. Its parameters are taken from
`model_default` in the reference benchmark's `network.py`: 8000 excitatory,
2000 inhibitory and 10,000 astrocytes, pairwise Bernoulli primary connections
at p = 0.1, random astrocyte pools of ten with p_third = 0.5, a 2000 Hz Poisson
drive, and no parameter randomisation. Anything not set there is a NEST default
and is left at the compiled-in default here.

```bash
for s in 1 2 3; do
  ./build/astrosimgpu --config config/paper_sparse.json -t 10000 -p 1000 -s $s -o /tmp/paper_s$s
done
```

| | mean firing rate |
|---|---|
| seed 1 | 4.7390 Hz |
| seed 2 | 4.7309 Hz |
| seed 3 | 4.7636 Hz |
| mean of three | 4.7445 Hz |
| paper, mean of nine | 4.76 spikes/s |

The agreement is 0.33 %, and the published value falls inside the seed-to-seed
spread of these three runs.

What that does and does not establish:

- It exercises the neuron model, the astrocyte model, the tripartite
  connectivity rule with random pools, the short-term plasticity, the Poisson
  input and the SIC coupling together, and they produce the same collective
  firing rate as NEST. Short-term plasticity in particular was the leading
  suspect for a systematic discrepancy, and at least in aggregate it is not
  one.
- It is one observable on one model. Agreement in mean firing rate does not
  establish agreement in the calcium statistics, in transient timing, or in
  the synchrony measures.
- It validates the machinery, not `config/use_case.json`. The two share the
  cell models and the connectivity rule but not the parameters.
- The paper gives no spread for its nine simulations, so it is not possible to
  say whether this result sits inside theirs.

Note that this agreement holds at `substeps = 1`, where the calcium transient
count is demonstrably not converged. The firing rate appears to be far less
sensitive to step size than the calcium measures are.

## Step size dependence

The reported values depend on the integration step. `config/use_case.json` at
60 s of model time, seed 1, varying only `substeps`:

| substeps | transients detected | mean pairwise correlation |
|---|---|---|
| 1 | 56 | 0.0106 |
| 2 | 63 | 0.0140 |
| 4 | 68 | 0.0170 |
| 8 | 71 | 0.0208 |

The transient count is converging: the increments are 7, 5 and 3, heading for
somewhere near 73 to 75. The default of `substeps = 1` therefore undercounts
transients by roughly 20 %.

The correlation is not converging in the same way. Its increments are 0.0034,
0.0030 and 0.0038, with no sign of shrinking. On a quantity this close to zero,
in a simulation driven by noise, that is more likely to be scatter in the
measure than error in the integrator. Distinguishing the two requires several
seeds at each level, which has not been done.

What this does and does not affect:

- **The regime transition is unaffected.** Asynchronous is 0.0106 to 0.0208
  across the whole range; bursting is 0.4177. The qualitative result has a
  twentyfold margin.
- **Quantitative values are not converged.** A number like "mean pairwise
  correlation 0.0106" is a property of this discretisation, not of the model.
  Comparing it against the reference implementation would compare two
  discretisation errors.

Use `substeps` of 4 or more for anything quantitative, and state the value
alongside the result. NEST uses an adaptive GSL solver here, which controls
this automatically; that is difference 1 in the README, and this table is what
makes it concrete.

There is a consequence for the GPU port. Raising `substeps` multiplies the
arithmetic per communication step without adding any communication, so the
update phase grows as a fraction of the run. A converged configuration is
therefore more offload-friendly than the default one, not less.

## What has not been checked

Stated explicitly, because the gap matters more than the agreement:

- **Only one published value has been compared.** The mean firing rate of the
  "Sparse" benchmark model agrees to 0.33 %, as above. Calcium transient
  durations, transient timing and the magnitude of the synchrony measures have
  not been compared against the published figures or against a NEST run.
- **The excitatory and inhibitory populations fire at very different rates**
  (roughly 0.1 Hz against 2 Hz). This follows from the parameters -- the
  inhibitory cells have a leak conductance of 4.3 nS against 18 nS and sit
  much closer to threshold under their Poisson drive -- but whether the
  reference shows the same asymmetry is unverified.
- **Short-term plasticity** is implemented in the published Tsodyks-Markram
  form, and NEST's `tsodyks_synapse` update order has still not been diffed
  against it line by line. The firing rate comparison above suggests the
  aggregate behaviour matches, which is weaker than a diff but not nothing.
- **NEST defaults** were read from source rather than from a running install.
  `scripts/dump_nest_defaults.py` produces a file to diff against
  `config/use_case.json`; one value, `delta_IP3`, remains unconfirmed.

## Errors found during validation

Recorded because each was silent, each changed the science rather than the
performance, and each is a plausible mistake for a reimplementation to make:

1. **Synaptic time constants taken from the wrong place.** The reference
   parameter file lists `tau_syn_ex = 2 ms` under its synaptic parameters, but
   passes it as the Tsodyks synapse's `tau_psc`; the neuron keeps the
   `aeif_cond_alpha_astro` default of 0.2 ms. Applying the 2 ms value to the
   neuron multiplied mean excitatory conductance by ten and put the network at
   22.7 Hz.
2. **Noise interval.** NEST's `noise_generator` holds its current constant for
   `dt`, which defaults to ten times the resolution. Redrawing every step cut
   the resulting membrane fluctuation by about the square root of ten.
3. **Shared noise.** That generator gives every target an independent current.
   A single shared draw removed the between-cell heterogeneity it exists to
   provide.
4. **Deduplicated SIC connections.** The third-factor rule attaches an
   astrocyte per primary connection and explicitly permits the same astrocyte
   to be attached repeatedly to the same target. Collapsing those duplicates
   divided the slow inward current by roughly sixteen.

Errors 2 and 3 together silenced the excitatory population entirely -- zero
spikes over 300 s of model time -- which made the neuron-to-astrocyte pathway
inert and left the default configuration behaving exactly like the
silent-neuron control. Error 4 then made the astrocyte-to-neuron weight have
no measurable effect on the network.

## Step size, with seeds

The earlier table used one seed, which cannot separate discretisation error from
the scatter of a noise-driven simulation. Five seeds per level, mean and
standard deviation, `config/use_case.json` at 60 s:

| substeps | transients | correlation | rate (Hz) |
|---|---|---|---|
| 1 | 55.2 +/- 7.1 | 0.0051 +/- 0.0067 | 0.4811 |
| 2 | 65.8 +/- 7.3 | 0.0076 +/- 0.0032 | 0.5009 |
| 4 | 66.2 +/- 8.6 | 0.0095 +/- 0.0043 | 0.5060 |
| 8 | 66.6 +/- 6.4 | 0.0100 +/- 0.0063 | 0.5094 |

Reproduce with `bash scripts/convergence.sh`.

**`substeps = 1` is under-resolved; `substeps = 2` is enough.** The transient
count moves 55.2 to 65.8 between substeps 1 and 2, which is 3.3 standard errors
and therefore systematic, then 0.4 and 0.4 across the next two doublings, well
inside the scatter.

**The correlation drift reported earlier was mostly noise.** Across five seeds
the shift from substeps 1 to 8 is 0.0049 against a standard error of 0.0030,
about 1.6 sigma. At this network size the measure cannot resolve a step-size
effect. The single-seed table should not have been read as one.

**Firing rate is 4 % low at `substeps = 1`** on this configuration, which raised
a question about the benchmark comparison above, since that was measured at the
same setting. It was checked and the concern does not apply:

| `config/paper_sparse.json` | mean firing rate | against 4.76 |
|---|---|---|
| substeps = 1, three seeds | 4.7445 | 0.33 % low |
| substeps = 2, three seeds | 4.7617 +/- 0.026 | 0.04 % low |

The published value falls inside the seed scatter at the converged setting, so
the agreement improves rather than degrades. The step sensitivity is real but
eleven times smaller here than on `use_case`, 0.36 % against 4.1 %: the
benchmark network is driven hard by a 2000 Hz Poisson input and sits far from
the marginal regime where the integrator matters.

Use `substeps = 2` for quantitative work. It converges the calcium statistics
and costs twice the arithmetic, not the four to eight times a less careful
reading of the single-seed table would have suggested.

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

- **No quantitative comparison against the reference implementation.** The
  regime transition reproduces qualitatively. Absolute firing rates, transient
  durations, and the magnitude of the correlation have not been compared
  against the published figures or against a NEST run.
- **The excitatory and inhibitory populations fire at very different rates**
  (roughly 0.1 Hz against 2 Hz). This follows from the parameters -- the
  inhibitory cells have a leak conductance of 4.3 nS against 18 nS and sit
  much closer to threshold under their Poisson drive -- but whether the
  reference shows the same asymmetry is unverified.
- **Short-term plasticity** is implemented in the published Tsodyks-Markram
  form; the reference uses NEST's `tsodyks_synapse`, whose update order has
  not been diffed against it.
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

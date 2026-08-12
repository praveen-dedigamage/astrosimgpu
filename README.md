# astrosimgpu

**Status: the default build is CPU-only.** A first offload path exists for the
astrocyte update, behind a build flag that is off by default. It has not been
compiled by an offloading compiler or run on a GPU. See "GPU offload" below and
`docs/gpu-port.md` for the plan.

A C++17 simulator for neuron-astrocyte networks. It contains:

- Adaptive exponential integrate-and-fire (AdEx) neurons with conductance-based
  synapses
- Li-Rinzel astrocytes with IP3-driven calcium dynamics
- A slow inward current (SIC) from astrocytes back to neurons
- Tripartite connectivity between the two populations

It reimplements the model from:

> Jiang H-J, Aćimović J, Manninen T, Ahokainen I, Stapmanns J, Lehtimäki M,
> Diesmann M, van Albada SJ, Plesser HE, Linne M-L (2025).
> *Modeling neuron-astrocyte interactions in neural networks using distributed
> simulation.* PLOS Computational Biology 21(9): e1013503.
> <https://doi.org/10.1371/journal.pcbi.1013503>

The reference implementation is Python running on the NEST simulator. It is
published at <https://doi.org/10.5281/zenodo.13757202>. This implementation is
standalone C++ and does not use NEST.

## Purpose

The reference implementation is the authority on the model. This one was
written for three reasons:

1. It builds with only a C++17 compiler. No NEST, MPI, GSL, or conda
   environment is needed.
2. The integrator, time step, and substepping are set in the configuration file
   rather than inside a simulator kernel.
3. The data layout is prepared for GPU offload. Populations are stored as
   structure-of-arrays and each cell update writes only its own array element.
   The astrocyte update has already been restructured into device-callable
   free functions.

Use the reference implementation for published results.

## Build

```bash
make -j
```

To build with OpenMP:

```bash
make OPENMP=1 -j
```

Apple clang does not include OpenMP, so on macOS the default build is
single-threaded. The OpenMP pragmas are ignored and results are unchanged.

CMake is also supported:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASTROSIMGPU_NATIVE=ON
cmake --build build -j
```

## GPU offload

The astrocyte update has been restructured so that it can run as an OpenMP
target region. This is the first step of the port described in
`docs/gpu-port.md`.

The per-cell code was moved out of `AstrocytePopulation` into free functions in
`include/astrosimgpu/astrocyte_kernel.hpp`. These take plain scalars only, with
no `this` pointer, no `std::vector`, and no reference into a class. A member
function cannot be offloaded, so this restructuring was needed before any
directive could be added. The random draws were moved to free functions in
`rng.hpp` for the same reason.

The host loop and the offloaded loop call the same `astro_advance` function.
Only the directive above the loop differs. The arithmetic is therefore covered
by the existing tests through the host path, and a test checks that
`rng_normal` matches `CounterRng` exactly.

To build with offload enabled:

```bash
make OFFLOAD=1 OFFLOAD_FLAGS="-mp=gpu -gpu=cc90" CXX=nvc++
```

Or with CMake, `-DASTROSIMGPU_OFFLOAD=ON` plus the offload flags for your
compiler.

Three limitations:

1. **Off by default.** The default build contains no device code.
2. **Astrocyte only.** The neuron update is 92 % of run time and is still
   host-only. It emits spikes, so the per-thread spike collection has to be
   reworked before it can be offloaded.
3. **Not tested on a GPU.** This code has never been compiled by an offloading
   compiler or run on a device. It compiles with the macro defined and the
   pragmas ignored, which shows only that the syntax is valid. The first real
   build should be expected to fail.

The restructuring was checked to reproduce the previous results exactly: mean
pairwise correlation 0.0106 in the asynchronous regime and 0.4177 in the
bursting regime, unchanged to four decimal places.

## Run

```bash
make test                                      # 86 component checks
./build/astrosimgpu --config config/quick.json # smoke test, about 0.3 s
./build/astrosimgpu --config config/use_case.json
```

Command line options:

```
-c, --config PATH     parameter file (JSON)
-o, --output DIR      output directory
-s, --seed N          random seed
-t, --sim-time MS     recorded duration
-p, --pre-time MS     discarded transient
    --threads N       OpenMP threads
    --no-analysis     skip the post-run summary
```

## Configurations

| File | Description |
|---|---|
| `config/use_case.json` | Block astrocyte pools, sparse asynchronous firing. Default. |
| `config/bursting.json` | Same, with a stronger neuron-to-astrocyte weight. The network becomes synchronised. |
| `config/no_spiking.json` | Neurons silent. Astrocytes receive only their own Poisson input. Control condition. |
| `config/random_pool.json` | Random astrocyte pools of five, lower recruitment probability, larger SIC weight. |
| `config/quick.json` | Small and short. For checking a build only. |

## Model equations

**Astrocyte** (Li & Rinzel 1994, extended by Nadkarni & Jung 2003). State
variables are `[Ca2+]`, `[IP3]` and `h_IP3R`:

```
d[Ca]/dt  = J_channel - J_pump + J_leak + J_noise
d[IP3]/dt = ([IP3]_0 - [IP3]) / tau_IP3 + delta_IP3 * J_syn(t)
dh/dt     = alpha_h (1 - h) - beta_h h

[Ca]_ER   = ([Ca]_tot - [Ca]) / r_ER,cyt
m_inf     = [IP3] / ([IP3] + Kd_IP3_1)
n_inf     = [Ca] / ([Ca] + Kd_act)
alpha_h   = k_IP3R Kd_inh ([IP3] + Kd_IP3_1) / ([IP3] + Kd_IP3_2)
beta_h    = k_IP3R [Ca]
J_channel = r_ER,cyt v_IP3R m_inf^3 n_inf^3 h^3 ([Ca]_ER - [Ca])
J_pump    = v_SERCA [Ca]^2 / (Km_SERCA^2 + [Ca]^2)
J_leak    = r_ER,cyt v_L ([Ca]_ER - [Ca])
```

Total calcium is conserved between cytosol and ER. Cytosolic calcium is
therefore held in the range `[0, Ca_tot]` after each step.

**SIC output.** The value is unitless. The pA unit comes from the
astrocyte-to-neuron weight:

```
F_SIC = SIC_scale * H(ln y) * ln y,    y = ([Ca] - SIC_th) / nM
```

`H` is the Heaviside function. No current is produced until calcium exceeds
the threshold by 1 nM.

**Neuron.** AdEx with alpha-shaped conductances and the astrocytic current:

```
C_m dV/dt = -g_L (V - E_L) + g_L Delta_T exp((V - V_th)/Delta_T)
            - g_ex (V - E_ex) - g_in (V - E_in) - w + I_e + I_stim + I_SIC
tau_w dw/dt = a (V - E_L) - w
```

When `V >= V_peak` the neuron spikes, `V` is set to `V_reset`, and `w` is
increased by `b`. Each conductance is an alpha function driven by a
two-variable cascade. It is normalised so that a spike of weight `W` gives a
peak conductance of `W` at one `tau_syn` after the spike arrives.

**Connectivity.** Neuron-to-neuron connections are drawn pairwise Bernoulli
with probability `p_primary`. Each connection then recruits an astrocyte with
probability `p_third_if_primary`. The astrocyte is taken from a pool assigned
to the postsynaptic neuron, either as a contiguous block or at random. A
recruited astrocyte receives the presynaptic spikes and sends a SIC to the
postsynaptic neuron.

## Output

CSV files are written to the configured output directory:

```
spikes.csv      time_ms,neuron
astrocytes.csv  time_ms,astrocyte,Ca,IP3
neurons.csv     time_ms,neuron,V_m,I_SIC
run.txt         network sizes, timings, and the calcium summary
```

`scripts/plot_results.py <dir>` produces a spike raster, the population firing
rate, and calcium traces. It requires numpy and matplotlib. The simulator does
not require either.

At the end of a run the program reports the number of astrocytes that produced
calcium transients, the mean transient duration, and the mean pairwise
correlation between transient onsets. The last of these is the measure of
astrocyte-supported synchronisation used in the reference analysis.

## Repository layout

```
include/astrosimgpu/   headers: cell models, network, analysis, config, RNG
src/                   implementation and command-line front end
config/                parameter sets, one per regime
tests/                 component tests, no external framework
scripts/               plotting, NEST default dump, Roihu build and baseline
docs/                  validation results, GPU port plan, Roihu notes
```

## Performance profile

Each run reports wall-clock time split by phase. The phases match the
decomposition used in the reference benchmarks (update, spike delivery, SIC
gathering and delivery), so the two profiles can be compared. Single-threaded
on `config/use_case.json`:

```
update astrocytes        4.3 %
update neurons          92.5 %
spike delivery           0.1 %
SIC gather+deliver       0.6 %
arrival application      1.6 %
```

Per-cell state update is 96.8 % of the run time. That work is parallel across
cells. Communication is 2.3 % at this network size. `docs/gpu-port.md` explains
what this means for a GPU port, including the speedup limit it implies.

## Parameter sources

The following come from the reference model's parameter file and the
regime-specific settings in its driver script:

- Fitted astrocyte parameters: `Ca_tot`, `IP3_0`, `tau_IP3`, `delta_IP3`
- Neuron parameters
- Synaptic weights and delays
- Connectivity probabilities
- Input rates

Remaining astrocyte parameters use the `astrocyte_lr_1994` defaults. Remaining
neuron parameters use the `aeif_cond_alpha_astro` defaults.

**One parameter is easy to get wrong.** The reference parameter file lists
`tau_syn_ex` and `tau_syn_in` under its synaptic parameters, both set to 2 ms.
These are passed to the Tsodyks synapse as `tau_psc`. They are not applied to
the neuron. The neuron keeps the `aeif_cond_alpha_astro` defaults of 0.2 ms and
2.0 ms. Using 2 ms for the neuron's excitatory conductance increases the mean
excitatory drive by a factor of ten and changes the network regime. Both values
are therefore stated explicitly in the neuron configurations here.

**Two properties of the background noise also matter and are not visible in the
parameter file.** NEST's `noise_generator` gives each target an independent
current, and holds that current constant for `dt`, which defaults to ten times
the resolution. Drawing a new value every step reduces the membrane potential
fluctuation by about a factor of sqrt(10). With these parameters that is enough
to stop the excitatory population from firing.

The remaining defaults were taken from the NEST source code, not from a running
installation. Verify them before making a quantitative comparison:

```bash
python scripts/dump_nest_defaults.py > nest_defaults.json   # requires NEST
```

Then compare against `config/use_case.json`. One value, `delta_IP3`, could not
be confirmed. Every configuration in this repository sets it explicitly.

## Differences from the reference implementation

The two simulators should agree statistically. They will not produce identical
traces.

1. **Integration.** Fixed-step RK4 with configurable substeps, rather than the
   adaptive GSL RKF45 solver NEST uses. The linear conductance cascade is
   propagated exactly instead of through the RK stages. Increase `substeps` to
   check whether a result depends on the step size.
2. **Short-term plasticity.** Implemented in the published Tsodyks-Markram
   form. NEST's `tsodyks_synapse` update order has not been compared against
   it. Set `synapse.stp.enabled` to `false` for static weights.
3. **Random numbers.** A different generator, so the same seed gives a
   different noise realisation. Runs are reproducible within this simulator
   only.
4. **Background noise.** Independent current per target, piecewise constant
   over `noise_dt`, defaulting to ten times the resolution. This matches NEST.
5. **Duplicate SIC connections are kept.** The third-factor rule attaches one
   astrocyte per primary connection, and the same astrocyte can be attached
   several times to the same target neuron. With the default parameters each
   neuron connects to its astrocyte about sixteen times. Removing the
   duplicates divides the slow inward current by the same factor. Set
   `connectivity.unique_third_out` to compare.
6. **No distributed execution.** Threads within one process only. There is no
   MPI support. Use the reference implementation for distributed runs.

## Licence

GPL-3.0. The model, its fitted parameters, and the analysis procedures come
from the reference implementation, which is GPL-3.0.

## References

- Jiang H-J *et al.* (2025) PLOS Comput Biol 21(9): e1013503.
- Li YX, Rinzel J (1994) *Equations for InsP3 receptor-mediated [Ca2+]i
  oscillations derived from a detailed kinetic model.* J Theor Biol
  166:461-473.
- De Young GW, Keizer J (1992) PNAS 89:9895-9899.
- Nadkarni S, Jung P (2003) Phys Rev Lett 91:268101.
- Brette R, Gerstner W (2005) *Adaptive exponential integrate-and-fire model.*
  J Neurophysiol 94:3637-3642.
- Tsodyks M, Uziel A, Markram H (2000) J Neurosci 20:RC50.

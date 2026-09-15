# Optimization roadmap

A staged plan for making the simulation loop faster, derived from the measured
profile (`docs/gpu-port.md`, `docs/profiling.md`) and confirmed independently
on Roihu (job 1293182 for the CUDA throughput sweep, job 1293360 for the nsys
phase breakdown, both on `rg2101`, one GH200). Work through it top to bottom;
each stage names what to change, why, the expected effect, and how to check it
didn't break correctness. Correctness first, performance later, throughout --
this project's own validation record (`docs/validation.md`) is a list of
changes that looked fine and were not.

## Stage 0 -- Decide the scaling regime (blocking decision, no code)

Everything below only pays off under **fixed in-degree** scaling. Under
**fixed connection probability** scaling, the delivery phases dominate
instead (96% of runtime at 1000 astro / 5000 neurons) and the update phases
this roadmap targets cap out near 4% -- offloading them buys ~1.04x, full
stop.

- [ ] Decide which scaling the target science actually uses (biologically,
      fixed in-degree is the defensible choice: a cortical neuron has ~10^4
      synapses regardless of brain size)
- [ ] Write the decision down somewhere durable (this file, or
      `docs/gpu-port.md`) so it isn't re-litigated per stage

**Do not start Stage 2 or later until this is settled.**

## Stage 1 -- Instrument before optimizing

- [ ] Add NVTX ranges around each phase already tracked by `PhaseProfile` in
      `Network::run` (`src/network.cpp`): `input_gen`, `update_astro`,
      `update_neuron`, `spike_cd`, `sic_gd`, `deliver`
- [ ] Gate the NVTX calls behind the CUDA/offload build flags so the default
      host-only build is untouched
- [ ] Re-run `scripts/roihu/nsys_profile.sbatch` and confirm the timeline now
      shows named ranges instead of requiring the phase-counter-to-row mental
      mapping we've been doing by hand

Cost: near zero, no logic change. Payoff: every report from here on
self-explains, and Stage 2's "did it work" check gets easier to read.

**Validation:** none needed -- this only adds instrumentation.

## Stage 2 -- Kill the two confirmed waste sources

Highest-confidence stage: both items were reproduced on our own Roihu run
this session, not just read from the docs.

### 2a. Parallelize / offload `drive_astrocytes`

- Evidence: 50.08% of wall time at 100k astrocytes (our nsys run,
  `nsys-1293360.out`); 720us vs. the kernel's 56us at 1M astrocytes
  (`docs/profiling.md`)
- Why it's easy: per-cell, no cross-cell state (`CounterRng` needs nothing
  shared between astrocytes)

- [ ] Quick win: add `#pragma omp parallel for` to the loop in
      `Network::drive_astrocytes` (`src/network.cpp`)
- [ ] Re-measure with `scripts/roihu/nsys_profile.sbatch` -- `background
      input` share should drop sharply
- [ ] Fuller fix: move Poisson generation onto the device entirely (new small
      kernel in `astrocyte_cuda.cu`), which also deletes the Host-to-Device
      transfer since there's nothing left to assemble on the host

### 2b. Stop moving the full astrocyte array every step

- Evidence: our nsys run showed H2D/D2H transfers of exactly 0.8MB each way
  (100,000 doubles) every step, while only 8,006 of 100,000 astrocytes
  (8%) are actually connected to a neuron
- `docs/experiments.md` already validated that `sic_interval = 10` leaves
  the regime-transition dynamics unchanged

- [ ] Gate `cuda_astro_pull_calcium` (called from `Network::run`) on
      `step % cfg_.syn.sic_interval == 0`, matching how `deliver_sic` is
      already gated
- [ ] Confirm no accuracy change (see validation below)
- [ ] Larger follow-on (optional, separate from this stage): move SIC
      generation onto the device so the transfer disappears rather than
      being merely gated -- this touches ring-buffer indexing in
      `deliver_sic`, so treat it as its own piece of work

**Validation for both 2a and 2b:** regime-transition mean pairwise
correlation must still match to 4 decimal places --
`0.0106` (`config/use_case.json`) and `0.4177` (`config/bursting.json`) --
exactly as confirmed on the current CUDA build. Also re-run
`make test` (86 checks) and `scripts/roihu/cuda_only.sbatch`.

## Stage 3 -- Offload the neuron update

- Evidence: the actual largest phase in realistic configurations -- 92.5% of
  wall time on one core, still 60.5% on 72 Grace cores, for
  `config/use_case.json` (`docs/gpu-port.md`). The astrocyte kernel was
  written first because it's simpler (3 state variables), not because it's
  the bigger win.
- Harder than Stage 2: `NeuronPopulation::update` emits spikes, so the
  per-thread spike collection (currently merged after an OpenMP loop, see
  `src/neuron.cpp`) needs a device-writable structure before it can move.

- [ ] Move the per-cell body of `NeuronPopulation::update` into free
      functions, the same way `astro_derivatives`/`astro_advance` were
      extracted into `astrocyte_kernel.hpp` (no `this`, no `std::vector`)
- [ ] Design a device-compatible spike collection scheme (atomics, a
      per-thread buffer with a compaction pass, or similar)
- [ ] Write `neuron_cuda.cu` mirroring `astrocyte_cuda.cu`'s structure:
      `CudaNeuron` opaque handle, `device_begin/end`, `device_push_input`,
      `device_pull_spikes` (or equivalent)
- [ ] Wire into `Network::run` the same way the astrocyte backend is wired

**Validation:** same as Stage 2 -- regime transition, 86 component tests,
plus a new test comparing spike timing/counts between host and device paths
if that isn't already covered.

**Do this only after Stage 2 is measured**, so the win gets attributed
correctly rather than confounded with Stage 2's changes.

## Stage 4 -- Keep everything resident, delivery included

- Evidence: even now, transfers are comparable in magnitude to the kernel
  itself (our nsys run: ~6.6us D2H + ~4.1us H2D vs. ~10us kernel at 100k
  astrocytes). Once both updates are on-device, per-step transfers -- not
  the kernels -- become the dominant cost.

- [ ] Move the ring buffers (`ring_exc_`, `ring_inh_`, `ring_sic_`,
      `ring_astro_` in `include/astrosimgpu/network.hpp`) to device memory
- [ ] Move `apply_arrivals`, `deliver_spikes`, `deliver_sic` to the device
- [ ] State should then stay resident for the entire run, transferring
      nothing per step except what's recorded (spikes/traces), and only on
      recording steps

**Validation:** same regression suite. This is the point where the
register-pressure question (Stage 5) starts to matter again, because by
then the kernel becomes the bottleneck rather than transfers.

## Stage 5 -- Kernel-internal tuning (lowest priority)

- Evidence (`docs/profiling.md`, reproducible with
  `scripts/roihu/ncu_profile.sbatch`): the astrocyte kernel is capped at
  31.25% theoretical occupancy by 94 registers/thread, not spilling, not
  divergent, not bandwidth-bound -- 42% of cycles stall on the
  fixed-latency RK4 dependency chain
- This addresses ~56us of the ~1,438us step gap -- roughly 25x less than
  Stages 2-4 combined

- [ ] Sweep `-maxrregcount` (94 -> 64 -> 48) on the CUDA build; watch for
      `Local Memory spill` appearing in `ncu` output (that would mean the
      sweep went too far)
- [ ] Evaluate single precision: halves register pressure and arithmetic
      cost, but the calcium ODE is moderately stiff
      (`Ca_ER = (Ca_tot - Ca) / ratio_ER_cyt` subtracts similar
      quantities) -- validate against the regime transition, not by eye
- [ ] If Stage 3's neuron kernel exists by this point, ask the same
      questions of it (`ncu_profile.sbatch` with `-k regex:neuron` or
      similar)

**Validation:** regime transition + 86 tests, plus explicitly check the
calcium statistics (transient count, correlation) aren't just
"close enough" -- `docs/validation.md`'s step-size table is the model for
how to report this kind of precision-vs-accuracy tradeoff.

## Net ordering

Stage 0 (gate) -> Stage 1 (near-zero cost, unblocks trustworthy
measurement) -> Stage 2 (biggest confirmed win, smallest risk) -> Stage 3
(biggest remaining win, largest effort) -> Stage 4 (only matters once 2+3
are done) -> Stage 5 (smallest win, most GPU-architecture-specific).

Re-run `scripts/roihu/nsys_profile.sbatch` after every stage and diff the
phase percentages against the previous stage's numbers -- that's the signal
for whether to move on or dig further into the current stage.

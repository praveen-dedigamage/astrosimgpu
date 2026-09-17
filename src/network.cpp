#include "astrosimgpu/network.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

// Labels the phases below on the nsys timeline. Header-only in NVTX v3 --
// nothing to link -- so this only needs the include path, wired in
// CMakeLists.txt for the CUDA build. A no-op everywhere else, so the default
// host-only build carries no dependency on it.
#if defined(ASTROSIMGPU_CUDA)
#include <nvtx3/nvToolsExt.h>
#define ASTROSIMGPU_NVTX_PUSH(name) nvtxRangePushA(name)
#define ASTROSIMGPU_NVTX_POP() nvtxRangePop()
#else
#define ASTROSIMGPU_NVTX_PUSH(name)
#define ASTROSIMGPU_NVTX_POP()
#endif

namespace astrosimgpu {

namespace {

// Flatten a per-source adjacency list into compressed row form.
template <typename T>
void flatten(const vec<vec<T>>& per_source, vec<index_t>& row_start, vec<T>& flat) {
  row_start.assign(per_source.size() + 1, 0);
  index_t total = 0;
  for (std::size_t s = 0; s < per_source.size(); ++s) {
    row_start[s] = total;
    total += static_cast<index_t>(per_source[s].size());
  }
  row_start[per_source.size()] = total;

  flat.clear();
  flat.reserve(total);
    for (const auto& row : per_source) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
}

int delay_to_steps(real delay_ms, real dt) {
  const int steps = static_cast<int>(delay_ms / dt + 0.5);
  return steps < 1 ? 1 : steps;
}

std::uint64_t pair_key(index_t a, index_t b) {
  return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
}

}  // namespace

void ConnectionSet::finalise(index_t sources, const StpParams& stp) {
  if (row_start.size() != sources + 1) {
    row_start.assign(sources + 1, size());
  }
  if (stp.enabled) {
    stp_x.assign(size(), 1.0);
    stp_u.assign(size(), stp.U);
    stp_t_last.assign(size(), -1e12);
  }
}

Network::Network(const ModelConfig& config) : cfg_(config) {}

void Network::build() {
  CounterRng rng(cfg_.seed, 0x5EED);

    astro_.build(cfg_.N.N_astro, cfg_.astro, cfg_.randomize_astro, cfg_.input_astro, rng);
  neurons_.build(cfg_.N.N_exc, cfg_.N.N_inh, cfg_.neuron_exc, cfg_.neuron_inh,
                 cfg_.randomize_neuron, cfg_.input_exc, cfg_.input_inh, rng);

  build_primary_connections(rng);

  stats_.n_astro = cfg_.N.N_astro;
  stats_.n_exc = cfg_.N.N_exc;
  stats_.n_inh = cfg_.N.N_inh;
  stats_.n_primary_exc = exc_primary_.size();
  stats_.n_primary_inh = inh_primary_.size();
  stats_.n_neuron_to_astro = neuron_astro_.size();
  stats_.n_astro_to_neuron = astro_neuron_.size();

  // astro_neuron_ is CSR keyed by astrocyte, so each row's span is that
  // astrocyte's actual out-degree -- what max_astro_out_degree caps.
  stats_.max_astro_out_degree = 0;
  for (index_t a = 0; a + 1 < static_cast<index_t>(astro_neuron_.row_start.size()); ++a) {
    const index_t degree = astro_neuron_.row_start[a + 1] - astro_neuron_.row_start[a];
    stats_.max_astro_out_degree = std::max(stats_.max_astro_out_degree, degree);
  }

  // Ring depth covers the longest delay.
  const real dt = cfg_.time.dt;
  int max_delay = 1;
  max_delay = std::max(max_delay, delay_to_steps(cfg_.syn.d_e, dt));
  max_delay = std::max(max_delay, delay_to_steps(cfg_.syn.d_i, dt));
  max_delay = std::max(max_delay, delay_to_steps(cfg_.syn.d_a2n, dt));
  ring_slots_ = max_delay + 1;

  // Collect once rather than rediscovering every step.
  sic_sources_.clear();
  for (index_t a = 0; a < cfg_.N.N_astro; ++a) {
    if (astro_neuron_.row_start[a] < astro_neuron_.row_start[a + 1]) {
      sic_sources_.push_back(a);
    }
  }
  {
    vec<char> seen(cfg_.N.N_astro, 0);
    astro_input_sinks_.clear();
    for (const index_t t : neuron_astro_.target) {
      if (!seen[t]) {
        seen[t] = 1;
        astro_input_sinks_.push_back(t);
      }
    }
    // Sorted so the scan runs forwards through memory.
    std::sort(astro_input_sinks_.begin(), astro_input_sinks_.end());
  }

  const index_t n_neurons = neurons_.size();
  ring_exc_.assign(ring_slots_, vec<real>(n_neurons, 0.0));
  ring_inh_.assign(ring_slots_, vec<real>(n_neurons, 0.0));
  ring_sic_.assign(ring_slots_, vec<real>(n_neurons, 0.0));
  ring_astro_.assign(ring_slots_, vec<real>(cfg_.N.N_astro, 0.0));
  ring_sic_pending_.assign(ring_slots_, 0);
  sic_delay_steps_ = delay_to_steps(cfg_.syn.d_a2n, dt);
}

void Network::build_primary_connections(CounterRng& rng) {
  const index_t n_exc = cfg_.N.N_exc;
  const index_t n_inh = cfg_.N.N_inh;
  const index_t n_neurons = n_exc + n_inh;
  const index_t n_astro = cfg_.N.N_astro;
  const real dt = cfg_.time.dt;

  vec<vec<index_t>> exc_targets(n_exc);
  vec<vec<index_t>> astro_targets(n_exc);
  vec<vec<index_t>> a2n_targets(n_astro);
  vec<vec<index_t>> inh_targets(n_inh);

  // Pools belong to the postsynaptic neuron and are fixed for the run. The
  // reference model makes two tripartite calls, so pools are per block.
  vec<vec<index_t>> pool(n_neurons);
  auto assign_pools = [&](index_t offset, index_t count) {
    for (index_t local = 0; local < count; ++local) {
      const index_t post = offset + local;
      pool[post].reserve(cfg_.conn.pool_size);
      if (cfg_.conn.pool_type == PoolType::Block) {
        const index_t start =
            n_astro == 0 ? 0 : static_cast<index_t>((static_cast<std::uint64_t>(local) *
                                                         n_astro) /
                               std::max<index_t>(count, 1));
        for (int k = 0; k < cfg_.conn.pool_size; ++k) {
          pool[post].push_back(static_cast<index_t>((start + k) % n_astro));
        }
      } else {
        // Sample without replacement.
        const int take = std::min<int>(cfg_.conn.pool_size, static_cast<int>(n_astro));
        if (take > 0 && static_cast<std::int64_t>(take) * 4 >=
                             static_cast<std::int64_t>(n_astro)) {
          // take is a large fraction of n_astro: rejection sampling below
          // would thrash on collisions, so fall back to the exact
          // shuffle. Only reached when n_astro itself is small (pool_size
          // is fixed and small in every config this repo uses), so the
          // O(n_astro) cost here stays cheap.
          vec<index_t> candidates(n_astro);
          for (index_t i = 0; i < n_astro; ++i) {
            candidates[i] = i;
          }
          for (int k = 0; k < take; ++k) {
            const auto pick =
                static_cast<std::size_t>(rng.uniform() * (candidates.size() - k)) + k;
            std::swap(candidates[k], candidates[std::min(pick, candidates.size() - 1)]);
            pool[post].push_back(candidates[k]);
          }
        } else {
          // The common case: pool_size is tiny next to n_astro, so
          // rejection sampling directly into pool[post] is O(pool_size)
          // expected work and never allocates an n_astro-sized array --
          // unlike the shuffle above, whose per-neuron O(n_astro) cost is
          // what made large sweeps (100k+ neurons) impractical to build.
          while (static_cast<int>(pool[post].size()) < take) {
            const auto candidate =
                static_cast<index_t>(rng.uniform() * static_cast<double>(n_astro)) % n_astro;
            bool dup = false;
            for (index_t existing : pool[post]) {
              if (existing == candidate) {
                dup = true;
                break;
              }
            }
            if (!dup) {
              pool[post].push_back(candidate);
            }
          }
        }
      }
    }
  };
  assign_pools(0, n_exc);
  assign_pools(n_exc, n_inh);

  // Duplicates are intended: the rule attaches an astrocyte per primary
  // connection, so a neuron can end up connected to the same astrocyte
  // several times. That sets the scale of the current it receives.
  std::unordered_set<std::uint64_t> a2n_seen;

  // Tracks how many times each astrocyte has been recruited as the third
  // factor so far, so max_astro_out_degree can be enforced below. Left at
  // size 0 (and the cap left inert) when the limit is disabled.
  vec<index_t> astro_out_degree;
  const bool cap_astro_degree = cfg_.conn.max_astro_out_degree > 0;
  if (cap_astro_degree) {
    astro_out_degree.assign(n_astro, 0);
  }

  // Distance to the next candidate independently included under
  // Bernoulli(p), i.e. how many excluded candidates come before the next
  // included one. Same distribution as testing every candidate one at a
  // time and keeping it with probability p, but costs one draw per
  // *included* candidate instead of one per candidate tested -- the
  // difference between O(expected connections) and O(population), which is
  // what makes generating connectivity for networks past a few thousand
  // cells practical (see docs/experiments.md and the neuron-scaling sweep).
  // `remaining` bounds the result so a very small p cannot push the
  // caller's running position past the end of its range.
  auto geometric_skip = [&](real p, index_t remaining) -> index_t {
    if (remaining == 0 || p >= 1.0) {
      return 0;
    }
    if (p <= 0.0) {
      return remaining;
    }
    real u = rng.uniform();
    if (u >= 1.0) {
      u = std::nextafter(static_cast<real>(1.0), static_cast<real>(0.0));
    }
    const real g = std::floor(std::log1p(-u) / std::log1p(-p));
    return (g >= static_cast<real>(remaining)) ? remaining : static_cast<index_t>(g);
  };

  auto connect_block = [&](index_t post_offset, index_t post_count) {
    for (index_t pre = 0; pre < n_exc; ++pre) {
      index_t pos = 0;
      while (pos < post_count) {
        pos += geometric_skip(cfg_.conn.p_primary, post_count - pos);
        if (pos >= post_count) {
          break;
        }
        const index_t post = post_offset + pos;
        ++pos;

        // The autapse slot, when it falls in range, is simply dropped here
        // rather than excluded from the trial beforehand as the old dense
        // sweep did -- a bias smaller than one connection in expectation,
        // negligible at any population size this generates.
        if (!cfg_.conn.allow_autapses && pre == post) {
          continue;
        }
        exc_targets[pre].push_back(post);

        if (n_astro == 0 || rng.uniform() >= cfg_.conn.p_third_if_primary) {
          continue;
        }
        const auto &p = pool[post];
        const index_t astro =
            p[static_cast<std::size_t>(rng.uniform() * p.size()) % p.size()];

        // The primary neuron-to-neuron synapse above still stands; only the
        // tripartite edge is dropped once this astrocyte is saturated.
        if (cap_astro_degree &&
            astro_out_degree[astro] >=
                static_cast<index_t>(cfg_.conn.max_astro_out_degree)) {
          continue;
        }

        astro_targets[pre].push_back(astro);
        if (!cfg_.conn.unique_third_out ||
            a2n_seen.insert(pair_key(astro, post)).second) {
          a2n_targets[astro].push_back(post);
        }
        if (cap_astro_degree) {
          ++astro_out_degree[astro];
        }
      }
    }
  };

  // Two calls, as the reference model does: excitatory targets then inhibitory.
  connect_block(0, n_exc);
  connect_block(n_exc, n_inh);

  // Inhibitory neurons project to both populations, no astrocytes.
  for (index_t pre = 0; pre < n_inh; ++pre) {
    const index_t pre_global = n_exc + pre;
    index_t pos = 0;
    while (pos < n_neurons) {
      pos += geometric_skip(cfg_.conn.p_primary, n_neurons - pos);
      if (pos >= n_neurons) {
        break;
      }
      const index_t post = pos;
      ++pos;
      if (!cfg_.conn.allow_autapses && pre_global == post) {
        continue;
      }
      inh_targets[pre].push_back(post);
    }
  }

  flatten(exc_targets, exc_primary_.row_start, exc_primary_.target);
  exc_primary_.weight.assign(exc_primary_.size(), cfg_.syn.w_e);
  exc_primary_.delay_steps.assign(exc_primary_.size(),
                                  delay_to_steps(cfg_.syn.d_e, dt));
  exc_primary_.finalise(n_exc, cfg_.syn.stp);

  flatten(inh_targets, inh_primary_.row_start, inh_primary_.target);
  inh_primary_.weight.assign(inh_primary_.size(), cfg_.syn.w_i);
  inh_primary_.delay_steps.assign(inh_primary_.size(),
                                  delay_to_steps(cfg_.syn.d_i, dt));
  inh_primary_.finalise(n_inh, cfg_.syn.stp);

  flatten(astro_targets, neuron_astro_.row_start, neuron_astro_.target);
  neuron_astro_.weight.assign(neuron_astro_.size(), cfg_.syn.w_n2a);
  neuron_astro_.delay_steps.assign(neuron_astro_.size(),
                                   delay_to_steps(cfg_.syn.d_e, dt));
  neuron_astro_.finalise(n_exc, cfg_.syn.stp);

  flatten(a2n_targets, astro_neuron_.row_start, astro_neuron_.target);
  astro_neuron_.weight.assign(astro_neuron_.size(), cfg_.syn.w_a2n);
  astro_neuron_.delay_steps.assign(astro_neuron_.size(),
                                   delay_to_steps(cfg_.syn.d_a2n, dt));
  // The SIC connection carries a continuous signal and has no plasticity.
  StpParams none;
  none.enabled = false;
  astro_neuron_.finalise(n_astro, none);
}

real Network::stp_weight(ConnectionSet &set, index_t synapse,real t_now) const {
  const StpParams &stp = cfg_.syn.stp;
  if (!stp.enabled) {
    return set.weight[synapse];
  }
  const real dt = t_now - set.stp_t_last[synapse];
  const real x_decay = std::exp(-dt / stp.tau_rec);
  const real u_decay = stp.tau_fac < 1e-10 ? 0.0 : std::exp(-dt / stp.tau_fac);

  real &x = set.stp_x[synapse];
  real &u = set.stp_u[synapse];

  x = 1.0 + (x - x * u - 1.0) * x_decay;
  u = stp.U + u * (1.0 - stp.U) * u_decay;
  set.stp_t_last[synapse] = t_now;

  return set.weight[synapse] * x * u;
}

void Network::deliver_spikes(const vec<Spike> &spikes, std::int64_t step) {
  const real t_now = static_cast<real>(step) * cfg_.time.dt;
  const index_t n_exc = cfg_.N.N_exc;
  const auto n_spikes = static_cast<std::int64_t>(spikes.size());

  // Each spike's synapse range is exclusive to its source, so stp_weight's
  // per-synapse state is race-free across threads. Different sources CAN
  // share a target, though. Atomics used to handle that, but their
  // accumulation order isn't guaranteed -- floating-point addition isn't
  // associative, so the result depended on real-time thread scheduling, not
  // just the input. Staging each thread's contributions in its own buffer
  // and merging them afterward in a fixed order (thread 0's, then thread
  // 1's, ...) removes that: no atomics, no scheduling-dependent result. With
  // schedule(static) the iteration range splits into contiguous per-thread
  // blocks in thread-index order, so this merge reconstructs the exact term
  // order a single serial loop over `spikes` would use -- the result is
  // therefore identical for any thread count, not just repeatable at a fixed
  // one.
  #pragma omp parallel
  {
#ifdef _OPENMP
    #pragma omp single
    {
      const std::size_t nt = static_cast<std::size_t>(omp_get_num_threads());
      exc_deltas_.resize(nt);
      astro_deltas_.resize(nt);
      inh_deltas_.resize(nt);
    }
    const int tid = omp_get_thread_num();
#else
    exc_deltas_.resize(1);
    astro_deltas_.resize(1);
    inh_deltas_.resize(1);
    const int tid = 0;
#endif
    vec<RingDelta> &exc_local = exc_deltas_[static_cast<std::size_t>(tid)];
    vec<RingDelta> &astro_local = astro_deltas_[static_cast<std::size_t>(tid)];
    vec<RingDelta> &inh_local = inh_deltas_[static_cast<std::size_t>(tid)];
    exc_local.clear();
    astro_local.clear();
    inh_local.clear();

    #pragma omp for schedule(static)
    for (std::int64_t idx = 0; idx < n_spikes; ++idx) {
      const Spike &s = spikes[idx];
      if (s.source < n_exc) {
        const index_t src = s.source;
        for (index_t k = exc_primary_.row_start[src];
             k < exc_primary_.row_start[src + 1]; ++k) {
          const int slot = static_cast<int>((step + exc_primary_.delay_steps[k]) %
                                            ring_slots_);
          exc_local.push_back({slot, exc_primary_.target[k], stp_weight(exc_primary_, k, t_now)});
        }
        for (index_t k = neuron_astro_.row_start[src];
             k < neuron_astro_.row_start[src + 1]; ++k) {
          const int slot = static_cast<int>(
              (step + neuron_astro_.delay_steps[k]) % ring_slots_);
          astro_local.push_back(
              {slot, neuron_astro_.target[k], stp_weight(neuron_astro_, k, t_now)});
        }
      } else {
        const index_t src = s.source - n_exc;
        for (index_t k = inh_primary_.row_start[src];
             k < inh_primary_.row_start[src + 1]; ++k) {
          const int slot = static_cast<int>((step + inh_primary_.delay_steps[k]) % ring_slots_);
          // Weights are negative; the ring stores magnitudes, so the sign
          // flip happens at merge time below (see the `-=` there), same as
          // the direct write this replaced.
          inh_local.push_back({slot, inh_primary_.target[k], stp_weight(inh_primary_, k, t_now)});
        }
      }
    }
  }  // implicit barrier: every thread's buffer is complete past this point

  // Serial on purpose -- this is what makes the result order-independent.
  // Each inner loop is itself in-order (push_back preserves the spikes-loop
  // order within a thread), so thread 0's block fully precedes thread 1's,
  // exactly like a plain sequential loop over `spikes` would.
  for (const vec<RingDelta> &bucket : exc_deltas_) {
    for (const RingDelta &d : bucket) {
      ring_exc_[static_cast<std::size_t>(d.slot)][d.target] += d.value;
    }
  }
  for (const vec<RingDelta> &bucket : astro_deltas_) {
    for (const RingDelta &d : bucket) {
      ring_astro_[static_cast<std::size_t>(d.slot)][d.target] += d.value;
    }
  }
  for (const vec<RingDelta> &bucket : inh_deltas_) {
    for (const RingDelta &d : bucket) {
      ring_inh_[static_cast<std::size_t>(d.slot)][d.target] -= d.value;
    }
  }
}

void Network::deliver_sic(std::int64_t step) {
  // Mark the arrival slot whether or not anything contributes. An exchange
  // with every astrocyte below threshold still tells the neuron its current
  // is zero, and treating that as "nothing arrived" would leave the last
  // value held indefinitely.
  ring_sic_pending_[(step + sic_delay_steps_) % ring_slots_] = 1;

  // Only astrocytes with an outgoing connection can contribute. Each source
  // owns an exclusive synapse range (no plasticity state here to race on,
  // unlike deliver_spikes), so only the ring write below needs an atomic.
  const auto n_sources = static_cast<std::int64_t>(sic_sources_.size());
  #pragma omp parallel for schedule(static)
  for (std::int64_t idx = 0; idx < n_sources; ++idx) {
    const index_t a = sic_sources_[idx];
    const real factor = astro_.sic_factor(a);
    if (factor == 0.0) {
      continue;
    }
    for (index_t k = astro_neuron_.row_start[a]; k < astro_neuron_.row_start[a + 1]; ++k) {
      const int slot = static_cast<int>((step + astro_neuron_.delay_steps[k]) % ring_slots_);
      const real contribution = astro_neuron_.weight[k] * factor;
      #pragma omp atomic update
      ring_sic_[slot][astro_neuron_.target[k]] += contribution;
    }
  }
}

void Network::apply_arrivals(std::int64_t step) {
  const int slot = static_cast<int>(step % ring_slots_);

  // With an exchange interval above one, most steps carry no new current and
  // the neuron holds the last one it was sent. Applying the empty slot would
  // zero it instead, which is not the same thing.
  const bool sic_arrives = ring_sic_pending_[slot] != 0;
  ring_sic_pending_[slot] = 0;

  vec<real> &ex = ring_exc_[slot];
  vec<real> &in = ring_inh_[slot];
  vec<real> &sic = ring_sic_[slot];
  const index_t n_neurons = neurons_.size();

  // Each iteration only touches its own cell's slots, so no races.
  #pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n_neurons; ++i) {
    if (ex[i] != 0.0) {
      neurons_.add_synaptic_input(i, ex[i]);
      ex[i] = 0.0;
    }
    if (in[i] != 0.0) {
      neurons_.add_synaptic_input(i, -in[i]);
      in[i] = 0.0;
    }
    if (sic_arrives) {
      neurons_.set_sic(i, sic[i]);
      sic[i] = 0.0;
    }
  }

  // Only cells some neuron projects to can have a pending entry.
  vec<real> &astro_in = ring_astro_[slot];
  for (const index_t a : astro_input_sinks_) {
    if (astro_in[a] != 0.0) {
      astro_.add_ip3_input(a, astro_in[a]);
      astro_in[a] = 0.0;
    }
  }
}

void Network::drive_astrocytes(std::int64_t step) {
  const real rate = cfg_.input_astro.poiss_rate;
  if (rate <= 0.0) {
    return;
  }
  const real lambda = rate * cfg_.time.dt * 1e-3;
  const index_t n_astro = astro_.size();
  #pragma omp parallel for
  for (index_t a = 0; a < n_astro; ++a) {
    CounterRng r(cfg_.seed ^ 0xC2B2AE3D27D4EB4FULL,
                 static_cast<std::uint64_t>(step) * 1000003ULL + a);
    const int events = r.poisson(lambda);
    if (events > 0) {
      astro_.add_ip3_input(a, cfg_.input_astro.poiss_weight * static_cast<real>(events));
    }
  }
}

void Network::run(Recorder &recorder) {
  const std::int64_t pre_steps = cfg_.time.pre_steps();
  const std::int64_t sim_steps = cfg_.time.sim_steps();
  const std::int64_t total = pre_steps + sim_steps;
  const real dt = cfg_.time.dt;

  const index_t astro_rec = std::min(cfg_.record_astro_max, astro_.size());
  const index_t neuron_rec = std::min(cfg_.record_neuron_max, neurons_.size());

  std::int64_t progress_mark = total / 10;
  if (progress_mark == 0) {
    progress_mark = 1;
  }

  // Timed over the recorded window only. The transient has cold caches and a
  // network below its working firing rate.
  using clock = std::chrono::steady_clock;
  auto tick = [](const clock::time_point &since) {
    return std::chrono::duration<double>(clock::now() - since).count();
  };
  // Timed directly. Prorating by step count assumed the transient cost the
  // same per step, which showed up as a negative "other" row.
  auto measured_start = clock::now();

  // State stays on the device for the run, so the per-step map clauses find
  // the arrays present and move nothing.
  astro_.device_begin();
  neurons_.device_begin();

  for (std::int64_t step = 0; step < total; ++step) {
    const bool measured = step >= pre_steps;
    if (step == pre_steps) {
      measured_start = clock::now();
    }

    auto t = clock::now();
    ASTROSIMGPU_NVTX_PUSH("apply_arrivals");
    apply_arrivals(step);
    ASTROSIMGPU_NVTX_POP();
    if (measured) {
      profile_.deliver += tick(t);
    }

    // Per-cell work with no communication, so timed apart from delivery.
    // CUDA: generated directly on the device (Stage 2 of docs/gpu-port.md),
    // so device_push_input has to land first -- this adds to it rather than
    // overwriting. Every other backend keeps the host loop, unchanged.
    t = clock::now();
    ASTROSIMGPU_NVTX_PUSH("drive_astrocytes");
#if defined(ASTROSIMGPU_CUDA)
    astro_.device_push_input();
    if (cfg_.input_astro.poiss_rate > 0.0) {
      const real lambda = cfg_.input_astro.poiss_rate * cfg_.time.dt * 1e-3;
      astro_.drive_device(step, cfg_.seed, lambda, cfg_.input_astro.poiss_weight);
    }
#else
    drive_astrocytes(step);
#endif
    ASTROSIMGPU_NVTX_POP();
    if (measured) {
      profile_.input_gen += tick(t);
    }

    t = clock::now();
    ASTROSIMGPU_NVTX_PUSH("update_astro");
#if !defined(ASTROSIMGPU_CUDA)
    astro_.device_push_input();
#endif
    astro_.update(cfg_.time, step, cfg_.seed);
    // deliver_sic reads calcium on the host.
    astro_.device_pull_calcium();
    astro_.clear_inputs(astro_input_sinks_);
    ASTROSIMGPU_NVTX_POP();
    if (measured) {
      profile_.update_astro += tick(t);
    }

    t = clock::now();
    ASTROSIMGPU_NVTX_PUSH("update_neuron");
    spike_buffer_.clear();
    neurons_.update(cfg_.time, step, cfg_.seed, spike_buffer_);
    ASTROSIMGPU_NVTX_POP();
    if (measured) {
      profile_.update_neuron += tick(t);
    }

    t = clock::now();
    ASTROSIMGPU_NVTX_PUSH("deliver_spikes");
    deliver_spikes(spike_buffer_, step);
    ASTROSIMGPU_NVTX_POP();
    if (measured) {
      profile_.spike_cd += tick(t);
    }

    t = clock::now();
    ASTROSIMGPU_NVTX_PUSH("deliver_sic");
    if (step % cfg_.syn.sic_interval == 0) {
      deliver_sic(step);
    }
    ASTROSIMGPU_NVTX_POP();
    if (measured) {
      profile_.sic_gd += tick(t);
    }

    if (step < pre_steps) {
      if (step % progress_mark == 0) {
        std::cout << "  transient "
                  << static_cast<int>(100.0 * step / pre_steps) << "%\r"
                  << std::flush;
      }
      continue;
    }

    const real t_ms = static_cast<real>(step) * dt;

    if (cfg_.record_spikes) {
      for (const Spike &s : spike_buffer_) {
        recorder.write_spike(t_ms, s.source);
      }
    }

    if ((step - pre_steps) % cfg_.record_every == 0) {
      if (cfg_.record_astro) {
        for (index_t a = 0; a < astro_rec; ++a) {
          recorder.write_astro(t_ms, a, astro_.Ca()[a], astro_.IP3()[a]);
        }
      }
      if (cfg_.record_neuron) {
        for (index_t i = 0; i < neuron_rec; ++i) {
          recorder.write_neuron(t_ms, i, neurons_.V()[i], neurons_.I_sic()[i]);
        }
      }
    }

    if (step % progress_mark == 0) {
      std::cout << "  simulating "
                << static_cast<int>(100.0 * (step - pre_steps) / sim_steps)
                << "%\r" << std::flush;
    }
  }
  astro_.device_end();
  neurons_.device_end();

  profile_.total = tick(measured_start);
  std::cout << "  done            " << std::endl;
}

} // namespace astrosimgpu

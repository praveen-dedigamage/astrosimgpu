#include "astrosimgpu/network.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <unordered_set>

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
                    n_astro == 0 ? 0
                                 : static_cast<index_t>((static_cast<std::uint64_t>(local) *
                                                         n_astro) /
                                                        std::max<index_t>(count, 1));
                for (int k = 0; k < cfg_.conn.pool_size; ++k) {
                    pool[post].push_back(static_cast<index_t>((start + k) % n_astro));
                }
            } else {
                // Sample without replacement.
                vec<index_t> candidates(n_astro);
                for (index_t i = 0; i < n_astro; ++i) {
                    candidates[i] = i;
                }
                const int take = std::min<int>(cfg_.conn.pool_size, static_cast<int>(n_astro));
                for (int k = 0; k < take; ++k) {
                    const auto pick =
                        static_cast<std::size_t>(rng.uniform() * (candidates.size() - k)) + k;
                    std::swap(candidates[k], candidates[std::min(pick, candidates.size() - 1)]);
                    pool[post].push_back(candidates[k]);
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

    auto connect_block = [&](index_t post_offset, index_t post_count) {
        for (index_t pre = 0; pre < n_exc; ++pre) {
            for (index_t local = 0; local < post_count; ++local) {
                const index_t post = post_offset + local;
                if (!cfg_.conn.allow_autapses && pre == post) {
                    continue;
                }
                if (rng.uniform() >= cfg_.conn.p_primary) {
                    continue;
                }
                exc_targets[pre].push_back(post);

                if (n_astro == 0 || rng.uniform() >= cfg_.conn.p_third_if_primary) {
                    continue;
                }
                const auto& p = pool[post];
                const index_t astro =
                    p[static_cast<std::size_t>(rng.uniform() * p.size()) % p.size()];

                astro_targets[pre].push_back(astro);
                if (!cfg_.conn.unique_third_out || a2n_seen.insert(pair_key(astro, post)).second) {
                    a2n_targets[astro].push_back(post);
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
        for (index_t post = 0; post < n_neurons; ++post) {
            if (!cfg_.conn.allow_autapses && pre_global == post) {
                continue;
            }
            if (rng.uniform() < cfg_.conn.p_primary) {
                inh_targets[pre].push_back(post);
            }
        }
    }

    flatten(exc_targets, exc_primary_.row_start, exc_primary_.target);
    exc_primary_.weight.assign(exc_primary_.size(), cfg_.syn.w_e);
    exc_primary_.delay_steps.assign(exc_primary_.size(), delay_to_steps(cfg_.syn.d_e, dt));
    exc_primary_.finalise(n_exc, cfg_.syn.stp);

    flatten(inh_targets, inh_primary_.row_start, inh_primary_.target);
    inh_primary_.weight.assign(inh_primary_.size(), cfg_.syn.w_i);
    inh_primary_.delay_steps.assign(inh_primary_.size(), delay_to_steps(cfg_.syn.d_i, dt));
    inh_primary_.finalise(n_inh, cfg_.syn.stp);

    flatten(astro_targets, neuron_astro_.row_start, neuron_astro_.target);
    neuron_astro_.weight.assign(neuron_astro_.size(), cfg_.syn.w_n2a);
    neuron_astro_.delay_steps.assign(neuron_astro_.size(), delay_to_steps(cfg_.syn.d_e, dt));
    neuron_astro_.finalise(n_exc, cfg_.syn.stp);

    flatten(a2n_targets, astro_neuron_.row_start, astro_neuron_.target);
    astro_neuron_.weight.assign(astro_neuron_.size(), cfg_.syn.w_a2n);
    astro_neuron_.delay_steps.assign(astro_neuron_.size(), delay_to_steps(cfg_.syn.d_a2n, dt));
    // The SIC connection carries a continuous signal and has no plasticity.
    StpParams none;
    none.enabled = false;
    astro_neuron_.finalise(n_astro, none);
}

real Network::stp_weight(ConnectionSet& set, index_t synapse, real t_now) const {
    const StpParams& stp = cfg_.syn.stp;
    if (!stp.enabled) {
        return set.weight[synapse];
    }
    const real dt = t_now - set.stp_t_last[synapse];
    const real x_decay = std::exp(-dt / stp.tau_rec);
    const real u_decay = stp.tau_fac < 1e-10 ? 0.0 : std::exp(-dt / stp.tau_fac);

    real& x = set.stp_x[synapse];
    real& u = set.stp_u[synapse];

    x = 1.0 + (x - x * u - 1.0) * x_decay;
    u = stp.U + u * (1.0 - stp.U) * u_decay;
    set.stp_t_last[synapse] = t_now;

    return set.weight[synapse] * x * u;
}

void Network::deliver_spikes(const vec<Spike>& spikes, std::int64_t step) {
    const real t_now = static_cast<real>(step) * cfg_.time.dt;
    const index_t n_exc = cfg_.N.N_exc;

    for (const Spike& s : spikes) {
        if (s.source < n_exc) {
            const index_t src = s.source;
            for (index_t k = exc_primary_.row_start[src]; k < exc_primary_.row_start[src + 1];
                 ++k) {
                const int slot =
                    static_cast<int>((step + exc_primary_.delay_steps[k]) % ring_slots_);
                ring_exc_[slot][exc_primary_.target[k]] += stp_weight(exc_primary_, k, t_now);
            }
            for (index_t k = neuron_astro_.row_start[src]; k < neuron_astro_.row_start[src + 1];
                 ++k) {
                const int slot =
                    static_cast<int>((step + neuron_astro_.delay_steps[k]) % ring_slots_);
                ring_astro_[slot][neuron_astro_.target[k]] += stp_weight(neuron_astro_, k, t_now);
            }
        } else {
            const index_t src = s.source - n_exc;
            for (index_t k = inh_primary_.row_start[src]; k < inh_primary_.row_start[src + 1];
                 ++k) {
                const int slot =
                    static_cast<int>((step + inh_primary_.delay_steps[k]) % ring_slots_);
                // Weights are negative; the ring stores magnitudes.
                ring_inh_[slot][inh_primary_.target[k]] -= stp_weight(inh_primary_, k, t_now);
            }
        }
    }
}

void Network::deliver_sic(std::int64_t step) {
    // Only astrocytes with an outgoing connection can contribute.
    for (const index_t a : sic_sources_) {
        const real factor = astro_.sic_factor(a);
        if (factor == 0.0) {
            continue;
        }
        for (index_t k = astro_neuron_.row_start[a]; k < astro_neuron_.row_start[a + 1]; ++k) {
            const int slot = static_cast<int>((step + astro_neuron_.delay_steps[k]) % ring_slots_);
            ring_sic_[slot][astro_neuron_.target[k]] += astro_neuron_.weight[k] * factor;
        }
    }
}

void Network::apply_arrivals(std::int64_t step) {
    const int slot = static_cast<int>(step % ring_slots_);

    vec<real>& ex = ring_exc_[slot];
    vec<real>& in = ring_inh_[slot];
    vec<real>& sic = ring_sic_[slot];
    const index_t n_neurons = neurons_.size();

    for (index_t i = 0; i < n_neurons; ++i) {
        if (ex[i] != 0.0) {
            neurons_.add_synaptic_input(i, ex[i]);
            ex[i] = 0.0;
        }
        if (in[i] != 0.0) {
            neurons_.add_synaptic_input(i, -in[i]);
            in[i] = 0.0;
        }
        // The SIC is continuous: whatever was scheduled for this step is the
        // current the neuron sees, and it is replaced rather than accumulated
        // across steps.
        neurons_.set_sic(i, sic[i]);
        sic[i] = 0.0;
    }

    // Only cells some neuron projects to can have a pending entry.
    vec<real>& astro_in = ring_astro_[slot];
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
    for (index_t a = 0; a < n_astro; ++a) {
        CounterRng r(cfg_.seed ^ 0xC2B2AE3D27D4EB4FULL,
                     static_cast<std::uint64_t>(step) * 1000003ULL + a);
        const int events = r.poisson(lambda);
        if (events > 0) {
            astro_.add_ip3_input(a, cfg_.input_astro.poiss_weight * static_cast<real>(events));
        }
    }
}

void Network::run(Recorder& recorder) {
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
    auto tick = [](const clock::time_point& since) {
        return std::chrono::duration<double>(clock::now() - since).count();
    };
    // Timed directly. Prorating by step count assumed the transient cost the
    // same per step, which showed up as a negative "other" row.
    auto measured_start = clock::now();

    // State stays on the device for the run, so the per-step map clauses find
    // the arrays present and move nothing.
    astro_.device_begin();

    for (std::int64_t step = 0; step < total; ++step) {
        const bool measured = step >= pre_steps;
        if (step == pre_steps) {
            measured_start = clock::now();
        }

        auto t = clock::now();
        apply_arrivals(step);
        if (measured) {
            profile_.deliver += tick(t);
        }

        // Per-cell work with no communication, so timed apart from delivery.
        t = clock::now();
        drive_astrocytes(step);
        if (measured) {
            profile_.input_gen += tick(t);
        }

        t = clock::now();
        astro_.device_push_input();
        astro_.update(cfg_.time, step, cfg_.seed);
        // deliver_sic reads calcium on the host.
        astro_.device_pull_calcium();
        astro_.clear_inputs(astro_input_sinks_);
        if (measured) {
            profile_.update_astro += tick(t);
        }

        t = clock::now();
        spike_buffer_.clear();
        neurons_.update(cfg_.time, step, cfg_.seed, spike_buffer_);
        if (measured) {
            profile_.update_neuron += tick(t);
        }

        t = clock::now();
        deliver_spikes(spike_buffer_, step);
        if (measured) {
            profile_.spike_cd += tick(t);
        }

        t = clock::now();
        deliver_sic(step);
        if (measured) {
            profile_.sic_gd += tick(t);
        }

        if (step < pre_steps) {
            if (step % progress_mark == 0) {
                std::cout << "  transient " << static_cast<int>(100.0 * step / pre_steps)
                          << "%\r" << std::flush;
            }
            continue;
        }

        const real t_ms = static_cast<real>(step) * dt;

        if (cfg_.record_spikes) {
            for (const Spike& s : spike_buffer_) {
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
            std::cout << "  simulating " << static_cast<int>(100.0 * (step - pre_steps) / sim_steps)
                      << "%\r" << std::flush;
        }
    }
    astro_.device_end();

    profile_.total = tick(measured_start);
    std::cout << "  done            " << std::endl;
}

}  // namespace astrosimgpu

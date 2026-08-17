#pragma once

#include <string>

#include "astrosimgpu/astrocyte.hpp"
#include "astrosimgpu/neuron.hpp"
#include "astrosimgpu/parameters.hpp"
#include "astrosimgpu/recorder.hpp"

namespace astrosimgpu {

// Compressed row storage, grouped by source: delivering one cell's output
// walks a contiguous run. STP state, when enabled, is parallel to `target`.
struct ConnectionSet {
    vec<index_t> row_start;  ///< size = number of sources + 1
    vec<index_t> target;
    vec<real> weight;
    vec<int> delay_steps;

    // Tsodyks-Markram state, one entry per synapse.
    vec<real> stp_x;
    vec<real> stp_u;
    vec<real> stp_t_last;

    [[nodiscard]] index_t size() const { return static_cast<index_t>(target.size()); }
    void finalise(index_t sources, const StpParams& stp);
};

// Wall clock per phase, using the same split as the reference benchmarks so
// the two profiles can be compared. update parallelises; the delivery phases
// carry the communication and do not.
struct PhaseProfile {
    double input_gen = 0.0;
    double update_astro = 0.0;
    double update_neuron = 0.0;
    double spike_cd = 0.0;
    double sic_gd = 0.0;
    double deliver = 0.0;
    double total = 0.0;

    [[nodiscard]] double accounted() const {
        return input_gen + update_astro + update_neuron + spike_cd + sic_gd + deliver;
    }
    [[nodiscard]] double other() const { return total - accounted(); }
};

struct NetworkStats {
    index_t n_astro = 0;
    index_t n_exc = 0;
    index_t n_inh = 0;
    index_t n_primary_exc = 0;
    index_t n_primary_inh = 0;
    index_t n_neuron_to_astro = 0;
    index_t n_astro_to_neuron = 0;
};

/// The tripartite neuron-astrocyte network and its time loop.
class Network {
public:
    explicit Network(const ModelConfig& config);

    void build();   // deterministic for a given seed
    void run(Recorder& recorder);

    [[nodiscard]] const NetworkStats& stats() const { return stats_; }
    [[nodiscard]] const PhaseProfile& profile() const { return profile_; }
    [[nodiscard]] const ModelConfig& config() const { return cfg_; }

private:
    void build_primary_connections(CounterRng& rng);
    void build_tripartite(index_t post_offset, index_t post_count, CounterRng& rng);
    void deliver_spikes(const vec<Spike>& spikes, std::int64_t step);
    void deliver_sic(std::int64_t step);
    void apply_arrivals(std::int64_t step);
    void drive_astrocytes(std::int64_t step);

    // Tsodyks-Markram: x <- 1 + (x - x u - 1) exp(-dt/tau_rec),
    //                   u <- U + u (1 - U) exp(-dt/tau_fac), in that order.
    // tau_fac = 0 gives pure depression.
    real stp_weight(ConnectionSet& set, index_t synapse, real t_now) const;

    ModelConfig cfg_;
    NetworkStats stats_{};
    PhaseProfile profile_{};

    AstrocytePopulation astro_;
    NeuronPopulation neurons_;

    ConnectionSet exc_primary_;   ///< excitatory neuron -> neuron
    ConnectionSet inh_primary_;   ///< inhibitory neuron -> neuron
    ConnectionSet neuron_astro_;  ///< excitatory neuron -> astrocyte
    ConnectionSet astro_neuron_;  ///< astrocyte -> neuron (SIC)

    // Pending input, [slot][cell]. A signal emitted at step t with delay d
    // lands in slot (t+d) % slots. No event queue, no sorting.
    vec<vec<real>> ring_exc_, ring_inh_, ring_sic_, ring_astro_;
    int ring_slots_ = 1;

    // Built once from the connectivity. Without these the delivery phases scan
    // the whole population every step to find the few cells that matter.
    vec<index_t> sic_sources_;        // have an outgoing SIC connection
    vec<index_t> astro_input_sinks_;  // are the target of some neuron

    vec<Spike> spike_buffer_;
};

}  // namespace astrosimgpu

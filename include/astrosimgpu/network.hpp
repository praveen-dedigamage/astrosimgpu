#pragma once

#include <string>

#include "astrosimgpu/astrocyte.hpp"
#include "astrosimgpu/neuron.hpp"
#include "astrosimgpu/parameters.hpp"
#include "astrosimgpu/recorder.hpp"

namespace astrosimgpu {

/// Connections held in compressed-row form, grouped by source.
///
/// Delivery walks one contiguous run per firing source, which is the access
/// pattern a spike-driven update wants and the one that ports cleanly to a
/// device kernel. Short-term plasticity state, when enabled, is per synapse
/// and sits in arrays parallel to `target`.
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

/// Wall-clock time attributed to each phase of state propagation.
///
/// The decomposition deliberately matches the one used in the reference
/// benchmarks, so a profile from this simulator can be read against the
/// published CPU numbers:
///
///   input_gen   drawing the background input for every cell
///   update      advancing the dynamical state of every cell
///   spike_cd    collocation and delivery of spikes
///   sic_gd      gathering and delivery of the astrocytic current
///   deliver     moving arrived input into the target cells
///
/// The split is the point of the instrumentation rather than a by-product:
/// `update` is per-cell and parallelises, while the delivery phases carry the
/// communication that does not, and knowing the ratio between them is what
/// decides where an offload is worth attempting.
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

/// Summary of what was built, printed at start-up and written to the run log.
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

    /// Build populations and connectivity. Deterministic for a given seed.
    void build();

    /// Run the pre-simulation transient followed by the recorded window.
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

    /// Effective weight of one synapse under short-term depression and
    /// facilitation (Tsodyks, Uziel & Markram, 2000):
    ///
    ///   x <- 1 + (x - x u - 1) exp(-dt / tau_rec)
    ///   u <- U + u (1 - U) exp(-dt / tau_fac)
    ///
    /// evaluated in that order at each presynaptic spike, with the release
    /// probability u and the available resources x multiplying the nominal
    /// weight. With tau_fac = 0 this reduces to pure depression.
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

    /// Ring buffers of pending input, indexed [slot][cell]. Separate rings for
    /// the two conductances and for the continuous astrocytic current, so no
    /// per-arrival sorting or event queue is needed.
    vec<vec<real>> ring_exc_, ring_inh_, ring_sic_, ring_astro_;
    int ring_slots_ = 1;

    /// Astrocytes that can take part in the delivery phases at all.
    ///
    /// Both lists are fixed by the connectivity and are built once. Without
    /// them the delivery loops scan the whole population every step to find
    /// the few cells that matter: at ten million astrocytes with eight
    /// thousand connections that was three quarters of the run time, and it
    /// scaled with the population rather than with the connectivity.
    vec<index_t> sic_sources_;        ///< have at least one outgoing SIC connection
    vec<index_t> astro_input_sinks_;  ///< are the target of at least one neuron

    vec<Spike> spike_buffer_;
};

}  // namespace astrosimgpu

#pragma once

#include <string>

#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

/// Li-Rinzel astrocyte parameters.
///
/// Defaults are the NEST astrocyte_lr_1994 defaults; the use case overrides
/// Ca_tot, IP3_0, tau_IP3, delta_IP3 and SIC_scale with fitted values.
/// Concentrations are in uM, rates in 1/ms.
struct AstrocyteParams {
    real Ca_tot = 2.0;            ///< total free calcium, cytosolic volume [uM]
    real IP3_0 = 0.16;            ///< baseline IP3 [uM]
    real Kd_IP3_1 = 0.13;         ///< first IP3R dissociation constant for IP3 [uM]
    real Kd_IP3_2 = 0.9434;       ///< second IP3R dissociation constant for IP3 [uM]
    real Kd_act = 0.08234;        ///< IP3R dissociation constant, Ca activation [uM]
    real Kd_inh = 1.049;          ///< IP3R dissociation constant, Ca inhibition [uM]
    real Km_SERCA = 0.1;          ///< SERCA half-activation constant [uM]
    real SIC_scale = 1.0;         ///< scale of the SIC output [unitless]
    real SIC_th = 0.19669;        ///< calcium threshold for SIC generation [uM]
    /// IP3 increment per synaptic input [uM]. This is the one default that
    /// could not be confirmed against a running NEST install; every shipped
    /// configuration sets it explicitly, so it is only reachable when a
    /// configuration omits it. See scripts/dump_nest_defaults.py.
    real delta_IP3 = 0.0002;
    real k_IP3R = 0.0002;         ///< IP3R binding constant for Ca inhibition [1/(uM ms)]
    real rate_IP3R = 0.006;       ///< max Ca release rate via IP3R [1/ms]
    real rate_L = 0.00011;        ///< ER-to-cytosol leak rate [1/ms]
    real rate_SERCA = 0.0009;     ///< max SERCA uptake rate [uM/ms]
    real ratio_ER_cyt = 0.185;    ///< ER-to-cytosol volume ratio [unitless]
    real tau_IP3 = 7142.0;        ///< IP3 decay time constant [ms]

    // Initial state.
    real Ca_init = 0.073;         ///< initial cytosolic calcium [uM]
    real h_init = 0.793;          ///< initial fraction of non-inactivated IP3R
    real IP3_init = 0.16;         ///< initial IP3 [uM]
};

/// AdEx neuron with alpha-shaped conductance synapses and an SIC input.
/// Defaults are the NEST aeif_cond_alpha_astro defaults.
struct NeuronParams {
    real C_m = 281.0;         ///< membrane capacitance [pF]
    real g_L = 30.0;          ///< leak conductance [nS]
    real E_L = -70.6;         ///< leak reversal potential [mV]
    real V_th = -50.4;        ///< spike initiation threshold [mV]
    real Delta_T = 2.0;       ///< slope factor [mV]
    real a = 4.0;             ///< subthreshold adaptation [nS]
    real b = 80.5;            ///< spike-triggered adaptation [pA]
    real tau_w = 144.0;       ///< adaptation time constant [ms]
    real V_reset = -60.0;     ///< reset potential [mV]
    real V_peak = 0.0;        ///< spike detection threshold [mV]
    real t_ref = 0.0;         ///< refractory period [ms]
    real E_ex = 0.0;          ///< excitatory reversal potential [mV]
    real E_in = -85.0;        ///< inhibitory reversal potential [mV]
    real tau_syn_ex = 0.2;    ///< excitatory synaptic time constant [ms]
    real tau_syn_in = 2.0;    ///< inhibitory synaptic time constant [ms]
    real I_e = 0.0;           ///< constant injected current [pA]
    real V_m_init = -70.6;    ///< initial membrane potential [mV]
};

/// Short-term plasticity, Tsodyks-Markram. NEST tsodyks_synapse defaults.
struct StpParams {
    bool enabled = true;
    real U = 0.5;          ///< utilisation of synaptic efficacy
    real tau_rec = 800.0;  ///< recovery time constant [ms]
    real tau_fac = 0.0;    ///< facilitation time constant [ms]
};

/// Synaptic weights and delays of the network.
struct SynapseParams {
    real w_e = 5.0;         ///< excitatory neuron-to-neuron weight [nS]
    real w_i = -5.0;        ///< inhibitory neuron-to-neuron weight [nS]
    real w_n2a = 0.2;       ///< neuron-to-astrocyte weight
    real w_a2n = 1.0;       ///< astrocyte-to-neuron SIC weight [pA]
    real d_e = 1.0;         ///< excitatory delay [ms]
    real d_i = 1.0;         ///< inhibitory delay [ms]
    real d_a2n = 1.0;       ///< astrocyte-to-neuron delay [ms]
    StpParams stp{};
    // Note: the synaptic time constants that shape the postsynaptic
    // conductance belong to the neuron, not here. The reference parameter
    // file lists tau_syn_ex and tau_syn_in under its synaptic parameters, but
    // passes them as the Tsodyks synapse's tau_psc; the neurons keep the
    // aeif_cond_alpha_astro defaults of 0.2 ms and 2.0 ms. Setting the
    // neuron's tau_syn_ex to 2 ms instead of 0.2 ms raises the mean
    // excitatory conductance tenfold and drives the network out of the
    // asynchronous regime entirely.
};

/// Tripartite connectivity.
enum class PoolType { Block, Random };

struct ConnectivityParams {
    real p_primary = 0.2;           ///< neuron-to-neuron connection probability
    real p_third_if_primary = 0.2;  ///< astrocyte recruitment given a primary connection
    int pool_size = 1;              ///< astrocytes available to one postsynaptic neuron
    PoolType pool_type = PoolType::Block;
    bool allow_autapses = false;

    /// Whether a given astrocyte-target pair may be connected more than once.
    ///
    /// The third-factor rule attaches an astrocyte per primary connection, so
    /// a neuron receiving several primary connections that each recruit the
    /// same astrocyte ends up connected to it several times. That is the
    /// intended behaviour -- a neuron interacting with one astrocyte at
    /// several synapses -- and it sets the scale of the SIC: collapsing the
    /// duplicates divides the current a neuron receives by the number of
    /// synapses that recruited the astrocyte, which here is around sixteen.
    bool unique_third_out = false;
};

/// Background drive.
///
/// Both generators are modelled on their NEST counterparts: the Poisson train
/// is an independent realisation per target, and so is the noise current
/// ("all targets of a noise generator receive different currents, but the
/// currents for all targets change at the same points in time").
///
/// The noise current is piecewise constant, held for noise_dt rather than
/// redrawn every step. That interval is part of the model, not a detail: the
/// low-frequency power of the injected current scales with it, and drawing
/// afresh every 0.1 ms instead of every 1 ms shrinks the membrane potential
/// fluctuation it produces by roughly the square root of ten -- enough, with
/// these parameters, to silence the excitatory population entirely.
struct InputParams {
    real poiss_rate = 0.0;       ///< Poisson rate [Hz]
    real poiss_weight = 1.0;     ///< weight of one Poisson event
    real gauss_noise_std = 0.0;  ///< std of the injected noise current [pA]
    real noise_dt = 0.0;         ///< [ms]; 0 selects the NEST default of 10 * dt
    bool independent_noise = true;
};

/// Gaussian spread applied to selected per-cell parameters.
struct RandomizeSpec {
    bool enabled = false;
    real lower = 1.0;  ///< lower bound as a fraction of the mean
    real upper = 1.0;  ///< upper bound as a fraction of the mean
    real var = 0.0;    ///< std as a fraction of |mean|
};

struct PopulationSizes {
    index_t N_astro = 100;
    index_t N_exc = 400;
    index_t N_inh = 100;
};

/// Post-run calcium analysis.
///
/// The defaults are the values the reference analysis uses: a 2 s window
/// advancing in 400 ms steps, and events closer than 2 s treated as one
/// interrupted transient rather than two.
struct AnalysisParams {
    bool enabled = true;
    real ca_threshold = -1.0;  ///< < 0 means take the astrocyte's SIC_th
    real merge_gap = 2000.0;   ///< [ms]
    real hist_window = 2000.0; ///< [ms]
    real hist_shift = 400.0;   ///< [ms]
};

/// Everything needed to build and run one simulation.
struct ModelConfig {
    PopulationSizes N{};
    TimeGrid time{};
    std::uint64_t seed = 1;

    AstrocyteParams astro{};
    NeuronParams neuron_exc{};
    NeuronParams neuron_inh{};
    SynapseParams syn{};
    ConnectivityParams conn{};

    InputParams input_astro{};
    InputParams input_exc{};
    InputParams input_inh{};

    RandomizeSpec randomize_astro{};
    RandomizeSpec randomize_neuron{};
    AnalysisParams analysis{};

    /// Recording. Cells are subsampled to keep output files manageable.
    bool record_spikes = true;
    bool record_astro = true;
    bool record_neuron = false;
    int record_every = 10;        ///< record every Nth step
    index_t record_astro_max = 100;
    index_t record_neuron_max = 50;

    std::string output_dir = "results";
};

/// Read a configuration from a JSON file. Missing keys keep their defaults.
/// Throws std::runtime_error on a malformed file or an unknown key.
ModelConfig load_config(const std::string& path);

}  // namespace astrosimgpu

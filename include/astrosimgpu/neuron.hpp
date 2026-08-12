#pragma once

#include "astrosimgpu/parameters.hpp"
#include "astrosimgpu/rng.hpp"
#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

/// Adaptive exponential integrate-and-fire neurons with alpha-shaped
/// conductance synapses and an astrocytic slow inward current.
///
/// Structure-of-arrays for the same reasons as the astrocyte population.
///
///   C_m dV/dt = -g_L (V - E_L) + g_L Delta_T exp((V - V_th)/Delta_T)
///               - g_ex (V - E_ex) - g_in (V - E_in) - w + I_e + I_stim + I_SIC
///   tau_w dw/dt = a (V - E_L) - w
///
/// Each conductance is an alpha function driven by a two-variable cascade,
///
///   d(dg)/dt = -dg / tau_syn
///   dg/dt    = dg - g / tau_syn
///
/// where an arriving spike of weight W adds W * e / tau_syn to dg, so the
/// conductance peaks at W one tau_syn after arrival.
///
/// On V >= V_peak the neuron emits a spike, V is reset to V_reset, the
/// adaptation variable is incremented by b, and the cell is refractory for
/// t_ref during which V is clamped.
class NeuronPopulation {
public:
    /// `exc_count` excitatory cells are laid out first, then `inh_count`
    /// inhibitory cells, so a global index maps to a population by comparison.
    void build(index_t exc_count, index_t inh_count, const NeuronParams& exc,
               const NeuronParams& inh, const RandomizeSpec& randomize, const InputParams& in_exc,
               const InputParams& in_inh, CounterRng& rng);

    /// Add synaptic drive arriving this step. Positive weights go to the
    /// excitatory conductance, negative weights to the inhibitory one with
    /// their magnitude, matching how NEST routes conductance-based input.
    void add_synaptic_input(index_t cell, real weight);

    /// Set the summed astrocytic current for this step [pA].
    void set_sic(index_t cell, real current) { I_sic_[cell] = current; }

    /// Advance one communication step and append emitted spikes to `out`.
    void update(const TimeGrid& time, std::int64_t step, std::uint64_t seed, vec<Spike>& out);

    [[nodiscard]] index_t size() const { return static_cast<index_t>(V_.size()); }
    [[nodiscard]] index_t exc_count() const { return exc_count_; }
    [[nodiscard]] bool is_excitatory(index_t cell) const { return cell < exc_count_; }

    [[nodiscard]] const vec<real>& V() const { return V_; }
    [[nodiscard]] const vec<real>& w() const { return w_; }
    [[nodiscard]] const vec<real>& I_sic() const { return I_sic_; }

private:
    void derivatives(index_t cell, real V, real w, real g_ex, real g_in, real I_ext, real& dV,
                     real& dw) const;

    /// Parameters are stored per cell rather than per population: the two
    /// populations differ in every field, and several fields are randomised.
    struct CellParams {
        real C_m, g_L, E_L, V_th, Delta_T, a, b, tau_w;
        real V_reset, V_peak, t_ref, E_ex, E_in;
        real tau_syn_ex, tau_syn_in, I_e;
    };

    index_t exc_count_ = 0;
    index_t inh_count_ = 0;

    vec<CellParams> p_;

    // State.
    vec<real> V_, w_, g_ex_, dg_ex_, g_in_, dg_in_, I_sic_;
    vec<int> refractory_steps_;

    // Drive accumulated for the current step.
    vec<real> exc_input_, inh_input_;

    InputParams input_exc_{}, input_inh_{};
    // Precomputed e / tau_syn for each cell, the alpha-function normalisation.
    vec<real> psc_init_ex_, psc_init_in_;
};

}  // namespace astrosimgpu

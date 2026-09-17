#pragma once

#include "astrosimgpu/neuron_kernel.hpp"
#include "astrosimgpu/parameters.hpp"
#include "astrosimgpu/rng.hpp"
#include "astrosimgpu/types.hpp"

#if defined(ASTROSIMGPU_CUDA)
#include "astrosimgpu/neuron_cuda.hpp"
#endif

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
    /// `inputs_on_device`: true when the caller (Network's Stage 4 CUDA
    /// delivery path) has already written this step's exc/inh/SIC drive
    /// directly into the device-resident buffers (see
    /// cuda_delivery_apply_arrivals in network_cuda.hpp); the usual
    /// cuda_neuron_push_input is skipped in that case, since it would
    /// stomp those writes with the stale host mirrors. Default false keeps
    /// every other backend's existing push-every-step behaviour.
    /// `pull_spikes`: whether to copy the device spike-flag array back to
    /// the host and append Spike entries to `out`. Stage 4's own
    /// deliver_spikes kernel reads the device flags directly and needs no
    /// host copy at all; the pull only has to happen when the caller still
    /// needs individual Spike entries (e.g. spikes.csv). Default true keeps
    /// every other backend's existing pull-every-step behaviour.
    void update(const TimeGrid& time, std::int64_t step, std::uint64_t seed, vec<Spike>& out,
                bool inputs_on_device = false, bool pull_spikes = true);

    // Device residency. No-ops in a host build. Keeping the state on the
    // device makes the per-step map clauses free, so only the synaptic input
    // (in) and the spike flags (out) actually move each step. Unlike
    // AstrocytePopulation, the push/launch/pull sequence has no other host
    // step in between (nothing needs the spike flags except update() itself,
    // the way deliver_sic needs calcium pulled ahead of it), so it is all
    // folded inside update() rather than split into calls Network::run() has
    // to sequence -- only the once-per-run device_begin/device_end are
    // called from there.
    void device_begin();
    void device_end();

#if defined(ASTROSIMGPU_CUDA)
    // Raw device pointers for Network's Stage 4 delivery kernels: deliver_spikes
    // reads the spike flags directly (no host round trip), and apply_arrivals
    // writes the per-step drive directly, both on the device. CUDA-only,
    // since there is no device pointer to hand back on any other backend.
    [[nodiscard]] const unsigned char* device_spiked() const {
        return cuda_neuron_device_spiked(cuda_);
    }
    [[nodiscard]] real* device_exc_input() { return cuda_neuron_device_exc_input(cuda_); }
    [[nodiscard]] real* device_inh_input() { return cuda_neuron_device_inh_input(cuda_); }
    [[nodiscard]] real* device_sic() { return cuda_neuron_device_sic(cuda_); }
#endif

    [[nodiscard]] index_t size() const { return static_cast<index_t>(V_.size()); }
    [[nodiscard]] index_t exc_count() const { return exc_count_; }
    [[nodiscard]] bool is_excitatory(index_t cell) const { return cell < exc_count_; }

    [[nodiscard]] const vec<real>& V() const { return V_; }
    [[nodiscard]] const vec<real>& w() const { return w_; }
    [[nodiscard]] const vec<real>& I_sic() const { return I_sic_; }

private:
#if defined(ASTROSIMGPU_CUDA)
    CudaNeuron* cuda_ = nullptr;
#endif

    index_t exc_count_ = 0;
    index_t inh_count_ = 0;

    // CellParams is defined in neuron_kernel.hpp: parameters are stored per
    // cell rather than per population, since the two populations differ in
    // every field and several fields (V_reset, b) are randomised.
    vec<CellParams> p_;

    // State.
    vec<real> V_, w_, g_ex_, dg_ex_, g_in_, dg_in_, I_sic_;
    vec<int> refractory_steps_;

    // Drive accumulated for the current step.
    vec<real> exc_input_, inh_input_;

    InputParams input_exc_{}, input_inh_{};
    // Precomputed e / tau_syn for each cell, the alpha-function normalisation.
    vec<real> psc_init_ex_, psc_init_in_;

    // One byte per cell, reused every step by the CUDA path so update() does
    // not allocate; unused on the host path.
    vec<unsigned char> spiked_buffer_;
};

}  // namespace astrosimgpu

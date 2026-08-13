#pragma once

#include "astrosimgpu/astrocyte_kernel.hpp"
#include "astrosimgpu/parameters.hpp"

#if defined(ASTROSIMGPU_KOKKOS)
#include <Kokkos_Core.hpp>
#endif
#include "astrosimgpu/rng.hpp"
#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

/// Population of Li-Rinzel astrocytes, stored as structure-of-arrays.
///
/// The layout is deliberate: every state variable and every per-cell parameter
/// lives in its own contiguous array, so the update loop reads and writes with
/// unit stride and each cell's update touches nothing but its own lane. That
/// is what makes the loop vectorise, and it is the layout a device kernel
/// would want if the population is later moved off the host.
///
/// Model, following Li & Rinzel (1994) as extended by Nadkarni & Jung (2003):
///
///   dCa/dt  = J_channel - J_pump + J_leak + J_noise
///   dIP3/dt = (IP3_0 - IP3) / tau_IP3 + delta_IP3 * J_syn(t)
///   dh/dt   = alpha_h * (1 - h) - beta_h * h
///
/// with
///
///   Ca_ER     = (Ca_tot - Ca) / ratio_ER_cyt
///   m_inf     = IP3 / (IP3 + Kd_IP3_1)
///   n_inf     = Ca / (Ca + Kd_act)
///   alpha_h   = k_IP3R * Kd_inh * (IP3 + Kd_IP3_1) / (IP3 + Kd_IP3_2)
///   beta_h    = k_IP3R * Ca
///   J_channel = ratio_ER_cyt * rate_IP3R * m_inf^3 * n_inf^3 * h^3 * (Ca_ER - Ca)
///   J_pump    = rate_SERCA * Ca^2 / (Km_SERCA^2 + Ca^2)
///   J_leak    = ratio_ER_cyt * rate_L * (Ca_ER - Ca)
///
/// Calcium is held in [0, Ca_tot] after every step, so the additive noise
/// cannot drive the concentration negative.
class AstrocytePopulation {
public:
    void build(index_t count, const AstrocyteParams& base, const RandomizeSpec& randomize,
               const InputParams& input, CounterRng& rng);

    /// Deposit IP3 from presynaptic spikes arriving this step.
    void add_ip3_input(index_t cell, real weight) { ip3_input_[cell] += weight; }

    /// Make the state resident on the device for the lifetime of a run.
    ///
    /// OpenMP map clauses are reference counted: once an array is present, a
    /// later map on a target region finds it and moves nothing. The per-step
    /// clauses in update() therefore become free, and only the two quantities
    /// that genuinely cross the boundary each step have to be moved.
    ///
    /// All four are no-ops in a host build.
    void device_begin();
    void device_end();

    /// Send this step's synaptic input to the device.
    void device_push_input();

    /// Retrieve calcium, which the host needs in order to deliver the SIC.
    void device_pull_calcium();

    /// Clear the host-side synaptic input for the listed cells.
    ///
    /// The device zeroes its own copy inside the kernel, but with the arrays
    /// resident nothing carries that back, so the host copy has to be cleared
    /// separately or it accumulates for the whole run. Only cells some neuron
    /// projects to can ever be non-zero, and passing that list keeps the cost
    /// proportional to the connectivity rather than the population.
    void clear_inputs(const vec<index_t>& cells);

    /// Advance every astrocyte by one communication step.
    /// `step` seeds the noise draw so a run is reproducible for a given seed.
    void update(const TimeGrid& time, std::int64_t step, std::uint64_t seed);

    /// SIC output factor F_SIC(Ca) = SIC_scale * H(ln y) * ln y,
    /// with y = (Ca - SIC_th) / nM. Unitless; the pA unit enters through the
    /// astrocyte-to-neuron weight.
    [[nodiscard]] real sic_factor(index_t cell) const;

    [[nodiscard]] index_t size() const { return static_cast<index_t>(Ca_.size()); }
    [[nodiscard]] const vec<real>& Ca() const { return Ca_; }
    [[nodiscard]] const vec<real>& IP3() const { return IP3_; }
    [[nodiscard]] const vec<real>& h() const { return h_; }
    [[nodiscard]] const AstrocyteParams& params() const { return p_; }

private:
    /// Shared parameters packed for passing into a kernel by value.
    [[nodiscard]] AstroConstants constants() const;

#if defined(ASTROSIMGPU_KOKKOS)
    /// Device copies of the per-cell arrays.
    ///
    /// The std::vectors above stay the canonical host storage, and these are
    /// synchronised at the same four points the OpenMP target path uses, so
    /// the two backends move data at identical moments and can be compared
    /// without the transfer pattern differing between them.
    using DeviceArray = Kokkos::View<real*>;
    DeviceArray d_Ca_, d_IP3_, d_h_, d_ip3_input_;
    DeviceArray d_Ca_tot_, d_IP3_0_, d_tau_IP3_, d_delta_IP3_;
    bool device_ready_ = false;
#endif

    AstrocyteParams p_{};
    InputParams input_{};

    // State.
    vec<real> Ca_, IP3_, h_;
    // Per-cell parameters that the use case randomises.
    vec<real> Ca_tot_, IP3_0_, tau_IP3_, delta_IP3_;
    // Spike-driven IP3 deposited during the current step.
    vec<real> ip3_input_;
};

}  // namespace astrosimgpu

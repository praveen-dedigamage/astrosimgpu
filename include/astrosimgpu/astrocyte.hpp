#pragma once

#include "astrosimgpu/astrocyte_kernel.hpp"
#include "astrosimgpu/parameters.hpp"

#if defined(ASTROSIMGPU_KOKKOS)
#include <Kokkos_Core.hpp>
#endif
#if defined(ASTROSIMGPU_CUDA)
#include "astrosimgpu/astrocyte_cuda.hpp"
#endif
#include "astrosimgpu/rng.hpp"
#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

// Li-Rinzel astrocytes. Equations in the README; kernel in astrocyte_kernel.hpp.
// Structure-of-arrays so the update loop is unit-stride and coalesces on GPU.
class AstrocytePopulation {
public:
    void build(index_t count, const AstrocyteParams& base, const RandomizeSpec& randomize,
               const InputParams& input, CounterRng& rng);

    void add_ip3_input(index_t cell, real weight) { ip3_input_[cell] += weight; }

    // Device residency. No-ops in a host build. Keeping the state on the
    // device makes the per-step map clauses free, so only the input (in) and
    // calcium (out) actually move each step.
    void device_begin();
    void device_end();
    void device_push_input();
    void device_pull_calcium();

    // The kernel zeroes its own copy of the input, but nothing carries that
    // back once the arrays are resident, so the host copy needs clearing or it
    // accumulates. Only cells with an incoming connection can be non-zero.
    void clear_inputs(const vec<index_t>& cells);

    // `step` seeds the noise, so a run is reproducible from its seed.
    void update(const TimeGrid& time, std::int64_t step, std::uint64_t seed);

    // Unitless; the pA comes from the astrocyte-to-neuron weight.
    [[nodiscard]] real sic_factor(index_t cell) const;

    [[nodiscard]] index_t size() const { return static_cast<index_t>(Ca_.size()); }
    [[nodiscard]] const vec<real>& Ca() const { return Ca_; }
    [[nodiscard]] const vec<real>& IP3() const { return IP3_; }
    [[nodiscard]] const vec<real>& h() const { return h_; }
    [[nodiscard]] const AstrocyteParams& params() const { return p_; }

private:
    [[nodiscard]] AstroConstants constants() const;

#if defined(ASTROSIMGPU_KOKKOS)
    // The vectors below stay canonical; these are synced at the same four
    // points as the OpenMP path so the backends stay comparable.
    using DeviceArray = Kokkos::View<real*>;
    DeviceArray d_Ca_, d_IP3_, d_h_, d_ip3_input_;
    DeviceArray d_Ca_tot_, d_IP3_0_, d_tau_IP3_, d_delta_IP3_;
    bool device_ready_ = false;
#endif
#if defined(ASTROSIMGPU_CUDA)
    CudaAstro* cuda_ = nullptr;
#endif

    AstrocyteParams p_{};
    InputParams input_{};

    vec<real> Ca_, IP3_, h_;                                  // state
    vec<real> Ca_tot_, IP3_0_, tau_IP3_, delta_IP3_;          // randomised per cell
    vec<real> ip3_input_;                                     // this step's spike input
};

}  // namespace astrosimgpu

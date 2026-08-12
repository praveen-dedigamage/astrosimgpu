#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace astrosimgpu {

/// Floating point type used throughout the simulator.
///
/// Kept as a single alias so the whole model can be switched to float for
/// GPU work without touching the model code.
using real = double;

using index_t = std::uint32_t;

/// Marks an absent index (e.g. an astrocyte pool slot that is not filled).
inline constexpr index_t no_index = static_cast<index_t>(-1);

template <typename T>
using vec = std::vector<T>;

/// A spike emitted by a neuron, in units of simulation steps.
struct Spike {
    std::int64_t step;   ///< step at which the spike was emitted
    index_t source;      ///< global neuron index
};

/// Time is measured in milliseconds everywhere, matching the reference model.
struct TimeGrid {
    real dt = 0.1;             ///< communication / recording step [ms]
    int substeps = 1;          ///< RK4 substeps taken within one dt
    real pre_sim_time = 0.0;   ///< transient discarded before recording [ms]
    real sim_time = 0.0;       ///< recorded simulation time [ms]

    [[nodiscard]] real h() const { return dt / static_cast<real>(substeps); }
    [[nodiscard]] std::int64_t pre_steps() const {
        return static_cast<std::int64_t>(pre_sim_time / dt + 0.5);
    }
    [[nodiscard]] std::int64_t sim_steps() const {
        return static_cast<std::int64_t>(sim_time / dt + 0.5);
    }
};

}  // namespace astrosimgpu

#pragma once

#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

/// Window over which a signal is analysed [ms].
struct Interval {
    real start = 0.0;
    real end = 0.0;
};

/// Onsets, offsets and durations of the events detected in one trace.
struct EventTrain {
    vec<real> onset;
    vec<real> offset;
    vec<real> duration;
};

/// Detect threshold crossings in a sampled trace.
///
/// An event starts at the first sample at or above `threshold` and ends at the
/// last sample above it. When `merge` is set, two events separated by less
/// than `max_gap` are treated as one interrupted event rather than two: noise
/// around the threshold otherwise fragments a single calcium transient into
/// several, which biases both the count and the duration statistics.
EventTrain detect_events(const vec<real>& times, const vec<real>& values, real threshold,
                         bool merge, real max_gap);

/// Rate of events in a sliding window, in events per ms.
///
/// The window of width `width` advances by `shift`, so consecutive bins
/// overlap. This is the representation the pairwise correlation is computed
/// on, following the reference analysis.
vec<real> sliding_window_rate(const vec<real>& event_times, const Interval& window, real width,
                              real shift);

/// Pearson correlation coefficient between two equal-length series.
/// Returns 0 when either series has no variance.
real pearson(const vec<real>& a, const vec<real>& b);

/// Pairwise Pearson correlation between the windowed rates of every cell.
///
/// `per_cell_events` holds one list of event times per cell. Cells with no
/// events contribute no pairs, matching the reference implementation, which
/// only correlates histograms that contain at least one event.
struct PairwiseCorrelation {
    vec<real> coefficient;
    vec<index_t> cell_a;
    vec<index_t> cell_b;

    [[nodiscard]] real mean() const;
};

PairwiseCorrelation pairwise_correlation(const vec<vec<real>>& per_cell_events,
                                         const Interval& window, real width, real shift);

/// Distance from each event of one cell to the nearest event of another,
/// pooled over all ordered cell pairs. A concentration of small distances is
/// the signature of synchronised activity.
vec<real> nearest_event_distances(const vec<vec<real>>& per_cell_events);

/// Mean and standard deviation of a sample. Returns {0, 0} when empty.
struct Summary {
    real mean = 0.0;
    real stddev = 0.0;
    std::size_t count = 0;
};

Summary summarise(const vec<real>& values);

}  // namespace astrosimgpu

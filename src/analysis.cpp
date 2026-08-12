#include "astrosimgpu/analysis.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>

namespace astrosimgpu {

EventTrain detect_events(const vec<real>& times, const vec<real>& values, real threshold,
                         bool merge, real max_gap) {
    EventTrain out;
    const std::size_t n = std::min(times.size(), values.size());
    if (n < 2) {
        return out;
    }

    vec<real> on, off;
    for (std::size_t i = 1; i < n; ++i) {
        if (values[i] >= threshold && values[i - 1] < threshold) {
            on.push_back(times[i]);
        }
        if (values[i] < threshold && values[i - 1] >= threshold) {
            off.push_back(times[i - 1]);
        }
    }
    // An event may already be running at the start of the window, or still be
    // running at the end; both are kept so durations are not silently dropped.
    if (values[0] >= threshold) {
        on.insert(on.begin(), times[0]);
    }
    if (values[n - 1] >= threshold) {
        off.push_back(times[n - 1]);
    }

    if (merge && !on.empty() && !off.empty()) {
        vec<real> merged_on{on.front()};
        vec<real> merged_off;
        const std::size_t pairs = std::min(on.size(), off.size());
        for (std::size_t i = 0; i + 1 < pairs; ++i) {
            if (on[i + 1] - off[i] >= max_gap) {
                merged_off.push_back(off[i]);
                merged_on.push_back(on[i + 1]);
            }
        }
        if (!off.empty()) {
            merged_off.push_back(off.back());
        }
        on.swap(merged_on);
        off.swap(merged_off);
    }

    out.onset = on;
    out.offset = off;
    const std::size_t pairs = std::min(out.onset.size(), out.offset.size());
    out.duration.reserve(pairs);
    for (std::size_t i = 0; i < pairs; ++i) {
        const real d = out.offset[i] - out.onset[i];
        if (d >= 0.0) {
            out.duration.push_back(d);
        }
    }
    return out;
}

vec<real> sliding_window_rate(const vec<real>& event_times, const Interval& window, real width,
                              real shift) {
    vec<real> hist;
    if (shift <= 0.0 || width <= 0.0 || window.end <= window.start) {
        return hist;
    }
    const auto steps = static_cast<std::int64_t>(
        std::floor((window.end - window.start - width) / shift));
    if (steps <= 0) {
        return hist;
    }
    hist.assign(static_cast<std::size_t>(steps), 0.0);
    if (event_times.empty()) {
        return hist;
    }

    // Events are sorted, so each window is a range scan rather than a full
    // pass over the event list.
    vec<real> sorted(event_times);
    std::sort(sorted.begin(), sorted.end());

    for (std::int64_t i = 0; i < steps; ++i) {
        const real lo = window.start + static_cast<real>(i) * shift;
        const real hi = lo + width;
        const auto first = std::lower_bound(sorted.begin(), sorted.end(), lo);
        const auto last = std::upper_bound(sorted.begin(), sorted.end(), hi);
        hist[static_cast<std::size_t>(i)] =
            static_cast<real>(std::distance(first, last)) / width;
    }
    return hist;
}

real pearson(const vec<real>& a, const vec<real>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    if (n < 2) {
        return 0.0;
    }
    real mean_a = 0.0, mean_b = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= static_cast<real>(n);
    mean_b /= static_cast<real>(n);

    real cov = 0.0, var_a = 0.0, var_b = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const real da = a[i] - mean_a;
        const real db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if (var_a <= 0.0 || var_b <= 0.0) {
        return 0.0;
    }
    return cov / std::sqrt(var_a * var_b);
}

real PairwiseCorrelation::mean() const {
    if (coefficient.empty()) {
        return 0.0;
    }
    real sum = 0.0;
    for (const real c : coefficient) {
        sum += c;
    }
    return sum / static_cast<real>(coefficient.size());
}

PairwiseCorrelation pairwise_correlation(const vec<vec<real>>& per_cell_events,
                                         const Interval& window, real width, real shift) {
    PairwiseCorrelation out;
    const std::size_t n = per_cell_events.size();

    vec<vec<real>> hists(n);
    vec<char> usable(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        hists[i] = sliding_window_rate(per_cell_events[i], window, width, shift);
        real sum = 0.0;
        for (const real v : hists[i]) {
            sum += v;
        }
        usable[i] = sum > 0.0 ? 1 : 0;
    }

    for (std::size_t i = 0; i + 1 < n; ++i) {
        if (!usable[i]) {
            continue;
        }
        for (std::size_t j = i + 1; j < n; ++j) {
            if (!usable[j]) {
                continue;
            }
            out.coefficient.push_back(pearson(hists[i], hists[j]));
            out.cell_a.push_back(static_cast<index_t>(i));
            out.cell_b.push_back(static_cast<index_t>(j));
        }
    }
    return out;
}

vec<real> nearest_event_distances(const vec<vec<real>>& per_cell_events) {
    vec<real> out;
    const std::size_t n = per_cell_events.size();

    vec<vec<real>> sorted(per_cell_events);
    for (auto& s : sorted) {
        std::sort(s.begin(), s.end());
    }

    for (std::size_t i = 0; i + 1 < n; ++i) {
        if (sorted[i].empty()) {
            continue;
        }
        for (std::size_t j = i + 1; j < n; ++j) {
            if (sorted[j].empty()) {
                continue;
            }
            for (const real e : sorted[i]) {
                const auto it = std::lower_bound(sorted[j].begin(), sorted[j].end(), e);
                real best = std::numeric_limits<real>::max();
                if (it != sorted[j].end()) {
                    best = std::min(best, std::abs(*it - e));
                }
                if (it != sorted[j].begin()) {
                    best = std::min(best, std::abs(*std::prev(it) - e));
                }
                out.push_back(best);
            }
        }
    }
    return out;
}

Summary summarise(const vec<real>& values) {
    Summary s;
    s.count = values.size();
    if (values.empty()) {
        return s;
    }
    real sum = 0.0;
    for (const real v : values) {
        sum += v;
    }
    s.mean = sum / static_cast<real>(values.size());

    real acc = 0.0;
    for (const real v : values) {
        const real d = v - s.mean;
        acc += d * d;
    }
    s.stddev = std::sqrt(acc / static_cast<real>(values.size()));
    return s;
}

}  // namespace astrosimgpu

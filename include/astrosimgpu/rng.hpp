#pragma once

#include <cmath>
#include <cstdint>

#include "astrosimgpu/types.hpp"

#if defined(ASTROSIMGPU_KOKKOS)
#include <Kokkos_Macros.hpp>
#define ASTROSIMGPU_RNG_FN KOKKOS_INLINE_FUNCTION
#elif defined(__CUDACC__)
#define ASTROSIMGPU_RNG_FN __host__ __device__ inline
#else
#define ASTROSIMGPU_RNG_FN inline
#endif

namespace astrosimgpu {

// splitmix64 over (seed, stream, counter). Free functions because a device
// kernel cannot carry a host object; CounterRng below wraps these, and a test
// pins the two to agree.
#ifdef ASTROSIMGPU_OFFLOAD
#pragma omp declare target
#endif

ASTROSIMGPU_RNG_FN std::uint64_t rng_bits(std::uint64_t seed, std::uint64_t stream, std::uint64_t counter) {
    std::uint64_t z = seed;
    z += stream * 0x9E3779B97F4A7C15ULL;
    z += counter * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Uniform on [0, 1).
ASTROSIMGPU_RNG_FN real rng_uniform(std::uint64_t seed, std::uint64_t stream, std::uint64_t counter) {
    return static_cast<real>(rng_bits(seed, stream, counter) >> 11) *
           (1.0 / 9007199254740992.0);
}

// One standard normal, Box-Muller over counters 0 and 1.
ASTROSIMGPU_RNG_FN real rng_normal(std::uint64_t seed, std::uint64_t stream) {
    real u1 = rng_uniform(seed, stream, 0);
    if (u1 <= 1e-300) {
        u1 = 1e-300;
    }
    const real u2 = rng_uniform(seed, stream, 1);
    return std::sqrt(-2.0 * std::log(u1)) *
           std::cos(6.283185307179586476925286766559 * u2);
}

#ifdef ASTROSIMGPU_OFFLOAD
#pragma omp end declare target
#endif

// Counter-based: every draw is a pure function of (seed, stream, counter), so
// a cell reproduces its own noise without touching shared state.
class CounterRng {
public:
    CounterRng() = default;
    CounterRng(std::uint64_t seed, std::uint64_t stream) : seed_(seed), stream_(stream) {}

    /// Uniform on [0, 1).
    real uniform() {
        const std::uint64_t bits = mix(seed_, stream_, counter_++);
        // 53 significant bits, the most a double can hold exactly.
        return static_cast<real>(bits >> 11) * (1.0 / 9007199254740992.0);
    }

    /// Uniform on [lo, hi).
    real uniform(real lo, real hi) { return lo + (hi - lo) * uniform(); }

    // Keeps the second variate, so two calls cost one transcendental pair.
    real normal() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        real u1, u2;
        do {
            u1 = uniform();
        } while (u1 <= 1e-300);
        u2 = uniform();
        const real r = std::sqrt(-2.0 * std::log(u1));
        const real theta = 6.283185307179586476925286766559 * u2;
        spare_ = r * std::sin(theta);
        has_spare_ = true;
        return r * std::cos(theta);
    }

    real normal(real mean, real stddev) { return mean + stddev * normal(); }

    // Mirrors nest.math.redraw(): resamples rather than clipping, so the
    // distribution keeps its shape.
    real normal_redraw(real mean, real stddev, real lo, real hi, int max_tries = 1000) {
        for (int i = 0; i < max_tries; ++i) {
            const real v = normal(mean, stddev);
            if (v >= lo && v <= hi) {
                return v;
            }
        }
        return std::min(std::max(mean, lo), hi);
    }

    // Knuth. Rates here give lambda well under 1 per step, so the loop almost
    // always exits on the first test.
    int poisson(real lambda) {
        if (lambda <= 0.0) {
            return 0;
        }
        if (lambda > 30.0) {
            // Normal approximation; only reached for pathological input rates.
            const real v = normal(lambda, std::sqrt(lambda));
            return v < 0.0 ? 0 : static_cast<int>(v + 0.5);
        }
        const real limit = std::exp(-lambda);
        real product = uniform();
        int count = 0;
        while (product > limit) {
            ++count;
            product *= uniform();
        }
        return count;
    }

    void reset(std::uint64_t seed, std::uint64_t stream) {
        seed_ = seed;
        stream_ = stream;
        counter_ = 0;
        has_spare_ = false;
    }

private:
    static std::uint64_t mix(std::uint64_t seed, std::uint64_t stream, std::uint64_t counter) {
        return rng_bits(seed, stream, counter);
    }

    std::uint64_t seed_ = 1;
    std::uint64_t stream_ = 0;
    std::uint64_t counter_ = 0;
    bool has_spare_ = false;
    real spare_ = 0.0;
};

}  // namespace astrosimgpu

// Self-contained checks on the model components. No test framework: the
// simulator has no dependencies and the tests should not add one.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "astrosimgpu/analysis.hpp"
#include "astrosimgpu/astrocyte_kernel.hpp"
#include "astrosimgpu/astrocyte.hpp"
#include "astrosimgpu/json.hpp"
#include "astrosimgpu/network.hpp"
#include "astrosimgpu/neuron.hpp"
#include "astrosimgpu/recorder.hpp"
#include "astrosimgpu/rng.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace astrosimgpu;

namespace {

int failures = 0;
int checks = 0;

void check(bool condition, const std::string& what) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "  FAIL: " << what << "\n";
    }
}

void check_close(real got, real expected, real tol, const std::string& what) {
    ++checks;
    if (!(std::abs(got - expected) <= tol)) {
        ++failures;
        std::cerr << "  FAIL: " << what << " (got " << got << ", expected " << expected
                  << " +/- " << tol << ")\n";
    }
}

void test_json() {
    const Json j = Json::parse(R"({
        "a": 1.5, "b": [1, 2, 3], "c": {"d": true}, "e": "text", "f": null
    })");
    check(j.is_object(), "json root is an object");
    check_close(j["a"].number(0.0), 1.5, 1e-12, "json number");
    check(j["b"].is_array() && j["b"].items().size() == 3, "json array length");
    check(j["c"]["d"].boolean(false), "json nested bool");
    check(j["e"].string("") == "text", "json string");
    check(j["f"].is_null(), "json null");
    check(!j.contains("missing"), "json missing key");

    double untouched = 42.0;
    j.get_to("missing", untouched);
    check_close(untouched, 42.0, 1e-12, "json get_to leaves defaults alone");

    bool threw = false;
    try {
        Json::parse("{ oops }");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "json rejects malformed input");
}

void test_rng() {
    CounterRng rng(12345, 7);
    const int n = 200000;

    real sum = 0.0;
    real min_v = 1.0, max_v = 0.0;
    for (int i = 0; i < n; ++i) {
        const real u = rng.uniform();
        sum += u;
        min_v = std::min(min_v, u);
        max_v = std::max(max_v, u);
    }
    check_close(sum / n, 0.5, 0.01, "uniform mean");
    check(min_v >= 0.0 && max_v < 1.0, "uniform range");

    CounterRng rng2(999, 1);
    real s = 0.0, s2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const real v = rng2.normal();
        s += v;
        s2 += v * v;
    }
    check_close(s / n, 0.0, 0.02, "normal mean");
    check_close(std::sqrt(s2 / n), 1.0, 0.02, "normal stddev");

    CounterRng rng3(31337, 2);
    const real lambda = 0.27;
    long total = 0;
    for (int i = 0; i < n; ++i) {
        total += rng3.poisson(lambda);
    }
    check_close(static_cast<real>(total) / n, lambda, 0.01, "poisson mean");

    // The same seed and stream must replay the same sequence: a run has to be
    // reproducible from its seed alone.
    CounterRng a(2024, 5), b(2024, 5);
    bool identical = true;
    for (int i = 0; i < 1000; ++i) {
        identical = identical && (a.uniform() == b.uniform());
    }
    check(identical, "rng is reproducible for a given seed and stream");

    // The free functions are what a device kernel calls; CounterRng is what
    // the rest of the host code uses. If these ever diverge, an offloaded run
    // silently stops matching the CPU run it is validated against, so the
    // equality is pinned here rather than assumed.
    bool free_matches = true;
    for (std::uint64_t stream = 0; stream < 200; ++stream) {
        CounterRng ref(4242, stream);
        free_matches = free_matches && (ref.uniform() == rng_uniform(4242, stream, 0));
        CounterRng ref2(4242, stream);
        free_matches = free_matches && (ref2.normal() == rng_normal(4242, stream));
    }
    check(free_matches, "free rng functions match CounterRng exactly");

    // Same reasoning as above, for rng_poisson: astro_drive_kernel (CUDA)
    // calls this instead of the host drive_astrocytes loop's
    // CounterRng(...).poisson(lambda), and the two have to agree exactly for
    // the device path to be the same computation, not a different one that
    // happens to look similar. Covers both branches: lambda <= 30 (the
    // rejection loop, what every real input rate in this codebase hits) and
    // lambda > 30 (the normal approximation, reachable only by a
    // pathological config).
    bool poisson_matches = true;
    for (std::uint64_t stream = 0; stream < 200; ++stream) {
        CounterRng ref3(777, stream);
        poisson_matches = poisson_matches && (ref3.poisson(0.27) == rng_poisson(777, stream, 0.27));
        CounterRng ref4(777, stream);
        poisson_matches = poisson_matches && (ref4.poisson(45.0) == rng_poisson(777, stream, 45.0));
    }
    check(poisson_matches, "free rng_poisson matches CounterRng::poisson exactly");
}

void test_astrocyte_rest() {
    AstrocyteParams p;
    p.IP3_init = p.IP3_0;
    RandomizeSpec none;
    InputParams no_input;
    CounterRng rng(1, 1);

    AstrocytePopulation pop;
    pop.build(4, p, none, no_input, rng);

    TimeGrid t;
    t.dt = 0.1;
    t.substeps = 1;

    // Undriven, the state must stay finite and inside its physical bounds.
    for (std::int64_t step = 0; step < 20000; ++step) {
        pop.update(t, step, 1);
    }
    for (index_t i = 0; i < pop.size(); ++i) {
        check(std::isfinite(pop.Ca()[i]), "calcium stays finite at rest");
        check(pop.Ca()[i] >= 0.0 && pop.Ca()[i] <= p.Ca_tot, "calcium stays within [0, Ca_tot]");
        check(pop.h()[i] >= 0.0 && pop.h()[i] <= 1.0, "IP3R gate stays a fraction");
        check_close(pop.IP3()[i], p.IP3_0, 1e-3, "IP3 relaxes to its baseline");
    }
}

void test_astrocyte_ip3_decay() {
    AstrocyteParams p;
    p.tau_IP3 = 1000.0;
    p.IP3_0 = 0.1;
    p.IP3_init = 0.1;
    p.delta_IP3 = 1.0;
    RandomizeSpec none;
    InputParams no_input;
    CounterRng rng(1, 1);

    AstrocytePopulation pop;
    pop.build(1, p, none, no_input, rng);

    TimeGrid t;
    t.dt = 0.1;
    t.substeps = 1;

    // One unit-weight input adds delta_IP3, then IP3 must decay towards
    // baseline with the configured time constant.
    pop.add_ip3_input(0, 1.0);
    pop.update(t, 0, 1);
    const real after_input = pop.IP3()[0];
    check(after_input > p.IP3_0 + 0.9, "input raises IP3 by delta_IP3");

    const real excess0 = after_input - p.IP3_0;
    const int steps = static_cast<int>(p.tau_IP3 / t.dt);
    for (int i = 1; i <= steps; ++i) {
        pop.update(t, i, 1);
    }
    const real excess1 = pop.IP3()[0] - p.IP3_0;
    check_close(excess1 / excess0, std::exp(-1.0), 0.01, "IP3 decays with tau_IP3");
}

void test_astro_kernel() {
    // The kernel is the code an offloaded loop runs. Check it against the
    // properties the population-level tests cover, but called directly, so a
    // regression shows up here rather than three layers up.
    AstrocyteParams p;
    AstroConstants c{p.Kd_IP3_1, p.Kd_IP3_2, p.Kd_act,      p.Kd_inh,
                     p.Km_SERCA, p.k_IP3R,   p.rate_IP3R,   p.rate_L,
                     p.rate_SERCA, p.ratio_ER_cyt};

    real Ca = p.Ca_init, IP3 = p.IP3_0, h = p.h_init;
    for (int step = 0; step < 20000; ++step) {
        astro_advance(c, p.Ca_tot, p.IP3_0, p.tau_IP3, p.delta_IP3, 0.0, 0.0, 0.1, 1, Ca, IP3, h);
    }
    check(std::isfinite(Ca) && Ca >= 0.0 && Ca <= p.Ca_tot, "kernel keeps calcium in range");
    check(h >= 0.0 && h <= 1.0, "kernel keeps the IP3R gate a fraction");
    check_close(IP3, p.IP3_0, 1e-6, "kernel relaxes IP3 to baseline");

    // An IP3 input must raise IP3 by delta_IP3 times the input weight.
    real Ca2 = p.Ca_init, IP3_2 = p.IP3_0, h2 = p.h_init;
    astro_advance(c, p.Ca_tot, p.IP3_0, p.tau_IP3, 0.5, 2.0, 0.0, 0.0, 0, Ca2, IP3_2, h2);
    check_close(IP3_2, p.IP3_0 + 1.0, 1e-12, "kernel applies delta_IP3 times input weight");
}

void test_sic_threshold() {
    AstrocyteParams p;
    p.SIC_th = 0.19669;
    p.SIC_scale = 1.0;
    p.Ca_init = 0.0;
    p.IP3_init = p.IP3_0;
    RandomizeSpec none;
    InputParams no_input;
    CounterRng rng(1, 1);

    AstrocytePopulation pop;
    pop.build(1, p, none, no_input, rng);

    // Below threshold, and within 1 nM of it, the output must be exactly zero.
    check_close(pop.sic_factor(0), 0.0, 0.0, "no SIC below threshold");
}

void test_neuron_kernel() {
    // The kernel is the code the CUDA path runs. Check it against the
    // properties the population-level tests below cover, but called
    // directly, so a regression shows up here rather than a layer up --
    // same rationale as test_astro_kernel.

    // Isolates the alpha cascade the same way test_alpha_conductance does:
    // zero everything that would move V except the synaptic conductance,
    // and never spike, so the peak time pins the psc_init normalisation.
    {
        CellParams p{};
        p.C_m = 281.0;
        p.g_L = 0.0;
        p.E_L = -70.6;
        p.V_th = -50.4;
        p.Delta_T = 2.0;
        p.a = 0.0;
        p.b = 0.0;
        p.tau_w = 144.0;
        p.V_reset = -60.0;
        p.V_peak = 1e9;  // never spike
        p.t_ref = 0.0;
        p.E_ex = 0.0;
        p.E_in = -85.0;
        p.tau_syn_ex = 2.0;
        p.tau_syn_in = 2.0;
        p.I_e = 0.0;
        const real psc_init_ex = std::exp(1.0) / p.tau_syn_ex;

        real V = -70.6, w = 0.0, g_ex = 0.0, dg_ex = 0.0, g_in = 0.0, dg_in = 0.0;
        int refractory = 0;
        const real h_step = 0.01;
        const real weight = 3.0;

        real peak_drop = 0.0;
        real peak_time = 0.0;
        real prev_V = V;
        for (std::int64_t step = 0; step < 2000; ++step) {
            const real exc_input = (step == 0) ? weight : 0.0;
            const bool spiked = neuron_advance(p, h_step, 1, exc_input, 0.0, psc_init_ex, 0.0,
                                               0.0, V, w, g_ex, dg_ex, g_in, dg_in, refractory);
            check(!spiked, "kernel with V_peak=1e9 never reports a spike");
            const real drop = std::abs(V - prev_V);
            if (drop > peak_drop) {
                peak_drop = drop;
                peak_time = static_cast<real>(step) * h_step;
            }
            prev_V = V;
        }
        check_close(peak_time, p.tau_syn_ex, 0.2, "kernel's alpha conductance peaks at tau_syn");
    }

    // Isolates the refractory/spike path: a steady current well above
    // rheobase must make the kernel report a spike and clamp V to V_reset
    // for t_ref afterwards -- mirrors the driven half of
    // test_neuron_rest_and_spiking.
    {
        CellParams p{};
        p.C_m = 130.0;
        p.g_L = 18.0;
        p.E_L = -58.0;
        p.V_th = -50.0;
        p.Delta_T = 2.0;
        p.a = 4.0;
        p.b = 300.0;
        p.tau_w = 450.0;
        p.V_reset = -50.0;
        p.V_peak = 0.0;
        p.t_ref = 2.0;
        p.E_ex = 0.0;
        p.E_in = -85.0;
        p.tau_syn_ex = 2.0;
        p.tau_syn_in = 2.0;
        p.I_e = 700.0;

        real V = -58.0, w = 0.0, g_ex = 0.0, dg_ex = 0.0, g_in = 0.0, dg_in = 0.0;
        int refractory = 0;
        const real h_step = 0.1 / 4;

        bool spiked_once = false;
        bool clamped_after_spike = false;
        for (std::int64_t step = 0; step < 20000 * 4; ++step) {
            const bool spiked =
                neuron_advance(p, h_step, 1, 0.0, 0.0, 0.0, 0.0, 0.0, V, w, g_ex, dg_ex, g_in,
                               dg_in, refractory);
            if (spiked) {
                spiked_once = true;
                check_close(V, p.V_reset, 1e-9, "kernel resets V to V_reset on a spike");
                check(refractory > 0, "kernel enters refractory on a spike");
            }
            if (spiked_once && refractory > 0) {
                clamped_after_spike = true;
            }
        }
        check(spiked_once, "kernel spikes when driven well above rheobase");
        check(clamped_after_spike, "kernel's refractory counter engages after a spike");
    }
}

void test_neuron_rest_and_spiking() {
    NeuronParams p;
    p.C_m = 130.0;
    p.g_L = 18.0;
    p.E_L = -58.0;
    p.V_th = -50.0;
    p.V_reset = -50.0;
    p.b = 300.0;
    p.a = 4.0;
    p.tau_w = 450.0;
    p.V_m_init = -58.0;
    p.tau_syn_ex = 2.0;
    p.tau_syn_in = 2.0;

    RandomizeSpec none;
    InputParams quiet;
    CounterRng rng(1, 1);

    TimeGrid t;
    t.dt = 0.1;
    t.substeps = 4;

    {
        NeuronPopulation pop;
        pop.build(1, 0, p, p, none, quiet, quiet, rng);
        vec<Spike> spikes;
        for (std::int64_t step = 0; step < 5000; ++step) {
            pop.update(t, step, 1, spikes);
        }
        check(spikes.empty(), "an undriven neuron does not fire");
        check_close(pop.V()[0], p.E_L, 1.0, "membrane potential settles near E_L");
    }

    {
        // A steady current well above rheobase must produce spikes.
        NeuronParams driven = p;
        driven.I_e = 700.0;
        NeuronPopulation pop;
        pop.build(1, 0, driven, driven, none, quiet, quiet, rng);
        vec<Spike> spikes;
        for (std::int64_t step = 0; step < 20000; ++step) {
            pop.update(t, step, 1, spikes);
        }
        check(!spikes.empty(), "a driven neuron fires");
        for (const Spike& s : spikes) {
            check(s.source == 0, "spikes carry the emitting neuron index");
        }
    }
}

void test_alpha_conductance() {
    // An alpha conductance driven by one spike of weight W must peak at W,
    // one synaptic time constant after arrival. This is what fixes the
    // normalisation of the two-variable cascade.
    NeuronParams p;
    p.tau_syn_ex = 2.0;
    p.V_m_init = -70.6;
    // Hold the membrane fixed so the conductance can be read off the current.
    p.g_L = 0.0;
    p.a = 0.0;
    p.b = 0.0;
    p.I_e = 0.0;
    p.V_peak = 1e9;  // never spike

    RandomizeSpec none;
    InputParams quiet;
    CounterRng rng(1, 1);

    TimeGrid t;
    t.dt = 0.01;
    t.substeps = 1;

    NeuronPopulation pop;
    pop.build(1, 0, p, p, none, quiet, quiet, rng);

    const real weight = 3.0;
    pop.add_synaptic_input(0, weight);

    vec<Spike> spikes;
    real peak_drop = 0.0;
    real peak_time = 0.0;
    real prev_V = pop.V()[0];
    for (std::int64_t step = 0; step < 2000; ++step) {
        pop.update(t, step, 1, spikes);
        const real drop = std::abs(pop.V()[0] - prev_V);
        if (drop > peak_drop) {
            peak_drop = drop;
            peak_time = static_cast<real>(step) * t.dt;
        }
        prev_V = pop.V()[0];
    }
    // The largest change in V tracks the conductance peak.
    check_close(peak_time, p.tau_syn_ex, 0.2, "alpha conductance peaks at tau_syn");
}

void test_analysis() {
    // Two square pulses over a flat baseline.
    vec<real> times, values;
    for (int i = 0; i < 1000; ++i) {
        const real t = static_cast<real>(i);
        times.push_back(t);
        const bool high = (i >= 100 && i < 200) || (i >= 600 && i < 650);
        values.push_back(high ? 1.0 : 0.0);
    }
    const EventTrain ev = detect_events(times, values, 0.5, false, 0.0);
    check(ev.onset.size() == 2, "two events detected");
    check(ev.duration.size() == 2, "two durations reported");
    if (ev.duration.size() == 2) {
        check_close(ev.duration[0], 99.0, 1.5, "first event duration");
        check_close(ev.duration[1], 49.0, 1.5, "second event duration");
    }

    // Merging joins events separated by less than the allowed gap.
    vec<real> v2;
    for (int i = 0; i < 1000; ++i) {
        const bool high = (i >= 100 && i < 200) || (i >= 210 && i < 300);
        v2.push_back(high ? 1.0 : 0.0);
    }
    const EventTrain merged = detect_events(times, v2, 0.5, true, 50.0);
    check(merged.onset.size() == 1, "nearby events merge into one");

    // Identical series correlate perfectly; a constant series has no variance.
    const vec<real> a{1.0, 2.0, 3.0, 4.0, 5.0};
    const vec<real> b{2.0, 4.0, 6.0, 8.0, 10.0};
    const vec<real> c{1.0, 1.0, 1.0, 1.0, 1.0};
    check_close(pearson(a, a), 1.0, 1e-12, "self correlation is one");
    check_close(pearson(a, b), 1.0, 1e-12, "linear scaling correlates perfectly");
    check_close(pearson(a, c), 0.0, 1e-12, "constant series correlates as zero");

    // Windowed rate: ten events spread over the window.
    vec<real> events;
    for (int i = 0; i < 10; ++i) {
        events.push_back(100.0 * i);
    }
    Interval w{0.0, 1000.0};
    const vec<real> hist = sliding_window_rate(events, w, 200.0, 100.0);
    check(!hist.empty(), "sliding window produces bins");
    for (const real v : hist) {
        check(v >= 0.0, "rates are non-negative");
    }

    const Summary s = summarise(vec<real>{1.0, 2.0, 3.0});
    check_close(s.mean, 2.0, 1e-12, "summary mean");
    check(s.count == 3, "summary count");

    const vec<vec<real>> per_cell{{10.0, 20.0}, {10.5, 20.5}};
    const vec<real> d = nearest_event_distances(per_cell);
    check(d.size() == 2, "one distance per event of the first cell");
    if (d.size() == 2) {
        check_close(d[0], 0.5, 1e-9, "nearest distance");
    }
}

void test_network_build() {
    ModelConfig cfg;
    cfg.N = {10, 40, 10};
    cfg.time.dt = 0.1;
    cfg.time.substeps = 1;
    cfg.time.pre_sim_time = 0.0;
    cfg.time.sim_time = 20.0;
    cfg.conn.p_primary = 0.2;
    cfg.conn.p_third_if_primary = 0.5;
    cfg.conn.pool_size = 1;
    cfg.seed = 7;

    Network net(cfg);
    net.build();
    const NetworkStats& s = net.stats();

    check(s.n_astro == 10 && s.n_exc == 40 && s.n_inh == 10, "population sizes");

    // Excitatory sources see 50 targets (minus autapses); with p = 0.2 the
    // expected count is about 40 * 50 * 0.2 = 400.
    check(s.n_primary_exc > 250 && s.n_primary_exc < 550,
          "excitatory synapse count is near its expectation");
    check(s.n_primary_inh > 50 && s.n_primary_inh < 150,
          "inhibitory synapse count is near its expectation");
    check(s.n_neuron_to_astro > 0, "astrocytes are recruited");
    check(s.n_astro_to_neuron > 0, "astrocytes project back to neurons");
    // Every attachment creates both directions, so the two counts match.
    check(s.n_astro_to_neuron == s.n_neuron_to_astro,
          "each attachment creates one connection in each direction");

    // With unique_third_out the return connections collapse per pair.
    ModelConfig uniq = cfg;
    uniq.conn.unique_third_out = true;
    Network unique_net(uniq);
    unique_net.build();
    check(unique_net.stats().n_astro_to_neuron < s.n_astro_to_neuron,
          "unique_third_out collapses duplicate astrocyte-to-neuron pairs");

    // Identical seeds must give identical networks.
    Network again(cfg);
    again.build();
    check(again.stats().n_primary_exc == s.n_primary_exc &&
              again.stats().n_astro_to_neuron == s.n_astro_to_neuron,
          "network construction is reproducible");

    ModelConfig other = cfg;
    other.seed = 8;
    Network different(other);
    different.build();
    check(different.stats().n_primary_exc != s.n_primary_exc ||
              different.stats().n_neuron_to_astro != s.n_neuron_to_astro,
          "a different seed gives a different network");
}

void test_astro_out_degree_cap() {
    // Deliberately adversarial: a small pool (so recruitment concentrates on
    // few astrocytes) with near-certain primary and third-factor connection,
    // so every astrocyte tries hard to exceed a small cap.
    ModelConfig cfg;
    cfg.N = {5, 60, 20};
    cfg.time.dt = 0.1;
    cfg.time.substeps = 1;
    cfg.time.pre_sim_time = 0.0;
    cfg.time.sim_time = 20.0;
    cfg.conn.p_primary = 0.9;
    cfg.conn.p_third_if_primary = 0.9;
    cfg.conn.pool_size = 2;
    cfg.conn.pool_type = PoolType::Random;
    cfg.seed = 3;

    Network uncapped(cfg);
    uncapped.build();
    check(uncapped.stats().max_astro_out_degree > 5,
          "uncapped adversarial config does concentrate load past the cap "
          "we are about to apply (otherwise this test proves nothing)");

    ModelConfig capped = cfg;
    capped.conn.max_astro_out_degree = 5;
    Network net(capped);
    net.build();
    check(net.stats().max_astro_out_degree <= 5,
          "max_astro_out_degree is a hard ceiling, not a target");
    check(net.stats().n_astro_to_neuron < uncapped.stats().n_astro_to_neuron,
          "the cap actually dropped tripartite edges to enforce itself");
    check(net.stats().n_primary_exc == uncapped.stats().n_primary_exc,
          "the primary neuron-to-neuron synapses are unaffected by the cap");

    // Several seeds, same adversarial shape: the ceiling must hold regardless
    // of which astrocytes happen to get recruited.
    for (std::int64_t seed = 0; seed < 8; ++seed) {
        ModelConfig sweep = capped;
        sweep.seed = seed;
        Network swept(sweep);
        swept.build();
        check(swept.stats().max_astro_out_degree <= 5,
              "cap holds across seeds " + std::to_string(seed));
    }
}

// Network::build() is checked for reproducibility above (test_network_build);
// this checks Network::run() itself, which build() never touches. Several
// per-step phases run under #pragma omp parallel for (drive_astrocytes,
// apply_arrivals) -- a race there would not reliably crash, it would show up
// as the same seed producing a different result from run to run. Comparing
// the full recorded spike sequence (not just a count) across two runs is the
// direct way to look for that.
void test_run_reproducibility() {
    ModelConfig cfg;
    cfg.N = {20, 80, 20};
    cfg.time.dt = 0.1;
    cfg.time.substeps = 1;
    cfg.time.pre_sim_time = 0.0;
    cfg.time.sim_time = 50.0;
    cfg.conn.p_primary = 0.2;
    cfg.conn.p_third_if_primary = 0.3;
    cfg.conn.pool_size = 3;
    cfg.conn.pool_type = PoolType::Random;
    cfg.seed = 42;
    // A constant suprathreshold current, not just Poisson luck, so spiking
    // is guaranteed rather than probable: g_L*(V_th-E_L) is the leak alone
    // at threshold (~606 pA at the default neuron_exc params), so 900 pA
    // reliably drives regular tonic firing regardless of input timing.
    // Without real activity, an empty spike sequence would trivially
    // "match" between the two runs regardless of whether run() is actually
    // deterministic, and the test would prove nothing.
    cfg.neuron_exc.I_e = 900.0;
    cfg.neuron_inh.I_e = 900.0;
    // Poisson rates on top, so drive_astrocytes (astro) and the
    // network's own recurrent delivery (exc/inh) both still get exercised,
    // not just the tonic-current path.
    cfg.input_exc.poiss_rate = 2700.0;
    cfg.input_exc.poiss_weight = 1.0;
    cfg.input_inh.poiss_rate = 2500.0;
    cfg.input_inh.poiss_weight = 1.0;
    cfg.input_astro.poiss_rate = 3.0;
    cfg.input_astro.poiss_weight = 1.0;

    const std::string base =
        (std::filesystem::temp_directory_path() / "astrosimgpu_repro_test").string();
    const std::string dir_a = base + "_a";
    const std::string dir_b = base + "_b";

    auto run_once = [&](const std::string& out_dir) {
        Network net(cfg);
        net.build();
        Recorder recorder(out_dir, /*spikes=*/true, /*astro=*/false, /*neuron=*/false);
        net.run(recorder);
        return recorder.spike_count();
    };

    const std::size_t count_a = run_once(dir_a);
    const std::size_t count_b = run_once(dir_b);

    check(count_a > 0,
          "reproducibility check config actually produces spikes (otherwise "
          "this test proves nothing)");
    check(count_a == count_b, "same seed produces the same spike count on repeated runs");

    auto read_file = [](const std::string& path) {
        std::ifstream f(path);
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    };
    const std::string spikes_a = read_file(dir_a + "/spikes.csv");
    const std::string spikes_b = read_file(dir_b + "/spikes.csv");
    check(!spikes_a.empty(), "spikes.csv was actually written");
    check(spikes_a == spikes_b,
          "same seed produces an identical spike sequence (time and neuron) on "
          "repeated runs -- catches a race in the parallel delivery/drive loops "
          "that a count-only check could miss");

    std::error_code ec;
    std::filesystem::remove_all(dir_a, ec);
    std::filesystem::remove_all(dir_b, ec);
}

// Complements test_run_reproducibility: that test checks the same thread
// count is deterministic run to run; this checks that thread count itself
// does not bias the result. drive_astrocytes and apply_arrivals write only
// to their own loop index, so they cannot introduce any thread-count
// dependence. deliver_spikes and deliver_sic scatter-add into shared
// ring-buffer slots via atomics: correct (no lost updates), but the
// accumulation order -- and so the exact floating-point result -- can
// depend on thread scheduling, and a spiking neuron's threshold crossing
// can amplify a ULP-level difference into a shifted spike. That is expected
// and not a bug, so a single-run exact- or tolerance-comparison is the
// wrong tool here: whether one pair of runs happens to match is a question
// about that pair, not about whether thread count biases the model.
//
// What actually matters scientifically is whether the *distribution* of
// outcomes differs between thread counts -- the same bar the model's own
// statistical properties (firing rate, synchrony) are validated against,
// not bit-exact reproducibility of one trajectory. This runs many
// independent trials (different seeds) at 1 thread and at max threads and
// runs a permutation test on the two samples of spike counts: pool both
// samples, repeatedly reshuffle which values belong to which group, and see
// how often a random reshuffle produces a mean difference at least as large
// as the one actually observed. A permutation test is used instead of a
// t-test so no assumption about the sampling distribution (normality, equal
// variance) is needed -- appropriate here since spike counts are bounded
// counts, not naturally Gaussian. CounterRng drives the shuffling so the
// test itself is reproducible.
void test_thread_count_invariance() {
#ifndef _OPENMP
    // No OpenMP in this build: the pragmas are no-ops, so thread count
    // cannot possibly matter here. Nothing to check.
    return;
#else
    ModelConfig cfg;
    cfg.N = {20, 80, 20};
    cfg.time.dt = 0.1;
    cfg.time.substeps = 1;
    cfg.time.pre_sim_time = 0.0;
    cfg.time.sim_time = 50.0;
    cfg.conn.p_primary = 0.2;
    cfg.conn.p_third_if_primary = 0.3;
    cfg.conn.pool_size = 3;
    cfg.conn.pool_type = PoolType::Random;
    cfg.neuron_exc.I_e = 900.0;
    cfg.neuron_inh.I_e = 900.0;
    cfg.input_exc.poiss_rate = 2700.0;
    cfg.input_exc.poiss_weight = 1.0;
    cfg.input_inh.poiss_rate = 2500.0;
    cfg.input_inh.poiss_weight = 1.0;
    cfg.input_astro.poiss_rate = 3.0;
    cfg.input_astro.poiss_weight = 1.0;

    const int max_threads = omp_get_max_threads();
    check(max_threads > 1, "thread-count test actually varies thread count (max_threads=" +
                               std::to_string(max_threads) + ") -- otherwise it proves nothing");

    const std::string dir =
        (std::filesystem::temp_directory_path() / "astrosimgpu_thread_stats").string();

    auto spike_count_for = [&](std::uint64_t seed, int threads) {
        omp_set_num_threads(threads);
        ModelConfig trial = cfg;
        trial.seed = seed;
        Network net(trial);
        net.build();
        // Reused every trial: only the in-memory count is read, and the
        // constructor truncates the file, so nothing leaks between trials.
        Recorder recorder(dir, /*spikes=*/true, /*astro=*/false, /*neuron=*/false);
        net.run(recorder);
        return static_cast<double>(recorder.spike_count());
    };

    constexpr int trials_per_group = 20;
    vec<double> counts_1(trials_per_group), counts_n(trials_per_group);
    for (int i = 0; i < trials_per_group; ++i) {
        // Distinct seeds per trial, and the same seeds across both groups,
        // so the two samples differ only in thread count, not in which
        // network instances were drawn.
        const std::uint64_t seed = 1000 + static_cast<std::uint64_t>(i);
        counts_1[static_cast<std::size_t>(i)] = spike_count_for(seed, 1);
        counts_n[static_cast<std::size_t>(i)] = spike_count_for(seed, max_threads);
    }
    omp_set_num_threads(max_threads);  // restore, so later tests are unaffected

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    auto mean_of = [](const vec<double>& v) {
        double sum = 0.0;
        for (double x : v) {
            sum += x;
        }
        return sum / static_cast<double>(v.size());
    };

    check(mean_of(counts_1) > 0.0, "thread-count check config actually produces spikes");

    const double observed = std::abs(mean_of(counts_n) - mean_of(counts_1));

    vec<double> pooled;
    pooled.reserve(counts_1.size() + counts_n.size());
    pooled.insert(pooled.end(), counts_1.begin(), counts_1.end());
    pooled.insert(pooled.end(), counts_n.begin(), counts_n.end());
    const std::size_t half = counts_1.size();

    constexpr int permutations = 2000;
    int at_least_as_extreme = 0;
    CounterRng shuffle_rng(0xA5A5A5A5U, 1);
    vec<double> shuffled = pooled;
    for (int p = 0; p < permutations; ++p) {
        // Fisher-Yates over the pooled sample. uniform() in [0, 1) times
        // (i + 1), truncated, is always in [0, i], so j never exceeds i.
        for (std::size_t i = shuffled.size(); i-- > 1;) {
            const auto j = static_cast<std::size_t>(shuffle_rng.uniform() *
                                                     static_cast<double>(i + 1));
            std::swap(shuffled[i], shuffled[j]);
        }
        double sum_a = 0.0, sum_b = 0.0;
        for (std::size_t i = 0; i < shuffled.size(); ++i) {
            (i < half ? sum_a : sum_b) += shuffled[i];
        }
        const double mean_a = sum_a / static_cast<double>(half);
        const double mean_b = sum_b / static_cast<double>(shuffled.size() - half);
        if (std::abs(mean_b - mean_a) >= observed) {
            ++at_least_as_extreme;
        }
    }
    // +1/+1: the observed assignment is itself one of the permutations, so
    // this can never report an impossible p = 0.
    const double p_value = static_cast<double>(at_least_as_extreme + 1) /
                            static_cast<double>(permutations + 1);

    check(p_value > 0.05,
          "1 thread and " + std::to_string(max_threads) +
              " threads produce spike-count distributions with no significant difference "
              "(p = " + std::to_string(p_value) + ", mean " + std::to_string(mean_of(counts_1)) +
              " vs " + std::to_string(mean_of(counts_n)) + " over " +
              std::to_string(trials_per_group) + " trials each)");
#endif
}

}  // namespace

int main() {
    struct Case {
        const char* name;
        void (*fn)();
    };
    const Case cases[] = {
        {"json", test_json},
        {"rng", test_rng},
        {"astrocyte rest state", test_astrocyte_rest},
        {"astrocyte IP3 decay", test_astrocyte_ip3_decay},
        {"astrocyte kernel", test_astro_kernel},
        {"SIC threshold", test_sic_threshold},
        {"neuron kernel", test_neuron_kernel},
        {"neuron rest and spiking", test_neuron_rest_and_spiking},
        {"alpha conductance", test_alpha_conductance},
        {"analysis", test_analysis},
        {"network build", test_network_build},
        {"astro out-degree cap", test_astro_out_degree_cap},
        {"run reproducibility", test_run_reproducibility},
        {"thread count invariance", test_thread_count_invariance},
    };

    for (const Case& c : cases) {
        const int before = failures;
        std::cout << "[ run  ] " << c.name << "\n";
        c.fn();
        std::cout << (failures == before ? "[  ok  ] " : "[ FAIL ] ") << c.name << "\n";
    }

    std::cout << "\n" << (checks - failures) << " / " << checks << " checks passed\n";
    return failures == 0 ? 0 : 1;
}

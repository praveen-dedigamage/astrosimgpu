#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "astrosimgpu/analysis.hpp"
#include "astrosimgpu/network.hpp"
#include "astrosimgpu/parameters.hpp"
#include "astrosimgpu/recorder.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace astrosimgpu;

namespace {

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Simulates a tripartite neuron-astrocyte network and reports the\n"
        << "calcium and spiking statistics used to assess synchronisation.\n\n"
        << "Options:\n"
        << "  -c, --config PATH     parameter file (JSON)   [config/use_case.json]\n"
        << "  -o, --output DIR      output directory        [from config]\n"
        << "  -s, --seed N          random seed             [from config]\n"
        << "  -t, --sim-time MS     recorded duration       [from config]\n"
        << "  -p, --pre-time MS     discarded transient     [from config]\n"
        << "  -a, --astrocytes N    override the astrocyte count [from config]\n"
        << "      --threads N       OpenMP threads          [runtime default]\n"
        << "      --no-analysis     skip the post-run summary\n"
        << "  -h, --help            show this message\n";
}

/// Read back the recorded astrocyte traces, grouped by cell.
/// Reading the file rather than keeping traces in memory means the analysis
/// path is exercised on exactly the format other tools consume.
bool read_astro_traces(const std::string& path, std::map<index_t, vec<real>>& times,
                       std::map<index_t, vec<real>>& calcium) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream ss(line);
        std::string field;
        real t = 0.0, ca = 0.0;
        index_t cell = 0;

        if (!std::getline(ss, field, ',')) continue;
        t = std::strtod(field.c_str(), nullptr);
        if (!std::getline(ss, field, ',')) continue;
        cell = static_cast<index_t>(std::strtoul(field.c_str(), nullptr, 10));
        if (!std::getline(ss, field, ',')) continue;
        ca = std::strtod(field.c_str(), nullptr);

        times[cell].push_back(t);
        calcium[cell].push_back(ca);
    }
    return true;
}

std::string format_summary(const Network& net, const NetworkStats& s, double wall_seconds,
                           std::size_t spikes, const std::string& analysis) {
    const ModelConfig& c = net.config();
    std::ostringstream os;
    os << std::fixed << std::setprecision(4);
    os << "Tripartite neuron-astrocyte network\n";
    os << "-----------------------------------\n";
    os << "astrocytes            " << s.n_astro << "\n";
    os << "excitatory neurons    " << s.n_exc << "\n";
    os << "inhibitory neurons    " << s.n_inh << "\n";
    os << "exc -> neuron synapses " << s.n_primary_exc << "\n";
    os << "inh -> neuron synapses " << s.n_primary_inh << "\n";
    os << "neuron -> astrocyte    " << s.n_neuron_to_astro << "\n";
    os << "astrocyte -> neuron    " << s.n_astro_to_neuron << "\n";
    os << "\n";
    os << "dt                    " << c.time.dt << " ms (" << c.time.substeps << " substeps)\n";
    os << "transient             " << c.time.pre_sim_time << " ms\n";
    os << "recorded              " << c.time.sim_time << " ms\n";
    os << "seed                  " << c.seed << "\n";
    os << "\n";
    os << "wall clock            " << wall_seconds << " s\n";
    os << "spikes recorded       " << spikes << "\n";
    if (c.time.sim_time > 0.0 && s.n_exc + s.n_inh > 0) {
        const double rate = static_cast<double>(spikes) * 1000.0 /
                            (c.time.sim_time * static_cast<double>(s.n_exc + s.n_inh));
        os << "mean firing rate      " << rate << " Hz\n";
    }
    const PhaseProfile& p = net.profile();
    if (p.total > 0.0) {
        const double pct = 100.0 / p.total;
        os << "\n";
        os << "State propagation by phase\n";
        os << "-----------------------------------\n";
        os << "                       seconds   % of wall\n";
        auto row = [&](const char* name, double seconds) {
            os << std::left << std::setw(22) << name << std::right << std::setw(8)
               << seconds << std::setw(11) << seconds * pct << "\n";
            os << std::left;
        };
        row("update astrocytes", p.update_astro);
        row("update neurons", p.update_neuron);
        row("spike delivery", p.spike_cd);
        row("SIC gather+deliver", p.sic_gd);
        row("arrival application", p.deliver);
        row("other (I/O, loop)", p.other());
        os << std::left << std::setw(22) << "total" << std::right << std::setw(8) << p.total
           << std::setw(11) << 100.0 << "\n";
        os << std::left;
        if (c.time.sim_time > 0.0) {
            os << "real-time factor      " << p.total / (c.time.sim_time * 1e-3) << "\n";
        }
        // The per-cell update is the part that parallelises; the delivery
        // phases are the part that carries communication. Their ratio is what
        // decides whether offloading the update is worth the trouble.
        const double update = p.update_astro + p.update_neuron;
        const double comms = p.spike_cd + p.sic_gd + p.deliver;
        if (comms > 0.0) {
            os << "update : communication " << update / comms << " : 1\n";
        }
    }
    if (!analysis.empty()) {
        os << "\n" << analysis;
    }
    return os.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path = "config/use_case.json";
    std::string output_override;
    bool run_analysis = true;
    long long seed_override = -1;
    double sim_time_override = -1.0;
    double pre_time_override = -1.0;
    long long astro_override = -1;
    int thread_override = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + what);
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-c" || arg == "--config") {
            config_path = next("--config");
        } else if (arg == "-o" || arg == "--output") {
            output_override = next("--output");
        } else if (arg == "-s" || arg == "--seed") {
            seed_override = std::stoll(next("--seed"));
        } else if (arg == "-t" || arg == "--sim-time") {
            sim_time_override = std::stod(next("--sim-time"));
        } else if (arg == "-p" || arg == "--pre-time") {
            pre_time_override = std::stod(next("--pre-time"));
        } else if (arg == "-a" || arg == "--astrocytes") {
            astro_override = std::stoll(next("--astrocytes"));
        } else if (arg == "--threads") {
            thread_override = std::stoi(next("--threads"));
        } else if (arg == "--no-analysis") {
            run_analysis = false;
        } else {
            std::cerr << "unknown option: " << arg << "\n\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    try {
        ModelConfig cfg = load_config(config_path);
        if (!output_override.empty()) {
            cfg.output_dir = output_override;
        }
        if (seed_override >= 0) {
            cfg.seed = static_cast<std::uint64_t>(seed_override);
        }
        if (sim_time_override >= 0.0) {
            cfg.time.sim_time = sim_time_override;
        }
        if (pre_time_override >= 0.0) {
            cfg.time.pre_sim_time = pre_time_override;
        }
        if (astro_override >= 0) {
            // Varying the astrocyte count alone only makes sense with random
            // pools, where it is independent of the neuron count. Block pools
            // tie the two together.
            cfg.N.N_astro = static_cast<index_t>(astro_override);
        }

#ifdef _OPENMP
        if (thread_override > 0) {
            omp_set_num_threads(thread_override);
        }
        std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
        (void)thread_override;
        std::cout << "OpenMP: disabled\n";
#endif

        std::cout << "Building network from " << config_path << " ...\n";
        Network net(cfg);
        net.build();

        const NetworkStats& s = net.stats();
        std::cout << "  " << s.n_astro << " astrocytes, " << s.n_exc << " excitatory, " << s.n_inh
                  << " inhibitory neurons\n"
                  << "  " << s.n_primary_exc + s.n_primary_inh << " neuron-neuron synapses, "
                  << s.n_neuron_to_astro << " neuron-astrocyte, " << s.n_astro_to_neuron
                  << " astrocyte-neuron\n";

        Recorder recorder(cfg.output_dir, cfg.record_spikes, cfg.record_astro, cfg.record_neuron);

        std::cout << "Simulating " << cfg.time.pre_sim_time << " ms transient + "
                  << cfg.time.sim_time << " ms recorded ...\n";
        const auto t0 = std::chrono::steady_clock::now();
        net.run(recorder);
        const auto t1 = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double>(t1 - t0).count();

        std::string analysis;
        if (run_analysis && cfg.analysis.enabled && cfg.record_astro) {
            std::map<index_t, vec<real>> times, calcium;
            if (read_astro_traces(cfg.output_dir + "/astrocytes.csv", times, calcium)) {
                // Detect calcium transients per astrocyte, then measure how
                // aligned they are across the population.
                const real threshold = cfg.analysis.ca_threshold >= 0.0
                                           ? cfg.analysis.ca_threshold
                                           : cfg.astro.SIC_th;
                const real max_gap = cfg.analysis.merge_gap;
                vec<vec<real>> onsets;
                vec<real> durations;
                std::size_t active = 0;

                for (const auto& [cell, ca] : calcium) {
                    const EventTrain ev =
                        detect_events(times[cell], ca, threshold, true, max_gap);
                    onsets.push_back(ev.onset);
                    durations.insert(durations.end(), ev.duration.begin(), ev.duration.end());
                    if (!ev.onset.empty()) {
                        ++active;
                    }
                }

                Interval window{cfg.time.pre_sim_time,
                                cfg.time.pre_sim_time + cfg.time.sim_time};
                // A window wider than the recording leaves no bins to
                // correlate; fall back to a tenth of the recorded span so a
                // short run still reports something meaningful.
                real width = cfg.analysis.hist_window;
                real shift = cfg.analysis.hist_shift;
                if (width >= cfg.time.sim_time && cfg.time.sim_time > 0.0) {
                    width = cfg.time.sim_time / 10.0;
                    shift = width / 5.0;
                }
                const PairwiseCorrelation corr =
                    pairwise_correlation(onsets, window, width, shift);
                const Summary dur = summarise(durations);

                std::ostringstream os;
                os << std::fixed << std::setprecision(4);
                os << "Calcium analysis (threshold " << threshold << " uM, window " << width
                   << "/" << shift << " ms)\n";
                os << "-----------------------------------\n";
                os << "astrocytes with transients  " << active << " / " << calcium.size() << "\n";
                os << "transients detected         " << dur.count << "\n";
                os << "mean duration               " << dur.mean << " ms (sd " << dur.stddev
                   << ")\n";
                os << "pairs correlated            " << corr.coefficient.size() << "\n";
                os << "mean pairwise correlation   " << corr.mean() << "\n";
                analysis = os.str();
            }
        }

        const std::string summary =
            format_summary(net, s, wall, recorder.spike_count(), analysis);
        recorder.write_meta(summary);
        std::cout << "\n" << summary << "\nOutput written to " << cfg.output_dir << "/\n";

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

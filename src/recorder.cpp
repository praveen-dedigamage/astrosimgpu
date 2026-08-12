#include "astrosimgpu/recorder.hpp"

#include <filesystem>
#include <stdexcept>

namespace astrosimgpu {

Recorder::Recorder(const std::string& output_dir, bool spikes, bool astro, bool neuron)
    : dir_(output_dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    if (ec) {
        throw std::runtime_error("cannot create output directory " + dir_ + ": " + ec.message());
    }

    if (spikes) {
        spikes_.open(dir_ + "/spikes.csv");
        if (!spikes_) {
            throw std::runtime_error("cannot open " + dir_ + "/spikes.csv");
        }
        spikes_ << "time_ms,neuron\n";
    }
    if (astro) {
        astro_.open(dir_ + "/astrocytes.csv");
        if (!astro_) {
            throw std::runtime_error("cannot open " + dir_ + "/astrocytes.csv");
        }
        astro_ << "time_ms,astrocyte,Ca,IP3\n";
    }
    if (neuron) {
        neuron_.open(dir_ + "/neurons.csv");
        if (!neuron_) {
            throw std::runtime_error("cannot open " + dir_ + "/neurons.csv");
        }
        neuron_ << "time_ms,neuron,V_m,I_SIC\n";
    }
}

Recorder::~Recorder() = default;

void Recorder::write_spike(real time_ms, index_t neuron) {
    if (spikes_.is_open()) {
        spikes_ << time_ms << ',' << neuron << '\n';
        ++spike_count_;
    }
}

void Recorder::write_astro(real time_ms, index_t astrocyte, real Ca, real IP3) {
    if (astro_.is_open()) {
        astro_ << time_ms << ',' << astrocyte << ',' << Ca << ',' << IP3 << '\n';
    }
}

void Recorder::write_neuron(real time_ms, index_t neuron, real V_m, real I_sic) {
    if (neuron_.is_open()) {
        neuron_ << time_ms << ',' << neuron << ',' << V_m << ',' << I_sic << '\n';
    }
}

void Recorder::write_meta(const std::string& text) {
    std::ofstream meta(dir_ + "/run.txt");
    if (meta) {
        meta << text;
    }
}

}  // namespace astrosimgpu

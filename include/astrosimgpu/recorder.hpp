#pragma once

#include <fstream>
#include <string>

#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

/// Writes simulation output as CSV, one file per signal.
///
/// The column layout matches what the reference analysis scripts expect, so
/// the same plotting and synchrony code can read either simulator's output:
///   spikes.csv        time_ms,neuron
///   astrocytes.csv    time_ms,astrocyte,Ca,IP3
///   neurons.csv       time_ms,neuron,V_m,I_SIC
class Recorder {
public:
    Recorder(const std::string& output_dir, bool spikes, bool astro, bool neuron);
    ~Recorder();

    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;

    void write_spike(real time_ms, index_t neuron);
    void write_astro(real time_ms, index_t astrocyte, real Ca, real IP3);
    void write_neuron(real time_ms, index_t neuron, real V_m, real I_sic);

    /// Free-form run metadata, written alongside the data.
    void write_meta(const std::string& text);

    [[nodiscard]] const std::string& directory() const { return dir_; }
    [[nodiscard]] std::size_t spike_count() const { return spike_count_; }

private:
    std::string dir_;
    std::ofstream spikes_, astro_, neuron_;
    std::size_t spike_count_ = 0;
};

}  // namespace astrosimgpu

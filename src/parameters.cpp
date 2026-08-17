#include "astrosimgpu/parameters.hpp"

#include <stdexcept>

#include "astrosimgpu/json.hpp"

namespace astrosimgpu {

namespace {

void load_astrocyte(const Json& j, AstrocyteParams& p) {
    if (!j.is_object()) {
        return;
    }
    j.get_to("Ca_tot", p.Ca_tot);
    j.get_to("IP3_0", p.IP3_0);
    j.get_to("Kd_IP3_1", p.Kd_IP3_1);
    j.get_to("Kd_IP3_2", p.Kd_IP3_2);
    j.get_to("Kd_act", p.Kd_act);
    j.get_to("Kd_inh", p.Kd_inh);
    j.get_to("Km_SERCA", p.Km_SERCA);
    j.get_to("SIC_scale", p.SIC_scale);
    j.get_to("SIC_th", p.SIC_th);
    j.get_to("delta_IP3", p.delta_IP3);
    j.get_to("k_IP3R", p.k_IP3R);
    j.get_to("rate_IP3R", p.rate_IP3R);
    j.get_to("rate_L", p.rate_L);
    j.get_to("rate_SERCA", p.rate_SERCA);
    j.get_to("ratio_ER_cyt", p.ratio_ER_cyt);
    j.get_to("tau_IP3", p.tau_IP3);
    j.get_to("Ca_init", p.Ca_init);
    j.get_to("h_init", p.h_init);
    j.get_to("IP3", p.IP3_init);
    j.get_to("IP3_init", p.IP3_init);
}

void load_neuron(const Json& j, NeuronParams& p) {
    if (!j.is_object()) {
        return;
    }
    j.get_to("C_m", p.C_m);
    j.get_to("g_L", p.g_L);
    j.get_to("E_L", p.E_L);
    j.get_to("V_th", p.V_th);
    j.get_to("Delta_T", p.Delta_T);
    j.get_to("a", p.a);
    j.get_to("b", p.b);
    j.get_to("tau_w", p.tau_w);
    j.get_to("V_reset", p.V_reset);
    j.get_to("V_peak", p.V_peak);
    j.get_to("t_ref", p.t_ref);
    j.get_to("E_ex", p.E_ex);
    j.get_to("E_in", p.E_in);
    j.get_to("tau_syn_ex", p.tau_syn_ex);
    j.get_to("tau_syn_in", p.tau_syn_in);
    j.get_to("I_e", p.I_e);
    j.get_to("V_m", p.V_m_init);
}

void load_input(const Json& j, InputParams& p) {
    if (!j.is_object()) {
        return;
    }
    j.get_to("poiss_rate", p.poiss_rate);
    j.get_to("poiss_weight", p.poiss_weight);
    j.get_to("gauss_noise_var", p.gauss_noise_std);
    j.get_to("gauss_noise_std", p.gauss_noise_std);
    j.get_to("noise_dt", p.noise_dt);
    j.get_to("independent_noise", p.independent_noise);
}

void load_randomize(const Json& j, RandomizeSpec& p) {
    if (!j.is_object()) {
        return;
    }
    p.enabled = true;
    j.get_to("enabled", p.enabled);
    j.get_to("lower", p.lower);
    j.get_to("upper", p.upper);
    j.get_to("var", p.var);
}

}  // namespace

ModelConfig load_config(const std::string& path) {
    const Json root = Json::parse_file(path);
    if (!root.is_object()) {
        throw std::runtime_error("configuration root must be a JSON object: " + path);
    }
    ModelConfig cfg;

    if (const Json& n = root["N"]; n.is_object()) {
        n.get_to("N_A", cfg.N.N_astro);
        n.get_to("N_E", cfg.N.N_exc);
        n.get_to("N_I", cfg.N.N_inh);
    }

    if (const Json& s = root["simulation"]; s.is_object()) {
        s.get_to("dt", cfg.time.dt);
        s.get_to("substeps", cfg.time.substeps);
        s.get_to("pre_sim_time", cfg.time.pre_sim_time);
        s.get_to("sim_time", cfg.time.sim_time);
        s.get_to("seed", cfg.seed);
        s.get_to("output_dir", cfg.output_dir);
        s.get_to("record_spikes", cfg.record_spikes);
        s.get_to("record_astro", cfg.record_astro);
        s.get_to("record_neuron", cfg.record_neuron);
        s.get_to("record_every", cfg.record_every);
        s.get_to("record_astro_max", cfg.record_astro_max);
        s.get_to("record_neuron_max", cfg.record_neuron_max);
    }

    load_astrocyte(root["astrocyte"], cfg.astro);
    load_neuron(root["neuron_exc"], cfg.neuron_exc);
    load_neuron(root["neuron_inh"], cfg.neuron_inh);

    load_input(root["input_astro"], cfg.input_astro);
    load_input(root["input_exc"], cfg.input_exc);
    load_input(root["input_inh"], cfg.input_inh);

    load_randomize(root["randomize_astro"], cfg.randomize_astro);
    load_randomize(root["randomize_neuron"], cfg.randomize_neuron);

    if (const Json& s = root["synapse"]; s.is_object()) {
        s.get_to("w_e", cfg.syn.w_e);
        s.get_to("w_i", cfg.syn.w_i);
        s.get_to("w_n2a", cfg.syn.w_n2a);
        s.get_to("w_a2n", cfg.syn.w_a2n);
        s.get_to("d_e", cfg.syn.d_e);
        s.get_to("d_i", cfg.syn.d_i);
        s.get_to("d_a2n", cfg.syn.d_a2n);
        s.get_to("sic_interval", cfg.syn.sic_interval);
        if (const Json& stp = s["stp"]; stp.is_object()) {
            stp.get_to("enabled", cfg.syn.stp.enabled);
            stp.get_to("U", cfg.syn.stp.U);
            stp.get_to("tau_rec", cfg.syn.stp.tau_rec);
            stp.get_to("tau_fac", cfg.syn.stp.tau_fac);
        }
    }

    if (const Json& a = root["analysis"]; a.is_object()) {
        a.get_to("enabled", cfg.analysis.enabled);
        a.get_to("ca_threshold", cfg.analysis.ca_threshold);
        a.get_to("merge_gap", cfg.analysis.merge_gap);
        a.get_to("hist_window", cfg.analysis.hist_window);
        a.get_to("hist_shift", cfg.analysis.hist_shift);
    }

    if (const Json& c = root["connectivity"]; c.is_object()) {
        c.get_to("p_primary", cfg.conn.p_primary);
        c.get_to("p_third_if_primary", cfg.conn.p_third_if_primary);
        c.get_to("pool_size", cfg.conn.pool_size);
        c.get_to("allow_autapses", cfg.conn.allow_autapses);
        c.get_to("unique_third_out", cfg.conn.unique_third_out);
        std::string pool_type = "block";
        c.get_to("pool_type", pool_type);
        if (pool_type == "block") {
            cfg.conn.pool_type = PoolType::Block;
        } else if (pool_type == "random") {
            cfg.conn.pool_type = PoolType::Random;
        } else {
            throw std::runtime_error("unknown pool_type '" + pool_type +
                                     "' (expected \"block\" or \"random\")");
        }
    }

    // An unset noise interval takes the noise_generator default of ten times
    // the resolution, resolved here so the rest of the code sees a real value.
    for (InputParams* in : {&cfg.input_astro, &cfg.input_exc, &cfg.input_inh}) {
        if (in->noise_dt <= 0.0) {
            in->noise_dt = 10.0 * cfg.time.dt;
        }
    }

    if (cfg.time.substeps < 1) {
        throw std::runtime_error("simulation.substeps must be at least 1");
    }
    if (cfg.time.dt <= 0.0) {
        throw std::runtime_error("simulation.dt must be positive");
    }
    if (cfg.record_every < 1) {
        throw std::runtime_error("simulation.record_every must be at least 1");
    }
    if (cfg.syn.sic_interval < 1) {
        throw std::runtime_error("synapse.sic_interval must be at least 1");
    }
    if (cfg.conn.pool_size < 1) {
        throw std::runtime_error("connectivity.pool_size must be at least 1");
    }

    return cfg;
}

}  // namespace astrosimgpu

#!/usr/bin/env python3
"""Generate the neuron-scaling sweep configs.

Fixed in-degree law (see docs/experiments.md and README.md "fixed
connection probability vs fixed in-degree"): as the network grows, hold
K_syn (synapses/neuron) and p_third_if_primary (fraction of synapses that
recruit an astrocyte) fixed, and grow N_A proportionally to the neuron
count so the average astrocyte out-degree also stays fixed. This is the
same law config/scale_1000.json already uses -- this script generalises
it to a full sweep instead of one hand-written point.

    p_primary   = K_SYN / N_E
    N_A         = N_total / ASTRO_RATIO      (ASTRO_RATIO=5 -> avg in-degree ~80)
    synapses   ~= K_SYN * N_total             (linear, not quadratic)

max_astro_out_degree is set as a safety cap (see include/astrosimgpu/
parameters.hpp) -- with uniform random pools at these sizes it should
never actually bind, but it is there so a lucky/adversarial draw cannot
produce an unbounded astrocyte.

Biology parameters (astrocyte/neuron/synapse/input blocks) are copied
unchanged from config/use_case.json -- only N, connectivity and the
measurement window change per size. The window is short (a few hundred
recorded steps) because this sweep is about per-step phase timing, not a
long correctness run; correctness itself is covered separately at N=500
by the existing regime-transition check against use_case.json/
bursting.json.

build_primary_connections generates connectivity with geometric-skip
sampling (O(expected connections) rather than a dense O(N_neurons x N_astro)
sweep), so generation time is no longer the limit at 1M-10M neurons -- it
takes seconds. Memory is: total connections scale as K_SYN * N (linear),
and each stored connection costs about 40 bytes with STP state (measured:
100,000 neurons ~= 575 MB peak). At 1,000,000 that is ~5.75 GB; at
10,000,000 it is ~57.5 GB. Both fit a GH200 node's host RAM, but 10M is a
large enough share of it to be worth watching in the job's memory request.
Neither fits comfortably on a laptop-class sandbox -- run those two sizes
on Roihu, not locally.
"""

import copy
import json
import os

K_SYN = 80.0              # target synapses per neuron, held fixed
P_THIRD = 0.2             # fraction of synapses that recruit an astrocyte
E_FRACTION = 0.8          # matches use_case.json's 400:100 split
ASTRO_RATIO = 5           # N_A = N_total / ASTRO_RATIO -> avg astro in-degree ~80
POOL_SIZE = 5             # random pool draw per neuron (kernel_scaling.json convention)
MAX_ASTRO_OUT_DEGREE = 10000

SIZES = [500, 1000, 2000, 5000, 10000, 100000, 1000000, 10000000]

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")

# Biology blocks, copied from config/use_case.json.
ASTROCYTE = {
    "Ca_tot": 1.8958264782990153,
    "IP3_0": 0.010925583871830699,
    "IP3": 0.010925583871830699,
    "tau_IP3": 1058.7589568400845,
    "delta_IP3": 0.04911592657233997,
    "SIC_scale": 1.0,
    "Kd_IP3_1": 0.13,
    "Kd_IP3_2": 0.9434,
    "Kd_act": 0.08234,
    "Kd_inh": 1.049,
    "Km_SERCA": 0.1,
    "SIC_th": 0.19669,
    "k_IP3R": 0.0002,
    "rate_IP3R": 0.006,
    "rate_L": 0.00011,
    "rate_SERCA": 0.0009,
    "ratio_ER_cyt": 0.185,
    "Ca_init": 0.073,
    "h_init": 0.793,
}
RANDOMIZE_ASTRO = {"enabled": True, "lower": 0.9, "upper": 1.05, "var": 0.1}
NEURON_EXC = {
    "Delta_T": 2.0, "a": 4.0, "tau_w": 450.0, "V_reset": -50.0, "b": 300.0,
    "V_m": -50.0, "C_m": 130.0, "g_L": 18.0, "E_L": -58.0, "V_th": -50.0,
    "V_peak": 0.0, "t_ref": 0.0, "E_ex": 0.0, "E_in": -85.0,
    "tau_syn_ex": 0.2, "tau_syn_in": 2.0,
}
NEURON_INH = {
    "Delta_T": 2.0, "a": -0.8, "tau_w": 264.0, "V_reset": -60.0, "b": 130.0,
    "V_m": -60.0, "C_m": 104.0, "g_L": 4.3, "E_L": -65.0, "V_th": -52.0,
    "V_peak": 0.0, "t_ref": 0.0, "E_ex": 0.0, "E_in": -85.0,
    "tau_syn_ex": 0.2, "tau_syn_in": 2.0,
}
RANDOMIZE_NEURON = {"enabled": True, "lower": 0.9, "upper": 1.1, "var": 0.05}
SYNAPSE = {
    "w_e": 5.0, "w_i": -5.0, "w_n2a": 0.2, "w_a2n": 1.0,
    "d_e": 1.0, "d_i": 1.0, "d_a2n": 1.0,
    "stp": {"enabled": True, "U": 0.5, "tau_rec": 800.0, "tau_fac": 0.0},
}
INPUT_ASTRO = {"poiss_rate": 3.029715502945856, "poiss_weight": 1.0, "gauss_noise_var": 0.0003}
INPUT_EXC = {"poiss_rate": 2700.0, "poiss_weight": 1.0, "gauss_noise_var": 100.0}
INPUT_INH = {"poiss_rate": 2500.0, "poiss_weight": 1.0, "gauss_noise_var": 100.0}


def build_config(n_total: int) -> dict:
    n_e = max(1, round(n_total * E_FRACTION))
    n_i = max(1, n_total - n_e)
    n_a = max(1, round(n_total / ASTRO_RATIO))
    p_primary = min(1.0, K_SYN / n_e)

    return {
        "N": {"N_A": n_a, "N_E": n_e, "N_I": n_i},
        "simulation": {
            "dt": 0.1,
            "substeps": 1,
            # Short window (matches nsys_profile.sbatch's convention): 50
            # transient + 200 recorded steps is enough for a stable
            # PhaseProfile percentage split, and keeps update_neuron's cost
            # -- linear in N, ~805 ms/step by 10,000,000 neurons -- from
            # blowing the 15-minute gputest budget on the largest sweep
            # points.
            "pre_sim_time": 5,
            "sim_time": 20,
            "seed": 1,
            "output_dir": f"results/neuron-scale-{n_total}",
            "record_spikes": False,
            "record_astro": False,
            "record_neuron": False,
            "record_every": 100,
            "record_astro_max": 100,
            "record_neuron_max": 50,
        },
        "astrocyte": copy.deepcopy(ASTROCYTE),
        "randomize_astro": copy.deepcopy(RANDOMIZE_ASTRO),
        "neuron_exc": copy.deepcopy(NEURON_EXC),
        "neuron_inh": copy.deepcopy(NEURON_INH),
        "randomize_neuron": copy.deepcopy(RANDOMIZE_NEURON),
        "synapse": copy.deepcopy(SYNAPSE),
        "connectivity": {
            "p_primary": p_primary,
            "p_third_if_primary": P_THIRD,
            "pool_size": POOL_SIZE,
            "pool_type": "random",
            "allow_autapses": False,
            "max_astro_out_degree": MAX_ASTRO_OUT_DEGREE,
        },
        "input_astro": copy.deepcopy(INPUT_ASTRO),
        "input_exc": copy.deepcopy(INPUT_EXC),
        "input_inh": copy.deepcopy(INPUT_INH),
    }


def main() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    written = []
    for n_total in SIZES:
        cfg = build_config(n_total)
        path = os.path.join(CONFIG_DIR, f"neuron_scale_{n_total}.json")
        header = (
            f"// Fixed-in-degree neuron-scaling sweep point: {n_total} total neurons.\n"
            f"// Generated by scripts/roihu/gen_neuron_scaling_configs.py -- do not\n"
            f"// hand-edit N or connectivity here, regenerate instead.\n"
            f"// K_syn={K_SYN}, p_third_if_primary={P_THIRD}, N_A=N/{ASTRO_RATIO},\n"
            f"// pool_type=random, pool_size={POOL_SIZE}, "
            f"max_astro_out_degree={MAX_ASTRO_OUT_DEGREE}.\n"
        )
        with open(path, "w") as f:
            f.write(header)
            json.dump(cfg, f, indent=2)
            f.write("\n")
        written.append(path)
        print(f"[+] {path}  N_E={cfg['N']['N_E']} N_I={cfg['N']['N_I']} "
              f"N_A={cfg['N']['N_A']} p_primary={cfg['connectivity']['p_primary']:.6f}")

    print(f"\nWrote {len(written)} configs.")


if __name__ == "__main__":
    main()

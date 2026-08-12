#!/usr/bin/env python3
"""Plot a run: spike raster, population rate, and astrocyte calcium traces.

    python scripts/plot_results.py results/use_case
    python scripts/plot_results.py results/use_case --out figures/use_case.png

Reads the CSV files the simulator writes. Only numpy and matplotlib are
required; the simulator itself needs neither.
"""

import argparse
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required for plotting: pip install matplotlib")


def load_spikes(path):
    if not os.path.exists(path):
        return np.empty(0), np.empty(0, dtype=int)
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        return np.empty(0), np.empty(0, dtype=int)
    return np.atleast_1d(data["time_ms"]), np.atleast_1d(data["neuron"]).astype(int)


def load_astro(path):
    if not os.path.exists(path):
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        return None
    return {
        "t": np.atleast_1d(data["time_ms"]),
        "cell": np.atleast_1d(data["astrocyte"]).astype(int),
        "Ca": np.atleast_1d(data["Ca"]),
        "IP3": np.atleast_1d(data["IP3"]),
    }


def population_rate(times, n_neurons, bin_ms=50.0):
    if times.size == 0:
        return np.empty(0), np.empty(0)
    lo, hi = times.min(), times.max()
    edges = np.arange(lo, hi + bin_ms, bin_ms)
    counts, _ = np.histogram(times, bins=edges)
    # spikes per bin -> Hz per neuron
    rate = counts / (bin_ms * 1e-3) / max(n_neurons, 1)
    return edges[:-1], rate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="directory written by the simulator")
    ap.add_argument("--out", default=None, help="output image (default: <results>/overview.png)")
    ap.add_argument("--max-traces", type=int, default=8,
                    help="astrocyte calcium traces to draw [8]")
    ap.add_argument("--max-raster", type=int, default=40000,
                    help="spikes to draw in the raster [40000]")
    args = ap.parse_args()

    spike_t, spike_id = load_spikes(os.path.join(args.results, "spikes.csv"))
    astro = load_astro(os.path.join(args.results, "astrocytes.csv"))

    if spike_t.size == 0 and astro is None:
        sys.exit(f"no data found in {args.results}")

    n_panels = 2 + (1 if astro is not None else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.0 * n_panels), sharex=True)
    axes = np.atleast_1d(axes)

    # Raster. Subsample rather than drawing every point, so a long run stays
    # legible and the file stays small.
    ax = axes[0]
    if spike_t.size:
        if spike_t.size > args.max_raster:
            keep = np.random.default_rng(0).choice(spike_t.size, args.max_raster, replace=False)
            keep.sort()
        else:
            keep = np.arange(spike_t.size)
        ax.plot(spike_t[keep] / 1000.0, spike_id[keep], ".", ms=0.7, color="#1f4e79",
                rasterized=True)
    ax.set_ylabel("neuron")
    ax.set_title(f"{os.path.basename(os.path.abspath(args.results))}: spiking and astrocyte calcium")

    # Population rate.
    ax = axes[1]
    n_neurons = int(spike_id.max()) + 1 if spike_id.size else 1
    t_rate, rate = population_rate(spike_t, n_neurons)
    if t_rate.size:
        ax.plot(t_rate / 1000.0, rate, lw=0.8, color="#a03020")
    ax.set_ylabel("rate [Hz/neuron]")

    # Calcium traces.
    if astro is not None:
        ax = axes[2]
        cells = np.unique(astro["cell"])[: args.max_traces]
        for c in cells:
            m = astro["cell"] == c
            ax.plot(astro["t"][m] / 1000.0, astro["Ca"][m], lw=0.7, alpha=0.85)
        ax.set_ylabel(r"[Ca$^{2+}$] [$\mu$M]")
        ax.set_xlabel("time [s]")
    else:
        axes[-1].set_xlabel("time [s]")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = args.out or os.path.join(args.results, "overview.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

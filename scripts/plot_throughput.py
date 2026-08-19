#!/usr/bin/env python3
"""Draw the throughput curve from docs/data/throughput.csv.

Writes SVG directly rather than using a plotting library, so the figure can be
regenerated with nothing but a Python interpreter -- the same constraint the
simulator itself is built under.

    python3 scripts/plot_throughput.py

Produces docs/img/throughput-light.svg and docs/img/throughput-dark.svg. The
README serves whichever matches the reader's theme.
"""

import csv
import math
import os

DATA = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "throughput.csv")
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "docs", "img")

# Categorical slots 1-3 of the reference palette, which validate across all
# pairs in both modes. Assigned in fixed order and never cycled: each series
# keeps its hue regardless of how many are present.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "axis": "#b4b2a9",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#2f2f2d",
        "axis": "#5f5e5a",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}

# Palette slots 1-4 in fixed order; a line chart uses the adjacent pairlist,
# which this order clears. The legend is set apart from the lines: the three
# device backends finish within three pixels of each other, so a label at the
# end of each line cannot be read.
SERIES = [
    ("host_us", "host, 72 Grace cores"),
    ("openmp_target_us", "OpenMP target"),
    ("kokkos_us", "Kokkos"),
    ("cuda_us", "native CUDA"),
]

W, H = 760, 430
L, R, T, B = 78, 190, 44, 58


def load():
    rows = []
    with open(DATA) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line)
    reader = csv.DictReader(rows)
    out = []
    for row in reader:
        rec = {"n": float(row["astrocytes"])}
        for key, _ in SERIES:
            v = (row.get(key) or "").strip()
            rec[key] = float(v) if v else None
        out.append(rec)
    return out


def make_svg(data, mode):
    t = THEMES[mode]
    xs = [d["n"] for d in data]
    vals = [d[k] for k, _ in SERIES for d in data if d[k] is not None]
    x0, x1 = math.log10(min(xs)), math.log10(max(xs))
    y0, y1 = math.floor(math.log10(min(vals))), math.ceil(math.log10(max(vals)))

    def px(n):
        return L + (math.log10(n) - x0) / (x1 - x0) * (W - L - R)

    def py(v):
        return T + (1 - (math.log10(v) - y0) / (y1 - y0)) * (H - T - B)

    s = []
    add = s.append
    add(f'<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="system-ui, -apple-system, Segoe UI, sans-serif" role="img" '
        f'aria-labelledby="title desc">')
    add('<title id="title">Astrocyte update cost per timestep against population size</title>')
    add('<desc id="desc">Log-log plot. The host cost rises steeply with population while '
        'the device cost starts higher but rises more slowly, so the two cross near forty '
        'thousand astrocytes. The three device backends lie almost on top of one '
        'another, within about ten per cent.</desc>')
    add(f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>')

    # Grid, recessive: decade lines only.
    for e in range(int(x0), int(x1) + 1):
        x = px(10 ** e)
        add(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H-B}" stroke="{t["grid"]}" stroke-width="1"/>')
        label = {2: "100", 3: "1k", 4: "10k", 5: "100k", 6: "1M", 7: "10M"}.get(e, str(10 ** e))
        add(f'<text x="{x:.1f}" y="{H-B+20}" fill="{t["muted"]}" font-size="12" '
            f'text-anchor="middle">{label}</text>')
    for e in range(int(y0), int(y1) + 1):
        y = py(10 ** e)
        add(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        lab = {0: "1", 1: "10", 2: "100", 3: "1000", 4: "10000"}.get(e, str(10 ** e))
        add(f'<text x="{L-10}" y="{y+4:.1f}" fill="{t["muted"]}" font-size="12" '
            f'text-anchor="end">{lab}</text>')

    add(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="{t["axis"]}" stroke-width="1"/>')
    add(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="{t["axis"]}" stroke-width="1"/>')

    add(f'<text x="{L}" y="{T-18}" fill="{t["text"]}" font-size="15" font-weight="500">'
        f'Astrocyte update, microseconds per timestep</text>')
    add(f'<text x="{(L+W-R)/2:.0f}" y="{H-14}" fill="{t["muted"]}" font-size="12" '
        f'text-anchor="middle">astrocytes</text>')

    # Crossover marker, drawn under the data so it never obscures a mark.
    cross = 27000
    add(f'<line x1="{px(cross):.1f}" y1="{T}" x2="{px(cross):.1f}" y2="{H-B}" '
        f'stroke="{t["axis"]}" stroke-width="1" stroke-dasharray="3 3"/>')
    add(f'<text x="{px(cross)+7:.1f}" y="{T+14}" fill="{t["muted"]}" font-size="12">'
        f'crossover ~27k</text>')

    entries = []
    for idx, (key, label) in enumerate(SERIES):
        pts = [(d["n"], d[key]) for d in data if d[key] is not None]
        if not pts:
            continue
        colour = t["series"][idx]
        path = " ".join(
            ("M" if i == 0 else "L") + f"{px(n):.1f},{py(v):.1f}" for i, (n, v) in enumerate(pts)
        )
        add(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>')
        for n, v in pts:
            # A surface ring keeps overlapping markers legible where the
            # series cross.
            add(f'<circle cx="{px(n):.1f}" cy="{py(v):.1f}" r="4.5" fill="{colour}" '
                f'stroke="{t["surface"]}" stroke-width="2"/>')
        entries.append((colour, label))

    # Legend in the right margin, in the order the series are declared, which is
    # also the order they appear at the right-hand end of the plot.
    for i, (colour, label) in enumerate(entries):
        y = T + 16 + i * 22
        add(f'<circle cx="{W-R+21}" cy="{y}" r="4.5" fill="{colour}"/>')
        add(f'<text x="{W-R+31}" y="{y+4}" fill="{t["text"]}" font-size="12">'
            f'{label}</text>')

    add('</svg>')
    return "\n".join(s)


def main():
    data = load()
    os.makedirs(OUTDIR, exist_ok=True)
    for mode in ("light", "dark"):
        path = os.path.join(OUTDIR, f"throughput-{mode}.svg")
        with open(path, "w") as fh:
            fh.write(make_svg(data, mode))
        print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()

#!/bin/bash
# Is the astrocytic current exchanged more often than the dynamics need?
#
#   bash scripts/sic_interval.sh
#   SEEDS="1 2 3 4 5" bash scripts/sic_interval.sh
#
# Calcium moves over seconds while the step is 0.1 ms, so the current may be
# sent far more often than it changes. Larger intervals hold the last value in
# between rather than sending zero.
#
# CPU only. Both regimes are run, because a coarser exchange might survive in
# the asynchronous one and break the synchronised one, where the astrocytic
# pathway is what drives the network.

set -uo pipefail

BIN=${BIN:-build/astrosimgpu}
SIMTIME=${SIMTIME:-60000}
INTERVALS=${INTERVALS:-"1 10 100 1000"}
SEEDS=${SEEDS:-"1 2 3"}
CONFIGS=${CONFIGS:-"use_case bursting"}
OUT=${OUT:-results/sic-interval}

[[ -x "$BIN" ]] || { echo "no binary at $BIN; run make first" >&2; exit 1; }
mkdir -p "$OUT"

raw="$OUT/raw.csv"
echo "config,interval,seed,correlation,rate,transients" > "$raw"

for cfg_name in $CONFIGS; do
    for k in $INTERVALS; do
        # sic_interval lives in the synapse block, so each value needs a file.
        cfg="$OUT/${cfg_name}_k$k.json"
        python3 - "$cfg_name" "$k" "$cfg" <<'PY'
import json, re, sys, pathlib
name, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
src = pathlib.Path(f"config/{name}.json").read_text()
cfg = json.loads(re.sub(r'^\s*//.*$', '', src, flags=re.M))
cfg.setdefault("synapse", {})["sic_interval"] = k
pathlib.Path(out).write_text(json.dumps(cfg, indent=2))
PY
        for seed in $SEEDS; do
            printf "%-9s k=%-5s seed=%-3s " "$cfg_name" "$k" "$seed"
            o=$("$BIN" --config "$cfg" -t "$SIMTIME" -s "$seed" \
                -o "$OUT/${cfg_name}_k${k}_s$seed" 2>/dev/null)
            co=$(awk '/mean pairwise correlation/ {print $NF}' <<< "$o")
            ra=$(awk '/mean firing rate/ {print $4}' <<< "$o")
            tr_=$(awk '/transients detected/ {print $3}' <<< "$o")
            echo "correlation=$co rate=$ra transients=$tr_"
            echo "$cfg_name,$k,$seed,${co:-},${ra:-},${tr_:-}" >> "$raw"
        done
    done
done

echo
echo "=============== mean and sd across seeds ==============="
printf "%-10s %-8s %20s %16s %14s\n" "config" "interval" "correlation" "rate Hz" "transients"
awk -F, 'NR>1 {
    key=$1" "$2; n[key]++; c[key]+=$4; c2[key]+=$4*$4; r[key]+=$5; t[key]+=$6
}
END {
    for (k in n) {
        split(k, p, " ")
        cm=c[k]/n[k]; csd=sqrt(c2[k]/n[k]-cm*cm)
        printf "%-10s %-8s %10.4f +/- %-7.4f %14.4f %12.1f\n", p[1], p[2], cm, csd, r[k]/n[k], t[k]/n[k]
    }
}' "$raw" | sort -k1,1 -k2,2n

echo
echo "The interval-1 row is the reference. A coarser interval that stays within"
echo "the +/- is not changing the dynamics; one that moves outside it is."
echo
echo "Raw values in $raw"

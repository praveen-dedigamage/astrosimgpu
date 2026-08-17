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
            # Stop on a failed run. Without this a crash and a run with no
            # transients are indistinguishable: both leave the fields empty and
            # awk reads empty as zero.
            if ! o=$("$BIN" --config "$cfg" -t "$SIMTIME" -s "$seed" \
                     -o "$OUT/${cfg_name}_k${k}_s$seed" 2>&1); then
                echo "FAILED"; echo "$o" | tail -5 >&2; exit 1
            fi
            co=$(awk '/mean pairwise correlation/ {print $NF}' <<< "$o")
            ra=$(awk '/mean firing rate/ {print $4}' <<< "$o")
            tr_=$(awk '/transients detected/ {print $3}' <<< "$o")
            echo "correlation=$co rate=$ra transients=$tr_"
            if [[ -z "$co" || -z "$ra" ]]; then
                echo "run produced no parseable output; stopping" >&2; exit 1
            fi
            echo "$cfg_name,$k,$seed,$co,$ra,$tr_" >> "$raw"
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

# The seeds differ enormously in how strongly the network synchronises, so the
# spread above is mostly between seeds and says little about the interval. The
# comparison is paired: same seed, different interval.
echo
echo "=============== paired against interval 1, same seed ==============="
printf "%-10s %-8s %22s %16s\n" "config" "interval" "correlation change" "rate change"
awk -F, 'NR>1 {
    if ($2 == 1) { c1[$1" "$3]=$4; r1[$1" "$3]=$5 }
    key=$1" "$2; rows[key]=1
}
END {
    while ((getline line < ARGV[1]) > 0) {
        n=split(line, f, ",")
        if (f[1]=="config" || n<5) continue
        s=f[1]" "f[3]
        if (!(s in c1)) continue
        dc = f[4]-c1[s]; dr = f[5]-r1[s]
        k=f[1]" "f[2]; m[k]++; sc[k]+=dc; sc2[k]+=dc*dc; sr[k]+=dr; sr2[k]+=dr*dr
    }
    for (k in m) {
        split(k, p, " ")
        if (p[2]==1) continue
        cm=sc[k]/m[k]; rm=sr[k]/m[k]
        printf "%-10s %-8s %+10.4f +/- %-8.4f %+8.4f +/- %-7.4f\n", p[1], p[2],
               cm, sqrt(sc2[k]/m[k]-cm*cm), rm, sqrt(sr2[k]/m[k]-rm*rm)
    }
}' "$raw" | sort -k1,1 -k2,2n

echo
echo "A change smaller than its own +/- is not a change. Read these rows, not the"
echo "means above: the seeds differ far more from each other than the interval"
echo "moves any one of them."
echo
echo "Raw values in $raw"

#!/bin/bash
# Does the answer depend on the integration step?
#
#   bash scripts/convergence.sh
#   SUBSTEPS="1 2 4 8 16" SEEDS="1 2 3 4 5" bash scripts/convergence.sh
#
# CPU only, no GPU needed. Reports mean and standard deviation across seeds at
# each substep level, because a single seed cannot tell discretisation error
# from the scatter of a noise-driven simulation. If the drift between levels is
# larger than the scatter within one, the step size matters.

set -uo pipefail

BIN=${BIN:-build/astrosimgpu}
CONFIG=${CONFIG:-config/use_case.json}
SIMTIME=${SIMTIME:-60000}
SUBSTEPS=${SUBSTEPS:-"1 2 4 8"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
OUT=${OUT:-results/convergence}

[[ -x "$BIN" ]] || { echo "no binary at $BIN; run make first" >&2; exit 1; }
mkdir -p "$OUT"

echo "config   : $CONFIG   -t $SIMTIME"
echo "substeps : $SUBSTEPS"
echo "seeds    : $SEEDS"
echo

raw="$OUT/raw.csv"
echo "substeps,seed,transients,correlation,rate" > "$raw"

for sub in $SUBSTEPS; do
    # substeps is a configuration field, so each level needs its own file.
    cfg="$OUT/sub$sub.json"
    sed "s/\"substeps\": *[0-9]*/\"substeps\": $sub/" "$CONFIG" > "$cfg"
    grep -q "\"substeps\": $sub" "$cfg" || { echo "substeps not set in $cfg" >&2; exit 1; }

    for seed in $SEEDS; do
        printf "substeps=%-3s seed=%-3s " "$sub" "$seed"
        if ! out=$("$BIN" --config "$cfg" -t "$SIMTIME" -s "$seed" \
                   -o "$OUT/s${sub}_$seed" 2>&1); then
            echo "FAILED"; echo "$out" | tail -5 >&2; exit 1
        fi
        tr_=$(awk '/transients detected/ {print $3}' <<< "$out")
        co=$(awk '/mean pairwise correlation/ {print $NF}' <<< "$out")
        ra=$(awk '/mean firing rate/ {print $4}' <<< "$out")
        echo "transients=$tr_ correlation=$co rate=$ra"
        if [[ -z "$co" || -z "$ra" ]]; then
            echo "run produced no parseable output; stopping" >&2; exit 1
        fi
        echo "$sub,$seed,$tr_,$co,$ra" >> "$raw"
    done
done

echo
echo "=============== mean and sd across seeds ==============="
printf "%-9s %18s %20s %16s\n" "substeps" "transients" "correlation" "rate Hz"
awk -F, 'NR>1 {
    n[$1]++; t[$1]+=$3; t2[$1]+=$3*$3; c[$1]+=$4; c2[$1]+=$4*$4; r[$1]+=$5
}
END {
    for (s in n) {
        tm=t[s]/n[s]; cm=c[s]/n[s]; rm=r[s]/n[s]
        tsd=sqrt(t2[s]/n[s]-tm*tm); csd=sqrt(c2[s]/n[s]-cm*cm)
        printf "%-9s %10.1f +/- %-4.1f %10.4f +/- %-7.4f %10.4f\n", s, tm, tsd, cm, csd, rm
    }
}' "$raw" | sort -n

echo
echo "Compare the change between rows against the +/- within a row. Drift larger"
echo "than the scatter means the step size is doing it; comparable means this"
echo "many seeds cannot tell."
echo
echo "Raw values in $raw"

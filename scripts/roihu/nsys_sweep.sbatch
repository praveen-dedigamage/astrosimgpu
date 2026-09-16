#!/bin/bash
#SBATCH --job-name=project_2020566-astrosimgpu-nsys-sweep
#SBATCH --account=project_2020566
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=nsys-sweep-%j.out
#SBATCH --error=nsys-sweep-%j.err

# nsys_profile.sbatch, but looped across the whole fixed-in-degree
# neuron-scaling sweep (config/neuron_scale_*.json) in one job, producing one
# .nsys-rep + .sqlite pair per size instead of one job per size -- this
# account allows only 2 concurrent jobs (running+pending), so 8 separate
# submissions would mean 8 rounds of "submit, wait, submit again". One job
# with a loop avoids that entirely.
#
# Build first, from roihu-gpu.csc.fi:
#
#   module load nvhpc/26.3
#   cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
#       -DASTROSIMGPU_CUDA=ON -DASTROSIMGPU_NATIVE=ON
#   cmake --build build-cuda -j16
#   python3 scripts/roihu/gen_neuron_scaling_configs.py   # regenerate if needed
#
# Then:
#
#   sbatch scripts/roihu/nsys_sweep.sbatch
#
# Override the size list or window on submission:
#
#   sbatch --export=ALL,SIZES="500 10000 1000000",SIM=10,PRE=2 \
#       scripts/roihu/nsys_sweep.sbatch

set -uo pipefail

module load nvhpc/26.3

# See nsys_profile.sbatch / neuron_scaling.sbatch for why this matters: idle
# OpenMP threads spin-wait by default, and on a 72-core node that inflates
# the measured cost of the serial delivery phases. Ruled out as an artifact
# (active vs passive gave the same real numbers once thread count was held
# constant), but there is no reason to pay the extra contention noise in a
# profiling run, so it stays set.
export OMP_WAIT_POLICY=PASSIVE

BIN=build-cuda/astrosimgpu
if [[ ! -x "$BIN" ]]; then
    echo "MISSING: $BIN -- build it first, see the header of this script" >&2
    exit 1
fi

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys not on PATH -- it ships with the CUDA toolkit / nvhpc module." >&2
    echo "Check 'module spider nsight-systems' or similar if this fails." >&2
    exit 1
fi

echo "account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "host   : $(hostname)   arch: $(uname -m)   cores: $(nproc)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nsys --version
echo

SIZES="${SIZES:-500 1000 2000 5000 10000 100000 1000000 10000000}"
SIM=${SIM:-20}
PRE=${PRE:-5}
CPU_SAMPLE=${CPU_SAMPLE:-1}
SAMPLE_FLAGS=()
if [[ "$CPU_SAMPLE" == "1" ]]; then
    SAMPLE_FLAGS=(--sample=cpu --backtrace=dwarf)
fi

# Kept separate from results/sweep_history.csv (neuron_scaling.sbatch's
# history): --sample=cpu adds real overhead to whichever phase happens to
# run while it interrupts threads, so a profiled run's us/step is not
# comparable to an unprofiled one -- mixing them into one file would
# reproduce the exact OMP_WAIT_POLICY confusion from earlier, just with a
# different cause. Compare nsys-sweep runs only against other nsys-sweep
# runs, and the plain sweep only against itself.
HISTORY="${HISTORY:-results/nsys_sweep_history.csv}"
HISTORY_COLS="timestamp,jobid,commit,neurons,N_A,synapses,max_astro_deg,astro_pct,neuron_pct,deliver_pct,other_pct,us_step,wall_s"
mkdir -p "$(dirname "$HISTORY")"
if [[ ! -f "$HISTORY" ]]; then
    echo "$HISTORY_COLS" > "$HISTORY"
fi
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

SUMMARY_ROWS=()

for n in $SIZES; do
    cfg="config/neuron_scale_${n}.json"
    if [[ ! -f "$cfg" ]]; then
        echo "MISSING: $cfg -- run scripts/roihu/gen_neuron_scaling_configs.py first" >&2
        continue
    fi

    out="astro_timeline_${n}_${SLURM_JOB_ID:-local}"
    echo
    echo "=============== N=$n ($cfg) ==============="
    echo "report : $PWD/${out}.nsys-rep"

    mkdir -p "/tmp/nsys.$$.${n}"

    # Captured to a file (rather than streaming straight to stdout) so the
    # summary table below can be built from it -- still printed in full via
    # the cat right after, so nothing that used to show up in this job's
    # log is lost.
    tmp_out="/tmp/nsys-sweep.$$.${n}.out"

    # Same invocation as nsys_profile.sbatch: srun --export=ALL wraps the
    # whole nsys call (CSC's documented pattern), -t adds nvtx to the
    # default trace set so the Network::run phase ranges show up, and
    # PYTHONNOUSERSITE=1 works around the utf-8-sig bug in Roihu's Python
    # during report post-processing.
    PYTHONNOUSERSITE=1 srun --export=ALL nsys profile -o "$out" -t cuda,nvtx,osrt \
        --stats=true --force-overwrite=true \
        "${SAMPLE_FLAGS[@]}" \
        "$BIN" --config "$cfg" \
        -t "$SIM" -p "$PRE" -o "/tmp/nsys.$$.${n}" > "$tmp_out" 2>&1
    cat "$tmp_out"

    PYTHONNOUSERSITE=1 nsys export --type sqlite --force-overwrite=true \
        -o "${out}.sqlite" "${out}.nsys-rep" || \
        echo "sqlite export failed for N=$n -- the .nsys-rep file is still usable in the GUI."

    # Same extraction as neuron_scaling.sbatch -- the target binary's own
    # "State propagation by phase" printout is still there ahead of nsys's
    # own --stats=true tables, just inside tmp_out instead of streamed
    # directly.
    n_a=$(awk '/^astrocytes / {print $2}' "$tmp_out")
    max_deg=$(awk '/max astro out-degree/ {print $4}' "$tmp_out")
    n2a=$(awk '/neuron -> astrocyte/ {print $4}' "$tmp_out")
    a2n=$(awk '/astrocyte -> neuron/ {print $4}' "$tmp_out")
    n_primary_exc=$(awk '/exc -> neuron synapses/ {print $5}' "$tmp_out")
    n_primary_inh=$(awk '/inh -> neuron synapses/ {print $5}' "$tmp_out")
    synapses=$(( ${n_primary_exc:-0} + ${n_primary_inh:-0} + ${n2a:-0} + ${a2n:-0} ))
    astro_pct=$(awk '/^update astrocytes/ {print $4}' "$tmp_out")
    neuron_pct=$(awk '/^update neurons/ {print $4}' "$tmp_out")
    deliver_pct=$(awk '/^spike delivery/ {s=$4} /^SIC gather\+deliver/ {s+=$4} /^arrival application/ {s+=$4} END {printf "%.4f", s}' "$tmp_out")
    other_pct=$(awk '/^other \(I\/O, loop\)/ {print $5}' "$tmp_out")
    total_s=$(awk '/^total / {print $2}' "$tmp_out")
    wall_s=$(awk '/wall clock/ {print $3}' "$tmp_out")
    recorded_ms=$(awk '/^recorded / {print $2}' "$tmp_out")
    dt=$(awk '/^dt / {print $2}' "$tmp_out")
    steps=$(awk -v r="$recorded_ms" -v d="$dt" 'BEGIN{printf "%d", r/d}')
    us_step=$(awk -v t="$total_s" -v s="$steps" 'BEGIN{ if (s>0) printf "%.2f", t*1e6/s; else print "n/a" }')

    prev_us_step=$(awk -F, -v n="$n" 'NR>1 && $4==n {prev=$12} END{if (prev!="") print prev}' "$HISTORY")
    if [[ -n "$prev_us_step" && "$prev_us_step" != "n/a" && "$us_step" != "n/a" ]]; then
        delta=$(awk -v p="$prev_us_step" -v c="$us_step" 'BEGIN{ printf "%+.1f%%", (p-c)/p*100 }')
    else
        delta="(none)"
        prev_us_step="-"
    fi

    SUMMARY_ROWS+=("$(printf "%-10s %10s %10s %14s %10s %10s %10s %10s %14s %12s %10s" \
        "$n" "${n_a:-?}" "${synapses:-?}" "${max_deg:-?}" "${astro_pct:-?}" \
        "${neuron_pct:-?}" "${deliver_pct:-?}" "${other_pct:-?}" "${us_step:-?}" \
        "$prev_us_step" "$delta")")

    echo "${TIMESTAMP},${SLURM_JOB_ID:-local},${COMMIT},${n},${n_a:-},${synapses:-}," \
         "${max_deg:-},${astro_pct:-},${neuron_pct:-},${deliver_pct:-},${other_pct:-}," \
         "${us_step:-},${wall_s:-}" | tr -d ' ' >> "$HISTORY"
done

echo
echo "=============== summary (all sizes, this job) ==============="
echo "history: $HISTORY (commit $COMMIT)"
printf "%-10s %10s %10s %14s %10s %10s %10s %10s %14s %12s %10s\n" \
    "neurons" "N_A" "synapses" "max_astro_deg" "astro%" "neuron%" "deliver%" "other%" "us/step" "prev_us/step" "delta"
printf '%.0s-' {1..135}; echo
for row in "${SUMMARY_ROWS[@]}"; do
    echo "$row"
done
echo
echo "delta is us/step change vs the most recent PRIOR nsys-sweep run of the"
echo "same size (positive = faster, negative = slower) -- compares only"
echo "against other profiled runs, never against results/sweep_history.csv's"
echo "unprofiled numbers, since --sample=cpu overhead makes the two"
echo "not directly comparable."
echo
echo "=============== done ==============="
echo "Wrote one astro_timeline_<N>_<jobid>.nsys-rep + .sqlite pair per size in $PWD."
echo "Copy them off Roihu with scp/rsync, e.g.:"
echo "  scp roihu:$PWD/astro_timeline_*_${SLURM_JOB_ID:-local}.nsys-rep ."
echo "  scp roihu:$PWD/astro_timeline_*_${SLURM_JOB_ID:-local}.sqlite ."

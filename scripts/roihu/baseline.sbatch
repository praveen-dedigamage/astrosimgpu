#!/bin/bash
#SBATCH --job-name=astrosimgpu-baseline
#SBATCH --account=project_XXXXXXX
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=baseline-%j.out
#SBATCH --error=baseline-%j.err

# CPU baseline on a Grace superchip, before any device code exists.
#
#   sbatch scripts/roihu/baseline.sbatch
#
# Set --account to your CSC project before submitting.
#
# The GPU reservation is not for computing: the Grace ARM cores only exist on
# the GPU nodes, so reserving one GH200 is how you get 72 of them. Each
# reserved GPU grants up to 72 CPU cores and 217 GiB of memory.
#
# gputest allows 15 minutes, which is ample -- the runs below are seconds of
# wall clock. Move to gpumedium only when you need longer.

set -euo pipefail

# Match whatever scripts/roihu/build.sh loaded.
if [[ -n "${COMPILER_MODULES:-}" ]]; then
    # shellcheck disable=SC2086
    module load $COMPILER_MODULES
fi

BIN="${BIN:-build-roihu/astrosimgpu}"
[[ -x "$BIN" ]] || BIN="build/astrosimgpu"

echo "host      : $(hostname)"
echo "arch      : $(uname -m)"
echo "cores     : $(nproc)"
echo "binary    : $BIN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# ---------------------------------------------------------------------------
# 1. Single-thread baseline.
#
# This is the number every later speedup is measured against, and it has to be
# taken on Grace rather than carried over from a laptop: different core,
# different cache, different memory system.
# ---------------------------------------------------------------------------
echo "=============== single thread ==============="
OMP_NUM_THREADS=1 "$BIN" --config config/use_case.json \
    -t 60000 -o "$PWD/results/roihu-1thread" --threads 1

# ---------------------------------------------------------------------------
# 2. OpenMP scaling across the superchip.
#
# The point is not the speedup but the shape of it. The per-cell update is the
# phase that should scale; the delivery phases are serial. Watching where the
# curve flattens says how much of the wall clock is genuinely parallel work,
# which is the same question the GPU port is asking -- just answered a day
# earlier and for free.
#
# If the update phase stops scaling well before 72 threads, the problem is the
# problem size, not the hardware: 500 cells over 72 threads is seven cells per
# thread. Scale the network up before concluding anything.
# ---------------------------------------------------------------------------
for threads in 2 4 8 16 32 72; do
    echo
    echo "=============== $threads threads ==============="
    OMP_NUM_THREADS=$threads "$BIN" --config config/use_case.json \
        -t 60000 -o "$PWD/results/roihu-${threads}thread" --threads "$threads"
done

# ---------------------------------------------------------------------------
# 3. A larger network at full width.
#
# The published benchmarks run 20,000 cells. At 500 the parallel region is too
# small to say anything about scaling, and the phase ratio that motivates the
# port is measured in the wrong regime.
# ---------------------------------------------------------------------------
echo
echo "=============== 20k cells, 72 threads ==============="
sed -e 's/"N_A": 100/"N_A": 4000/' \
    -e 's/"N_E": 400/"N_E": 16000/' \
    -e 's/"N_I": 100/"N_I": 4000/' \
    config/use_case.json > "$PWD/config-20k.json"

OMP_NUM_THREADS=72 "$BIN" --config "$PWD/config-20k.json" \
    -t 10000 -p 2000 -o "$PWD/results/roihu-20k" --threads 72

echo
echo "Done. Phase breakdowns are in each results/*/run.txt"

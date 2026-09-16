#!/bin/bash
#SBATCH --job-name=project_2020568-astrosimgpu-cuda
#SBATCH --account=project_2020568
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=cuda-%j.out
#SBATCH --error=cuda-%j.err

# Native CUDA backend only: correctness, then astrocyte-update throughput
# against population size. No comparison against the other backends -- that
# is what scripts/roihu/three_way.sbatch is for.
#
# Build first, from roihu-gpu.csc.fi:
#
#   module load nvhpc/26.3
#   cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
#       -DASTROSIMGPU_CUDA=ON -DASTROSIMGPU_NATIVE=ON
#   cmake --build build-cuda -j16
#
# Then:
#
#   sbatch scripts/roihu/cuda_only.sbatch
#
# Override the sweep on submission, e.g. a wider range or a longer window:
#
#   sbatch --export=ALL,SIZES="1000 100000 10000000",SIM=2000 \
#       scripts/roihu/cuda_only.sbatch

set -uo pipefail

module load nvhpc/26.3

# Idle OpenMP threads spin-wait by default. deliver_spikes/apply_arrivals
# run serially between two large parallel regions (drive_astrocytes,
# update_neuron) every step; on a 72-core node the 71 idle spinning
# threads contend for memory/cache bandwidth with that serial code and
# inflate its measured cost by roughly 10x -- confirmed empirically: at
# 10,000 neurons deliver phases measured 47.37% active vs 4.84% passive,
# update_neuron 34.21% vs 93.09%, the latter matching the host-only build.
# This has nothing to do with the model; it is purely a wait-policy
# artifact on many-core hardware.
export OMP_WAIT_POLICY=PASSIVE

BIN=build-cuda/astrosimgpu
TESTS=build-cuda/astrosimgpu_tests

for f in "$BIN" "$TESTS"; do
    if [[ ! -x "$f" ]]; then
        echo "MISSING: $f -- build it first, see the header of this script" >&2
        exit 1
    fi
done

echo "account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "host   : $(hostname)   arch: $(uname -m)   cores: $(nproc)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

# Confirm the binary really is the CUDA backend before trusting anything
# below. A CMake misconfiguration builds the host path silently.
echo "=============== backend check ==============="
srun --export=ALL "$BIN" --config config/quick.json -o /tmp/cuda-backend-check.$$ >/dev/null 2>&1
grep 'astrocyte backend' "/tmp/cuda-backend-check.$$/run.txt"
echo

# Correctness first: a fast wrong answer is not a result.
echo "=============== component tests ==============="
srun --export=ALL "$TESTS"
echo

# The regime transition is the model-level check: near-zero correlation
# becomes 0.34 when the neuron-to-astrocyte weight is raised. Values are from
# docs/validation.md.
echo "=============== regime transition ==============="
printf "%-10s %14s %14s\n" "config" "correlation" "expected"

async=$(srun --export=ALL "$BIN" --config config/use_case.json -t 60000 -o results/cuda-async \
    2>/dev/null | awk '/mean pairwise/ {print $NF}')
printf "%-10s %14s %14s\n" "async" "${async:-FAILED}" "0.0106"

burst=$(srun --export=ALL "$BIN" --config config/bursting.json -t 60000 -o results/cuda-burst \
    2>/dev/null | awk '/mean pairwise/ {print $NF}')
printf "%-10s %14s %14s\n" "bursting" "${burst:-FAILED}" "0.4177"
echo

# Astrocyte update cost against population size. Neurons are held fixed
# (config/kernel_scaling.json), so only the offloaded loop grows; flat means
# the cost is per launch, linear means it is per cell. See docs/gpu-port.md
# for the same sweep against the other backends.
PRE=${PRE:-1000}
SIM=${SIM:-5000}
DT=0.1
STEPS=$(awk -v s="$SIM" -v d="$DT" 'BEGIN{printf "%d", s/d}')
SIZES="${SIZES:-100 1000 10000 40000 100000 1000000}"

echo "=============== astrocyte update, CUDA, us/step ==============="
echo "steps: $STEPS measured (${PRE}ms transient discarded, ${SIM}ms recorded, dt ${DT}ms)"
printf "%-12s %14s %14s\n" "astrocytes" "total_s" "us/step"

for n in $SIZES; do
    t=$(srun --export=ALL "$BIN" --config config/kernel_scaling.json --astrocytes "$n" \
        -t "$SIM" -p "$PRE" -o "results/cuda-ks-$n" 2>/dev/null \
        | awk '/update astrocytes/ {print $3}')
    [[ -z "$t" ]] && t=0
    us=$(awk -v t="$t" -v s="$STEPS" 'BEGIN{printf "%.2f", t*1e6/s}')
    printf "%-12s %14s %14s\n" "$n" "$t" "$us"
done

echo
echo "Full phase breakdowns are in each results/cuda-*/run.txt."

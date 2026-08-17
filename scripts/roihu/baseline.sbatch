#!/bin/bash
#SBATCH --job-name=astrosimgpu-baseline
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=baseline-%j.out
#SBATCH --error=baseline-%j.err

# The account is not set here. Export SBATCH_ACCOUNT (and SLURM_ACCOUNT for
# srun) instead, so switching project does not mean editing every script:
#
#   export SBATCH_ACCOUNT=project_XXXXXXX

# CPU baseline on Grace.
#
# The GPU reservation is only how you get at the Grace cores -- they exist only
# on the GPU nodes. One GH200 grants 72 of them.

set -euo pipefail

# Same modules as the build.
if [[ -n "${COMPILER_MODULES:-}" ]]; then
    # shellcheck disable=SC2086
    module load $COMPILER_MODULES
fi

BIN="${BIN:-build-roihu/astrosimgpu}"
[[ -x "$BIN" ]] || BIN="build/astrosimgpu"

echo "account   : ${SLURM_JOB_ACCOUNT:-unknown}"
echo "host      : $(hostname)"
echo "arch      : $(uname -m)"
echo "cores     : $(nproc)"
echo "binary    : $BIN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# Single thread: the number later speedups are measured against. Has to be
# taken on Grace, not carried over from another machine.
echo "=============== single thread ==============="
OMP_NUM_THREADS=1 "$BIN" --config config/use_case.json \
    -t 60000 -o "$PWD/results/roihu-1thread" --threads 1

# Thread scaling. The shape matters more than the number: the update should
# scale, delivery should not. If it flattens well before 72 threads, the network
# is too small rather than the hardware being at fault.
for threads in 2 4 8 16 32 72; do
    echo
    echo "=============== $threads threads ==============="
    OMP_NUM_THREADS=$threads "$BIN" --config config/use_case.json \
        -t 60000 -o "$PWD/results/roihu-${threads}thread" --threads "$threads"
done

# 20k cells, matching the published benchmarks. At 500 the parallel region is
# too small to say anything.
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

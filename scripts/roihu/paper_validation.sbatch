#!/bin/bash
#SBATCH --job-name=project_2020566-astrosimgpu-paper
#SBATCH --account=project_2020566
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=paper-%j.out
#SBATCH --error=paper-%j.err

# Reproduce the published "Sparse" benchmark firing rate (docs/validation.md)
# against the *current* CUDA build, not the host build that check was
# originally run against. The host build reproduced Jiang et al. (2025)'s
# 4.76 spikes/s to 0.04% at substeps=2, and 0.33% at the default substeps=1;
# that check predates Stage 3/4, and nothing since has confirmed the neuron
# kernel and the fully-resident delivery pipeline still land on the same
# collective firing rate rather than merely being internally self-consistent.
# This exercises the neuron model, the astrocyte model, the tripartite
# connectivity with random pools, short-term plasticity (enabled in this
# config), Poisson input and SIC coupling together -- the same combination
# docs/validation.md calls out as the point of this check.
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
#   sbatch scripts/roihu/paper_validation.sbatch

set -uo pipefail

module load nvhpc/26.3
export OMP_WAIT_POLICY=PASSIVE

BIN=build-cuda/astrosimgpu
for f in "$BIN"; do
    if [[ ! -x "$f" ]]; then
        echo "MISSING: $f -- build it first, see the header of this script" >&2
        exit 1
    fi
done

echo "account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "host   : $(hostname)   arch: $(uname -m)   cores: $(nproc)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

echo "=============== backend check ==============="
srun --export=ALL "$BIN" --config config/quick.json -o /tmp/paper-backend-check.$$ >/dev/null 2>&1
grep 'astrocyte backend' "/tmp/paper-backend-check.$$/run.txt"
echo

echo "=============== paper_sparse, substeps=1 (default config), 3 seeds ==============="
printf "%-8s %16s\n" "seed" "mean firing rate (Hz)"
sum1=0
for s in 1 2 3; do
    r=$(srun --export=ALL "$BIN" --config config/paper_sparse.json -s "$s" \
        -o "results/paper-cuda-ss1-s${s}-${SLURM_JOB_ID:-local}" 2>/dev/null \
        | awk '/mean firing rate/ {print $4}')
    printf "%-8s %16s\n" "$s" "${r:-FAILED}"
    sum1=$(awk -v a="$sum1" -v b="${r:-0}" 'BEGIN{printf "%.6f", a+b}')
done
mean1=$(awk -v s="$sum1" 'BEGIN{printf "%.4f", s/3}')
echo "mean of three (substeps=1): $mean1 Hz"
echo "host reference (substeps=1, docs/validation.md): 4.7445 Hz (0.33% from paper's 4.76)"
echo "paper, mean of nine: 4.76 spikes/s"
echo

echo "=============== paper_sparse, substeps=2 (converged), 1 seed ==============="
CFG2="/tmp/paper_sparse_substeps2.$$.json"
sed 's/"substeps": 1/"substeps": 2/' config/paper_sparse.json > "$CFG2"
r2=$(srun --export=ALL "$BIN" --config "$CFG2" -s 1 \
    -o "results/paper-cuda-ss2-s1-${SLURM_JOB_ID:-local}" 2>/dev/null \
    | awk '/mean firing rate/ {print $4}')
echo "seed 1, substeps=2: ${r2:-FAILED} Hz"
echo "host reference (substeps=2, docs/validation.md): 4.7617 Hz (0.04% from paper's 4.76)"
rm -f "$CFG2"
echo

echo "=============== done ==============="
echo "Compare the CUDA figures above against docs/validation.md's host figures."
echo "Agreement close to the host numbers (not just close to 4.76 itself) is"
echo "what confirms the CUDA delivery/neuron kernels reproduce the same"
echo "collective behaviour as the reference host implementation, not merely a"
echo "coincidentally-similar firing rate."

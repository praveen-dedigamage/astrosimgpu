#!/bin/bash
#SBATCH --job-name=astrosimgpu-backends
#SBATCH --partition=gpumedium
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=backends-%j.out
#SBATCH --error=backends-%j.err

# The account is not set here. Export SBATCH_ACCOUNT (and SLURM_ACCOUNT for
# srun) instead, so switching project does not mean editing every script:
#
#   export SBATCH_ACCOUNT=project_XXXXXXX

# Compare all backends on one node, in one job. Any binary that is missing is
# skipped. Build them first:
#
#   make OPENMP=1 CXX=nvc++ BUILD=build-cpu -j16
#   make OFFLOAD=1 CXX=nvc++ BUILD=build-gpu OFFLOAD_FLAGS="-mp=gpu -gpu=cc90" -j16
#   bash scripts/roihu/build_kokkos.sh
#   cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DASTROSIMGPU_CUDA=ON
#   cmake --build build-cuda -j16
#
# Correctness first: a backend that computes something else is not a data point
# about performance, and it fails silently.

set -uo pipefail

module load nvhpc/26.3

# Pin for everything, not just Kokkos which is the only one that warns. An
# unpinned host baseline flatters the device.
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

SIZES="${SIZES:-1000 40000 1000000}"
PRE=${PRE:-100}
SIM=${SIM:-500}
STEPS=$(awk -v s=$SIM -v d=0.1 'BEGIN{printf "%d", s/d}')

declare -a NAMES=("host" "openmp-target" "kokkos" "cuda")
declare -a BINS=("build-cpu/astrosimgpu" "build-gpu/astrosimgpu" "build-kokkos/astrosimgpu" \
                 "build-cuda/astrosimgpu")
INDICES="${!NAMES[@]}"

echo "account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "host   : $(hostname)   cores: $(nproc)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "steps  : $STEPS measured"
echo "binding: OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"
echo

for i in $INDICES; do
    if [[ -x "${BINS[$i]}" ]]; then
        printf "%-14s %s\n" "${NAMES[$i]}" "$(./${BINS[$i]} --config config/quick.json \
            -o /tmp/bk.$$.$i 2>/dev/null >/dev/null; \
            grep -h 'astrocyte backend' /tmp/bk.$$.$i/run.txt | sed 's/astrocyte backend *//')"
    else
        printf "%-14s MISSING (%s)\n" "${NAMES[$i]}" "${BINS[$i]}"
    fi
done
echo

# The regime transition is the check: 0.0106 asynchronous, 0.4177 bursting.
echo "=============== correctness ==============="
printf "%-14s %14s %14s\n" "backend" "use_case" "bursting"
for i in $INDICES; do
    [[ -x "${BINS[$i]}" ]] || continue
    a=$(OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72} ./${BINS[$i]} --config config/use_case.json \
        -t 60000 -o "results/3w-${NAMES[$i]}-async" 2>/dev/null | awk '/mean pairwise/ {print $NF}')
    b=$(OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72} ./${BINS[$i]} --config config/bursting.json \
        -t 60000 -o "results/3w-${NAMES[$i]}-burst" 2>/dev/null | awk '/mean pairwise/ {print $NF}')
    printf "%-14s %14s %14s\n" "${NAMES[$i]}" "${a:-FAILED}" "${b:-FAILED}"
done
echo "expected       0.0106 and 0.4177"
echo

# Only the astrocyte update is dispatched differently, so only it is comparable.
echo "=============== astrocyte update, us/step ==============="
printf "%-12s %14s %14s %14s %14s\n" \
    "astrocytes" "host" "openmp-target" "kokkos" "cuda"
for n in $SIZES; do
    line=$(printf "%-12s" "$n")
    for i in $INDICES; do
        if [[ -x "${BINS[$i]}" ]]; then
            t=$(OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72} ./${BINS[$i]} \
                --config config/kernel_scaling.json --astrocytes "$n" -t "$SIM" -p "$PRE" \
                -o "results/3w-${NAMES[$i]}-$n" 2>/dev/null | awk '/update astrocytes/ {print $3}')
            v=$(awk -v t="${t:-0}" -v s="$STEPS" 'BEGIN{printf "%.1f", t*1e6/s}')
        else
            v="-"
        fi
        line="$line $(printf '%14s' "$v")"
    done
    echo "$line"
done

echo
echo "All backends share the per-cell function and move data at the same points,"
echo "so any difference is dispatch. CUDA is the reference: if the portable"
echo "routes match it, portability is free here."

#!/bin/bash
#SBATCH --job-name=astrosimgpu-scaling
#SBATCH --account=project_2003397
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=scaling-%j.out
#SBATCH --error=scaling-%j.err

# Astrocyte kernel cost against population size. Neurons are fixed, so the host
# work is constant and only the offloaded loop grows. Reported per timestep:
# flat means the cost is per launch, linear means it is per byte.

set -uo pipefail

module load nvhpc/26.3

CPU_BIN=build-cpu/astrosimgpu
GPU_BIN=build-gpu/astrosimgpu
CONFIG=config/kernel_scaling.json

# Fewer steps are fine at large populations: one step costs milliseconds there.
#   sbatch --export=ALL,SIZES=10000000,PRE=100,SIM=500 scripts/roihu/kernel_scaling.sbatch
PRE=${PRE:-1000}
SIM=${SIM:-5000}
DT=0.1
# Recorded window only. The timers start after the transient.
STEPS=$(awk -v s=$SIM -v d=$DT 'BEGIN{printf "%d", s/d}')

SIZES="${SIZES:-100 1000 10000 40000}"

echo "host   : $(hostname)   cores: $(nproc)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "steps  : $STEPS measured   (${PRE}ms transient discarded, ${SIM}ms recorded, dt ${DT}ms)"
echo "threads: ${SLURM_CPUS_PER_TASK:-72} requested for the host runs"
# Confirm the host build really is threaded. A serial run would look plausible
# and mean something else.
"$CPU_BIN" --config "$CONFIG" --astrocytes 100 -t 1 -p 0 -o /tmp/threadcheck.$$ 2>/dev/null \
    | grep -i "openmp threads" || echo "threads: NOT REPORTED -- host build may lack OpenMP"
echo

printf "%-10s %14s %14s %14s %14s %10s\n" \
    "astrocytes" "cpu_total_s" "gpu_total_s" "cpu_us/step" "gpu_us/step" "speedup"

for n in $SIZES; do
    cpu=$(OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72} "$CPU_BIN" --config "$CONFIG" \
            --astrocytes "$n" -t "$SIM" -p "$PRE" -o "results/ks-cpu-$n" 2>/dev/null \
          | awk '/update astrocytes/ {print $3}')
    gpu=$(OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72} "$GPU_BIN" --config "$CONFIG" \
            --astrocytes "$n" -t "$SIM" -p "$PRE" -o "results/ks-gpu-$n" 2>/dev/null \
          | awk '/update astrocytes/ {print $3}')

    [[ -z "$cpu" ]] && cpu=0
    [[ -z "$gpu" ]] && gpu=0

    awk -v n="$n" -v c="$cpu" -v g="$gpu" -v s="$STEPS" 'BEGIN {
        cus = c * 1e6 / s
        gus = g * 1e6 / s
        sp  = (g > 0) ? c / g : 0
        printf "%-10s %14.4f %14.4f %14.2f %14.2f %9.2fx\n", n, c, g, cus, gus, sp
    }'
done

echo
echo "Read gpu_us/step down the page: flat as the population grows tenfold means"
echo "the cost is per launch."

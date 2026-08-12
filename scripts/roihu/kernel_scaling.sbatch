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

# Experiment 1: is the offload cost per launch, or per byte?
#
# The neuron population is fixed and only the astrocyte count varies, so the
# host work is constant and the only thing changing is the size of the
# offloaded loop. Time is reported per timestep, which is what makes the
# question answerable:
#
#   flat as the population grows   -> the cost is launch and mapping overhead
#   growing linearly               -> the cost is transfer volume
#   falling                        -> the device was starved at small sizes
#
# Build both variants first, as in gpu_offload_test.sbatch.

set -uo pipefail

module load nvhpc/26.3

CPU_BIN=build-cpu/astrosimgpu
GPU_BIN=build-gpu/astrosimgpu
CONFIG=config/kernel_scaling.json

PRE=1000
SIM=5000
DT=0.1
STEPS=$(awk -v p=$PRE -v s=$SIM -v d=$DT 'BEGIN{printf "%d", (p+s)/d}')

SIZES="${SIZES:-100 1000 10000 40000}"

echo "host   : $(hostname)   cores: $(nproc)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "steps  : $STEPS   (pre ${PRE}ms + sim ${SIM}ms at dt ${DT}ms)"
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
echo "Read the gpu_us/step column down the page. If it barely moves while the"
echo "population grows tenfold, the cost is per launch and keeping the arrays"
echo "resident on the device is the whole problem."

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

# The measurement is a cost per timestep, so the number of steps only has to
# be large enough to average over. At very large populations a single step
# costs milliseconds, and 60,000 of them is a wasted hour.
#
#   sbatch --export=ALL,SIZES=10000000,PRE=100,SIM=500 scripts/roihu/kernel_scaling.sbatch
PRE=${PRE:-1000}
SIM=${SIM:-5000}
DT=0.1
# Only the recorded window, not the transient. The phase timers start at the
# end of the transient, so dividing by the total step count understates every
# figure by the ratio between them.
STEPS=$(awk -v s=$SIM -v d=$DT 'BEGIN{printf "%d", s/d}')

SIZES="${SIZES:-100 1000 10000 40000}"

echo "host   : $(hostname)   cores: $(nproc)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "steps  : $STEPS measured   (${PRE}ms transient discarded, ${SIM}ms recorded, dt ${DT}ms)"
echo "threads: ${SLURM_CPUS_PER_TASK:-72} requested for the host runs"
# Confirmed rather than assumed: the host figures are quoted as a many-core
# baseline, and a serial run would produce a plausible-looking number that
# means something entirely different.
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
echo "Read the gpu_us/step column down the page. If it barely moves while the"
echo "population grows tenfold, the cost is per launch and keeping the arrays"
echo "resident on the device is the whole problem."

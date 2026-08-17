#!/bin/bash
#SBATCH --job-name=astrosimgpu-offload
#SBATCH --account=project_2003397
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=offload-%j.out
#SBATCH --error=offload-%j.err

# Validate the offload against the host build, same machine, same job.
#
# Build both variants on the login node first:
#
#   module load nvhpc/26.3
#   make distclean
#   make OPENMP=1 CXX=nvc++ BUILD=build-cpu -j16
#   make OFFLOAD=1 CXX=nvc++ BUILD=build-gpu \
#        OFFLOAD_FLAGS="-mp=gpu -gpu=cc90" -j16
#
# Then:
#
#   sbatch scripts/roihu/gpu_offload_test.sbatch
#
# gres names are case-sensitive: gpu:GH200:1 matches nothing and the job is
# admitted with zero cores. Do not request memory; it comes with the GPU.

set -uo pipefail

module load nvhpc/26.3

CPU_BIN=build-cpu/astrosimgpu
GPU_BIN=build-gpu/astrosimgpu
GPU_TESTS=build-gpu/astrosimgpu_tests

# Which configuration to compare, and for how long. Override on submission:
#
#   sbatch --export=ALL,CONFIG=config/scale_1000.json,SIMTIME=10000 \
#          scripts/roihu/gpu_offload_test.sbatch
CONFIG=${CONFIG:-config/use_case.json}
SIMTIME=${SIMTIME:-60000}

echo "config  : $CONFIG   sim-time: $SIMTIME ms"
echo "host    : $(hostname)"
echo "arch    : $(uname -m)"
echo "cores   : $(nproc)"
echo
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

for f in "$CPU_BIN" "$GPU_BIN" "$GPU_TESTS"; do
    if [[ ! -x "$f" ]]; then
        echo "MISSING: $f -- build both variants first, see the header of this script" >&2
        exit 1
    fi
done

# Component tests first: a failure here names the quantity that is wrong.
echo "=============== component tests (offload build) ==============="
"$GPU_TESTS"
TESTS_RC=$?
echo


echo "=============== CPU build ==============="
CPU_START=$SECONDS
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} "$CPU_BIN" \
    --config "$CONFIG" -t "$SIMTIME" -o results/cmp-cpu > /dev/null
CPU_TIME=$((SECONDS - CPU_START))
CPU_VAL=$(grep 'mean pairwise' results/cmp-cpu/run.txt | awk '{print $NF}')
grep -E 'update astrocytes|update neurons|total|real-time' results/cmp-cpu/run.txt
echo "wall: ${CPU_TIME}s   correlation: $CPU_VAL"
echo

echo "=============== offload build ==============="
GPU_START=$SECONDS
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} "$GPU_BIN" \
    --config "$CONFIG" -t "$SIMTIME" -o results/cmp-gpu > /dev/null
GPU_TIME=$((SECONDS - GPU_START))
GPU_VAL=$(grep 'mean pairwise' results/cmp-gpu/run.txt | awk '{print $NF}')
grep -E 'update astrocytes|update neurons|total|real-time' results/cmp-gpu/run.txt
echo "wall: ${GPU_TIME}s   correlation: $GPU_VAL"
echo

# Correctness is what this job is for; a large speedup is not expected. The two
# builds are checked against each other rather than a constant, because a wrong
# map direction leaves the output plausible but different.
echo "=============== result ==============="
echo "config               : $CONFIG  (-t $SIMTIME)"
echo "CPU build            : $CPU_VAL"
echo "offload build        : $GPU_VAL"
echo

if [[ "$CPU_VAL" == "$GPU_VAL" ]]; then
    echo "PASS: offloaded run matches the CPU run"
else
    echo "FAIL: results differ -- check the map clauses in src/astrocyte.cpp"
fi

if [[ $TESTS_RC -ne 0 ]]; then
    echo "NOTE: component tests failed, see above"
fi

echo
echo "Only the astrocyte update is offloaded. The neuron update is still on"
echo "the host, so the totals include host work in both columns. Compare the"
echo "'update astrocytes' rows against each other, not the totals."

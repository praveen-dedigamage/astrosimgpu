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

# Validate the OpenMP target offload of the astrocyte update against the CPU
# build, on the same machine and in the same job.
#
# Build both variants on the login node first:
#
#   module load nvhpc/26.3
#   make clean
#   make OPENMP=1 BUILD=build-cpu -j16
#   make OFFLOAD=1 CXX=nvc++ BUILD=build-gpu \
#        OFFLOAD_FLAGS="-mp=gpu -gpu=cc90 -Minfo=mp" -j16
#
# Then:
#
#   sbatch scripts/roihu/gpu_offload_test.sbatch
#
# The gres name is lowercase gh200. Slurm gres names are case-sensitive, and
# gpu:GH200:1 matches nothing: the job is then admitted with zero cores and
# fails with a message about the request being unsupported.
#
# Memory is not requested. It comes with the GPU allocation, and asking for it
# is overridden.
#
# Change --account above if this project has no Roihu GPU allocation.

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

# The check is that the two builds agree with each other. A hardcoded value
# would only be valid for one configuration, and the failure being guarded
# against is a wrong map direction, which makes the offloaded run disagree with
# the host run while still producing plausible output.

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

# ---------------------------------------------------------------------------
# 1. Component tests, built with offload enabled.
#
# These run on the device wherever the offloaded path is reached, so a failure
# here localises the problem to a specific quantity rather than to the whole
# simulation.
# ---------------------------------------------------------------------------
echo "=============== component tests (offload build) ==============="
"$GPU_TESTS"
TESTS_RC=$?
echo

# ---------------------------------------------------------------------------
# 2. Same simulation both ways.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3. Verdict.
#
# Only the astrocyte update is offloaded, and it is 4.3% of the run time on
# this configuration. A large speedup is not expected and its absence is not a
# problem. Correctness is what this job is for.
# ---------------------------------------------------------------------------
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

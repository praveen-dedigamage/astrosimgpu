#!/bin/bash
#SBATCH --job-name=project_2020566-astrosimgpu-ncu
#SBATCH --account=project_2020566
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=ncu-%j.out
#SBATCH --error=ncu-%j.err

# Nsight Compute: one kernel, in detail -- registers, occupancy, stall
# reasons, memory throughput. This is what produced the "limited by
# registers" finding in docs/profiling.md (94 registers/thread, 31.25%
# theoretical occupancy, 42% of cycles stalled on a fixed-latency execution
# dependency through the RK4 stages).
#
# Unlike nsys, --set full REPLAYS each kernel launch about 40 times to
# collect every metric, so keep the run short -- this does not represent
# real per-step timing, only what happens inside one launch.
#
# Build the CUDA backend first:
#
#   module load nvhpc/26.3
#   cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
#       -DASTROSIMGPU_CUDA=ON -DASTROSIMGPU_NATIVE=ON
#   cmake --build build-cuda -j16
#
# Then:
#
#   sbatch scripts/roihu/ncu_profile.sbatch
#
# Override the population size on submission:
#
#   sbatch --export=ALL,ASTRO=100000 scripts/roihu/ncu_profile.sbatch
#
# Override which kernel(s) get profiled (default: astro, matching the astro
# kernels this script was written for). KERNEL is an ncu -k regex, so it
# matches by substring against the kernel's name -- "neuron" catches
# neuron_update_kernel, "astro" catches astro_update_kernel and
# astro_drive_kernel, "astro|neuron" catches all three in one run:
#
#   sbatch --export=ALL,CONFIG=config/neuron_scale_10000000.json,ASTRO="",KERNEL=neuron \
#       scripts/roihu/ncu_profile.sbatch
#
# The full report (astro_kernel.ncu-rep, tens of MB) is not printed here --
# copy it off Roihu and open it in the Nsight Compute GUI for the guided
# analysis (it points at the specific stall/occupancy issue, section by
# section). This job prints --print-summary=per-kernel to stdout, which is
# enough to read the headline numbers without the GUI.

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
if [[ ! -x "$BIN" ]]; then
    echo "MISSING: $BIN -- build it first, see the header of this script" >&2
    exit 1
fi

if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu not on PATH -- it ships with the CUDA toolkit / nvhpc module." >&2
    echo "Check 'module spider nsight-compute' or similar if this fails." >&2
    exit 1
fi

echo "account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "host   : $(hostname)   arch: $(uname -m)   cores: $(nproc)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
ncu --version
echo

# Very short window: --set full replays every launch, so keep the number of
# launches small. The kernel's per-launch shape does not depend on how many
# steps run, only on the population size.
#
# CONFIG picks which network to profile -- default is the astro-only
# kernel_scaling.json sweep (neurons fixed at 500), which is why ASTRO
# defaults to overriding N_A to 1,000,000. Point CONFIG at a fixed-in-degree
# config instead (e.g. config/neuron_scale_500.json) to profile the
# astrocyte kernel's actual shape under a network where neurons scale too;
# pass ASTRO="" in that case so the config's own N_A is used rather than
# overridden.
CONFIG=${CONFIG:-config/kernel_scaling.json}
ASTRO=${ASTRO-1000000}
SIM=${SIM:-2}
PRE=${PRE:-1}
KERNEL=${KERNEL:-astro}
OUT=${OUT:-astro_kernel_${SLURM_JOB_ID:-local}}
ASTRO_FLAGS=()
if [[ -n "$ASTRO" ]]; then
    ASTRO_FLAGS=(--astrocytes "$ASTRO")
fi

echo "config : $CONFIG ${ASTRO_FLAGS[*]} -t $SIM -p $PRE"
echo "kernel : regex:$KERNEL"
echo "report : $PWD/${OUT}.ncu-rep"
echo

mkdir -p /tmp/ncu.$$

# srun (CSC's documented pattern for this cluster) wraps the whole ncu
# invocation; ncu launches the target binary as its child.
PYTHONNOUSERSITE=1 srun --export=ALL ncu --set full -k "regex:$KERNEL" -o "$OUT" --force-overwrite \
    --print-summary per-kernel \
    "$BIN" --config "$CONFIG" "${ASTRO_FLAGS[@]}" \
    -t "$SIM" -p "$PRE" -o "/tmp/ncu.$$"

echo
echo "=============== what to look for ==============="
echo "In the summary above, or in ${OUT}.ncu-rep in the GUI:"
echo "  - Compute (SM) / Memory throughput -- low on both means latency"
echo "    bound (too few resident warps), not bandwidth or divergence bound."
echo "  - Achieved Occupancy vs Theoretical Occupancy -- close together means"
echo "    the ceiling is register allocation, not scheduling or imbalance."
echo "  - Registers Per Thread -- 94 gave 31.25% theoretical occupancy on the"
echo "    reference GH200. A -maxrregcount sweep trades this against spilling"
echo "    (Local Memory spill should stay at zero if the sweep is safe)."
echo "  - Warp State (Stall) breakdown -- a large 'Stall Wait' / short-scoreboard"
echo "    share here means the RK4 stage sequence itself is the latency chain."
echo
echo "Copy ${OUT}.ncu-rep off Roihu to inspect it section by section in the"
echo "Nsight Compute GUI -- it will point at whichever of these is binding."

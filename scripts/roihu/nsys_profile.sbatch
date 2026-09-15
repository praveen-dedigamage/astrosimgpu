#!/bin/bash
#SBATCH --job-name=astrosimgpu-nsys
#SBATCH --account=project_2020568
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=nsys-%j.out
#SBATCH --error=nsys-%j.err

# Nsight Systems: where one step goes in time (host and device), as opposed to
# Nsight Compute (ncu), which profiles a single kernel launch in detail. See
# docs/profiling.md -- this reproduces the "device is idle most of the step"
# measurement from there.
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
#   sbatch scripts/roihu/nsys_profile.sbatch
#
# Override the population size or window on submission:
#
#   sbatch --export=ALL,ASTRO=100000,SIM=10,PRE=2 scripts/roihu/nsys_profile.sbatch
#
# The report itself (astro_timeline.nsys-rep, in $PWD) is not printed here --
# copy it off Roihu and open it in the Nsight Systems GUI for the visual
# timeline. This job prints the --stats=true text summary to stdout, which is
# enough to read the kernel/transfer/idle split without the GUI.

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

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys not on PATH -- it ships with the CUDA toolkit / nvhpc module." >&2
    echo "Check 'module spider nsight-systems' or similar if this fails." >&2
    exit 1
fi

echo "account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "host   : $(hostname)   arch: $(uname -m)   cores: $(nproc)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nsys --version
echo

# Short window: this profiles the whole step at normal speed (no replay, so
# unlike ncu it does not distort per-step timing), and the pattern repeats
# every step, so seconds are enough to see the shape.
#
# CONFIG picks which network to profile -- default is the astro-only
# kernel_scaling.json sweep (neurons fixed at 500), which is why ASTRO
# defaults to overriding N_A to 1,000,000, matching the documented baseline
# in docs/profiling.md. Point CONFIG at a fixed-in-degree config instead
# (e.g. config/neuron_scale_500.json) to profile a network where neurons
# scale too; pass ASTRO="" in that case so the config's own N_A is used
# rather than overridden.
CONFIG=${CONFIG:-config/kernel_scaling.json}
ASTRO=${ASTRO-1000000}
SIM=${SIM:-20}
PRE=${PRE:-5}
ASTRO_FLAGS=()
if [[ -n "$ASTRO" ]]; then
    ASTRO_FLAGS=(--astrocytes "$ASTRO")
fi

# CPU call-stack sampling, on by default: this is what shows *which host
# function* (drive_astrocytes, deliver_spikes, ...) is running during the
# idle-GPU gap between kernel launches, on the same timeline as the CUDA
# activity below. Set CPU_SAMPLE=0 to skip it (smaller, faster report) if you
# only want the GPU-side tables.
CPU_SAMPLE=${CPU_SAMPLE:-1}
SAMPLE_FLAGS=()
if [[ "$CPU_SAMPLE" == "1" ]]; then
    SAMPLE_FLAGS=(--sample=cpu --backtrace=dwarf)
fi
OUT=${OUT:-astro_timeline}

echo "config : $CONFIG ${ASTRO_FLAGS[*]} -t $SIM -p $PRE"
echo "report : $PWD/${OUT}.nsys-rep"
echo

mkdir -p /tmp/nsys.$$

# PYTHONNOUSERSITE=1 works around 'unknown encoding: utf-8-sig' when nsys
# post-processes the report with Roihu's Python; see docs/profiling.md.
# -t adds nvtx explicitly to CSC's own recommended trace set (nvtx,cuda) --
# nvtx picks up the phase ranges just added to Network::run; osrt is kept on
# top of CSC's default so the OS-runtime/futex thread-state view still shows.
# srun (CSC's documented pattern for this cluster) wraps the whole nsys
# invocation, not just the target binary -- nsys launches it as a child.
PYTHONNOUSERSITE=1 srun --export=ALL nsys profile -o "$OUT" -t cuda,nvtx,osrt \
    --stats=true --force-overwrite=true \
    "${SAMPLE_FLAGS[@]}" \
    "$BIN" --config "$CONFIG" "${ASTRO_FLAGS[@]}" \
    -t "$SIM" -p "$PRE" -o "/tmp/nsys.$$"

echo
echo "=============== summary ==============="
echo "Kernel / memcpy time is in the --stats=true tables above:"
echo "  cudaapisum / gpukernsum / gpumemtimesum are the ones docs/profiling.md"
echo "  reads (kernel share, transfer share, and by elimination, idle share)."
echo "  nvtx_sum, new since the phase ranges were added to Network::run, gives"
echo "  the same six-phase split PhaseProfile prints, but per-launch instead"
echo "  of only totalled at the end of the run."
if [[ "$CPU_SAMPLE" == "1" ]]; then
    echo
    echo "CPU call stacks were also sampled (--sample=cpu). They are not in the"
    echo "text tables above -- open ${OUT}.nsys-rep in the Nsight Systems GUI"
    echo "and look at the CPU row under the process: each sample is a stack,"
    echo "so you can see which host function (e.g. drive_astrocytes) is"
    echo "running during the gaps between astro_update_kernel launches."
fi
echo

# The .sqlite export lets you query specific rows directly (this is how the
# per-operation table in docs/profiling.md was built) without re-running nsys
# stats with different --report flags each time.
PYTHONNOUSERSITE=1 nsys export --type sqlite --force-overwrite=true \
    -o "${OUT}.sqlite" "${OUT}.nsys-rep" || \
    echo "sqlite export failed -- the .nsys-rep file is still usable in the GUI."

echo
echo "Wrote: ${OUT}.nsys-rep (open in the Nsight Systems GUI for the timeline)"
echo "       ${OUT}.sqlite   (query with sqlite3, e.g.:"
echo "                        sqlite3 ${OUT}.sqlite \\"
echo "                          \"select name, sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL \\"
echo "                           join StringIds on nameId=id group by name;\")"
echo
echo "Copy both files off Roihu with scp/rsync to inspect locally."

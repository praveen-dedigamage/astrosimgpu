#!/bin/bash
#SBATCH --job-name=astrosimgpu-nsys
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=nsys-%j.out
#SBATCH --error=nsys-%j.err

# The account is not set here. Export SBATCH_ACCOUNT (and SLURM_ACCOUNT for
# srun) instead, so switching project does not mean editing every script:
#
#   export SBATCH_ACCOUNT=project_XXXXXXX

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
ASTRO=${ASTRO:-1000000}
SIM=${SIM:-20}
PRE=${PRE:-5}
OUT=${OUT:-astro_timeline}

echo "config : config/kernel_scaling.json --astrocytes $ASTRO -t $SIM -p $PRE"
echo "report : $PWD/${OUT}.nsys-rep"
echo

mkdir -p /tmp/nsys.$$

# PYTHONNOUSERSITE=1 works around 'unknown encoding: utf-8-sig' when nsys
# post-processes the report with Roihu's Python; see docs/profiling.md.
PYTHONNOUSERSITE=1 nsys profile -o "$OUT" --stats=true --force-overwrite=true \
    "$BIN" --config config/kernel_scaling.json --astrocytes "$ASTRO" \
    -t "$SIM" -p "$PRE" -o "/tmp/nsys.$$"

echo
echo "=============== summary ==============="
echo "Kernel / memcpy time is in the --stats=true tables above:"
echo "  cudaapisum / gpukernsum / gpumemtimesum are the ones docs/profiling.md"
echo "  reads (kernel share, transfer share, and by elimination, idle share)."
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

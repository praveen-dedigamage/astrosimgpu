#!/bin/bash
# Build astrosimgpu on Roihu's Grace (ARM) side.
#
#   ssh roihu-gpu.csc.fi          # NOT roihu-cpu.csc.fi
#   bash scripts/roihu/build.sh
#
# Roihu has two login nodes with different architectures, and binaries do not
# cross between them: "Software compiled on Roihu-CPU nodes only works on
# Roihu-CPU nodes." The GPU nodes are the ARM ones, so everything that will
# ever run beside a GH200 has to be built from roihu-gpu.csc.fi.

set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: this is $(uname -m), not aarch64." >&2
    echo "You are on the x86 login node. Log in to roihu-gpu.csc.fi instead." >&2
    exit 1
fi

echo "== architecture: $(uname -m) =="

# Module names are not hard-coded: CSC renames and re-versions them, and a
# stale name here would cost more time than the discovery does. Find what is
# actually installed with:
#
#   module spider nvhpc
#   module spider cuda
#   module spider gcc
#
# then set COMPILER_MODULES to the result, e.g.
#   export COMPILER_MODULES="nvhpc/24.7 cuda/12.6"
COMPILER_MODULES="${COMPILER_MODULES:-}"

if [[ -n "$COMPILER_MODULES" ]]; then
    # shellcheck disable=SC2086
    module load $COMPILER_MODULES
    echo "== loaded: $COMPILER_MODULES =="
else
    echo "== no COMPILER_MODULES set; using the default environment =="
    echo "   run 'module spider nvhpc' and re-run with:"
    echo "   COMPILER_MODULES=\"nvhpc/<version>\" bash scripts/roihu/build.sh"
fi

echo "== compiler =="
${CXX:-c++} --version | head -2

# OpenMP matters here in a way it does not on a laptop: a Grace superchip has
# 72 cores, and the per-cell update is the phase meant to use them. Building
# single-threaded would make the baseline meaningless.
BUILD_DIR="${BUILD_DIR:-build-roihu}"

if command -v cmake >/dev/null 2>&1; then
    echo "== cmake build into $BUILD_DIR =="
    cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DASTROSIMGPU_OPENMP=ON \
        -DASTROSIMGPU_NATIVE=ON
    cmake --build "$BUILD_DIR" -j 16
    BIN="$BUILD_DIR/astrosimgpu"
    TESTS="$BUILD_DIR/astrosimgpu_tests"
else
    echo "== no cmake; falling back to make =="
    make OPENMP=1 -j 16
    BIN="build/astrosimgpu"
    TESTS="build/astrosimgpu_tests"
fi

echo "== tests =="
"$TESTS"

echo
echo "Built: $BIN"
echo "Next:  sbatch scripts/roihu/baseline.sbatch"

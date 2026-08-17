#!/bin/bash
# Build on Roihu's Grace (ARM) side.
#
#   ssh roihu-gpu.csc.fi          # not roihu-cpu
#   bash scripts/roihu/build.sh
#
# Binaries do not cross between Roihu's two login architectures, and the GPU
# nodes are the ARM ones.

set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: this is $(uname -m), not aarch64." >&2
    echo "You are on the x86 login node. Log in to roihu-gpu.csc.fi instead." >&2
    exit 1
fi

echo "== architecture: $(uname -m) =="

# Module names change, so find them rather than hard-coding:
#   module spider nvhpc
# then e.g. export COMPILER_MODULES="nvhpc/26.3"
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

BUILD_DIR="${BUILD_DIR:-build-roihu}"

# OpenMP on: a Grace superchip has 72 cores and a serial baseline is useless.

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

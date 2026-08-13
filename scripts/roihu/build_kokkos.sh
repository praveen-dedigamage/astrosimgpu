#!/bin/bash
# Build Kokkos for GH200 and then astrosimgpu against it.
#
#   ssh roihu-gpu.csc.fi          # the ARM login node, not roihu-cpu
#   bash scripts/roihu/build_kokkos.sh
#
# Set KOKKOS_ROOT to an existing installation to skip building Kokkos:
#
#   KOKKOS_ROOT=/path/to/kokkos/install bash scripts/roihu/build_kokkos.sh
#
# The course repository from CSC's Portable GPU Programming ships Kokkos
# exercises and may already have an installation worth pointing at.

set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: this is $(uname -m). Log in to roihu-gpu.csc.fi." >&2
    exit 1
fi

module load nvhpc/26.3

PREFIX="${PREFIX:-$PWD/_kokkos}"
KOKKOS_ROOT="${KOKKOS_ROOT:-}"

if [[ -z "$KOKKOS_ROOT" ]]; then
    echo "== no KOKKOS_ROOT set; building Kokkos into $PREFIX =="

    if [[ ! -d "$PREFIX/src" ]]; then
        git clone --depth 1 --branch master https://github.com/kokkos/kokkos.git "$PREFIX/src"
    fi

    # Kokkos_ARCH_HOPPER90 is the GH200's compute capability. CUDA_LAMBDA is
    # required because the update is dispatched through a lambda; without it
    # the failure appears as template errors rather than a missing option.
    cmake -S "$PREFIX/src" -B "$PREFIX/build" \
        -DCMAKE_INSTALL_PREFIX="$PREFIX/install" \
        -DCMAKE_BUILD_TYPE=Release \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ENABLE_CUDA_LAMBDA=ON \
        -DKokkos_ENABLE_OPENMP=ON \
        -DKokkos_ARCH_HOPPER90=ON

    cmake --build "$PREFIX/build" -j16 --target install
    KOKKOS_ROOT="$PREFIX/install"
fi

echo
echo "== Kokkos at $KOKKOS_ROOT =="

# Kokkos records the compiler it was built with, and mixing compilers between
# Kokkos and its consumer produces link errors rather than a clear message.
cmake -S . -B build-kokkos \
    -DCMAKE_BUILD_TYPE=Release \
    -DASTROSIMGPU_KOKKOS=ON \
    -DKokkos_ROOT="$KOKKOS_ROOT"

cmake --build build-kokkos -j16

echo
echo "== build complete =="

# A CUDA-enabled Kokkos binary links against libcuda.so.1, the driver library,
# which is only present on nodes that have a GPU. Running it on the login node
# fails before main() with a missing shared library, so the check happens in an
# allocation rather than here.
if [[ -n "${SLURM_JOB_ID:-}" ]] || command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "== tests =="
    ./build-kokkos/astrosimgpu_tests
    ./build-kokkos/astrosimgpu --config config/quick.json -o /tmp/kokkos-check.$$ >/dev/null
    grep -E "astrocyte backend" /tmp/kokkos-check.$$/run.txt
else
    cat <<'EOF'

No GPU on this node, so the binary cannot start: a CUDA build links against
the driver library, which only exists where a GPU does. Check it in an
allocation:

  srun --account=project_2003397 --partition=gputest --time=00:10:00 \
       --gres=gpu:gh200:1 --ntasks-per-node=1 --cpus-per-task=72 \
       ./build-kokkos/astrosimgpu_tests

Then: sbatch scripts/roihu/three_way.sbatch
EOF
fi

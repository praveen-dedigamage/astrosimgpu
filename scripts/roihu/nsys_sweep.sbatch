#!/bin/bash
#SBATCH --job-name=astrosimgpu-nsys-sweep
#SBATCH --account=project_2020568
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=nsys-sweep-%j.out
#SBATCH --error=nsys-sweep-%j.err

# nsys_profile.sbatch, but looped across the whole fixed-in-degree
# neuron-scaling sweep (config/neuron_scale_*.json) in one job, producing one
# .nsys-rep + .sqlite pair per size instead of one job per size -- this
# account allows only 2 concurrent jobs (running+pending), so 8 separate
# submissions would mean 8 rounds of "submit, wait, submit again". One job
# with a loop avoids that entirely.
#
# Build first, from roihu-gpu.csc.fi:
#
#   module load nvhpc/26.3
#   cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
#       -DASTROSIMGPU_CUDA=ON -DASTROSIMGPU_NATIVE=ON
#   cmake --build build-cuda -j16
#   python3 scripts/roihu/gen_neuron_scaling_configs.py   # regenerate if needed
#
# Then:
#
#   sbatch scripts/roihu/nsys_sweep.sbatch
#
# Override the size list or window on submission:
#
#   sbatch --export=ALL,SIZES="500 10000 1000000",SIM=10,PRE=2 \
#       scripts/roihu/nsys_sweep.sbatch

set -uo pipefail

module load nvhpc/26.3

# See nsys_profile.sbatch / neuron_scaling.sbatch for why this matters: idle
# OpenMP threads spin-wait by default, and on a 72-core node that inflates
# the measured cost of the serial delivery phases. Ruled out as an artifact
# (active vs passive gave the same real numbers once thread count was held
# constant), but there is no reason to pay the extra contention noise in a
# profiling run, so it stays set.
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

SIZES="${SIZES:-500 1000 2000 5000 10000 100000 1000000 10000000}"
SIM=${SIM:-20}
PRE=${PRE:-5}
CPU_SAMPLE=${CPU_SAMPLE:-1}
SAMPLE_FLAGS=()
if [[ "$CPU_SAMPLE" == "1" ]]; then
    SAMPLE_FLAGS=(--sample=cpu --backtrace=dwarf)
fi

for n in $SIZES; do
    cfg="config/neuron_scale_${n}.json"
    if [[ ! -f "$cfg" ]]; then
        echo "MISSING: $cfg -- run scripts/roihu/gen_neuron_scaling_configs.py first" >&2
        continue
    fi

    out="astro_timeline_${n}"
    echo
    echo "=============== N=$n ($cfg) ==============="
    echo "report : $PWD/${out}.nsys-rep"

    mkdir -p "/tmp/nsys.$$.${n}"

    # Same invocation as nsys_profile.sbatch: srun --export=ALL wraps the
    # whole nsys call (CSC's documented pattern), -t adds nvtx to the
    # default trace set so the Network::run phase ranges show up, and
    # PYTHONNOUSERSITE=1 works around the utf-8-sig bug in Roihu's Python
    # during report post-processing.
    PYTHONNOUSERSITE=1 srun --export=ALL nsys profile -o "$out" -t cuda,nvtx,osrt \
        --stats=true --force-overwrite=true \
        "${SAMPLE_FLAGS[@]}" \
        "$BIN" --config "$cfg" \
        -t "$SIM" -p "$PRE" -o "/tmp/nsys.$$.${n}"

    PYTHONNOUSERSITE=1 nsys export --type sqlite --force-overwrite=true \
        -o "${out}.sqlite" "${out}.nsys-rep" || \
        echo "sqlite export failed for N=$n -- the .nsys-rep file is still usable in the GUI."
done

echo
echo "=============== done ==============="
echo "Wrote one astro_timeline_<N>.nsys-rep + .sqlite pair per size in $PWD."
echo "Copy them off Roihu with scp/rsync, e.g.:"
echo "  scp roihu:$PWD/astro_timeline_*.nsys-rep roihu:$PWD/astro_timeline_*.sqlite ."

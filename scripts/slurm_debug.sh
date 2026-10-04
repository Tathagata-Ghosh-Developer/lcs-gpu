#!/bin/bash
#SBATCH --job-name=LCS_Debug
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=lcs_debug_%j.txt
#SBATCH --error=lcs_debug_err_%j.txt

# Debug script: compares OpenMP/CUDA outputs against serial reference.

set -e

cd "${LCS_ROOT:-${SLURM_SUBMIT_DIR:-.}}" || { echo "ERROR: Cannot cd to work directory"; exit 1; }

echo "======================================================"
echo "  LCS DEBUG - CORRECTNESS VERIFICATION"
echo "======================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================"
echo ""

echo "--- SYSTEM INFO ---"
echo "CPU:     $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "CPUs:    $(nproc) available"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "GPU: Waiting for driver..."
echo ""

echo "Building executables..."
make clean 2>/dev/null || true
make all GPU_ARCH=sm_86

if [ ! -f ./bin/lcs_serial ]; then
    echo "ERROR: lcs_serial was not built!"
    ls -la
    exit 1
fi
if [ ! -f ./bin/lcs_openmp ]; then
    echo "ERROR: lcs_openmp was not built!"
    ls -la
    exit 1
fi
if [ ! -f ./bin/lcs_cuda ]; then
    echo "ERROR: lcs_cuda was not built!"
    ls -la
    exit 1
fi
echo "Build complete. Executables verified."
echo ""

# Shared-seed correctness test with debug-size input.
echo "======================================================"
echo "  Serial/OpenMP/CUDA Correctness Test (N=2000)"
echo "  Shared-seed comparison"
echo "======================================================"
export OMP_NUM_THREADS=8
SEED=${SLURM_JOB_ID:-$RANDOM}

SERIAL_RESULT=$(./bin/lcs_serial 100000000 1 ${SEED} | grep -oP 'Result: \K[0-9]+' | head -1)
OPENMP_RESULT=$(./bin/lcs_openmp 100000000 1 ${SEED} | grep -oP 'Result: \K[0-9]+' | head -1)
CUDA_RESULT=$(./bin/lcs_cuda 100000000 1 ${SEED} | grep -oP 'Result: \K[0-9]+' | head -1)

echo "Seed:   ${SEED}"
echo "Serial: ${SERIAL_RESULT}"
echo "OpenMP: ${OPENMP_RESULT}"
echo "CUDA:   ${CUDA_RESULT}"

if [ "${OPENMP_RESULT}" = "${SERIAL_RESULT}" ]; then
    OMP_STATUS=0
else
    OMP_STATUS=1
fi

if [ "${CUDA_RESULT}" = "${SERIAL_RESULT}" ]; then
    CUDA_STATUS=0
else
    CUDA_STATUS=1
fi
echo ""

echo "======================================================"
echo "  DEBUG SUMMARY"
echo "======================================================"
if [ ${OMP_STATUS} -eq 0 ]; then
    echo "OpenMP: PASSED ✓"
else
    echo "OpenMP: FAILED ✗"
fi

if [ ${CUDA_STATUS} -eq 0 ]; then
    echo "CUDA:   PASSED ✓"
else
    echo "CUDA:   FAILED ✗"
fi
echo "======================================================"
echo "Debug completed at: $(date)"
echo "======================================================"

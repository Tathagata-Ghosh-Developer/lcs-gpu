#!/bin/bash
#SBATCH --job-name=LCS_Profile
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=lcs_profile_%j.txt
#SBATCH --error=lcs_profile_err_%j.txt

# Profiling script for OpenMP scaling and CUDA timeline/benchmark analysis.

set -e

cd "${LCS_ROOT:-${SLURM_SUBMIT_DIR:-.}}" || { echo "ERROR: Cannot cd to work directory"; exit 1; }

echo "======================================================"
echo "  LCS PROFILING - OpenMP & CUDA Analysis"
echo "======================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

echo "--- AVAILABLE PROFILERS ---"
HAVE_NSYS=0
HAVE_PERF=0
HAVE_TAU=0

if command -v nsys &> /dev/null; then
    echo "Nsight Systems: $(nsys --version 2>&1 | head -1)"
    HAVE_NSYS=1
else
    echo "Nsight Systems: Not found"
fi

if command -v perf &> /dev/null; then
    echo "Linux perf: Available"
    HAVE_PERF=1
else
    echo "Linux perf: Not found"
fi

if command -v tau_cxx.sh &> /dev/null || [ -n "${TAU_ROOT}" ]; then
    echo "TAU Profiler: Available (TAU_ROOT=${TAU_ROOT:-auto})"
    HAVE_TAU=1
else
    echo "TAU Profiler: Not found (set TAU_ROOT if available)"
fi

echo ""

echo "--- SYSTEM INFO ---"
CORES=$(nproc)
echo "CPU:     $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "CPUs:    ${CORES} available"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "GPU info pending..."
echo ""

PROFILE_DIR="Profile/profiles_${SLURM_JOB_ID:-local}"
mkdir -p ${PROFILE_DIR}
echo "Output directory: ${PROFILE_DIR}/"
echo ""

echo "Building executables..."
make clean 2>/dev/null || true
make all GPU_ARCH=sm_86

if [ ! -f ./bin/lcs_openmp ] || [ ! -f ./bin/lcs_cuda ]; then
    echo "ERROR: Build failed!"
    ls -la
    exit 1
fi
echo "Build complete. Executables verified."
echo ""

SIZE=100000000
WARMUP_RUNS=5

export OMP_PROC_BIND=close
export OMP_PLACES=cores

THREAD_CONFIGS="16 32 36 64 72"

# OpenMP TAU profiling profile (single measured run per thread)
echo "======================================================"
echo "  OPENMP TAU PROFILE - SINGLE RUN PER THREAD"
echo "======================================================"
echo ""
echo "Thread counts: ${THREAD_CONFIGS}"
echo "Warm-up runs (not measured): ${WARMUP_RUNS}"
echo ""

get_time() {
    grep -oP 'Time: \K[0-9.]+' | head -1
}

OMP_PROFILE_FILE="${PROFILE_DIR}/openmp_scaling.csv"
echo "threads,time_seconds,profile_dir" > ${OMP_PROFILE_FILE}

run_openmp_warmups() {
    local threads="$1"
    echo "Running ${WARMUP_RUNS} OpenMP warm-up runs (${threads} threads)..."
    export OMP_NUM_THREADS="${threads}"
    for i in $(seq 1 ${WARMUP_RUNS}); do
        OUTPUT=$(./bin/lcs_openmp ${SIZE} 0)
        TIME=$(echo "${OUTPUT}" | get_time)
        echo "  Warm-up ${i}/${WARMUP_RUNS}: ${TIME} s"
    done
}

run_cuda_warmups() {
    echo "Running ${WARMUP_RUNS} CUDA warm-up runs..."
    for i in $(seq 1 ${WARMUP_RUNS}); do
        OUTPUT=$(./bin/lcs_cuda ${SIZE} 0)
        TIME=$(echo "${OUTPUT}" | get_time)
        echo "  Warm-up ${i}/${WARMUP_RUNS}: ${TIME} s"
    done
}

# TAU profiling
if [ ${HAVE_TAU} -eq 1 ]; then
    echo "======================================================"
    echo "  TAU PROFILER - OpenMP Function-Level Analysis"
    echo "======================================================"
    
    echo "Building OpenMP with TAU instrumentation..."
    make lcs_openmp_tau 2>/dev/null || {
        echo "TAU build failed - attempting manual compilation..."
        if [ -f "${TAU_ROOT}/bin/tau_cxx.sh" ]; then
            ${TAU_ROOT}/bin/tau_cxx.sh -O3 -fopenmp -march=native -std=c++17 -DUSE_TAU \
                src/lcs_openmp.cpp -o bin/lcs_openmp_tau
        fi
    }
    
    if [ -f ./bin/lcs_openmp_tau ]; then
        echo "Running TAU-instrumented single-run profiles after warm-ups..."
        
        export TAU_PROFILE=1
        export TAU_TRACE=0

        for THREADS in ${THREAD_CONFIGS}; do
            echo "--- TAU OpenMP ${THREADS} Threads ---"
            run_openmp_warmups "${THREADS}"
            export OMP_NUM_THREADS=${THREADS}
            export PROFILEDIR="${PROFILE_DIR}/tau_profiles_t${THREADS}"
            mkdir -p ${PROFILEDIR}

            OUTPUT=$(./bin/lcs_openmp_tau ${SIZE} 0)
            TIME=$(echo "${OUTPUT}" | get_time)
            echo "  Profiled run: ${TIME} s"
            echo "${THREADS},${TIME},${PROFILEDIR}" >> ${OMP_PROFILE_FILE}

            if command -v pprof &> /dev/null; then
                echo "  TAU summary with pprof for ${THREADS} threads:"
                pprof -a ${PROFILEDIR} || true
            fi
            echo ""
        done

        echo "OpenMP TAU profile index saved to: ${OMP_PROFILE_FILE}"
    else
        echo "TAU-instrumented build not available. Skipping TAU profiling."
    fi
    echo ""
fi

# CUDA timeline profile
if [ ${HAVE_NSYS} -eq 1 ]; then
    echo "======================================================"
    echo "  NSIGHT SYSTEMS - CUDA Timeline Profiling"
    echo "======================================================"
    echo "Captures: CUDA API, Kernels, Memory transfers, NVTX markers"
    echo ""
    
    run_cuda_warmups
    echo "Running single detailed timeline profile (post warm-up)..."
    nsys profile \
        --trace=cuda,osrt \
        --stats=true \
        --force-overwrite=true \
        -o ${PROFILE_DIR}/lcs_cuda_nsys \
        ./bin/lcs_cuda ${SIZE} 0 | tee ${PROFILE_DIR}/cuda_nsys_single_run.log
    
    echo ""
    echo "Report saved: ${PROFILE_DIR}/lcs_cuda_nsys.nsys-rep"
    echo "View with: nsys-ui ${PROFILE_DIR}/lcs_cuda_nsys.nsys-rep"
    echo ""
fi

# CUDA single-run timing summary (from Nsight-invoked run)
echo "======================================================"
echo "  CUDA SINGLE-RUN SUMMARY"
echo "======================================================"

CUDA_PROFILE_FILE="${PROFILE_DIR}/cuda_benchmark.csv"
echo "run,time_seconds" > ${CUDA_PROFILE_FILE}

CUDA_TIME=""
if [ -f "${PROFILE_DIR}/cuda_nsys_single_run.log" ]; then
    CUDA_TIME=$(grep -oP 'Time: \K[0-9.]+' ${PROFILE_DIR}/cuda_nsys_single_run.log | head -1)
fi

if [ -n "${CUDA_TIME}" ]; then
    echo "1,${CUDA_TIME}" >> ${CUDA_PROFILE_FILE}
    echo "  Single profiled run time: ${CUDA_TIME} s"
else
    echo "  CUDA profiled run time not found in log."
fi

echo ""
echo "CUDA results saved to: ${CUDA_PROFILE_FILE}"
echo ""

# Speedup summary
echo "======================================================"
echo "  SPEEDUP SUMMARY"
echo "======================================================"

CUDA_MEAN="${CUDA_TIME}"

BEST_OMP_MEAN="999999"
BEST_OMP_THREADS=""

for THREADS in ${THREAD_CONFIGS}; do
    OMP_MEAN=$(grep "^${THREADS}," ${OMP_PROFILE_FILE} | cut -d',' -f2 | awk '{sum+=$1} END {if (NR>0) print sum/NR}')
    if [ -n "${OMP_MEAN}" ]; then
        IS_BETTER=$(echo "${OMP_MEAN} ${BEST_OMP_MEAN}" | awk '{print ($1 < $2) ? 1 : 0}')
        if [ "${IS_BETTER}" -eq 1 ]; then
            BEST_OMP_MEAN="${OMP_MEAN}"
            BEST_OMP_THREADS="${THREADS}"
        fi
    fi
done

if [ -n "${BEST_OMP_THREADS}" ]; then
    SPEEDUP=$(echo "${BEST_OMP_MEAN} ${CUDA_MEAN}" | awk '{printf "%.2f", $1/$2}')
    echo ""
    echo "  >>> GPU Speedup over Best OpenMP (${BEST_OMP_THREADS} threads): ${SPEEDUP}x <<<"
    echo ""
fi

echo "All thread configurations:"
for THREADS in ${THREAD_CONFIGS}; do
    OMP_MEAN=$(grep "^${THREADS}," ${OMP_PROFILE_FILE} | cut -d',' -f2 | awk '{sum+=$1} END {if (NR>0) print sum/NR}')
    if [ -n "${OMP_MEAN}" ]; then
        SPEEDUP=$(echo "${OMP_MEAN} ${CUDA_MEAN}" | awk '{printf "%.2f", $1/$2}')
        echo "  GPU over OpenMP (${THREADS} threads): ${SPEEDUP}x"
    fi
done

# Local analysis instructions
echo ""
echo "======================================================"
echo "  ANALYZE PROFILES LOCALLY WITH NSIGHT"
echo "======================================================"
echo "  Copy profile files to your local machine:"
echo ""
echo "  rsync -av <user>@<login-host>:<repo-dir>/${PROFILE_DIR}/ ."
echo ""
echo "  Then open in Nsight Systems GUI:"
echo "    - Windows: nsys-ui.exe lcs_cuda_nsys.nsys-rep"
echo "    - Linux:   nsys-ui lcs_cuda_nsys.nsys-rep"
echo ""
echo "  NOTE: Nsight Compute (ncu) requires admin permissions"
echo "        on the cluster (ERR_NVGPUCTRPERM). Use nsys instead."
echo ""

# Summary
echo "======================================================"
echo "  PROFILING SUMMARY"
echo "======================================================"
echo "Output directory: ${PROFILE_DIR}/"
echo ""
echo "Files generated:"
ls -la ${PROFILE_DIR}/ 2>/dev/null || echo "  (none)"
echo ""
echo "To analyze:"
echo "  - OpenMP scaling: ${OMP_PROFILE_FILE}"
echo "  - CUDA benchmark: ${CUDA_PROFILE_FILE}"
if [ ${HAVE_NSYS} -eq 1 ]; then
    echo "  - CUDA timeline: ${PROFILE_DIR}/lcs_cuda_nsys.nsys-rep"
fi
echo ""
echo "======================================================"
echo "Profiling completed at: $(date)"
echo "======================================================"

#!/bin/bash
#SBATCH --job-name=LCS
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=Outputs/lcs_output_%j.txt
#SBATCH --error=Errors/lcs_error_%j.txt

# Pipeline script: correctness check, benchmark sweep, and CUDA profiling.

set -e

cd "${LCS_ROOT:-${SLURM_SUBMIT_DIR:-.}}" || { echo "ERROR: Cannot cd to work directory"; exit 1; }

mkdir -p Outputs Errors Profile

JOB_TAG=${SLURM_JOB_ID:-local}
DATA_DIR="Outputs/data_${JOB_TAG}"
mkdir -p "${DATA_DIR}"

echo "============================================================"
echo "          OpenMP vs CUDA"
echo "============================================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Work Dir:  $(pwd)"
echo ""

echo "--- SYSTEM INFO ---"
echo "CPU:       $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "Cores:     $(nproc) available"
echo "Memory:    $(free -h | grep Mem | awk '{print $2}')"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Checking...')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  (GPU info available after driver loads)"
echo ""

echo "--- BUILDING ---"
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

SIZE=100000000
RUNS=30
WARMUP_RUNS=5
THREAD_CONFIGS="16 32 36 64 72"
GUSTAFSON_BASE_SIZES="1000 10000 100000 1000000 10000000 100000000 1000000000 10000000000 100000000000 1000000000000"

# To avoid memory/time failures in the scaling sweep, very large sizes are recorded as skipped.
MAX_FEASIBLE_SIZE=100000000

export OMP_PROC_BIND=close
export OMP_PLACES=cores

STRONG_SCALING_CSV="${DATA_DIR}/strong_scaling_${JOB_TAG}.csv"
GUSTAFSON_CSV="${DATA_DIR}/gustafson_scaling_${JOB_TAG}.csv"
CUDA_ROOFLINE_LOG="${DATA_DIR}/cuda_roofline_raw_${JOB_TAG}.log"
CUDA_PHASE_CSV="${DATA_DIR}/cuda_phase_times_${JOB_TAG}.csv"
TAU_SINGLE_CSV="${DATA_DIR}/tau_single_run_${JOB_TAG}.csv"

echo "threads,mean_time_s,speedup_vs_16,ideal_speedup_vs_16,efficiency_percent" > "${STRONG_SCALING_CSV}"
echo "model,threads,base_size,scaled_size,time_s,status" > "${GUSTAFSON_CSV}"
echo "stage,run,size,phase_h2d_s,phase_sa_s,phase_kernel_s,phase_d2h_s,total_s" > "${CUDA_PHASE_CSV}"
echo "threads,time_s,profile_dir" > "${TAU_SINGLE_CSV}"

get_time() {
    grep -oP 'Time: \K[0-9.]+' | head -1
}

get_phase_time() {
    local phase_name="$1"
    grep -oP "${phase_name}: \K[0-9.]+" | head -1
}

run_openmp_warmups() {
    local threads="$1"
    local size="$2"
    echo "Running ${WARMUP_RUNS} OpenMP warm-up runs (${threads} threads, size=${size})..."
    export OMP_NUM_THREADS="${threads}"
    for i in $(seq 1 ${WARMUP_RUNS}); do
        output=$(./bin/lcs_openmp "${size}" 0)
        warm_time=$(echo "${output}" | get_time)
        echo "  Warm-up ${i}/${WARMUP_RUNS}: ${warm_time} s"
    done
}

run_cuda_warmups() {
    local size="$1"
    local stage="$2"
    echo "Running ${WARMUP_RUNS} CUDA warm-up runs (size=${size})..."
    for i in $(seq 1 ${WARMUP_RUNS}); do
        output=$(./bin/lcs_cuda "${size}" 0)
        warm_time=$(echo "${output}" | get_time)
        h2d=$(echo "${output}" | get_phase_time 'Phase Time H2D')
        sa=$(echo "${output}" | get_phase_time 'Phase Time SA')
        kernel=$(echo "${output}" | get_phase_time 'Phase Time Kernel')
        d2h=$(echo "${output}" | get_phase_time 'Phase Time D2H')
        echo "  Warm-up ${i}/${WARMUP_RUNS}: ${warm_time} s"
        {
            echo "### ${stage} CUDA WARMUP ${i} ###"
            echo "${output}"
            echo ""
        } >> "${CUDA_ROOFLINE_LOG}"
        echo "${stage}_warmup,${i},${size},${h2d},${sa},${kernel},${d2h},${warm_time}" >> "${CUDA_PHASE_CSV}"
    done
}

# Phase 1: correctness with shared seed and debug-size input.
echo "############################################################"
echo "PHASE 1: PROOF OF CORRECTNESS"
echo "  Running with verification enabled"
echo "############################################################"
echo ""

SEED=${SLURM_JOB_ID:-$RANDOM}
echo "Using shared seed: ${SEED}"
SERIAL_RESULT=$(./bin/lcs_serial ${SIZE} 1 ${SEED} | grep -oP 'Result: \K[0-9]+' | head -1)
OPENMP_RESULT=$(./bin/lcs_openmp ${SIZE} 1 ${SEED} | grep -oP 'Result: \K[0-9]+' | head -1)
CUDA_RESULT=$(./bin/lcs_cuda ${SIZE} 1 ${SEED} | grep -oP 'Result: \K[0-9]+' | head -1)

echo "Serial: ${SERIAL_RESULT}"
echo "OpenMP: ${OPENMP_RESULT}"
echo "CUDA:   ${CUDA_RESULT}"

if [ "${OPENMP_RESULT}" != "${SERIAL_RESULT}" ]; then
    echo "ERROR: OpenMP result mismatch against serial reference"
    exit 1
fi

if [ "${CUDA_RESULT}" != "${SERIAL_RESULT}" ]; then
    echo "ERROR: CUDA result mismatch against serial reference"
    exit 1
fi
echo ""

echo ">>> All correctness tests PASSED <<<"
echo ""

# Phase 2: statistical benchmarking.
echo "############################################################"
echo "PHASE 2: PERFORMANCE BENCHMARK (${RUNS} Runs Each)"
echo "  Size: ${SIZE} (100 Million) elements"
echo "  Warm-up runs: ${WARMUP_RUNS} (not measured)"
echo "  Thread configs: ${THREAD_CONFIGS}"
echo "############################################################"
echo ""

calc_stats() {
    echo "$@" | awk '{
        n = NF; sum = 0
        for (i = 1; i <= n; i++) sum += $i
        mean = sum / n
        sumsq = 0
        for (i = 1; i <= n; i++) sumsq += ($i - mean)^2
        if (n > 1) stddev = sqrt(sumsq / (n - 1)); else stddev = 0
        stderr = stddev / sqrt(n)
        ci95 = 1.96 * stderr
        printf "Mean: %.4f s | StdDev: %.4f | 95%% CI: ±%.4f", mean, stddev, ci95
    }'
}

get_mean() { echo "$@" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}'; }

declare -A OMP_MEANS

for THREADS in ${THREAD_CONFIGS}; do
    echo "--- OpenMP (${THREADS} Threads) ---"
    echo "    Algorithm: 7-mer hash index + SIMD extension"

    run_openmp_warmups "${THREADS}" "${SIZE}"
    export OMP_NUM_THREADS=${THREADS}
    
    TIMES=""
    for i in $(seq 1 ${RUNS}); do
        printf "  Run %2d/%d: " $i ${RUNS}
        OUTPUT=$(./bin/lcs_openmp ${SIZE} 0)
        TIME=$(echo "${OUTPUT}" | grep -oP 'Time: \K[0-9.]+' | head -1)
        TIMES="${TIMES} ${TIME}"
        echo "${TIME} s"
    done
    
    STATS=$(calc_stats ${TIMES})
    MEAN=$(get_mean ${TIMES})
    OMP_MEANS[${THREADS}]=${MEAN}
    echo "  >>> ${STATS}"
    echo ""
done

echo "--- CUDA GPU ---"
echo "    Algorithm: prefix-doubling suffix array + GPU LCP/max kernel"

CUDA_TIMES=""
run_cuda_warmups "${SIZE}" "benchmark"
for i in $(seq 1 ${RUNS}); do
    printf "  Run %2d/%d: " $i ${RUNS}
    OUTPUT=$(./bin/lcs_cuda ${SIZE} 0)
    TIME=$(echo "${OUTPUT}" | grep -oP 'Time: \K[0-9.]+' | head -1)
    H2D=$(echo "${OUTPUT}" | get_phase_time 'Phase Time H2D')
    SA=$(echo "${OUTPUT}" | get_phase_time 'Phase Time SA')
    KERNEL=$(echo "${OUTPUT}" | get_phase_time 'Phase Time Kernel')
    D2H=$(echo "${OUTPUT}" | get_phase_time 'Phase Time D2H')

    {
        echo "### BENCHMARK CUDA RUN ${i} ###"
        echo "${OUTPUT}"
        echo ""
    } >> "${CUDA_ROOFLINE_LOG}"
    echo "benchmark,${i},${SIZE},${H2D},${SA},${KERNEL},${D2H},${TIME}" >> "${CUDA_PHASE_CSV}"

    CUDA_TIMES="${CUDA_TIMES} ${TIME}"
    echo "${TIME} s"
done

CUDA_STATS=$(calc_stats ${CUDA_TIMES})
CUDA_MEAN=$(get_mean ${CUDA_TIMES})
echo "  >>> ${CUDA_STATS}"
echo ""

# Phase 2b: speedup summary.
echo "############################################################"
echo "SPEEDUP ANALYSIS"
echo "############################################################"
echo ""

BEST_OMP_THREADS=""
BEST_OMP_TIME="999999"
for THREADS in ${THREAD_CONFIGS}; do
    if [ -n "${OMP_MEANS[${THREADS}]}" ]; then
        IS_BETTER=$(echo "${OMP_MEANS[${THREADS}]} ${BEST_OMP_TIME}" | awk '{print ($1 < $2) ? 1 : 0}')
        if [ "${IS_BETTER}" -eq 1 ]; then
            BEST_OMP_TIME="${OMP_MEANS[${THREADS}]}"
            BEST_OMP_THREADS="${THREADS}"
        fi
    fi
done

echo "Best OpenMP config: ${BEST_OMP_THREADS} threads (Mean: ${BEST_OMP_TIME} s)"
echo "CUDA GPU Mean: ${CUDA_MEAN} s"
echo ""

SPEEDUP=$(echo "${BEST_OMP_TIME} ${CUDA_MEAN}" | awk '{printf "%.2f", $1/$2}')
echo ">>> GPU SPEEDUP over Best OpenMP (${BEST_OMP_THREADS} threads): ${SPEEDUP}x <<<"
echo ""

echo "All speedups (GPU over OpenMP):"
for THREADS in ${THREAD_CONFIGS}; do
    if [ -n "${OMP_MEANS[${THREADS}]}" ]; then
        SP=$(echo "${OMP_MEANS[${THREADS}]} ${CUDA_MEAN}" | awk '{printf "%.2f", $1/$2}')
        echo "  vs ${THREADS} threads: ${SP}x"
    fi
done
echo ""

BASELINE_THREADS=16
BASELINE_TIME="${OMP_MEANS[${BASELINE_THREADS}]}"
if [ -n "${BASELINE_TIME}" ]; then
    for THREADS in ${THREAD_CONFIGS}; do
        if [ -n "${OMP_MEANS[${THREADS}]}" ]; then
            CURR_TIME="${OMP_MEANS[${THREADS}]}"
            SCALE_SPEEDUP=$(echo "${BASELINE_TIME} ${CURR_TIME}" | awk '{printf "%.4f", $1/$2}')
            IDEAL_SPEEDUP=$(echo "${THREADS} ${BASELINE_THREADS}" | awk '{printf "%.4f", $1/$2}')
            EFFICIENCY=$(echo "${SCALE_SPEEDUP} ${IDEAL_SPEEDUP}" | awk '{printf "%.2f", ($1/$2)*100}')
            echo "${THREADS},${CURR_TIME},${SCALE_SPEEDUP},${IDEAL_SPEEDUP},${EFFICIENCY}" >> "${STRONG_SCALING_CSV}"
        fi
    done
fi

echo "Strong-scaling CSV: ${STRONG_SCALING_CSV}"
echo "CUDA roofline raw log: ${CUDA_ROOFLINE_LOG}"
echo "CUDA phase-time CSV: ${CUDA_PHASE_CSV}"
echo ""

echo "############################################################"
echo "PHASE 2C: GUSTAFSON-LAW SCALING DATA"
echo "  Base sizes: ${GUSTAFSON_BASE_SIZES}"
echo "############################################################"
echo ""

for BASE_SIZE in ${GUSTAFSON_BASE_SIZES}; do
    if [ "${BASE_SIZE}" -le "${MAX_FEASIBLE_SIZE}" ]; then
        SERIAL_OUT=$(./bin/lcs_serial ${BASE_SIZE} 0)
        SERIAL_TIME=$(echo "${SERIAL_OUT}" | get_time)
        echo "serial,1,${BASE_SIZE},${BASE_SIZE},${SERIAL_TIME},ok" >> "${GUSTAFSON_CSV}"

        CUDA_OUT=$(./bin/lcs_cuda ${BASE_SIZE} 0)
        CUDA_TIME=$(echo "${CUDA_OUT}" | get_time)
        H2D=$(echo "${CUDA_OUT}" | get_phase_time 'Phase Time H2D')
        SA=$(echo "${CUDA_OUT}" | get_phase_time 'Phase Time SA')
        KERNEL=$(echo "${CUDA_OUT}" | get_phase_time 'Phase Time Kernel')
        D2H=$(echo "${CUDA_OUT}" | get_phase_time 'Phase Time D2H')
        {
            echo "### GUSTAFSON CUDA BASE_SIZE ${BASE_SIZE} ###"
            echo "${CUDA_OUT}"
            echo ""
        } >> "${CUDA_ROOFLINE_LOG}"
        echo "gustafson,1,${BASE_SIZE},${BASE_SIZE},${CUDA_TIME},ok" >> "${GUSTAFSON_CSV}"
        echo "gustafson,0,${BASE_SIZE},${H2D},${SA},${KERNEL},${D2H},${CUDA_TIME}" >> "${CUDA_PHASE_CSV}"
    else
        echo "serial,1,${BASE_SIZE},${BASE_SIZE},NA,skipped_size_cap" >> "${GUSTAFSON_CSV}"
        echo "gustafson,1,${BASE_SIZE},${BASE_SIZE},NA,skipped_size_cap" >> "${GUSTAFSON_CSV}"
    fi

    for THREADS in ${THREAD_CONFIGS}; do
        SCALED_SIZE=$(echo "${BASE_SIZE} ${THREADS}" | awk '{printf "%.0f", $1*$2}')
        if [ "${SCALED_SIZE}" -le "${MAX_FEASIBLE_SIZE}" ]; then
            export OMP_NUM_THREADS=${THREADS}
            OPENMP_OUT=$(./bin/lcs_openmp ${SCALED_SIZE} 0)
            OPENMP_TIME=$(echo "${OPENMP_OUT}" | get_time)
            echo "openmp,${THREADS},${BASE_SIZE},${SCALED_SIZE},${OPENMP_TIME},ok" >> "${GUSTAFSON_CSV}"
        else
            echo "openmp,${THREADS},${BASE_SIZE},${SCALED_SIZE},NA,skipped_size_cap" >> "${GUSTAFSON_CSV}"
        fi
    done
done

echo "Gustafson scaling CSV: ${GUSTAFSON_CSV}"
echo ""

# Phase 3: Nsight Systems profiling.
echo "############################################################"
echo "PHASE 3: PROFILING"
echo "############################################################"
echo ""

PROFILE_DIR="Profile/profiles_${SLURM_JOB_ID:-local}"
mkdir -p ${PROFILE_DIR}

HAVE_TAU=0
if command -v tau_cxx.sh &> /dev/null || [ -n "${TAU_ROOT}" ]; then
    HAVE_TAU=1
fi

if [ ${HAVE_TAU} -eq 1 ]; then
    echo "Building OpenMP with TAU instrumentation..."
    make lcs_openmp_tau 2>/dev/null || {
        echo "TAU build failed - attempting manual compilation..."
        if [ -n "${TAU_ROOT}" ] && [ -f "${TAU_ROOT}/bin/tau_cxx.sh" ]; then
            ${TAU_ROOT}/bin/tau_cxx.sh -O3 -fopenmp -march=native -std=c++17 -DUSE_TAU \
                src/lcs_openmp.cpp -o bin/lcs_openmp_tau
        fi
    }

    if [ -f ./bin/lcs_openmp_tau ]; then
        echo "TAU single-run profiling per thread after warm-ups..."
        export TAU_PROFILE=1
        export TAU_TRACE=0
        for THREADS in ${THREAD_CONFIGS}; do
            run_openmp_warmups "${THREADS}" "${SIZE}"
            export OMP_NUM_THREADS=${THREADS}
            export PROFILEDIR="${PROFILE_DIR}/tau_profiles_t${THREADS}"
            mkdir -p "${PROFILEDIR}"
            TAU_OUTPUT=$(./bin/lcs_openmp_tau ${SIZE} 0)
            TAU_TIME=$(echo "${TAU_OUTPUT}" | get_time)
            echo "  TAU profile (${THREADS} threads): ${TAU_TIME} s"
            echo "${THREADS},${TAU_TIME},${PROFILEDIR}" >> "${TAU_SINGLE_CSV}"
        done
    else
        echo "TAU-instrumented build not available. Skipping TAU profiling in this run."
    fi
fi

if command -v nsys &> /dev/null; then
    run_cuda_warmups "${SIZE}" "profile"
    echo "Running Nsight Systems profiling..."
    nsys profile \
        --trace=cuda,osrt \
        --stats=true \
        --force-overwrite=true \
        --output=${PROFILE_DIR}/lcs_cuda_nsys \
        ./bin/lcs_cuda ${SIZE} 0 | tee "${DATA_DIR}/nsys_single_run_${JOB_TAG}.log"

    NSYS_TOTAL=$(grep -oP 'Time: \K[0-9.]+' "${DATA_DIR}/nsys_single_run_${JOB_TAG}.log" | head -1)
    NSYS_H2D=$(grep -oP 'Phase Time H2D: \K[0-9.]+' "${DATA_DIR}/nsys_single_run_${JOB_TAG}.log" | head -1)
    NSYS_SA=$(grep -oP 'Phase Time SA: \K[0-9.]+' "${DATA_DIR}/nsys_single_run_${JOB_TAG}.log" | head -1)
    NSYS_KERNEL=$(grep -oP 'Phase Time Kernel: \K[0-9.]+' "${DATA_DIR}/nsys_single_run_${JOB_TAG}.log" | head -1)
    NSYS_D2H=$(grep -oP 'Phase Time D2H: \K[0-9.]+' "${DATA_DIR}/nsys_single_run_${JOB_TAG}.log" | head -1)
    {
        echo "### NSYS SINGLE PROFILE RUN ###"
        cat "${DATA_DIR}/nsys_single_run_${JOB_TAG}.log"
        echo ""
    } >> "${CUDA_ROOFLINE_LOG}"
    echo "profile_nsys,1,${SIZE},${NSYS_H2D},${NSYS_SA},${NSYS_KERNEL},${NSYS_D2H},${NSYS_TOTAL}" >> "${CUDA_PHASE_CSV}"

    echo ""
    echo "Profile saved: ${PROFILE_DIR}/lcs_cuda_nsys.nsys-rep"
    echo ""
    echo "To analyze locally:"
    echo "  rsync -av <user>@<login-host>:<repo-dir>/${PROFILE_DIR}/ ."
    echo "  nsys-ui lcs_cuda_nsys.nsys-rep"
else
    echo "nsys not found - skipping profiling"
    echo "To profile manually: nsys profile ./bin/lcs_cuda ${SIZE} 0"
fi

echo ""
echo "Data products for plots/analysis:"
echo "  - ${STRONG_SCALING_CSV}"
echo "  - ${GUSTAFSON_CSV}"
echo "  - ${TAU_SINGLE_CSV}"
echo "  - ${CUDA_PHASE_CSV}"
echo "  - ${CUDA_ROOFLINE_LOG}"

echo ""
echo "============================================================"
echo "  JOB COMPLETED SUCCESSFULLY"
echo "  Finished at: $(date)"
echo "============================================================"

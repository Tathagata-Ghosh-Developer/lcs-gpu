#!/bin/bash
#SBATCH --job-name=LCS_Benchmark
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=lcs_benchmark_%j.txt
#SBATCH --error=lcs_benchmark_err_%j.txt

# Benchmark script: serial/OpenMP/CUDA timing with confidence intervals.

set -e

cd "${LCS_ROOT:-${SLURM_SUBMIT_DIR:-.}}" || { echo "ERROR: Cannot cd to work directory"; exit 1; }

echo "======================================================"
echo "  LCS BENCHMARK - STATISTICAL PERFORMANCE ANALYSIS"
echo "======================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

echo "--- SYSTEM INFO ---"
CORES=$(nproc)
echo "CPU:     $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "CPUs:    ${CORES} available (36 physical cores x 2 threads)"
echo "Memory:  $(free -h | grep Mem | awk '{print $2}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: Waiting for driver..."
echo "======================================================"
echo ""

echo "Building executables..."
make clean 2>/dev/null || true
make all GPU_ARCH=sm_86

# Recover if any target is unexpectedly missing after make all.
if [ ! -x ./bin/lcs_serial ]; then
    echo "lcs_serial missing after make all; rebuilding target..."
    make lcs_serial
fi
if [ ! -x ./bin/lcs_openmp ]; then
    echo "lcs_openmp missing after make all; rebuilding target..."
    make lcs_openmp
fi
if [ ! -x ./bin/lcs_cuda ]; then
    echo "lcs_cuda missing after make all; rebuilding target..."
    make lcs_cuda GPU_ARCH=sm_86
fi

if [ ! -x ./bin/lcs_serial ] || [ ! -x ./bin/lcs_openmp ] || [ ! -x ./bin/lcs_cuda ]; then
    echo "ERROR: Build failed! Missing executables:"
    [ ! -x ./bin/lcs_serial ] && echo "  - lcs_serial"
    [ ! -x ./bin/lcs_openmp ] && echo "  - lcs_openmp"
    [ ! -x ./bin/lcs_cuda ] && echo "  - lcs_cuda"
    ls -la
    exit 1
fi
echo "Build complete. Executables verified."
echo ""

# Configuration
SIZE=100000000
RUNS=30
WARMUP_RUNS=5
SERIAL_RUNS=5

THREAD_CONFIGS="16 32 36 64 72"

echo "======================================================"
echo "  BENCHMARK CONFIGURATION"
echo "======================================================"
echo "  Input Size:       ${SIZE} (100 Million elements)"
echo "  Warm-up runs:     ${WARMUP_RUNS} (not measured)"
echo "  Serial runs:      ${SERIAL_RUNS}"
echo "  Runs per config:  ${RUNS} (for statistical significance)"
echo "  Thread configs:   ${THREAD_CONFIGS}"
echo "  Algorithms:       serial SA-IS+LCP / OpenMP k-mer index / CUDA prefix doubling"
echo "======================================================"
echo ""

export OMP_PROC_BIND=close
export OMP_PLACES=cores

get_time() {
    grep -oP 'Time: \K[0-9.]+' | head -1
}

OMP_RESULTS_FILE="/tmp/omp_results_${SLURM_JOB_ID:-$$}.txt"
CUDA_RESULTS_FILE="/tmp/cuda_results_${SLURM_JOB_ID:-$$}.txt"
SERIAL_RESULTS_FILE="/tmp/serial_results_${SLURM_JOB_ID:-$$}.txt"
> ${OMP_RESULTS_FILE}
> ${CUDA_RESULTS_FILE}
> ${SERIAL_RESULTS_FILE}

# Serial reference benchmark
echo "======================================================"
echo "  SERIAL BENCHMARK - REFERENCE CPU"
echo "  Runs: ${SERIAL_RUNS}"
echo "======================================================"

SERIAL_TIMES=""
for i in $(seq 1 ${SERIAL_RUNS}); do
    printf "  Run %2d/%d: " $i ${SERIAL_RUNS}
    OUTPUT=$(./bin/lcs_serial ${SIZE} 0)
    TIME=$(echo "${OUTPUT}" | get_time)
    SERIAL_TIMES="${SERIAL_TIMES} ${TIME}"
    echo "${TIME} s"
done

SERIAL_STATS=$(echo ${SERIAL_TIMES} | awk '{
    n = NF
    sum = 0
    for (i = 1; i <= n; i++) sum += $i
    mean = sum / n

    sumsq = 0
    for (i = 1; i <= n; i++) sumsq += ($i - mean)^2
    if (n > 1) stddev = sqrt(sumsq / (n - 1)); else stddev = 0
    stderr = stddev / sqrt(n)
    ci95 = 1.96 * stderr

    printf "%.6f %.6f %.6f %.6f", mean, stddev, stderr, ci95
}')

echo "SERIAL ${SERIAL_STATS}" >> ${SERIAL_RESULTS_FILE}
read SERIAL_MEAN SERIAL_STDDEV SERIAL_STDERR SERIAL_CI95 <<< "${SERIAL_STATS}"

echo ""
echo "  Summary (Serial CPU):"
echo "    Mean:     ${SERIAL_MEAN} s"
echo "    Std Dev:  ${SERIAL_STDDEV} s"
echo "    95% CI:   +/-${SERIAL_CI95} s"
echo ""

# CUDA benchmark
echo "======================================================"
echo "  CUDA BENCHMARK - GPU"
echo "  Algorithm: prefix-doubling suffix array + GPU LCP/max kernel"
echo "  Runs: ${RUNS}"
echo "======================================================"

CUDA_TIMES=""
echo "Running ${WARMUP_RUNS} CUDA warm-up runs (not measured)..."
for i in $(seq 1 ${WARMUP_RUNS}); do
    OUTPUT=$(./bin/lcs_cuda ${SIZE} 0)
    TIME=$(echo "${OUTPUT}" | get_time)
    echo "  Warm-up ${i}/${WARMUP_RUNS}: ${TIME} s"
done

for i in $(seq 1 ${RUNS}); do
    printf "  Run %2d/%d: " $i ${RUNS}
    OUTPUT=$(./bin/lcs_cuda ${SIZE} 0)
    TIME=$(echo "${OUTPUT}" | get_time)
    CUDA_TIMES="${CUDA_TIMES} ${TIME}"
    echo "${TIME} s"
done

CUDA_STATS=$(echo ${CUDA_TIMES} | awk '{
    n = NF
    sum = 0
    for (i = 1; i <= n; i++) sum += $i
    mean = sum / n
    
    sumsq = 0
    for (i = 1; i <= n; i++) sumsq += ($i - mean)^2
    stddev = sqrt(sumsq / (n - 1))
    stderr = stddev / sqrt(n)
    ci95 = 1.96 * stderr
    
    printf "%.6f %.6f %.6f %.6f", mean, stddev, stderr, ci95
}')

echo "CUDA ${CUDA_STATS}" >> ${CUDA_RESULTS_FILE}

read CUDA_MEAN CUDA_STDDEV CUDA_STDERR CUDA_CI95 <<< "${CUDA_STATS}"

echo ""
echo "  Summary (CUDA GPU):"
echo "    Mean:     ${CUDA_MEAN} s"
echo "    Std Dev:  ${CUDA_STDDEV} s"
echo "    95% CI:   +/-${CUDA_CI95} s"
echo ""

# OpenMP benchmarks across thread counts
for THREADS in ${THREAD_CONFIGS}; do
    echo "======================================================"
    echo "  OPENMP BENCHMARK - ${THREADS} THREADS"
    echo "  Algorithm: 7-mer hash index + SIMD extension"
    echo "  Runs: ${RUNS}"
    echo "======================================================"
    export OMP_NUM_THREADS=${THREADS}

    echo "Running ${WARMUP_RUNS} warm-up runs (not measured)..."
    for i in $(seq 1 ${WARMUP_RUNS}); do
        OUTPUT=$(./bin/lcs_openmp ${SIZE} 0)
        TIME=$(echo "${OUTPUT}" | get_time)
        echo "  Warm-up ${i}/${WARMUP_RUNS}: ${TIME} s"
    done
    
    TIMES=""
    for i in $(seq 1 ${RUNS}); do
        printf "  Run %2d/%d: " $i ${RUNS}
        OUTPUT=$(./bin/lcs_openmp ${SIZE} 0)
        TIME=$(echo "${OUTPUT}" | get_time)
        TIMES="${TIMES} ${TIME}"
        echo "${TIME} s"
    done
    
    STATS=$(echo ${TIMES} | awk '{
        n = NF
        sum = 0
        for (i = 1; i <= n; i++) sum += $i
        mean = sum / n
        
        sumsq = 0
        for (i = 1; i <= n; i++) sumsq += ($i - mean)^2
        stddev = sqrt(sumsq / (n - 1))
        stderr = stddev / sqrt(n)
        ci95 = 1.96 * stderr
        
        printf "%.6f %.6f %.6f %.6f", mean, stddev, stderr, ci95
    }')
    
    echo "${THREADS} ${STATS}" >> ${OMP_RESULTS_FILE}
    
    read _ MEAN STDDEV STDERR CI95 <<< "${THREADS} ${STATS}"
    echo ""
    echo "  Summary (${THREADS} threads):"
    echo "    Mean:     ${MEAN} s"
    echo "    Std Dev:  ${STDDEV} s"
    echo "    95% CI:   +/-${CI95} s"
    echo ""
done


# Results summary and speedup analysis
echo "======================================================"
echo "  COMPREHENSIVE BENCHMARK SUMMARY"
echo "======================================================"
echo ""
echo "Configuration     | Mean (s)   | Std Dev  | 95% CI      | Speedup vs GPU"
echo "------------------|------------|----------|-------------|---------------"

SERIAL_SPEEDUP=$(echo "${SERIAL_MEAN} ${CUDA_MEAN}" | awk '{printf "%.2f", $1/$2}')
printf "Serial CPU        | %10.4f | %8.4f | +/-%-8.4f | %sx\n" \
    ${SERIAL_MEAN} ${SERIAL_STDDEV} ${SERIAL_CI95} ${SERIAL_SPEEDUP}

BEST_OMP_MEAN="999999"
BEST_OMP_THREADS=""

while read line; do
    THREADS=$(echo $line | awk '{print $1}')
    MEAN=$(echo $line | awk '{print $2}')
    STDDEV=$(echo $line | awk '{print $3}')
    CI95=$(echo $line | awk '{print $5}')
    
    SPEEDUP=$(echo "${MEAN} ${CUDA_MEAN}" | awk '{printf "%.2f", $1/$2}')
    
    printf "OpenMP %-3d threads | %10.4f | %8.4f | +/-%-8.4f | %sx\n" \
           ${THREADS} ${MEAN} ${STDDEV} ${CI95} ${SPEEDUP}
    
    IS_BETTER=$(echo "${MEAN} ${BEST_OMP_MEAN}" | awk '{print ($1 < $2) ? 1 : 0}')
    if [ "${IS_BETTER}" -eq 1 ]; then
        BEST_OMP_MEAN="${MEAN}"
        BEST_OMP_THREADS="${THREADS}"
    fi
done < ${OMP_RESULTS_FILE}

printf "CUDA GPU          | %10.4f | %8.4f | +/-%-8.4f | 1.00x (baseline)\n" \
       ${CUDA_MEAN} ${CUDA_STDDEV} ${CUDA_CI95}

echo ""
echo "======================================================"
echo "  KEY SPEEDUP METRICS"
echo "======================================================"

# GPU speedup over best OpenMP across all configured thread counts
if [ -n "${BEST_OMP_THREADS}" ]; then
    SPEEDUP_GPU_BEST=$(echo "${BEST_OMP_MEAN} ${CUDA_MEAN}" | awk '{printf "%.2f", $1/$2}')
    echo ""
    echo "  >>> GPU Speedup over Best OpenMP (${BEST_OMP_THREADS} threads): ${SPEEDUP_GPU_BEST}x <<<"
    echo ""
fi

# All thread configs vs CUDA
echo "  Detailed Speedups:"
while read line; do
    THREADS=$(echo $line | awk '{print $1}')
    MEAN=$(echo $line | awk '{print $2}')
    SPEEDUP=$(echo "${MEAN} ${CUDA_MEAN}" | awk '{printf "%.2f", $1/$2}')
    echo "    GPU over OpenMP (${THREADS} threads): ${SPEEDUP}x"
done < ${OMP_RESULTS_FILE}

echo ""
echo "======================================================"
echo "  OPENMP SCALING ANALYSIS"
echo "======================================================"

BASELINE_MEAN=$(head -1 ${OMP_RESULTS_FILE} | awk '{print $2}')
BASELINE_THREADS=$(head -1 ${OMP_RESULTS_FILE} | awk '{print $1}')

while read line; do
    THREADS=$(echo $line | awk '{print $1}')
    MEAN=$(echo $line | awk '{print $2}')
    SPEEDUP=$(echo "${BASELINE_MEAN} ${MEAN}" | awk '{printf "%.2f", $1/$2}')
    IDEAL=$(echo "${THREADS} ${BASELINE_THREADS}" | awk '{printf "%.2f", $1/$2}')
    EFFICIENCY=$(echo "${SPEEDUP} ${IDEAL}" | awk '{printf "%.1f", ($1/$2)*100}')
    echo "  ${THREADS} threads: ${SPEEDUP}x speedup (ideal: ${IDEAL}x, efficiency: ${EFFICIENCY}%)"
done < ${OMP_RESULTS_FILE}

echo ""
echo "======================================================"
echo "  STATISTICAL NOTES"
echo "======================================================"
echo "  - ${RUNS} runs per configuration for statistical significance"
echo "  - By Central Limit Theorem, sample mean approaches population mean"
echo "  - 95% Confidence Interval: Mean +/- 1.96 x Standard Error"
echo "  - Standard Error = StdDev / sqrt(n)"
echo ""

rm -f ${OMP_RESULTS_FILE} ${CUDA_RESULTS_FILE} ${SERIAL_RESULTS_FILE}

echo "======================================================"
echo "  Benchmark completed at: $(date)"
echo "======================================================"

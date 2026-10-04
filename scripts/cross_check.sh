#!/bin/bash
# Seeded cross-check of the benchmark (non-debug) code paths.
#
# slurm_debug.sh runs every solver in debug mode (N=2000), where the OpenMP
# solver switches to an exact O(N^2) diagonal scan. This script instead runs
# the fast paths on shared seeded inputs and compares them with the SA-IS
# serial reference. CUDA is included when bin/lcs_cuda exists.
#
# Usage: scripts/cross_check.sh [sizes...]   (default: 1000 100000 1000000 10000000)

set -euo pipefail

cd "$(dirname "$0")/.."

SIZES=${*:-"1000 100000 1000000 10000000"}
SEEDS="1 42 2026"
THREADS=${OMP_NUM_THREADS:-8}

result() { grep -oP 'Result: \K[0-9]+' | head -1; }

for exe in bin/lcs_serial bin/lcs_openmp; do
    [ -x "$exe" ] || { echo "missing $exe - run 'make cpu' first"; exit 1; }
done
HAVE_CUDA=0
[ -x bin/lcs_cuda ] && HAVE_CUDA=1

fail=0
printf "%-10s %-6s %-7s %-7s %-7s %s\n" "size" "seed" "serial" "openmp" "cuda" "status"
for n in ${SIZES}; do
    for seed in ${SEEDS}; do
        s=$(./bin/lcs_serial "$n" 0 "$seed" | result)
        o=$(OMP_NUM_THREADS=${THREADS} ./bin/lcs_openmp "$n" 0 "$seed" | result)
        c="-"
        [ ${HAVE_CUDA} -eq 1 ] && c=$(./bin/lcs_cuda "$n" 0 "$seed" | result)
        status=OK
        [ "$o" = "$s" ] || status=MISMATCH
        [ ${HAVE_CUDA} -eq 0 ] || [ "$c" = "$s" ] || status=MISMATCH
        [ "$status" = OK ] || fail=1
        printf "%-10s %-6s %-7s %-7s %-7s %s\n" "$n" "$seed" "$s" "$o" "$c" "$status"
    done
done

if [ ${fail} -ne 0 ]; then
    echo "cross-check FAILED"
    exit 1
fi
echo "cross-check passed"

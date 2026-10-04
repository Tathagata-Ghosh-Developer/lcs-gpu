# Build targets for serial/OpenMP/CUDA longest-common-substring executables.

CXX := g++
NVCC := nvcc

SRC_DIR := src
BIN_DIR := bin

# Host compiler flags
CXXFLAGS := -O3 -fopenmp -march=native -std=c++17
CXXFLAGS_DEBUG := -O0 -g -fopenmp -std=c++17

# CUDA flags (default GPU architecture: sm_86)
GPU_ARCH ?= sm_86
NVCCFLAGS := -O3 -arch=$(GPU_ARCH) -lineinfo --use_fast_math -std=c++17 \
             -Xcompiler -fopenmp \
             --extended-lambda --expt-relaxed-constexpr \
             -maxrregcount=32
NVCCFLAGS_DEBUG := -O0 -g -G -arch=$(GPU_ARCH) -std=c++17 -Xcompiler -fopenmp \
                   --extended-lambda --expt-relaxed-constexpr

# NVTX support for CUDA profiling builds
NVTX_FLAGS := -DUSE_NVTX
NVTX_LDFLAGS := -lnvToolsExt

# TAU support for OpenMP profiling builds
TAU_MAKEFILE ?= $(TAU_ROOT)/lib/Makefile.tau-papi-ompt-pdt-openmp
TAU_CXX = tau_cxx.sh
TAU_FLAGS := -DUSE_TAU

SERIAL := $(BIN_DIR)/lcs_serial
OPENMP := $(BIN_DIR)/lcs_openmp
CUDA   := $(BIN_DIR)/lcs_cuda
OPENMP_TAU := $(BIN_DIR)/lcs_openmp_tau

.PHONY: all cpu clean help verify profile-tau lcs_serial lcs_openmp lcs_cuda lcs_openmp_tau

all: $(SERIAL) $(OPENMP) $(CUDA)
	@echo ""
	@echo "Build complete! Usage:"
	@echo "  ./bin/lcs_serial <size> [0|1] [seed]"
	@echo "  ./bin/lcs_openmp <size> [0|1]   # 0=benchmark, 1=debug"
	@echo "  ./bin/lcs_cuda <size> [0|1]"

# CPU-only build for machines without the CUDA toolkit
cpu: $(SERIAL) $(OPENMP)

lcs_serial: $(SERIAL)
lcs_openmp: $(OPENMP)
lcs_cuda: $(CUDA)
lcs_openmp_tau: $(OPENMP_TAU)

$(BIN_DIR):
	mkdir -p $@

$(SERIAL): $(SRC_DIR)/lcs_serial.cpp $(SRC_DIR)/random_text.hpp | $(BIN_DIR)
	@echo "Building Serial checker (SA-IS + LCP)..."
	$(CXX) $(CXXFLAGS) $< -o $@
	@test -f $@ && echo "  [OK] $@" || (echo "  [FAIL] $@" && exit 1)

$(OPENMP): $(SRC_DIR)/lcs_openmp.cpp $(SRC_DIR)/random_text.hpp | $(BIN_DIR)
	@echo "Building OpenMP (k-mer hash index + SIMD extension)..."
	$(CXX) $(CXXFLAGS) $< -o $@
	@test -f $@ && echo "  [OK] $@" || (echo "  [FAIL] $@" && exit 1)

$(CUDA): $(SRC_DIR)/lcs_cuda.cu $(SRC_DIR)/random_text.hpp | $(BIN_DIR)
	@echo "Building CUDA (Prefix Doubling + Thrust + GPU LCP Reduction)..."
	$(NVCC) $(NVCCFLAGS) $(NVTX_FLAGS) $< -o $@ $(NVTX_LDFLAGS) 2>/dev/null || \
	$(NVCC) $(NVCCFLAGS) $< -o $@
	@test -f $@ && echo "  [OK] $@" || (echo "  [FAIL] $@" && exit 1)

$(OPENMP_TAU): $(SRC_DIR)/lcs_openmp.cpp $(SRC_DIR)/random_text.hpp | $(BIN_DIR)
	@echo "Building OpenMP with TAU instrumentation..."
	$(TAU_CXX) $(CXXFLAGS) $(TAU_FLAGS) $< -o $@
	@test -f $@ && echo "  [OK] $@ (TAU-instrumented)" || (echo "  [FAIL] $@" && exit 1)

profile-tau: $(OPENMP_TAU)
	@echo "Run: ./$(OPENMP_TAU) <size> then analyze with pprof/paraprof"

# Quick correctness check against seeded serial reference
verify: all
	@echo ""
	@echo "Running correctness verification against serial (N=2000)..."
	@SEED=$$(date +%s); \
	SERIAL_RESULT=$$(./$(SERIAL) 100000000 1 $$SEED | grep -oP 'Result: \K[0-9]+' | head -1); \
	OPENMP_RESULT=$$(./$(OPENMP) 100000000 1 $$SEED | grep -oP 'Result: \K[0-9]+' | head -1); \
	CUDA_RESULT=$$(./$(CUDA) 100000000 1 $$SEED | grep -oP 'Result: \K[0-9]+' | head -1); \
	echo "Seed: $$SEED"; \
	echo "Serial: $$SERIAL_RESULT | OpenMP: $$OPENMP_RESULT | CUDA: $$CUDA_RESULT"; \
	[ "$$OPENMP_RESULT" = "$$SERIAL_RESULT" ] || (echo "[FAIL] OpenMP mismatch" && exit 1); \
	[ "$$CUDA_RESULT" = "$$SERIAL_RESULT" ] || (echo "[FAIL] CUDA mismatch" && exit 1)
	@echo ""
	@echo "All tests PASSED!"

clean:
	rm -rf $(BIN_DIR)
	rm -f *.nsys-rep *.ncu-rep *.sqlite
	rm -f *_output.txt
	rm -rf profile.* MULTI__* tauprofile.*

help:
	@echo "============================================================"
	@echo "LCS - Longest Common Substring of two random strings"
	@echo "============================================================"
	@echo ""
	@echo "Implementations:"
	@echo "  - Serial: SA-IS suffix array + Kasai LCP, O(N) (reference)"
	@echo "  - OpenMP: 7-mer hash index + AVX2 match extension"
	@echo "  - CUDA:   prefix-doubling suffix array (Thrust radix sort)"
	@echo "            + fused LCP / atomicMax kernel"
	@echo ""
	@echo "Build Targets:"
	@echo "  make all      - Build lcs_serial, lcs_openmp and lcs_cuda"
	@echo "  make cpu      - Build lcs_serial and lcs_openmp only (no nvcc)"
	@echo "  make verify   - Build and run correctness tests"
	@echo "  make clean    - Remove all executables"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  ./bin/lcs_serial <size> [mode] [seed]"
	@echo "  ./bin/lcs_openmp <size> [mode] [seed]"
	@echo "  ./bin/lcs_cuda <size> [mode] [seed]"
	@echo ""
	@echo "Examples:"
	@echo "  ./bin/lcs_openmp 100000000 0     # Benchmark with 100M elements"
	@echo "  ./bin/lcs_serial 100000000 1 42  # Serial reference on debug-size input"
	@echo "  ./bin/lcs_cuda 100000000 1 42    # Same seeded input for correctness check"
	@echo ""
	@echo "GPU Architecture: $(GPU_ARCH)"
	@echo "============================================================"

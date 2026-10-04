#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>

#include "random_text.hpp"

namespace Config {
    constexpr int BLOCK_SIZE = 256;
    constexpr std::size_t DEBUG_SIZE = 2000;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void generate_seeded_text(std::vector<char>& data, std::size_t size, uint64_t seed) {
    data.resize(size);
    for (std::size_t i = 0; i < size; ++i) {
        uint64_t rnd = splitmix64(seed + i);
        data[i] = static_cast<char>('A' + static_cast<int>(rnd % 26ULL));
    }
}

__global__ void pack_keys_kernel(const int* __restrict__ rank,
                                 unsigned long long* __restrict__ keys,
                                 int n,
                                 int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long r1 = static_cast<unsigned long long>(rank[idx]);
        unsigned long long r2 = (idx + k < n) ? static_cast<unsigned long long>(rank[idx + k]) : 0ULL;
        keys[idx] = (r1 << 32) | r2;
    }
}

__global__ void lcp_and_reduce_kernel(const unsigned char* __restrict__ text,
                                      const int* __restrict__ sa,
                                      int n,
                                      int split_idx,
                                      int* __restrict__ d_max) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int idx = tid + 1; idx < n; idx += stride) {
        int sa_curr = sa[idx];
        int sa_prev = sa[idx - 1];
        bool curr_in_x = (sa_curr < split_idx);
        bool prev_in_x = (sa_prev < split_idx);

        if (curr_in_x == prev_in_x) continue;

        int len = 0;
        while (sa_curr + len < n && sa_prev + len < n && text[sa_curr + len] == text[sa_prev + len]) {
            ++len;
        }
        if (len > 0) atomicMax(d_max, len);
    }
}

int get_optimal_blocks() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (" << prop.multiProcessorCount << " SMs)\n";
    int blocks = (prop.maxThreadsPerMultiProcessor / Config::BLOCK_SIZE) * prop.multiProcessorCount;
    return std::max(1, std::min(blocks, 4096));
}

int solve_gpu(const std::vector<char>& h_x,
              const std::vector<char>& h_y,
              int num_blocks,
              double& h2d_s,
              double& sa_s,
              double& kernel_s,
              double& d2h_s) {
    int n_x = static_cast<int>(h_x.size());
    int n_y = static_cast<int>(h_y.size());
    int n = n_x + 1 + n_y + 1;

    std::vector<unsigned char> h_text(n);
    std::memcpy(h_text.data(), h_x.data(), static_cast<std::size_t>(n_x));
    h_text[n_x] = 1;
    std::memcpy(h_text.data() + n_x + 1, h_y.data(), static_cast<std::size_t>(n_y));
    h_text[n - 1] = 0;

    unsigned char* d_text = nullptr;
    int* d_sa = nullptr;
    int* d_rank = nullptr;
    int* d_new_rank = nullptr;
    int* d_max = nullptr;
    unsigned long long* d_keys = nullptr;

    CUDA_CHECK(cudaMalloc(&d_text, static_cast<std::size_t>(n) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_sa, static_cast<std::size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rank, static_cast<std::size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_rank, static_cast<std::size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keys, static_cast<std::size_t>(n) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(int)));

    int block_size = Config::BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;

    auto t_h2d_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_text, h_text.data(), static_cast<std::size_t>(n) * sizeof(unsigned char), cudaMemcpyHostToDevice));
    thrust::sequence(thrust::device, d_sa, d_sa + n);
    thrust::copy(thrust::device, d_text, d_text + n, d_rank);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_h2d_end = std::chrono::high_resolution_clock::now();

    auto t_sa_start = std::chrono::high_resolution_clock::now();
    thrust::device_ptr<unsigned long long> dev_keys(d_keys);
    thrust::device_ptr<int> dev_new_rank(d_new_rank);

    for (int k = 1; k < n; k <<= 1) {
        thrust::sequence(thrust::device, d_sa, d_sa + n);

        pack_keys_kernel<<<grid_size, block_size>>>(d_rank, d_keys, n, k);
        CUDA_CHECK(cudaGetLastError());

        thrust::sort_by_key(thrust::device, dev_keys, dev_keys + n, d_sa);
        thrust::adjacent_difference(
            thrust::device,
            dev_keys,
            dev_keys + n,
            dev_new_rank,
            thrust::not_equal_to<unsigned long long>());
        int one = 1;
        CUDA_CHECK(cudaMemcpy(d_new_rank, &one, sizeof(int), cudaMemcpyHostToDevice));
        thrust::inclusive_scan(thrust::device, dev_new_rank, dev_new_rank + n, dev_new_rank);
        thrust::scatter(thrust::device, dev_new_rank, dev_new_rank + n, d_sa, d_rank);

        int max_rank = 0;
        CUDA_CHECK(cudaMemcpy(&max_rank, d_new_rank + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
        if (max_rank == n) break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_sa_end = std::chrono::high_resolution_clock::now();

    auto t_kernel_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemset(d_max, 0, sizeof(int)));
    lcp_and_reduce_kernel<<<num_blocks, block_size>>>(d_text, d_sa, n, n_x, d_max);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_kernel_end = std::chrono::high_resolution_clock::now();

    int result = 0;
    auto t_d2h_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(&result, d_max, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_d2h_end = std::chrono::high_resolution_clock::now();

    h2d_s = std::chrono::duration<double>(t_h2d_end - t_h2d_start).count();
    sa_s = std::chrono::duration<double>(t_sa_end - t_sa_start).count();
    kernel_s = std::chrono::duration<double>(t_kernel_end - t_kernel_start).count();
    d2h_s = std::chrono::duration<double>(t_d2h_end - t_d2h_start).count();

    cudaFree(d_text);
    cudaFree(d_sa);
    cudaFree(d_rank);
    cudaFree(d_new_rank);
    cudaFree(d_keys);
    cudaFree(d_max);

    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size> [debug: 0|1] [seed]\n";
        return 1;
    }

    std::size_t n = std::stoull(argv[1]);
    bool debug_mode = (argc > 2) && (std::stoi(argv[2]) == 1);
    bool has_seed = argc > 3;
    uint64_t seed = has_seed ? std::stoull(argv[3]) : 0ULL;

    if (debug_mode) n = Config::DEBUG_SIZE;

    int num_blocks = get_optimal_blocks();
    std::cout << "Algorithm: Prefix Doubling + Thrust Radix Sort + GPU LCP Reduce\n";
    std::cout << "Size: " << n << " elements\n";

    std::vector<char> x, y;
    if (has_seed) {
        generate_seeded_text(x, n, seed ^ 0x12345678ULL);
        generate_seeded_text(y, n, seed ^ 0x87654321ULL);
    } else {
        generate_random_text(x, n);
        generate_random_text(y, n);
    }

    auto total_start = std::chrono::high_resolution_clock::now();
    double h2d_s = 0.0;
    double sa_s = 0.0;
    double kernel_s = 0.0;
    double d2h_s = 0.0;
    int result = solve_gpu(x, y, num_blocks, h2d_s, sa_s, kernel_s, d2h_s);
    auto total_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> total_diff = total_end - total_start;
    std::cout << "Phase Time H2D: " << h2d_s << " s\n";
    std::cout << "Phase Time SA: " << sa_s << " s\n";
    std::cout << "Phase Time Kernel: " << kernel_s << " s\n";
    std::cout << "Phase Time D2H: " << d2h_s << " s\n";
    std::cout << "Time: " << total_diff.count() << " s\n";
    std::cout << "Result: " << result << "\n";
    return 0;
}

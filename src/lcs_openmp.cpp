#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <omp.h>

#include "random_text.hpp"

#ifdef USE_TAU
#include <TAU.h>
#define TAU_PHASE_START(name) TAU_PHASE_CREATE_DYNAMIC(tau_phase_##__LINE__, name, "", TAU_DEFAULT)
#define TAU_PHASE_STOP(name)
#else
#define TAU_PHASE_START(name)
#define TAU_PHASE_STOP(name)
#endif

using IndexT = int32_t;

namespace Config {
    constexpr std::size_t DEBUG_SIZE = 2000;
    constexpr int KMER_LEN = 7;
    constexpr uint64_t POS_MASK_27 = (1ULL << 27) - 1ULL;
}

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void generate_seeded_text(std::vector<char>& data, std::size_t size, uint64_t seed) {
    data.resize(size);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < size; ++i) {
        uint64_t rnd = splitmix64(seed + i);
        data[i] = static_cast<char>('A' + static_cast<int>(rnd % 26ULL));
    }
}

static inline int lcp_simd(const unsigned char* a, const unsigned char* b, int max_len) {
    int i = 0;
#if defined(__AVX2__)
    constexpr uint32_t full = 0xFFFFFFFFu;
    for (; i + 32 <= max_len; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i cmp = _mm256_cmpeq_epi8(va, vb);
        uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(cmp));
        if (mask != full) {
            return i + __builtin_ctz(~mask);
        }
    }
#endif
    while (i < max_len && a[i] == b[i]) ++i;
    return i;
}

static inline int lcs_diagonal_exact_small(const std::vector<unsigned char>& x, const std::vector<unsigned char>& y) {
    int n = static_cast<int>(x.size());
    int m = static_cast<int>(y.size());
    int best = 0;

    for (int d = -(m - 1); d < n; ++d) {
        int i0 = std::max(0, d);
        int i1 = std::min(n, m + d);
        int run = 0;
        for (int i = i0; i < i1; ++i) {
            int j = i - d;
            if (x[i] == y[j]) {
                ++run;
                if (run > best) best = run;
            } else {
                run = 0;
            }
        }
    }
    return best;
}

static inline uint64_t next_pow2_u64(uint64_t x) {
    if (x <= 1) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

static inline uint64_t encode_kmer(const unsigned char* p, int k) {
    uint64_t code = 0;
    for (int i = 0; i < k; ++i) {
        code = (code << 5) | static_cast<uint64_t>(p[i] - 'A');
    }
    return code;
}

class LCS_OpenMP_Fast {
    std::vector<unsigned char> x;
    std::vector<unsigned char> y;
    bool safe_mode;

public:
    LCS_OpenMP_Fast(const std::vector<char>& X, const std::vector<char>& Y, bool debug_mode)
        : x(X.begin(), X.end()), y(Y.begin(), Y.end()), safe_mode(debug_mode) {}

    int solve() {
        if (safe_mode) {
            TAU_PHASE_START("OpenMP: Debug Exact Diagonal");
            int ans = lcs_diagonal_exact_small(x, y);
            TAU_PHASE_STOP("OpenMP: Debug Exact Diagonal");
            return ans;
        }

        TAU_PHASE_START("OpenMP: Kmer Index + SIMD Extend");

        const int n = static_cast<int>(x.size());
        const int m = static_cast<int>(y.size());
        const int k = Config::KMER_LEN;

        if (n < k || m < k) {
            int ans = lcs_diagonal_exact_small(x, y);
            TAU_PHASE_STOP("OpenMP: Kmer Index + SIMD Extend");
            return ans;
        }

        const int nx = n - k + 1;
        const int my = m - k + 1;

        std::vector<uint64_t> xcodes(static_cast<std::size_t>(nx));
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < nx; ++i) {
            xcodes[i] = encode_kmer(x.data() + i, k);
        }

        uint64_t cap = next_pow2_u64(static_cast<uint64_t>(nx) * 2ULL);
        if (cap < (1ULL << 20)) cap = (1ULL << 20);
        std::vector<uint64_t> table(static_cast<std::size_t>(cap), 0ULL);
        const uint64_t mask = cap - 1ULL;

        for (int i = 0; i < nx; ++i) {
            uint64_t key = xcodes[static_cast<std::size_t>(i)] + 1ULL;
            uint64_t entry = (key << 27) | static_cast<uint64_t>(i);
            uint64_t slot = splitmix64(key) & mask;
            while (table[static_cast<std::size_t>(slot)] != 0ULL) {
                slot = (slot + 1ULL) & mask;
            }
            table[static_cast<std::size_t>(slot)] = entry;
        }

        int global_best = 0;

        #pragma omp parallel for schedule(dynamic, 32768) reduction(max:global_best)
        for (int j = 0; j < my; ++j) {
            uint64_t code = encode_kmer(y.data() + j, k);
            uint64_t key = code + 1ULL;
            uint64_t slot = splitmix64(key) & mask;

            int local_best = 0;
            while (true) {
                uint64_t entry = table[static_cast<std::size_t>(slot)];
                if (entry == 0ULL) break;

                uint64_t ekey = entry >> 27;
                if (ekey == key) {
                    int i = static_cast<int>(entry & Config::POS_MASK_27);

                    int fwd_limit = std::min(n - (i + k), m - (j + k));
                    int fwd = (fwd_limit > 0) ? lcp_simd(x.data() + i + k, y.data() + j + k, fwd_limit) : 0;

                    int bi = i;
                    int bj = j;
                    int back = 0;
                    while (bi > 0 && bj > 0 && x[bi - 1] == y[bj - 1]) {
                        --bi;
                        --bj;
                        ++back;
                    }

                    int len = back + k + fwd;
                    if (len > local_best) local_best = len;
                }
                slot = (slot + 1ULL) & mask;
            }

            if (local_best > global_best) global_best = local_best;
        }

        TAU_PHASE_STOP("OpenMP: Kmer Index + SIMD Extend");

        if (global_best == 0) {
            TAU_PHASE_START("OpenMP: Fallback Small Exact");
            int ans = lcs_diagonal_exact_small(x, y);
            TAU_PHASE_STOP("OpenMP: Fallback Small Exact");
            return ans;
        }

        return global_best;
    }
};

int main(int argc, char** argv) {
#ifdef USE_TAU
    TAU_PROFILE_SET_NODE(0);
#endif

    omp_set_dynamic(0);
    omp_set_schedule(omp_sched_dynamic, 4096);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size> [debug: 0|1] [seed]\n";
        return 1;
    }

    std::size_t n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    bool debug_mode = (argc > 2) && (std::atoi(argv[2]) == 1);
    bool has_seed = argc > 3;
    uint64_t seed = has_seed ? static_cast<uint64_t>(std::strtoull(argv[3], nullptr, 10)) : 0ULL;

    if (debug_mode) n = Config::DEBUG_SIZE;

    std::cout << "Threads: " << omp_get_max_threads() << "\n";
    std::cout << "Algorithm: Parallel K-mer Index + SIMD Extend (Debug exact mode)\n";
    std::cout << "Size: " << n << " elements\n";

    std::vector<char> X, Y;
    if (has_seed) {
        generate_seeded_text(X, n, seed ^ 0x12345678ULL);
        generate_seeded_text(Y, n, seed ^ 0x87654321ULL);
    } else {
        generate_random_text(X, n);
        generate_random_text(Y, n);
    }

    auto start = std::chrono::high_resolution_clock::now();
    LCS_OpenMP_Fast solver(X, Y, debug_mode);
    int result = solver.solve();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " s\n";
    std::cout << "Result: " << result << "\n";
    return 0;
}

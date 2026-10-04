#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "random_text.hpp"

namespace Config {
    constexpr std::size_t DEBUG_SIZE = 2000;
}

namespace SAIS {
    unsigned char mask[] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
    #define tget(i) ((t[(i) / 8] & mask[(i) % 8]) ? 1 : 0)
    #define tset(i, b) t[(i) / 8] = (b) ? (mask[(i) % 8] | t[(i) / 8]) : ((~mask[(i) % 8]) & t[(i) / 8])
    #define chr(i) (cs == sizeof(int) ? ((int*)s)[i] : ((unsigned char*)s)[i])
    #define isLMS(i) (i > 0 && tget(i) && !tget(i - 1))

    void getBuckets(unsigned char* s, int* bkt, int n, int K, int cs, bool end) {
        int sum = 0;
        for (int i = 0; i <= K; i++) bkt[i] = 0;
        for (int i = 0; i < n; i++) bkt[chr(i)]++;
        for (int i = 0; i <= K; i++) {
            sum += bkt[i];
            bkt[i] = end ? sum : sum - bkt[i];
        }
    }

    void induceSAl(unsigned char* t, int* SA, unsigned char* s, int* bkt, int n, int K, int cs, bool end) {
        getBuckets(s, bkt, n, K, cs, end);
        for (int i = 0; i < n; i++) {
            int j = SA[i] - 1;
            if (j >= 0 && !tget(j)) SA[bkt[chr(j)]++] = j;
        }
    }

    void induceSAs(unsigned char* t, int* SA, unsigned char* s, int* bkt, int n, int K, int cs, bool end) {
        getBuckets(s, bkt, n, K, cs, end);
        for (int i = n - 1; i >= 0; i--) {
            int j = SA[i] - 1;
            if (j >= 0 && tget(j)) SA[--bkt[chr(j)]] = j;
        }
    }

    void SA_IS(unsigned char* s, int* SA, int n, int K, int cs) {
        unsigned char* t = static_cast<unsigned char*>(std::malloc(n / 8 + 1));
        int* bkt;

        tset(n - 2, 0);
        tset(n - 1, 1);
        for (int i = n - 3; i >= 0; i--) {
            tset(i, (chr(i) < chr(i + 1) || (chr(i) == chr(i + 1) && tget(i + 1))));
        }

        bkt = static_cast<int*>(std::malloc(sizeof(int) * (K + 1)));
        getBuckets(s, bkt, n, K, cs, true);
        for (int i = 0; i < n; i++) SA[i] = -1;
        for (int i = 1; i < n; i++) {
            if (isLMS(i)) SA[--bkt[chr(i)]] = i;
        }

        induceSAl(t, SA, s, bkt, n, K, cs, false);
        induceSAs(t, SA, s, bkt, n, K, cs, true);
        std::free(bkt);

        int n1 = 0;
        for (int i = 0; i < n; i++) {
            if (isLMS(SA[i])) SA[n1++] = SA[i];
        }

        for (int i = n1; i < n; i++) SA[i] = -1;
        int name = 0, prev = -1;
        for (int i = 0; i < n1; i++) {
            int pos = SA[i];
            bool diff = false;
            for (int d = 0; d < n; d++) {
                if (prev == -1 || chr(pos + d) != chr(prev + d) || tget(pos + d) != tget(prev + d)) {
                    diff = true;
                    break;
                }
                if (d > 0 && (isLMS(pos + d) || isLMS(prev + d))) break;
            }
            if (diff) {
                name++;
                prev = pos;
            }
            pos = (pos % 2 == 0) ? pos / 2 : (pos - 1) / 2;
            SA[n1 + pos] = name - 1;
        }

        for (int i = n - 1, j = n - 1; i >= n1; i--) {
            if (SA[i] >= 0) SA[j--] = SA[i];
        }

        int* SA1 = SA;
        int* s1 = SA + n - n1;
        if (name < n1) SA_IS(reinterpret_cast<unsigned char*>(s1), SA1, n1, name - 1, sizeof(int));
        else for (int i = 0; i < n1; i++) SA1[s1[i]] = i;

        bkt = static_cast<int*>(std::malloc(sizeof(int) * (K + 1)));
        getBuckets(s, bkt, n, K, cs, true);
        for (int i = 1, j = 0; i < n; i++) if (isLMS(i)) s1[j++] = i;
        for (int i = 0; i < n1; i++) SA[i] = s1[SA[i]];
        for (int i = n1; i < n; i++) SA[i] = -1;
        for (int i = n1 - 1; i >= 0; i--) {
            int j = SA[i];
            SA[i] = -1;
            SA[--bkt[chr(j)]] = j;
        }

        induceSAl(t, SA, s, bkt, n, K, cs, false);
        induceSAs(t, SA, s, bkt, n, K, cs, true);

        std::free(bkt);
        std::free(t);
    }
}

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

std::vector<int> build_lcp(const std::vector<int>& text, const std::vector<int>& sa) {
    int n = static_cast<int>(text.size());
    std::vector<int> rank(n);
    for (int i = 0; i < n; i++) rank[sa[i]] = i;

    std::vector<int> lcp(n, 0);
    int h = 0;
    for (int i = 0; i < n; i++) {
        if (rank[i] == 0) continue;
        int j = sa[rank[i] - 1];
        while (i + h < n && j + h < n && text[i + h] == text[j + h]) ++h;
        lcp[rank[i]] = h;
        if (h > 0) --h;
    }
    return lcp;
}

int solve_serial(const std::vector<char>& X, const std::vector<char>& Y) {
    std::size_t n = X.size() + Y.size() + 2;
    std::vector<int> text(n);

    std::size_t idx = 0;
    for (char c : X) text[idx++] = static_cast<unsigned char>(c);
    text[idx++] = 1;
    for (char c : Y) text[idx++] = static_cast<unsigned char>(c);
    text[idx++] = 0;

    std::vector<int> sa(n);
    SAIS::SA_IS(reinterpret_cast<unsigned char*>(text.data()), sa.data(), static_cast<int>(n), 128, sizeof(int));
    std::vector<int> lcp = build_lcp(text, sa);

    int max_len = 0;
    int split = static_cast<int>(X.size());
    for (std::size_t i = 1; i < n; ++i) {
        bool a = sa[i] < split;
        bool b = sa[i - 1] < split;
        if (a != b) max_len = std::max(max_len, lcp[i]);
    }
    return max_len;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size> [debug: 0|1] [seed]\n";
        return 1;
    }

    std::size_t n = std::stoull(argv[1]);
    bool debug_mode = (argc > 2) && (std::stoi(argv[2]) == 1);
    bool has_seed = argc > 3;
    uint64_t seed = has_seed ? std::stoull(argv[3]) : 0ULL;

    if (debug_mode) n = Config::DEBUG_SIZE;

    std::vector<char> X, Y;
    if (has_seed) {
        generate_seeded_text(X, n, seed ^ 0x12345678ULL);
        generate_seeded_text(Y, n, seed ^ 0x87654321ULL);
    } else {
        generate_random_text(X, n);
        generate_random_text(Y, n);
    }

    auto start = std::chrono::high_resolution_clock::now();
    int result = solve_serial(X, Y);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " s\n";
    std::cout << "Result: " << result << "\n";
    return 0;
}

#ifndef RANDOM_TEXT_HPP
#define RANDOM_TEXT_HPP

#include <cstddef>   // size_t
#include <random>    // mt19937, uniform_int_distribution
#include <omp.h>     // OpenMP

// Template: Works with any container that has resize() and operator[]
// CharT: Character type (char, unsigned char, etc.)
template <typename Container, typename CharT = char>
void generate_random_text(Container& data, std::size_t size, 
                          CharT minChar = 'A', CharT maxChar = 'Z') {
    data.resize(size);

    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() ^ static_cast<unsigned>(omp_get_thread_num()));
        std::uniform_int_distribution<int> dis(minChar, maxChar);

        #pragma omp for
        for (std::size_t i = 0; i < size; ++i) {
            data[i] = static_cast<CharT>(dis(gen));
        }
    }
}

// Overload for raw pointer (used by CUDA pinned memory)
template <typename CharT = char>
void generate_random_text(CharT* data, std::size_t size,
                          CharT minChar = 'A', CharT maxChar = 'Z') {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() ^ static_cast<unsigned>(omp_get_thread_num()));
        std::uniform_int_distribution<int> dis(minChar, maxChar);

        #pragma omp for
        for (std::size_t i = 0; i < size; ++i) {
            data[i] = static_cast<CharT>(dis(gen));
        }
    }
}

#endif

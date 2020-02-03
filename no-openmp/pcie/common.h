#include <cuda_runtime.h>
#include <stdio.h>
#include <limits>

typedef float tx_type;

#define DEFAULT_COUNT 10000000

#define CUDACHECK(e) cudaCheck(e, __FILE__, __LINE__)
inline void cudaCheck(cudaError_t e, const char *filename, const int lineno) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d : \"%s\"\n", filename, lineno,
                cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
bool checkClose(const T* test, const T* ref, size_t N) {
    const T diff = std::numeric_limits<T>::epsilon();
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(test[i] - ref[i]) > diff) {
            std::cerr << "data mismatch: test[" << i << "]=" << test[i];
            std::cerr << " ref[" << i << "]=" << ref[i] << std::endl;
            return false;
        }
    }
    return true;
}

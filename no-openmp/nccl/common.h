#include <cstdio>
#include "nccl.h"

#define CUDACHECK(r) cudaCheck(r, __FILE__, __LINE__)
inline void cudaCheck(cudaError_t e, const char *filename, const int lineno) {
    if (e != cudaSuccess) {
        printf("Failed: CUDA error %s:%d '%s'\n", filename, lineno,
                cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

#define NCCLCHECK(r) ncclCheck(r, __FILE__, __LINE__)
inline void ncclCheck(ncclResult_t r, const char *filename, const int lineno) {
    if (r != ncclSuccess) {
        printf("Failed: NCCL error %s:%d '%s'\n", filename, lineno,
                ncclGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

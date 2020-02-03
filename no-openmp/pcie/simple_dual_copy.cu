#include <cuda_runtime.h>
#include <iostream>

#include "common.h"

int main(int argc, char *argv[]) {
    const size_t count = 10000000;
    const size_t bufSize = count * sizeof(tx_type);

    cudaStream_t s[2];
    cudaEvent_t start[2];
    cudaEvent_t stop[2];

    tx_type *d_buffers[2];

    for (int d = 0; d < 2; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaStreamCreateWithFlags(&s[d], cudaStreamNonBlocking));
        CUDACHECK(cudaEventCreate(&start[d]));
        CUDACHECK(cudaEventCreate(&stop[d]));
        CUDACHECK(cudaMalloc(&d_buffers[d], bufSize));
    }

    tx_type *h_original, *h_result;
    CUDACHECK(cudaMallocHost(&h_original, bufSize));
    CUDACHECK(cudaMallocHost(&h_result, bufSize));

    for (size_t i = 0; i < count; ++i) {
        h_original[i] = i;
    }
    CUDACHECK(cudaMemcpy(d_buffers[0], h_original, bufSize, cudaMemcpyDefault));

    const size_t half = count / 2;
    const size_t halfBufSize = half * sizeof(tx_type);

    for (int d = 0; d < 2; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaEventRecord(start[d], s[d]));
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemcpyAsync(d_buffers[1], d_buffers[0],
                halfBufSize, cudaMemcpyDefault, s[0]));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMemcpyAsync(d_buffers[1]+half, d_buffers[0]+half,
                (bufSize - halfBufSize), cudaMemcpyDefault, s[1]));
    //CUDACHECK(cudaSetDevice(0));
    //CUDACHECK(cudaMemcpyPeerAsync(d_buffers[1], 1, d_buffers[0], 0,
    //            halfBufSize, s[0]));
    //CUDACHECK(cudaSetDevice(1));
    //CUDACHECK(cudaMemcpyPeerAsync(d_buffers[1]+half, 1, d_buffers[0]+half, 0,
    //            (bufSize - halfBufSize), s[1]));

    for (int d = 0; d < 2; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaEventRecord(stop[d], s[d]));
    }

    for (int d = 0; d < 2; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaDeviceSynchronize());
        float ms;
        CUDACHECK(cudaEventElapsedTime(&ms, start[d], stop[d]));
        std::cout << ms << " ms" << std::endl;
    }

    CUDACHECK(cudaMemcpy(h_result, d_buffers[1], bufSize, cudaMemcpyDefault));
    for (size_t i = 0; i < count; ++i) {
        if (h_original[i] != h_result[i]) {
            std::cerr << "data mismatch at idx=" << i << std::endl;
            break;
        }
    }

    for (int d = 0; d < 2; ++d) {
        CUDACHECK(cudaFree(d_buffers[d]));
    }
    CUDACHECK(cudaFreeHost(h_result));
    CUDACHECK(cudaFreeHost(h_original));

    return 0;
}

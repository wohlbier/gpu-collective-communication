#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

#include "common.h"

int main(int argc, char *argv[]) {
    std::cout << "starting nstream" << std::endl;

    const int nDevs = 2;
    const int N = atoi(argv[1]);
    const int nStreams = atoi(argv[2]);

    // make sure we have enough devices
    int sysDevices;
    CUDACHECK(cudaGetDeviceCount(&sysDevices));

    if (sysDevices < nDevs) {
        std::cerr << "need at least 2 cuda devices" << std::endl;
        exit(EXIT_FAILURE);
    }

    // create space for things
    const size_t numBytes = N * sizeof(tx_type);
    const size_t chunkCount = N / nStreams;
    tx_type *h_data;
    CUDACHECK(cudaMallocHost(&h_data, numBytes));

    tx_type *d_data[nDevs];
    cudaStream_t s[nDevs][nStreams];
    cudaEvent_t start[nDevs][nStreams];
    cudaEvent_t stop[nDevs][nStreams];

    for (int i = 0; i < nDevs; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(&d_data[i], numBytes));
        for (int j = 0; j < nStreams; ++j) {
            CUDACHECK(cudaStreamCreate(&s[i][j]));
            CUDACHECK(cudaEventCreate(&start[i][j]));
            CUDACHECK(cudaEventCreate(&stop[i][j]));
        }
    }

    // set host data to be something to check delivery
    CUDACHECK(cudaMemset(h_data, 1, numBytes));
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemcpy(d_data[0], h_data, numBytes, cudaMemcpyDefault));

    // do the transfers
    for (int i = 1; i < nDevs; ++i) {
        CUDACHECK(cudaSetDevice(i));
        for (int j = 0; j < nStreams; ++j) {
            CUDACHECK(cudaEventRecord(start[i][j], s[i][j]));
            CUDACHECK(cudaMemcpyAsync(d_data[i]+j*chunkCount, d_data[0]+j*chunkCount, chunkCount * sizeof(tx_type), cudaMemcpyDefault, s[i][j]));
            CUDACHECK(cudaEventRecord(stop[i][j], s[i][j]));
        }
    }

    // print timing info
    for (int dev = 1; dev < nDevs; ++dev) {
        CUDACHECK(cudaSetDevice(dev));
        CUDACHECK(cudaDeviceSynchronize());
        for (int i = 0; i < nStreams; ++i) {
            float ms;
            CUDACHECK(cudaEventElapsedTime(&ms, start[dev][i], stop[dev][i]));
            printf("time for dev %d stream %d: %f\n", dev, i, ms);
        }
    }

    for (int i=0; i<nDevs; ++i) CUDACHECK(cudaFree(d_data[i]));

    return 0;
}

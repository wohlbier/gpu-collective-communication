#include <cuda_runtime.h>
#include <iostream>

#include "common.h"

/**
 * Gear things up as a send - see if we can actually do better with having
 * GPUs in a push/pull config instead of just pull
 */

int main(int argc, char *argv[]) {
    const int nDevs = 2;

    // make sure we have enough devices
    int numDevices;
    CUDACHECK(cudaGetDeviceCount(&numDevices));

    if (numDevices < nDevs) {
        std::cout << "not enough devices" << std::endl;
        exit(EXIT_FAILURE);
    }

    // some arg parsing
    const size_t N = atoi(argv[1]);
    int nStreams = 1;
    if (argc > 2)
        nStreams = atoi(argv[2]);

    // divide among devs and then streams
    int devChunkSize = N / nDevs;
    int streamChunkSize = devChunkSize / nStreams;

    tx_type *h_data;
    CUDACHECK(cudaMallocHost(&h_data, N * sizeof(tx_type)));
    CUDACHECK(cudaMemset(h_data, 1., N * sizeof(tx_type)));

    tx_type *d_data[nDevs];
    cudaEvent_t *start[nDevs];
    cudaEvent_t *stop[nDevs];
    cudaStream_t **s = (cudaStream_t**)malloc(nDevs * sizeof(cudaStream_t*));
    for (int device = 0; device < nDevs; ++device) {
        CUDACHECK(cudaSetDevice(device));
        CUDACHECK(cudaMalloc(&d_data[device], N * sizeof(tx_type)));

        s[device] = (cudaStream_t*)malloc(nStreams * sizeof(cudaStream_t));
        start[device] = (cudaEvent_t*)malloc(nStreams * sizeof(cudaStream_t));
        stop[device] = (cudaEvent_t*)malloc(nStreams * sizeof(cudaStream_t));
        for (int i = 0; i < nStreams; ++i) {
            CUDACHECK(cudaStreamCreate(s[device]+i));
            CUDACHECK(cudaEventCreate(start[device]+i));
            CUDACHECK(cudaEventCreate(stop[device]+i));
        }
    }

    CUDACHECK(cudaMemcpy(d_data[0], h_data, N * sizeof(tx_type), cudaMemcpyDefault));


    // do the copy
    // how many ways can I split this up two ways???
    int fDev = 0, fStream = 0, fEvent = 0;
    int sDev = 1, sStream = 0, sEvent = 0;
    CUDACHECK(cudaSetDevice(fDev));
    CUDACHECK(cudaEventRecord(start[fDev][fEvent], s[fDev][fStream]));
    CUDACHECK(cudaSetDevice(sDev));
    CUDACHECK(cudaEventRecord(start[sDev][sEvent], s[sDev][sStream]));

    CUDACHECK(cudaSetDevice(fDev));
    CUDACHECK(cudaMemcpyAsync(d_data[1], d_data[0],
                devChunkSize * sizeof(tx_type), cudaMemcpyDefault, s[fDev][fStream]));
    CUDACHECK(cudaSetDevice(sDev));
    CUDACHECK(cudaMemcpyAsync(d_data[1] + devChunkSize, d_data[0] + devChunkSize,
                (N - devChunkSize) * sizeof(tx_type), cudaMemcpyDefault, s[sDev][sStream]));

    CUDACHECK(cudaEventRecord(stop[fDev][fEvent], s[fDev][fStream]));
    CUDACHECK(cudaEventRecord(stop[sDev][sEvent], s[sDev][sStream]));


    // error checking
    tx_type *h_result;
    CUDACHECK(cudaMallocHost(&h_result, N * sizeof(tx_type)));
    CUDACHECK(cudaMemcpy(h_result, d_data[1], N * sizeof(tx_type), cudaMemcpyDefault));

    for (size_t i = 0; i < N; ++i) {
        if (h_result[i] != h_data[i]) {
            std::cout << "copy failed" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    for (int dev = 0; dev < nDevs; ++dev) {
        CUDACHECK(cudaSetDevice(dev));
        CUDACHECK(cudaDeviceSynchronize());
    }

    float ms;
    CUDACHECK(cudaEventElapsedTime(&ms, start[fDev][fEvent], stop[fDev][fEvent]));
    printf("first copy took %f ms\n", ms);

    CUDACHECK(cudaEventElapsedTime(&ms, start[sDev][sEvent], stop[sDev][sEvent]));
    printf("second copy took %f ms\n", ms);

    for (int device = 0; device < nDevs; ++device) {
        CUDACHECK(cudaFree(d_data[device]));
    }

    CUDACHECK(cudaFreeHost(h_result));
    CUDACHECK(cudaFreeHost(h_data));

    for (int dev = 0; dev < nDevs; ++dev) {
        free(s[dev]);
        free(start[dev]);
        free(stop[dev]);
    }
    free(s);

    return 0;
}

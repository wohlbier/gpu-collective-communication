#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>

#include "common.h"

#define THREADS_PER_BLOCK 256

static const size_t defaultCount = 10000000;

// push kernel
//  - push blockDim elements at a time, post on a status buffer
template <typename T>
__global__
void send(T* dst, const T* src, const size_t nelems, volatile size_t *counter) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // get how many steps we need to do
    size_t nSteps = ((nelems - 1) / THREADS_PER_BLOCK) + 1;

    size_t idx = tid;
    for (size_t i = 0; i < nSteps; ++i) {
        if (idx < nelems) {
            dst[idx] = src[idx];
            idx += THREADS_PER_BLOCK;
        }
        __threadfence_system();
        if (tid == 0) {
            *counter += THREADS_PER_BLOCK;
        }
    }
}

// pull kernel
//  - possibly launch this for as many elements that we have and gate the
//    threads on the status buffer
template <typename T>
__global__
void recv(T* dst, const T* src, const size_t nelems, volatile size_t *counter) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nelems) {
        while ((*counter) < tid);
        dst[tid] = src[tid];
    }
}

int main(int argc, char *argv[]) {
    std::cout << "starting coop_multi_device" << std::endl;

    const size_t COUNT = (argc > 1) ? atoi(argv[1]) : defaultCount;
    const size_t BUF_SIZE = COUNT * sizeof(float);

    // make sure we have two, and only two devs right now
    int nDevs;
    CUDACHECK(cudaGetDeviceCount(&nDevs));
    if (nDevs < 2) {
        std::cerr << "need two devices to do this" << std::endl;
        exit(EXIT_FAILURE);
    }

    // regardless of how many we have, just use two devices
    nDevs = 2;

    // create some host space for data
    float *h_data;
    CUDACHECK(cudaMallocHost(&h_data, BUF_SIZE));

    // host space for staging
    float *h_staging;
    size_t *h_counter;
    CUDACHECK(cudaMallocHost(&h_staging, BUF_SIZE));
    CUDACHECK(cudaMallocHost(&h_counter, sizeof(size_t)));

    // device things
    std::vector<float*> d_buffers(nDevs);
    std::vector<cudaEvent_t> start(nDevs);
    std::vector<cudaEvent_t> stop(nDevs);
    std::vector<cudaStream_t> stream(nDevs);

    for (int d = 0; d < nDevs; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaMalloc(&d_buffers[d], BUF_SIZE));
        CUDACHECK(cudaEventCreate(&start[d]));
        CUDACHECK(cudaEventCreate(&stop[d]));
        CUDACHECK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
    }

    // init data to send
    for (size_t i = 0; i < COUNT; ++i) {
        h_data[i] = i;
    }
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemcpy(d_buffers[0], h_data, BUF_SIZE, cudaMemcpyDefault));

    for (int d = 0; d < nDevs; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaEventRecord(start[d], stream[d]));
    }

    CUDACHECK(cudaSetDevice(0));
    dim3 dimBlock(THREADS_PER_BLOCK);
    dim3 dimGrid(1);
    send<<<dimGrid, dimBlock, 0, stream[0]>>>(h_staging, d_buffers[0], COUNT, h_counter);
    CUDACHECK(cudaGetLastError());

    CUDACHECK(cudaSetDevice(1));
    dimGrid = dim3(((COUNT - 1) / THREADS_PER_BLOCK) + 1);
    recv<<<dimGrid, dimBlock, 0, stream[1]>>>(d_buffers[1], h_staging, COUNT, h_counter);
    CUDACHECK(cudaGetLastError());

    for (int d = 0; d < nDevs; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaEventRecord(stop[d], stream[d]));
    }

    // wait for completion and report times
    for (int d = 0; d < nDevs; ++d) {
        std::cout << "dev " << d << std::endl;
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaStreamSynchronize(stream[d]));

        float ms;
        CUDACHECK(cudaEventElapsedTime(&ms, start[d], stop[d]));
        std::cout << "dev " << d << " took " << ms << " ms" << std::endl;
    }

    // check that the data got there. overwrite staging since we're done with it
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMemcpy(h_staging, d_buffers[1], BUF_SIZE, cudaMemcpyDefault));
    for (size_t i = 0; i < COUNT; ++i) {
        if (h_staging[i] - h_data[i] > std::numeric_limits<float>::epsilon()) {
            std::cerr << "data mismatch at " << i
                << ", orig=" << h_data[i] << ", test=" << h_staging[i] << std::endl;
            break;
        }
    }

    // cleanup
    for (int d = 0; d < nDevs; ++d) CUDACHECK(cudaFree(d_buffers[d]));
    CUDACHECK(cudaFreeHost(h_counter));
    CUDACHECK(cudaFreeHost(h_staging));
    CUDACHECK(cudaFreeHost(h_data));

    return 0;
}

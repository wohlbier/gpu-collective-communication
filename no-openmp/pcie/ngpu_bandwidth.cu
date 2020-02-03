#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

#include "common.h"

static const size_t COUNT = 10000000;

std::string horLine = "------------------------------------------------------------\n";

void reportBandwidth(std::string testStr, float *ms, const int N) {
    // pull out the max time - bandwidth determined by slowest copy
    float max = *(std::max_element(ms, ms + N));
    size_t bytes = COUNT * sizeof(tx_type);
    float bw = (bytes/(1024*1024)) / max;

    std::cout << horLine << testStr << std::endl;
    std::cout << "\tbw: " << bw << " GBps" << std::endl;
    std::cout << horLine << std::endl;
}

void exitGpuCount(const int need, const int have) {
    std::cout << "not enough GPUs to continue" << std::endl;
    std::cout << "\tneed: " << need << ", have: " << have << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // get how many devices I have to work with
    int nDevs;
    CUDACHECK(cudaGetDeviceCount(&nDevs));

    // need at least 2 GPUs from here on out
    if (nDevs < 2) {
        exitGpuCount(2, nDevs);
    }

    /*
     * Copy 0 -> 1
     *  - First do solely 0 -> 1
     *  - Then split evenly so that 0 pushes and 1 pulls
     */

    std::vector<tx_type*> d_buffers(nDevs);
    std::vector<cudaStream_t> stream0(nDevs);
    std::vector<cudaStream_t> stream1(nDevs);
    std::vector<cudaEvent_t> start(nDevs);
    std::vector<cudaEvent_t> stop(nDevs);

    for (int i = 0; i < nDevs; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreateWithFlags(&stream0[i], cudaStreamNonBlocking));
        CUDACHECK(cudaStreamCreateWithFlags(&stream1[i], cudaStreamNonBlocking));
        CUDACHECK(cudaEventCreate(&start[i]));
        CUDACHECK(cudaEventCreate(&stop[i]));
    }

    const size_t numBytes = COUNT * sizeof(tx_type);
    tx_type *h_data, *h_result;
    CUDACHECK(cudaMallocHost(&h_data, numBytes));
    CUDACHECK(cudaMallocHost(&h_result, numBytes));
    CUDACHECK(cudaMemset(h_data, 1., numBytes));
    CUDACHECK(cudaMemset(h_result, 0., numBytes));

    for (int i = 0; i < 2; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(&d_buffers[i], numBytes));
    }
    CUDACHECK(cudaMemcpy(d_buffers[0], h_data, numBytes, cudaMemcpyDefault));

    // copy
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(start[0]));
    CUDACHECK(cudaMemcpyAsync(d_buffers[1], d_buffers[0], numBytes, cudaMemcpyDefault));
    CUDACHECK(cudaEventRecord(stop[0]));

    // get timing
    float ms;
    CUDACHECK(cudaEventSynchronize(stop[0]));
    CUDACHECK(cudaEventElapsedTime(&ms, start[0], stop[0]));
    reportBandwidth("0 push all to 1", &ms, 1);

    // make sure the data got there properly
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMemcpy(h_result, d_buffers[1], numBytes, cudaMemcpyDefault));
    for (size_t i = 0; i < COUNT; ++i) {
        if (h_result[i] != h_data[i]) {
            fprintf(stderr, "data mismatch: orig[%ld]=%f, copy[%ld]=%f\n",
                    i, h_data[i], i, h_result[i]);
            exit(EXIT_FAILURE);
        }
    }

    // Even split between 0 push to 1 and 1 pull from 0

    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMemset(d_buffers[1], 0, numBytes));

    const size_t half = COUNT / 2;
    const size_t halfBytes = half * sizeof(tx_type);
    float twoMs[2] = { 0 };

    //// start the timers
    //for (int i = 0; i < 2; ++i) {
    //    CUDACHECK(cudaSetDevice(i));
    //    CUDACHECK(cudaEventRecord(start[i]));
    //}

    //// push first half with dev 0
    //CUDACHECK(cudaSetDevice(0));
    //CUDACHECK(cudaMemcpyAsync(d_buffers[1], d_buffers[0], halfBytes, cudaMemcpyDefault));

    //// pull second half with dev 1
    //CUDACHECK(cudaSetDevice(1));
    //CUDACHECK(cudaMemcpyAsync(d_buffers[1] + half, d_buffers[0] + half,
    //            numBytes - halfBytes, cudaMemcpyDefault));

    //// stop timers
    //for (int i = 0; i < 2; ++i) {
    //    CUDACHECK(cudaSetDevice(i));
    //    CUDACHECK(cudaEventRecord(stop[i]));
    //}

    //// get timing
    //for (int i = 0; i < 2; ++i) {
    //    CUDACHECK(cudaSetDevice(i));
    //    CUDACHECK(cudaEventSynchronize(stop[i]));
    //    CUDACHECK(cudaEventElapsedTime(&twoMs[i], start[i], stop[i]));
    //}
    //reportBandwidth("0 push half, 1 pull half", twoMs, 2);

    CUDACHECK(cudaSetDevice(0));

    // create some new events for this particular one
    std::vector<cudaEvent_t> twoStart(2);
    std::vector<cudaEvent_t> twoStop(2);

    for (int i = 0; i < 2; ++i) {
        CUDACHECK(cudaEventCreate(&twoStart[i]));
        CUDACHECK(cudaEventCreate(&twoStop[i]));
    }

    CUDACHECK(cudaEventRecord(twoStart[0], stream0[0]));
    CUDACHECK(cudaEventRecord(twoStart[1], stream1[0]));

    CUDACHECK(cudaMemcpyAsync(d_buffers[1], d_buffers[0], halfBytes,
                cudaMemcpyDefault, stream0[0]));
    CUDACHECK(cudaMemcpyAsync(d_buffers[1]+half, d_buffers[0]+half, halfBytes,
                cudaMemcpyDefault, stream1[0]));

    CUDACHECK(cudaEventRecord(twoStop[0], stream0[0]));
    CUDACHECK(cudaEventRecord(twoStop[1], stream1[0]));

    CUDACHECK(cudaStreamSynchronize(stream0[0]));
    CUDACHECK(cudaStreamSynchronize(stream1[0]));

    CUDACHECK(cudaEventElapsedTime(&twoMs[0], twoStart[0], twoStop[0]));
    CUDACHECK(cudaEventElapsedTime(&twoMs[1], twoStart[1], twoStop[1]));

    std::cout << twoMs[0] << "  " << twoMs[1] << std::endl;
    reportBandwidth("half and half", twoMs, 2);

    // make sure the data got there properly
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMemcpy(h_result, d_buffers[1], numBytes, cudaMemcpyDefault));
    for (size_t i = 0; i < COUNT; ++i) {
        if (h_result[i] != h_data[i]) {
            fprintf(stderr, "data mismatch: orig[%ld]=%f, copy[%ld]=%f\n",
                    i, h_data[i], i, h_result[i]);
            exit(EXIT_FAILURE);
        }
    }

    /*
     * cleanup
     */

    // cuda things
    CUDACHECK(cudaFreeHost(h_data));
    CUDACHECK(cudaFreeHost(h_result));
    for (int i = 0; i < 2; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(d_buffers[i]));
    }

    return 0;
}

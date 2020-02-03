#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdio>

#include "common.h"

/**
 * Need temporary buffers on each device that are big enough to hold the
 * total amount of data being transferred, i.e. count * numDevices.
 */
template <typename T>
void scatterMST(T **d_sendbuf, T **d_recvbuf, size_t count, int root,
        int left, int right, cudaStream_t *stream, cudaEvent_t *event)
{
    const size_t numBytes = count * sizeof(T);

    // if we are done, copy into my receive buffer
    if (left == right) {
        CUDACHECK(cudaMemcpyAsync(d_recvbuf[root], d_sendbuf[root]+count*root,
                    numBytes, cudaMemcpyDefault, stream[root]));
        return;
    }

    const int mid = ((left + right - 1) / 2) + 1;
    int sendTo, leftRoot = left, rightRoot = right;
    size_t offset, size;
    if (root < mid) {
        sendTo = right;
        leftRoot = root;
        offset = mid*count;
        size = (right-mid+1)*numBytes;
    }
    else {
        sendTo = left;
        rightRoot = root;
        offset = left*count;
        size = (mid-left) * numBytes;
    }

    // mark receiver to wait until root has its data
    CUDACHECK(cudaEventRecord(event[root], stream[root]));
    CUDACHECK(cudaStreamWaitEvent(stream[sendTo], event[root], 0));

    CUDACHECK(cudaMemcpyAsync(d_sendbuf[sendTo]+offset, d_sendbuf[root]+offset,
                size, cudaMemcpyDefault, stream[sendTo]));

    // do the recursive calls
    scatterMST(d_sendbuf, d_recvbuf, count, leftRoot, left, mid-1, stream, event);
    scatterMST(d_sendbuf, d_recvbuf, count, rightRoot, mid, right, stream, event);
}

/**
 * Different from normal bucket in that in sends both directions simultaneously.
 * In this one, temps only need to be twice the size of send count.
 */
template <typename T>
void scatterRing(T **d_buffers, T **d_temps, size_t count, int root)
{
    return;
}

int main(int argc, char *argv[]) {
    std::cout << "starting scatter" << std::endl;

    const size_t defaultScatterCount = 1000000;
    const int nIters = 10;

    const size_t COUNT = (argc > 1) ? atoi(argv[1]) : defaultScatterCount;
    const size_t BUF_SIZE = COUNT * sizeof(float);

    int nDevsPresent;
    CUDACHECK(cudaGetDeviceCount(&nDevsPresent));

    const int nDevs = (argc > 2) ? atoi(argv[2]) : nDevsPresent;
    if (nDevs > nDevsPresent) {
        std::cerr << nDevs << " devices requested, only " << nDevsPresent
            << " present" << std::endl;
        exit(EXIT_FAILURE);
    }

    const int root = (argc > 3) ? atoi(argv[3]) : 0;

    std::vector<float*> d_sendbuf(nDevs);
    std::vector<float*> d_recvbuf(nDevs);
    std::vector<cudaStream_t> stream(nDevs);
    std::vector<cudaEvent_t> start(nDevs);
    std::vector<cudaEvent_t> stop(nDevs);
    std::vector<cudaEvent_t> event(nDevs);
    std::vector<float> times(nDevs);

    for (int d = 0; d < nDevs; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaMalloc(&d_sendbuf[d], nDevs * BUF_SIZE));
        CUDACHECK(cudaMalloc(&d_recvbuf[d], BUF_SIZE));
        CUDACHECK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
        CUDACHECK(cudaEventCreate(&start[d]));
        CUDACHECK(cudaEventCreate(&stop[d]));
        CUDACHECK(cudaEventCreate(&event[d]));
        for (int p = 0; p < nDevs; ++p) {
            int canAccessPeer;
            if (p != d) {
                CUDACHECK(cudaDeviceCanAccessPeer(&canAccessPeer, d, p));
                if (canAccessPeer) CUDACHECK(cudaDeviceEnablePeerAccess(p, 0));
            }
        }
        times[d] = 0;
    }

    float *h_buffer, *h_result;
    CUDACHECK(cudaMallocHost(&h_buffer, nDevs * BUF_SIZE));
    CUDACHECK(cudaMallocHost(&h_result, BUF_SIZE));
    for (size_t i = 0; i < nDevs * COUNT; ++i) {
        h_buffer[i] = i;
    }

    for (int i = 0; i < nIters; ++i) {
        for (int d = 0; d < nDevs; ++d) {
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaMemset(d_recvbuf[d], 0, BUF_SIZE));
            CUDACHECK(cudaMemset(d_sendbuf[d], 0, nDevs * BUF_SIZE));
        }
        CUDACHECK(cudaMemcpy(d_sendbuf[root], h_buffer, nDevs * BUF_SIZE, cudaMemcpyDefault));

        for (int d = 0; d < nDevs; ++d) {
            CUDACHECK(cudaEventRecord(start[d], stream[d]));
        }

        // scatter
        scatterMST(&d_sendbuf[0], &d_recvbuf[0], COUNT, root, 0, nDevs-1,
                &stream[0], &event[0]);

        for (int d = 0; d < nDevs; ++d) {
            CUDACHECK(cudaEventRecord(stop[d], stream[d]));
        }

        for (int d = 0; d < nDevs; ++d) {
            CUDACHECK(cudaStreamSynchronize(stream[d]));

            float ms;
            CUDACHECK(cudaEventElapsedTime(&ms, start[d], stop[d]));
            times[d] += ms;

            CUDACHECK(cudaMemcpy(h_result, d_recvbuf[d], BUF_SIZE, cudaMemcpyDefault));
            checkClose(h_result, h_buffer+COUNT*d, COUNT);
        }
    }

    // make sure nothing went horribly wrong
    CUDACHECK(cudaGetLastError());

    for (int d = 0; d < nDevs; ++d) {
        printf("dev %d: %f ms\n", d, times[d] / nIters);
    }

    // cleanup
    for (int d = 0; d < nDevs; ++d) {
        CUDACHECK(cudaFree(d_sendbuf[d]));
        CUDACHECK(cudaFree(d_recvbuf[d]));
    }
    CUDACHECK(cudaFreeHost(h_result));
    CUDACHECK(cudaFreeHost(h_buffer));

    return 0;
}

/**
 * Testing sending from one to N GPUs in a variety of different ways
 */

#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <iostream>
#include <thread>

#include "common.h"

static const size_t defaultCount = 10000000;

template <typename T>
void cudaCopy(int device, cudaStream_t stream, T* dest, const T* src, size_t bytes) {
    CUDACHECK(cudaSetDevice(device));
    CUDACHECK(cudaMemcpyAsync(dest, src, bytes, cudaMemcpyDefault, stream));
}

template <typename T>
void cudaCopyRoot(int dev, int nDevs, cudaStream_t *streams,
        T** dest, const T* src, size_t bytes)
{
    CUDACHECK(cudaSetDevice(dev));
    for (int d = 0; d < nDevs; ++d) {
        if (d != dev)
            CUDACHECK(cudaMemcpyAsync(dest[d], src, bytes, cudaMemcpyDefault, streams[d]));
    }
}

template <typename T>
void broadcastMST(T **d_buffers, size_t count, int root, int left, int right,
        cudaStream_t *stream, cudaEvent_t *event)
{
    // exit condition
    if (left == right) return;

    const size_t numBytes = count * sizeof(T);

    // get the midpoint of this segment
    int mid = ((left + right - 1) / 2) + 1;

    if (root < mid) {
        // send to right
        CUDACHECK(cudaMemcpyAsync(d_buffers[right], d_buffers[root], numBytes,
                    cudaMemcpyDefault, stream[root]));

        // make the next copy wait on this copy to finish
        CUDACHECK(cudaEventRecord(event[root], stream[root]));
        CUDACHECK(cudaStreamWaitEvent(stream[right], event[root], 0));

        broadcastMST(d_buffers, count, root, left, mid-1, stream, event);
        broadcastMST(d_buffers, count, right, mid, right, stream, event);
    }
    else {
        // send to left
        CUDACHECK(cudaMemcpyAsync(d_buffers[left], d_buffers[root], numBytes,
                    cudaMemcpyDefault, stream[root]));

        // make the next copy wait on this copy to finish
        CUDACHECK(cudaEventRecord(event[root], stream[root]));
        CUDACHECK(cudaStreamWaitEvent(stream[left], event[root], 0));

        broadcastMST(d_buffers, count, root, mid, right, stream, event);
        broadcastMST(d_buffers, count, left, left, mid-1, stream, event);
    }
}

/**
 * Notes
 *  - All timers are started in a group before the copies so that any
 *    serialization is caught by the later timers.
 */

using Float = double;

int main(int argc, char *argv[]) {
    const size_t COUNT = (argc > 1) ? atoi(argv[1]) : defaultCount;
    const size_t BUF_SIZE = COUNT * sizeof(Float);
    const int nIters = 10;

    int nDevs;
    CUDACHECK(cudaGetDeviceCount(&nDevs));

    std::vector<Float*> d_buffers(nDevs);

    for (int d = 0; d < nDevs; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaMalloc(&d_buffers[d], BUF_SIZE));
        for (int p = 0; p < nDevs; ++p) {
            if (p != d) {
                int canAccessPeer;
                CUDACHECK(cudaDeviceCanAccessPeer(&canAccessPeer, d, p));
                if (canAccessPeer) {
                    std::cout << "enabling peer access from dev " << d << " to " << p << std::endl;
                    CUDACHECK(cudaDeviceEnablePeerAccess(p, 0));
                }
            }
        }
    }

    Float *h_buffer, *h_result;
    CUDACHECK(cudaMallocHost(&h_buffer, BUF_SIZE));
    CUDACHECK(cudaMallocHost(&h_result, BUF_SIZE));
    for (size_t i = 0; i < COUNT; ++i) {
        h_buffer[i] = i;
    }
    CUDACHECK(cudaMemcpy(d_buffers[0], h_buffer, BUF_SIZE, cudaMemcpyDefault));


    /*
     * Source GPU pushing with stream concurrency
     */

    {
        std::cout << std::endl;
        std::cout << "Source push with N streams" << std::endl;

        std::vector<cudaStream_t> streams(nDevs);
        std::vector<cudaEvent_t> start(nDevs);
        std::vector<cudaEvent_t> stop(nDevs);
        std::vector<float> times(nDevs);

        CUDACHECK(cudaSetDevice(0));
        for (int d = 0; d < nDevs; ++d) {
            CUDACHECK(cudaStreamCreateWithFlags(&streams[d], cudaStreamNonBlocking));
            CUDACHECK(cudaEventCreate(&start[d]));
            CUDACHECK(cudaEventCreate(&stop[d]));
        }

        for (int iter = 0; iter < nIters; ++iter) {
            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaMemset(d_buffers[d], 0, BUF_SIZE));
            }

            CUDACHECK(cudaSetDevice(0));
            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaEventRecord(start[d], streams[d]));
            }

            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaMemcpyAsync(d_buffers[d], d_buffers[0], BUF_SIZE,
                            cudaMemcpyDefault, streams[d]));
            }

            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaEventRecord(stop[d], streams[d]));
            }

            for (int d = 1; d < nDevs; ++d) {
                float ms;
                CUDACHECK(cudaStreamSynchronize(streams[d]));
                CUDACHECK(cudaEventElapsedTime(&ms, start[d], stop[d]));
                times[d] += ms;
            }
        }

        for (int d = 1; d < nDevs; ++d) {
            printf("dev %d: %f ms\n", d, times[d] / nIters);
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaMemcpy(h_result, d_buffers[d], BUF_SIZE, cudaMemcpyDefault));
            checkClose(h_result, h_buffer, COUNT);
        }

        CUDACHECK(cudaSetDevice(0));
        for (int d = 0; d < nDevs; ++d) {
            CUDACHECK(cudaStreamDestroy(streams[d]));
            CUDACHECK(cudaEventDestroy(start[d]));
            CUDACHECK(cudaEventDestroy(stop[d]));
        }
    }


    /*
     * Dest GPUs pulling on a single stream per GPU
     */

    {
        std::cout << std::endl << "Destinations pull" << std::endl;

        std::vector<cudaEvent_t> start(nDevs);
        std::vector<cudaEvent_t> stop(nDevs);
        std::vector<cudaStream_t> stream(nDevs);
        std::vector<float> times(nDevs);

        for (int d = 0; d < nDevs; ++d) {
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
            CUDACHECK(cudaEventCreate(&start[d]));
            CUDACHECK(cudaEventCreate(&stop[d]));
        }

        for (int iter = 0; iter < nIters; ++iter) {
            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaMemset(d_buffers[d], 0, BUF_SIZE));
            }

            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaEventRecord(start[d], stream[d]));
            }

            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaMemcpyAsync(d_buffers[d], d_buffers[0], BUF_SIZE,
                            cudaMemcpyDefault, stream[d]));
            }

            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaEventRecord(stop[d], stream[d]));
            }

            for (int d = 1; d < nDevs; ++d) {
                float ms;
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaStreamSynchronize(stream[d]));
                CUDACHECK(cudaEventElapsedTime(&ms, start[d], stop[d]));
                times[d] += ms;
            }
        }

        for (int d = 1; d < nDevs; ++d) {
            printf("dev %d: %f ms\n", d, times[d] / nIters);
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaMemcpy(h_result, d_buffers[d], BUF_SIZE, cudaMemcpyDefault));
            checkClose(h_result, h_buffer, COUNT);
        }

        for (int d = 0; d < nDevs; ++d) {
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaStreamDestroy(stream[d]));
            CUDACHECK(cudaEventDestroy(start[d]));
            CUDACHECK(cudaEventDestroy(stop[d]));
        }
    }


    /*
     * Launch a thread per device to do the sends (like NCCL)
     */

    {
        std::cout << std::endl << "Thread per device to pull" << std::endl;

        std::vector<std::thread> threads(nDevs);
        std::vector<float> times(nDevs);
        std::vector<cudaStream_t> stream(nDevs);
        std::vector<cudaEvent_t> start(nDevs);
        std::vector<cudaEvent_t> stop(nDevs);

        for (int d = 0; d < nDevs; ++d) {
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
            CUDACHECK(cudaEventCreate(&start[d]));
            CUDACHECK(cudaEventCreate(&stop[d]));
            times[d] = 0.f;
        }

        for (int i = 0; i < nIters; ++i) {
            for (int d = 0; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaEventRecord(start[d], stream[d]));
            }

            for (int d = 0; d < nDevs; ++d) {
                threads[d] = std::thread(cudaCopy<Float>, d, stream[d], d_buffers[d], d_buffers[0], BUF_SIZE);
            }

            for (auto &thread : threads) {
                thread.join();
            }

            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaEventRecord(stop[d], stream[d]));
            }

            for (int d = 1; d < nDevs; ++d) {
                float ms;
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaStreamSynchronize(stream[d]));
                CUDACHECK(cudaEventElapsedTime(&ms, start[d], stop[d]));
                times[d] += ms;
            }
        }

        for (int d = 1; d < nDevs; ++d) {
            printf("dev %d: %f ms\n", d, times[d] / nIters);
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaMemcpy(h_result, d_buffers[d], BUF_SIZE, cudaMemcpyDefault));
            checkClose(h_result, h_buffer, COUNT);
            CUDACHECK(cudaEventDestroy(stop[d]));
            CUDACHECK(cudaEventDestroy(start[d]));
            CUDACHECK(cudaStreamDestroy(stream[d]));
        }
    }


    /*
     * Dest/source in push/pull (half each)
     */

    const size_t HALF_COUNT = COUNT / 2;
    const size_t HALF_BUF_SIZE = HALF_COUNT * sizeof(Float);

    {
        std::cout << std::endl << "Dest/source push/pull, split half/half" << std::endl;

        std::vector<cudaStream_t> stream(nDevs);
        std::vector<cudaStream_t> stream0(nDevs);
        std::vector<cudaEvent_t> start(nDevs);
        std::vector<cudaEvent_t> start0(nDevs);
        std::vector<cudaEvent_t> stop(nDevs);
        std::vector<cudaEvent_t> stop0(nDevs);
        std::vector<float> times(nDevs);
        std::vector<float> times0(nDevs);

        CUDACHECK(cudaSetDevice(0));
        for (int d = 0; d < nDevs; ++d) {
            CUDACHECK(cudaStreamCreate(&stream0[d]));
            CUDACHECK(cudaEventCreate(&start0[d]));
            CUDACHECK(cudaEventCreate(&stop0[d]));
            times0[d] = 0.f;
        }

        for (int d = 1; d < nDevs; ++d) {
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
            CUDACHECK(cudaEventCreate(&start[d]));
            CUDACHECK(cudaEventCreate(&stop[d]));
            times[d] = 0.f;
        }

        for (int i = 0; i < nIters; ++i) {
            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaEventRecord(start[d], stream[d]));
            }

            CUDACHECK(cudaSetDevice(0));
            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaEventRecord(start0[d], stream0[d]));
            }

            for (int d = 1; d < nDevs; ++d) {
                // push from 0
                CUDACHECK(cudaSetDevice(0));
                CUDACHECK(cudaMemcpyAsync(d_buffers[d], d_buffers[0],
                            HALF_BUF_SIZE, cudaMemcpyDefault, stream0[d]));

                // pull from dev d
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaMemcpyAsync(d_buffers[d]+HALF_COUNT, d_buffers[0]+HALF_COUNT,
                            (BUF_SIZE-HALF_BUF_SIZE), cudaMemcpyDefault, stream[d]));
            }

            CUDACHECK(cudaSetDevice(0));
            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaEventRecord(stop0[d], stream0[d]));
            }

            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaEventRecord(stop[d], stream[d]));
            }

            for (int d = 1; d < nDevs; ++d) {
                float ms;
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaStreamSynchronize(stream[d]));
                CUDACHECK(cudaEventElapsedTime(&ms, start[d], stop[d]));
                times[d] += ms;
            }

            CUDACHECK(cudaSetDevice(0));
            for (int d = 1; d < nDevs; ++d) {
                float ms;
                CUDACHECK(cudaStreamSynchronize(stream0[d]));
                CUDACHECK(cudaEventElapsedTime(&ms, start0[d], stop0[d]));
                times0[d] += ms;
            }
        }

        for (int d = 1; d < nDevs; ++d) {
            printf("dev 0 to dev %d: %f ms\n", d, times0[d] / nIters);
        }

        for (int d = 1; d < nDevs; ++d) {
            printf("dev %d: %f ms\n", d, times[d] / nIters);
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaMemcpy(h_result, d_buffers[d], BUF_SIZE, cudaMemcpyDefault));
            checkClose(h_result, h_buffer, COUNT);
            CUDACHECK(cudaEventDestroy(stop[d]));
            CUDACHECK(cudaEventDestroy(start[d]));
            CUDACHECK(cudaStreamDestroy(stream[d]));
        }
    }


    /*
     * Dest/source in push/pull (half each), with separate threads per device
     */

    {
        std::cout << std::endl
            << "Dest/source push/pull, separate threads per device" << std::endl;

        std::vector<cudaStream_t> stream(nDevs);
        std::vector<cudaStream_t> stream0(nDevs);
        std::vector<cudaEvent_t> start(nDevs);
        std::vector<cudaEvent_t> start0(nDevs);
        std::vector<cudaEvent_t> stop(nDevs);
        std::vector<cudaEvent_t> stop0(nDevs);
        std::vector<float> times(nDevs);
        std::vector<float> times0(nDevs);
        std::vector<std::thread> threads(nDevs);

        CUDACHECK(cudaSetDevice(0));
        for (int d = 0; d < nDevs; ++d) {
            CUDACHECK(cudaStreamCreate(&stream0[d]));
            CUDACHECK(cudaEventCreate(&start0[d]));
            CUDACHECK(cudaEventCreate(&stop0[d]));
            times0[d] = 0.f;
        }

        for (int d = 1; d < nDevs; ++d) {
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
            CUDACHECK(cudaEventCreate(&start[d]));
            CUDACHECK(cudaEventCreate(&stop[d]));
            times[d] = 0.f;
        }

        for (int i = 0; i < nIters; ++i) {
            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaEventRecord(start[d], stream[d]));
            }

            CUDACHECK(cudaSetDevice(0));
            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaEventRecord(start0[d], stream0[d]));
            }

            std::thread root(cudaCopyRoot<Float>, 0, nDevs, &stream0[0],
                    &d_buffers[0], d_buffers[0], HALF_BUF_SIZE);

            for (int d = 1; d < nDevs; ++d) {
                threads[d] = std::thread(cudaCopy<Float>, d, stream[d],
                        d_buffers[d]+HALF_COUNT, d_buffers[0]+HALF_COUNT,
                        (BUF_SIZE-HALF_BUF_SIZE));
            }

            root.join();
            for (int d = 1; d < nDevs; ++d) {
                threads[d].join();
            }

            CUDACHECK(cudaSetDevice(0));
            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaEventRecord(stop0[d], stream0[d]));
            }

            for (int d = 1; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaEventRecord(stop[d], stream[d]));
            }

            for (int d = 1; d < nDevs; ++d) {
                float ms;
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaStreamSynchronize(stream[d]));
                CUDACHECK(cudaEventElapsedTime(&ms, start[d], stop[d]));
                times[d] += ms;
            }

            CUDACHECK(cudaSetDevice(0));
            for (int d = 1; d < nDevs; ++d) {
                float ms;
                CUDACHECK(cudaStreamSynchronize(stream0[d]));
                CUDACHECK(cudaEventElapsedTime(&ms, start0[d], stop0[d]));
                times0[d] += ms;
            }
        }

        CUDACHECK(cudaSetDevice(0));
        for (int d = 1; d < nDevs; ++d) {
            printf("dev 0 to dev %d: %f ms\n", d, times0[d] / nIters);
            CUDACHECK(cudaEventDestroy(start0[d]));
            CUDACHECK(cudaEventDestroy(stop0[d]));
            CUDACHECK(cudaStreamDestroy(stream0[d]));
        }

        for (int d = 1; d < nDevs; ++d) {
            printf("dev %d: %f ms\n", d, times[d] / nIters);
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaMemcpy(h_result, d_buffers[d], BUF_SIZE, cudaMemcpyDefault));
            checkClose(h_result, h_buffer, COUNT);
            CUDACHECK(cudaEventDestroy(stop[d]));
            CUDACHECK(cudaEventDestroy(start[d]));
            CUDACHECK(cudaStreamDestroy(stream[d]));
        }
    }


    /*
     * Send via MST
     */

    {
        std::cout << std::endl << "MST broadcast" << std::endl;

        std::vector<cudaStream_t> stream(nDevs);
        std::vector<cudaEvent_t> event(nDevs);
        std::vector<cudaEvent_t> start(nDevs);
        std::vector<cudaEvent_t> stop(nDevs);
        std::vector<float> times(nDevs);

        for (int d = 0; d < nDevs; ++d) {
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
            CUDACHECK(cudaEventCreate(&event[d]));
            CUDACHECK(cudaEventCreate(&start[d]));
            CUDACHECK(cudaEventCreate(&stop[d]));
            times[d] = 0;
        }

        for (int iter = 0; iter < nIters; ++iter) {
            for (int d = 0; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaEventRecord(start[d], stream[d]));
            }

            broadcastMST(&d_buffers[0], COUNT, 0, 0, nDevs-1, &stream[0], &event[0]);

            for (int d = 0; d < nDevs; ++d) {
                CUDACHECK(cudaSetDevice(d));
                CUDACHECK(cudaEventRecord(stop[d], stream[d]));
                CUDACHECK(cudaStreamSynchronize(stream[d]));
                float ms;
                CUDACHECK(cudaEventElapsedTime(&ms, start[d], stop[d]));
                times[d] += ms;
            }
        }

        for (int d = 0; d < nDevs; ++d) {
            printf("dev %d: %f ms\n", d, times[d] / nIters);
            CUDACHECK(cudaSetDevice(d));
            CUDACHECK(cudaMemcpy(h_result, d_buffers[d], BUF_SIZE, cudaMemcpyDefault));
            checkClose(h_result, h_buffer, COUNT);
            CUDACHECK(cudaEventDestroy(stop[d]));
            CUDACHECK(cudaEventDestroy(start[d]));
            CUDACHECK(cudaEventDestroy(event[d]));
            CUDACHECK(cudaStreamDestroy(stream[d]));
        }
    }


    /*
     * Clean up
     */

    for (int d = 0; d < nDevs; ++d) {
        CUDACHECK(cudaFree(d_buffers[d]));
    }
    CUDACHECK(cudaFreeHost(h_buffer));
    CUDACHECK(cudaFreeHost(h_result));

    return 0;
}

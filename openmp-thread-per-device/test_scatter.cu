/**
 * A test program to evaluate the gpu collective communications
 */

#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <mpi.h>
#include <omp.h>

#include "collectives/collectives.h"

inline
unsigned long long rdtsc() {
    unsigned a, d;

    __asm__ volatile("rdtsc" : "=a" (a), "=d" (d));

    return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}

__global__
void square(double *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * data[idx];
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    cudaMPI_Init(MPI_COMM_WORLD);

    const int N = atoi(argv[1]);

    CUDAMPI_START_DEVICE_THREADS

        // get my rank
        int rank;
        cudaMPI_Comm_rank(&rank);

        // allocate some space on my device
        double *d_data;
        CUDACHECK( cudaMalloc(&d_data, N * sizeof(double)) );
        CUDACHECK( cudaMemset(d_data, 0, N * sizeof(double)) );

        // allocate some space on host for error checking
        double *h_data;
        CUDACHECK( cudaMallocHost(&h_data, N * sizeof(double)) );

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // if I am rank 0, generate data for everyone
        if (rank == 0) {
            int cuda_size;
            cudaMPI_Comm_size(MPI_COMM_WORLD, &cuda_size);

            int sendcount = N * cuda_size;
            double *d_sendbuf;
            double *h_sendbuf;
            CUDACHECK( cudaMalloc(&d_sendbuf, sendcount * sizeof(double)) );
            CUDACHECK( cudaMallocHost(&h_sendbuf, sendcount * sizeof(double)) );

            int i;
            for (i = 0; i < sendcount; ++i) {
                h_sendbuf[i] = i;
            }

            CUDACHECK( cudaMemcpy(d_sendbuf, h_sendbuf, sendcount * sizeof(double), cudaMemcpyDefault) );

            CUDACHECK( cudaEventRecord(start) );
            cudaMPI_Scatter(d_sendbuf, N, d_data, N, 0, MPI_COMM_WORLD);
            CUDACHECK( cudaEventRecord(stop) );

            CUDACHECK( cudaFreeHost(h_sendbuf) );
            CUDACHECK( cudaFree(d_sendbuf) );
        }
        else {
            CUDACHECK( cudaEventRecord(start) );
            cudaMPI_Scatter(NULL, 0, d_data, N, 0, MPI_COMM_WORLD);
            CUDACHECK( cudaEventRecord(stop) );
        }

        float time;
        CUDACHECK( cudaDeviceSynchronize() );
        CUDACHECK( cudaEventElapsedTime(&time, start, stop) );
        printf("rank %d took %f ms for the scatter\n", rank, time);

        // run a kernel on said data
        dim3 gridDim(128);
        dim3 blockDim(((N - 1) / gridDim.x) + 1);
        square<<<gridDim, blockDim>>>(d_data, N);

        // do a print to show the results
        CUDACHECK( cudaMemcpy(h_data, d_data, N * sizeof(double), cudaMemcpyDefault) );

        printf("rank %d results\n", rank);
        int i;
        for (i = 0; i < N; ++i) {
            printf("%lf, ", h_data[i]);
        }
        printf("\n");

        CUDACHECK( cudaFree(d_data) );
        CUDACHECK( cudaFreeHost(h_data) );

    CUDAMPI_STOP_DEVICE_THREADS

    cudaMPI_Finalize();
    MPI_Finalize();
}

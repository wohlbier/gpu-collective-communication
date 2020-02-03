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

        // if I am rank 0, fill data with something
        if (rank == 0) {
            int i;
            for (i = 0; i < N; ++i) {
                h_data[i] = i;
            }

            CUDACHECK( cudaMemcpy(d_data, h_data, N * sizeof(double), cudaMemcpyDefault) );
        }

        unsigned long long end, start;
        start = rdtsc();
        cudaMPI_Bcast(d_data, N, 0, MPI_COMM_WORLD);
        end = rdtsc();

        printf("bcast took %lld cycles on rank %d\n", (end - start), rank);

        // check non-root ranks to verify data
        if (rank != 0) {
            CUDACHECK( cudaMemcpy(h_data, d_data, N * sizeof(double), cudaMemcpyDefault) );
            int i;
            for (i = 0; i < N; ++i) {
                if (h_data[i] != i) {
                    fprintf(stderr, "data mismatch on rank %d: data[%d]=%lf\n",
                            rank, i, h_data[i]);
                    exit(EXIT_FAILURE);
                }
            }
        }

        // run a kernel on said data
        dim3 gridDim(128);
        dim3 blockDim(((N - 1) / gridDim.x) + 1);
        square<<<gridDim, blockDim>>>(d_data, N);

        CUDACHECK( cudaFree(d_data) );
        CUDACHECK( cudaFreeHost(h_data) );

    CUDAMPI_STOP_DEVICE_THREADS

    cudaMPI_Finalize();
    MPI_Finalize();
}

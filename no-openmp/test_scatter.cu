/**
 * A test program to evaluate the gpu collective communications
 */

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "collectives/collectives.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    cudaMPI_Init(MPI_COMM_WORLD);

    if (argc < 2) {
        fprintf(stderr, "give a value for N\n");
        exit(EXIT_FAILURE);
    }
    const size_t N = atoi(argv[1]);
    size_t i;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int gpu_size;
    cudaMPI_Comm_size(MPI_COMM_WORLD, &gpu_size);

    // find how many GPUs are connected here
    int deviceCount;
    CUDACHECK( cudaGetDeviceCount(&deviceCount) );

    printf("rank %d has %d cuda device(s)\n", rank, deviceCount);

    // create some timers
    cudaEvent_t *start = (cudaEvent_t*)malloc(deviceCount * sizeof(cudaEvent_t));
    cudaEvent_t *stop = (cudaEvent_t*)malloc(deviceCount * sizeof(cudaEvent_t));
    int device;
    for (device = 0; device < deviceCount; ++device) {
        CUDACHECK( cudaSetDevice(device) );
        CUDACHECK( cudaEventCreate(start+device) );
        CUDACHECK( cudaEventCreate(stop+device) );
    }

    double **d_data = NULL;
    CUDACHECK( cudaMallocHost((void **)&d_data, deviceCount * sizeof(double*)) );
    for (device = 0; device < deviceCount; ++device) {
        CUDACHECK( cudaSetDevice(device) );
        CUDACHECK( cudaMalloc((void **)&(d_data[device]), N * sizeof(double)) );
    }

    // allocate some host space for data filling/checking
    double *h_data = NULL;
    CUDACHECK( cudaMallocHost((void **)&h_data, N * sizeof(double)) );

    if (rank == 0) {
        CUDACHECK( cudaFree(d_data[0]) );
        CUDACHECK( cudaMalloc(&d_data[0], N * gpu_size * sizeof(double)) );

        int val = 0;
        int gpu;
        for (gpu = 0; gpu < gpu_size; ++gpu) {
            for (i = 0; i < N; ++i) {
                h_data[i] = (double)val++;
            }

            CUDACHECK( cudaMemcpy(d_data[0] + gpu*N, h_data,
                            N * sizeof(double), cudaMemcpyDefault) );
        }

        // zero out other buffers
        for (device = 1; device < deviceCount; ++device) {
            CUDACHECK( cudaMemset(d_data[device], 0, N * sizeof(double)) );
        }
    }
    else {
        // zero out other buffers
        for (device = 0; device < deviceCount; ++device) {
            CUDACHECK( cudaMemset(d_data[device], 0, N * sizeof(double)) );
        }
    }

    // time the scatter
    for (device = 0; device < deviceCount; ++device) {
        CUDACHECK( cudaSetDevice(device) );
        CUDACHECK( cudaEventRecord(start[device]) );
    }

    cudaMPI_Scatter(d_data[0], N, d_data, N, 0, 0);

    for (device = 0; device < deviceCount; ++device) {
        CUDACHECK( cudaSetDevice(device) );
        CUDACHECK( cudaEventRecord(stop[device]) );
        CUDACHECK( cudaDeviceSynchronize() );
    }

    // check first 10 elements of each device
    for (device = 0; device < deviceCount; ++device) {
        printf("device %d data: ", device);
        CUDACHECK( cudaMemcpy(h_data, d_data[device], 10 * sizeof(double), cudaMemcpyDefault) );
        for (i = 0; i < 10; ++i) {
            printf("%f ", h_data[i]);
        }
        printf("\n");
    }

    for (device = 0; device < deviceCount; ++device) {
        float time;
        CUDACHECK( cudaEventElapsedTime(&time, start[device], stop[device]) );
        printf("dev %d time %f ms\n", device, time);
    }

    for (device = 0; device < deviceCount; ++device) {
        CUDACHECK( cudaFree(d_data[device]) );
    }

    CUDACHECK( cudaFreeHost(d_data) );
    CUDACHECK( cudaFreeHost(h_data) );

    cudaMPI_Finalize();
    MPI_Finalize();
}

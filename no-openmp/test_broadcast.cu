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


    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // find how many GPUs are connected here
    int deviceCount;
    CUDACHECK( cudaGetDeviceCount(&deviceCount) );
    printf("rank %d has %d cuda device(s)\n", rank, deviceCount);


    double **sendbuf = (double**)malloc(deviceCount * sizeof(double*));
    double **recvbuf = (double**)malloc(deviceCount * sizeof(double*));
    cudaEvent_t *start = (cudaEvent_t*)malloc(deviceCount * sizeof(cudaEvent_t));
    cudaEvent_t *stop = (cudaEvent_t*)malloc(deviceCount * sizeof(cudaEvent_t));
    cudaStream_t *s = (cudaStream_t*)malloc(deviceCount * sizeof(cudaStream_t));
    for (int device = 0; device < deviceCount; ++device) {
        CUDACHECK( cudaSetDevice(device) );
        CUDACHECK( cudaEventCreate(start+device) );
        CUDACHECK( cudaEventCreate(stop+device) );
        CUDACHECK( cudaStreamCreate(s+device) );
        CUDACHECK( cudaMalloc(&sendbuf[device], N * sizeof(double)) );
        CUDACHECK( cudaMalloc(&recvbuf[device], N * sizeof(double)) );
        CUDACHECK( cudaMemset(sendbuf[device], 0, N * sizeof(double)) );
        CUDACHECK( cudaMemset(recvbuf[device], 0, N * sizeof(double)) );
    }


    // allocate some host space for data filling/checking
    double *h_data = NULL;
    CUDACHECK( cudaMallocHost((void **)&h_data, N * sizeof(double)) );
    if (rank == 0) {
        // initialize some data on rank 0
        for (size_t i = 0; i < N; ++i) h_data[i] = (double)i;
        CUDACHECK( cudaMemcpy(sendbuf[0], h_data, N * sizeof(double), cudaMemcpyDefault) );
    }

    for (int device = 0; device < deviceCount; ++device) {
        CUDACHECK( cudaSetDevice(device) );
        CUDACHECK( cudaEventRecord(start[device]) );
    }

    cudaMPI_Bcast(sendbuf[0], recvbuf, N, 0, MPI_COMM_WORLD, s);

    // stop timers and synchronize
    for (int device = 0; device < deviceCount; ++device) {
        CUDACHECK( cudaSetDevice(device) );
        CUDACHECK( cudaEventRecord(stop[device]) );
        CUDACHECK( cudaDeviceSynchronize() );
    }

    // make sure the broadcast was successful
    for (int device = 0; device < deviceCount; ++device) {
        // skip root GPU
        memset(h_data, 0, N * sizeof(double));
        CUDACHECK( cudaMemcpy(h_data, recvbuf[device], N * sizeof(double), cudaMemcpyDefault) );

        for (size_t i = 0; i < N; ++i) {
            if (h_data[i] != (double)i) {
                fprintf(stderr, "data mismatch: h_data[%d]=%lf\n", (int)i, h_data[i]);
                exit(EXIT_FAILURE);
            }
        }
    }

    for (int device = 0; device < deviceCount; ++device) {
        float time;
        CUDACHECK( cudaEventElapsedTime(&time, start[device], stop[device]) );
        printf("dev %d time %f ms\n", device, time);
    }

    for (int device = 0; device < deviceCount; ++device) {
        CUDACHECK( cudaFree(sendbuf[device]) );
        CUDACHECK( cudaFree(recvbuf[device]) );
    }

    CUDACHECK( cudaFreeHost(h_data) );

    free(sendbuf);
    free(recvbuf);
    free(start);
    free(stop);
    free(s);

    cudaMPI_Finalize();
    MPI_Finalize();
}

#include <stdio.h>
#include <mpi.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <limits>

#define CUDACHECK(cmd) do {                                 \
    cudaError_t e = cmd;                                    \
    if( e != cudaSuccess ) {                                \
        printf("Failed: Cuda error %s:%d '%s'\n",           \
                __FILE__,__LINE__,cudaGetErrorString(e));   \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)


#define NCCLCHECK(cmd) do {                                 \
    ncclResult_t r = cmd;                                   \
    if (r!= ncclSuccess) {                                  \
        printf("Failed, NCCL error %s:%d '%s'\n",           \
                __FILE__,__LINE__,ncclGetErrorString(r));   \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)


/**
 * Scatter will be composed of a broadcast and then all-gather
 */
int main(int argc, char* argv[])
{
    if (argc < 2) {
        fprintf(stderr, "need a value for N\n");
        exit(EXIT_FAILURE);
    }
    const size_t sizePerDevice = atoi(argv[1]);

    int nDev = 0;
    if (argc > 2) {
        nDev = atoi(argv[2]);
    }
    int deviceCount;
    CUDACHECK( cudaGetDeviceCount(&deviceCount) );
    if (deviceCount < nDev || nDev == 0) {
        fprintf(stderr, "changing nDev from %d to %d\n", nDev, deviceCount);
        nDev = deviceCount;
    }

    const size_t size = sizePerDevice * nDev;


    // init MPI
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


    ncclUniqueId id;
    if (mpi_rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);


    int *device_counts = (int*)malloc(mpi_size * sizeof(int));
    MPI_Allgather(&nDev, 1, MPI_INT,
            device_counts, 1, MPI_INT, MPI_COMM_WORLD);
    int device_rank_start = 0, total_device_count = 0;
    for (int i = 0; i < mpi_size; ++i) {
        if (i < mpi_rank) device_rank_start += device_counts[i];
        total_device_count += device_counts[i];
    }


    //initializing NCCL, group API is required around ncclCommInitRank as it is
    //called across multiple GPUs in each thread/process
    ncclComm_t *comms = (ncclComm_t*)malloc(nDev * sizeof(ncclComm_t));
    NCCLCHECK(ncclGroupStart());
    for (int i=0; i<nDev; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommInitRank(comms+i, total_device_count, id, device_rank_start++));
    }
    NCCLCHECK(ncclGroupEnd());


    // set up host buffers to initialize data and check results
    float *h_buffer, *h_result;
    CUDACHECK(cudaMallocHost(&h_buffer, size * sizeof(float)));
    CUDACHECK(cudaMallocHost(&h_result, sizePerDevice * sizeof(float)));
    CUDACHECK(cudaMemset(h_result, 0, sizePerDevice * sizeof(float)));
    for (size_t i = 0; i < size; ++i) {
        h_buffer[i] = i;
    }


    //allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** tempbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
    cudaEvent_t* start = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);
    cudaEvent_t* stop = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);


    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(tempbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, sizePerDevice * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(tempbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, sizePerDevice * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s+i));
        CUDACHECK(cudaEventCreate(start+i));
        CUDACHECK(cudaEventCreate(stop+i));
    }


    // start timers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaEventRecord(start[i], s[i]));
    }


    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
        NCCLCHECK(ncclBroadcast((const void*)sendbuff[i], (void*)tempbuff[i],
                    size, ncclFloat, 0, comms[i], s[i]));
        NCCLCHECK(ncclReduceScatter((const void*)tempbuff[i],
                    (void*)recvbuff[i],sizePerDevice, ncclFloat, ncclMin,
                    comms[i], s[i]));
    }
    NCCLCHECK(ncclGroupEnd());


    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaEventRecord(stop[i], s[i]));
    }


    // check data
    float diff = std::numeric_limits<float>::epsilon();
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
        CUDACHECK(cudaMemcpy(h_result, recvbuff[i],
                    sizePerDevice * sizeof(float), cudaMemcpyDefault));

        for (size_t j = 0; j < sizePerDevice; ++j) {
            if ((h_result[j] - h_buffer[i*sizePerDevice + j]) > diff) {
                printf("dev %d: data mismatch at %ld: ref=%f, test=%f\n",
                        i, j, h_buffer[i*sizePerDevice + j], h_result[j]);
                break;
            }
        }
    }


    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }
    free(sendbuff);
    free(recvbuff);


    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);


    // report timings
    for (int i = 0; i < nDev; ++i) {
        float time;
        CUDACHECK(cudaEventElapsedTime(&time, start[i], stop[i]));
        printf("dev %d: %f ms\n", i, time);
    }


    free(comms);


    MPI_Finalize();


    printf("[MPI Rank %d] Success \n", mpi_rank);
    return 0;
}

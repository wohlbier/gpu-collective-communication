#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"

#include "common.h"


int main(int argc, char* argv[])
{
    // test data size
    const int defaultSize = 1000000;
    const int size = (argc > 1) ? atoi(argv[1]) : defaultSize;

    // number of devices
    int nSysDev;
    CUDACHECK(cudaGetDeviceCount(&nSysDev));
    const int nDev = (argc > 2) ? atoi(argv[2]) : nSysDev;
    if (nDev > nSysDev) {
        fprintf(stderr, "requested %d devs, only %d present\n", nDev, nSysDev);
        exit(EXIT_FAILURE);
    }


    ncclComm_t comms[nDev];
    int devs[nDev];
    for (int i = 0; i < nDev; ++i) devs[i] = i;


    //allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
    cudaEvent_t* start = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);
    cudaEvent_t* stop = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);


    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s+i));
        CUDACHECK(cudaEventCreate(start+i));
        CUDACHECK(cudaEventCreate(stop+i));
    }


    //initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));


    for (int d = 0; d < nDev; ++d)
        CUDACHECK(cudaEventRecord(start[d], s[d]));


    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
        NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i],
                    size, ncclFloat, ncclSum, comms[i], s[i]));
    NCCLCHECK(ncclGroupEnd());


    for (int d = 0; d < nDev; ++d)
        CUDACHECK(cudaEventRecord(stop[d], s[d]));


    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
        float ms;
        CUDACHECK(cudaEventElapsedTime(&ms, start[i], stop[i]));
        printf("dev %d: %f ms\n", i, ms);
    }


    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }
    free(s);
    free(start);
    free(stop);
    free(sendbuff);
    free(recvbuff);


    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);


    printf("Success \n");
    return 0;
}

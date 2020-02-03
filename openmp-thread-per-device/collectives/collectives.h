/**
 * Place to declare all collective communication prototypes
 */

#ifndef CUDAMPI_COLLECTIVES_H
#define CUDAMPI_COLLECTIVES_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUDACHECK(result) cudaCheck(result, __FILE__, __LINE__)
inline void cudaCheck(cudaError_t code, const char *filename, int const line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(\"%s\")\n", filename, line,
                code, cudaGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}

/**
 * Define a structure to hold GPU data (how many, what their IDs are, etc.)
 */
typedef struct cudaMPI_Comm {
} cudaMPI_Comm;

extern int total_num_devices;
extern int *device_counts;
extern int *device_rank_to_node;
extern int *device_id_to_rank;
extern int **has_peer_access;

void cudaMPI_Init(MPI_Comm comm);
void cudaMPI_Finalize(void);

void cudaMPI_Comm_size(MPI_Comm comm, int *size);
void cudaMPI_Comm_rank(int *rank);

#define CUDAMPI_START_DEVICE_THREADS\
    {\
        int _device_count;\
        CUDACHECK( cudaGetDeviceCount(&_device_count) );\
        omp_set_num_threads(_device_count);\
    }\
    _Pragma("omp parallel")\
    {\
        CUDACHECK( cudaSetDevice(omp_get_thread_num()) );

#define CUDAMPI_STOP_DEVICE_THREADS \
    }

/**
 * Emulates normal MPI Broadcast with the view of each GPU as a process.
 * I.e. the root GPU will send data to all other GPUs currently executing
 * within this MPI "program".
 *
 * data: array of pointers to buffers on GPUs of this host
 * count: number of elements in the buffers
 * root: rank of the broadcast root - i.e. gpu dev id that is broadcasting
 */
void cudaMPI_Bcast(double *buffer, int count,
        int root, MPI_Comm comm);

void cudaMPI_Scatter(const double *sendbuf, int sendcount,
        double *recvbuf, int recvcount,
        int root, MPI_Comm comm);

#ifdef __cplusplus
};
#endif

#endif /* CUDAMPI_COLLECTIVES_H */

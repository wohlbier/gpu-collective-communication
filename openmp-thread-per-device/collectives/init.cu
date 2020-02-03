#include <mpi.h>
#include <cuda.h>
#include <stdio.h>
#include <omp.h>

// this should not be a thing - this should be a common.h, cudaMPI.h, etc.
#include "./collectives.h"

int *device_counts;
int *device_rank_to_node;
int *device_id_to_rank;
int **has_peer_access;
int total_num_devices;

void cudaMPI_Init(MPI_Comm comm) {
    // things that need to happen in here:
    //  - report how many GPUs I have
    //  - assign each GPU a unique ID
    //  - keep track of which node owns each GPU
    //  - create map of who has peer access
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int num_devices;
    CUDACHECK( cudaGetDeviceCount(&num_devices) );

    // create space to hold how many GPUs each node has.
    // this can be used for the sendcounts variable in the appropriate MPI
    // routines
    device_counts = (int*)malloc(size * sizeof(int));

    MPI_Allgather(&num_devices, 1, MPI_INT,
            device_counts, 1, MPI_INT, MPI_COMM_WORLD);

    total_num_devices = 0;
    int i;
    for (i = 0; i < size; ++i) {
        total_num_devices += *(device_counts + i);
    }

    // create the map from device rank to MPI rank
    device_rank_to_node = (int*)malloc(total_num_devices * sizeof(int));
    device_id_to_rank = (int*)malloc(num_devices * sizeof(int));

    int j;
    int device_rank = 0;
    for (i = 0; i < size; ++i) {
        for (j = 0; j < device_counts[i]; ++j) {
            if (i == rank) {
                device_id_to_rank[j] = device_rank;
            }
            device_rank_to_node[device_rank++] = i;
        }
    }

    // figure out who has peer access to each other
    int device, peer;
    has_peer_access = (int**)malloc(num_devices * sizeof(int*));
    has_peer_access[0] = (int*)malloc(num_devices * num_devices * sizeof(int));

    for (device = 0; device < num_devices; ++device) {
        CUDACHECK( cudaSetDevice(device) );
        if (device != 0) {
            has_peer_access[device] = has_peer_access[0] + num_devices*device;
        }

        for (peer = 0; peer < num_devices; ++peer) {
            if (peer != device) {
                int *canAccessPeer = &has_peer_access[device][peer];
                CUDACHECK( cudaDeviceCanAccessPeer(canAccessPeer, device, peer) );

                if (*canAccessPeer) {
                    CUDACHECK( cudaDeviceEnablePeerAccess(peer, 0) );
                    printf("%d has peer access to %d\n", device, peer);
                }
            }
        }
    }
}

void cudaMPI_Comm_size(MPI_Comm comm, int *size) {
    *size = total_num_devices;
}

void cudaMPI_Comm_rank(int *rank) {
    *rank = device_id_to_rank[omp_get_thread_num()];
}

void cudaMPI_Finalize(void) {
    free(device_rank_to_node);
    free(device_id_to_rank);
    free(has_peer_access[0]);
    free(has_peer_access);
}

#include <cuda.h>
#include <mpi.h>

#include "./collectives.h"

/**
 * I am a single MPI process with N GPUs. I need to send to every other GPU
 * I know about - i.e. all GPUs on my node as well as other nodes.
 */
__host__
void cudaMPI_Scatter(const double *sendbuf, int sendcount,
        double **recvbuf, int recvcount,
        int root, MPI_Comm comm)
{
    int mpi_rank, mpi_size, mpi_root;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    mpi_root = device_rank_to_node[root];

    int i;
    double *host_temp = NULL;
    const double *src_buf;

    if (mpi_rank == mpi_root) {
        src_buf = sendbuf;

        if (mpi_size > 1) {
            // create host space to send and receive over MPI
            int data_size = sendcount * total_num_devices * sizeof(double);
            CUDACHECK( cudaMallocHost(&host_temp, data_size) );
            CUDACHECK( cudaMemcpy(host_temp, sendbuf, data_size, cudaMemcpyDeviceToHost) );

            int *sendcounts = (int*)malloc(mpi_size * sizeof(int));
            int *displs = (int*)malloc(mpi_size * sizeof(int));

            sendcounts[0] = device_counts[0] * sendcount;
            displs[0] = 0;

            for (i = 1; i < mpi_size; ++i) {
                sendcounts[i] = device_counts[i] * sendcount;
                displs[i] = displs[i-1] + sendcounts[i-1];
            }

            // don't send things back to myself
            sendcounts[mpi_rank] = 0;

            MPI_Scatterv(host_temp, sendcounts, displs, MPI_DOUBLE,
                    host_temp, recvcount, MPI_DOUBLE, mpi_rank, MPI_COMM_WORLD);

            free(displs);
            free(sendcounts);
        }
    }
    else {
        // will only hit this case when mpi_size > 1
        int data_size = recvcount * device_counts[mpi_rank] * sizeof(double);
        CUDACHECK( cudaMallocHost(&host_temp, data_size) );

        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                host_temp, recvcount, MPI_DOUBLE, mpi_root, MPI_COMM_WORLD);

        src_buf = host_temp;
    }

    // copy into GPU buffers. can't use async here cause we wanna free
    // the host buffer before we leave
    int device;
    for (device = 0; device < device_counts[mpi_rank]; ++device) {
        int offset = device_id_to_rank[device] * recvcount;
        CUDACHECK( cudaMemcpy(recvbuf[device], src_buf + offset, recvcount, cudaMemcpyDefault) );
    }

    if (host_temp) {
        CUDACHECK( cudaFree(host_temp) );
    }
}

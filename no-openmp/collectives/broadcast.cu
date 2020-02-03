#include <cuda_runtime.h>
#include <mpi.h>

#include "./collectives.h"

__host__
void cudaMPI_Bcast(double *sendbuf, double **recvbuf, size_t count,
        /* datatype, */ int root, MPI_Comm comm, cudaStream_t *s)
{
    int mpi_rank, mpi_size, mpi_root;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    mpi_root = device_rank_to_node[root];

    const double *src_buf = NULL;
    int device = 0;

    if (mpi_rank == mpi_root) {
        src_buf = sendbuf;

        if (mpi_size > 1) {
            // send over MPI
            double *host_temp;
            CUDACHECK( cudaMallocHost(&host_temp, count * sizeof(double)) );
            CUDACHECK( cudaMemcpy(host_temp, sendbuf, count * sizeof(double),
                        cudaMemcpyDefault) );

            MPI_Bcast(host_temp, count, MPI_DOUBLE, mpi_root, comm);

            CUDACHECK( cudaFreeHost(host_temp) );
        }
    }
    else {
        // set up to receive from MPI
        double *host_temp;
        CUDACHECK( cudaSetDevice(device) );
        CUDACHECK( cudaMallocHost(&host_temp, count * sizeof(double)) );
        
        MPI_Bcast(host_temp, count, MPI_DOUBLE, mpi_root, comm);

        // I want to copy to one device, then have everyone else copy from it
        // that way I can free the host and have all others be async
        CUDACHECK( cudaMemcpy(recvbuf[device], host_temp, count * sizeof(double),
                        cudaMemcpyDefault) );
        CUDACHECK( cudaFreeHost(host_temp) );
        src_buf = recvbuf[device++];
    }

    // TODO: do mst bcast with local GPUs

    for ( ; device < device_counts[mpi_rank]; ++device) {
        CUDACHECK( cudaSetDevice(device) );
        if (recvbuf[device] != src_buf) {
            CUDACHECK(cudaMemcpyAsync(recvbuf[device], src_buf,
                        count * sizeof(double), cudaMemcpyDefault, s[device]));
        }
    }
}

#include <cuda.h>
#include <mpi.h>
#include <omp.h>

#include "./collectives.h"

const double *src_buf = NULL;

__host__
void cudaMPI_Scatter(const double *sendbuf, int sendcount,
        double *recvbuf, int recvcount,
        int root, MPI_Comm comm)
{
    int tid = omp_get_thread_num();
    int rank = device_id_to_rank[tid];
    int mpi_root = device_rank_to_node[root];

    int mpi_rank, mpi_size;
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    if (rank == root) {
        src_buf = sendbuf;
    }

    double *host_tmp = NULL;
    if (mpi_size > 1) {
        if (rank == root) {
            // allocate space for all GPUs
            int total_count = total_num_devices * sendcount;
            CUDACHECK( cudaMallocHost(&host_tmp, total_count * sizeof(double)) );
            CUDACHECK( cudaMemcpy(host_tmp, sendbuf, total_count * sizeof(double), cudaMemcpyDefault) );

            int *sendcounts = (int*)malloc(mpi_size * sizeof(int));
            int *displs = (int*)malloc(mpi_size * sizeof(int));
            displs[0] = 0;

            int i;
            for (i = 0; i < mpi_size; ++i) {
                sendcounts[i] = device_counts[i] * sendcount;
                if (i > 0) {
                    displs[i] = displs[i-1] + sendcounts[i-1];
                }
            }

            // don't waste time sending things back here
            sendcounts[mpi_root] = 0;

            MPI_Scatterv(host_tmp, sendcounts, displs, MPI_DOUBLE,
                    host_tmp, 0, MPI_DOUBLE, mpi_root, comm);

            free(displs);
            free(sendcounts);
        }
        else {
            if (tid == 0) {
                CUDACHECK( cudaMallocHost(&host_tmp, recvcount * sizeof(double)) );

                MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                        host_tmp, recvcount, MPI_DOUBLE, mpi_root, comm);

                src_buf = host_tmp;
            }
        }
    }

    #pragma omp barrier

    // I want to put offset calculation here. If I am on the same MPI node,
    // this should be as simple as (src + (rank * sendcount)).
    // However, if I am not on the same MPI node, this should be
    // (src + (tid * sendcount))
    const double *mysrcbuf = (mpi_rank == mpi_root) ?
        src_buf + (rank * recvcount) : src_buf + (tid * recvcount);

    // TODO: bucket algorithm here
    CUDACHECK( cudaMemcpy(recvbuf, mysrcbuf, recvcount * sizeof(double), cudaMemcpyDefault) );

    #pragma omp barrier

    // clear out src pointer
    if (tid == 0) {
        src_buf = NULL;
    }

    if (host_tmp) {
        CUDACHECK( cudaFreeHost((host_tmp)) );
        host_tmp = NULL;
    }
}

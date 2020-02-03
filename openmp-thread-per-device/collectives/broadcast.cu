/**
 * Implementation of the broadcast collective communication for GPU
 */

#include <cuda.h>
#include <mpi.h>
#include <omp.h>

#include "./collectives.h"

static double *src_buf = NULL;

__host__
void cudaMPI_Bcast(double *buffer, int count,
        int root, MPI_Comm comm)
{
    int tid = omp_get_thread_num();
    int gpu_rank = device_id_to_rank[tid];

    int mpi_rank = device_rank_to_node[gpu_rank];
    int mpi_root = device_rank_to_node[root];

    int mpi_size;
    MPI_Comm_size(comm, &mpi_size);

    if (gpu_rank == root) {
        // I have the data, set src_buf for my node
        src_buf = buffer;
    }

    // only deal with MPI if there is more than one node
    if (mpi_size > 1) {
        if (mpi_rank == mpi_root) {
            // this node has the data and will do the sending
            if (gpu_rank == root) {
                // bring data to host to send out
                double *host_tmp;
                CUDACHECK( cudaMallocHost(&host_tmp, count * sizeof(double)) );
                CUDACHECK( cudaMemcpy(host_tmp, buffer, count * sizeof(double), cudaMemcpyDefault) );

                // send out data
                MPI_Bcast(host_tmp, count, MPI_DOUBLE, mpi_root, comm);

                // free host space
                CUDACHECK( cudaFreeHost(host_tmp) );
            }
        }
        else {
            // only thread 0 should actually make the MPI call
            if (tid == 0) {
                CUDACHECK( cudaMallocHost(&src_buf, count * sizeof(double)) );

                MPI_Bcast(src_buf, count, MPI_DOUBLE, mpi_root, comm);
            }
        }
    }

    // barrier to wait for src_buf to be ready
    #pragma omp barrier

    // copy data into my buffer
    if (gpu_rank != root) {
        CUDACHECK( cudaMemcpy(buffer, src_buf, count * sizeof(double), cudaMemcpyDefault) );
    }

    if (mpi_rank != mpi_root) {
        // do another barrier here so that tid 0 can free the data when done
        #pragma omp barrier

        if (tid == 0) {
            CUDACHECK( cudaFreeHost(src_buf) );
        }
    }
}

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-synchronization-cg
// typedef struct CUDA_LAUNCH_PARAMS_st {
//     CUfunction function;
//     unsigned int gridDimX;
//     unsigned int gridDimY;
//     unsigned int gridDimZ;
//     unsigned int blockDimX;
//     unsigned int blockDimY;
//     unsigned int blockDimZ;
//     unsigned int sharedMemBytes;
//     CUstream hStream;
//     void **kernelParams;
// } CUDA_LAUNCH_PARAMS;

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define CUDACHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

// rotate rows up from some row offset `rotate`
/*
e.g.
rotate 1
row 0 | a b c d   ->    row 1 | e f g h
row 1 | e f g h   ->    row 2 | i j k l
row 2 | i j k l   ->    row 3 | m n o p
row 3 | m n o p   ->    row 0 | a b c d
*/
#define DEBUG 0

// these are somewhat arbitrarily picked.
// tries to use fewer threads if there isn't enough data.
#define MAX_BLOCKS 32*1024
#define COPY_THREADS 4*32

#define max(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

#define min(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

// rotates used by bucket algo.
__global__ void rotate(
  double *dst, double *src, size_t width, int height, int rotate
) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  size_t n = (width*height);

  rotate = rotate % height;

  for (
    int dst_offset = index, src_offset = (index + rotate*width) % n;
    dst_offset < n;
    dst_offset += num_threads,
    src_offset = (src_offset + num_threads) % n
  ) {
    dst[dst_offset] = src[src_offset];
  }
}
__global__ void rotate_rev(
  double *dst, double *src, size_t width, int height, int rotate
) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  size_t n = (width*height);

  // row offset is going to be indexed forward,
  int row_offset = index % width;
  // but each successive col will precede the last.
  int col_offset = index / width;
  /*
  for rotate=2, height=4
  dst 0 1 2 3 | 4 5 6 7 | 8 9 A B | C D E F
  src 8 9 A B | 4 5 6 7 | 0 1 2 3 | C D E F
  */
  int num_cols = num_threads / width;
  src += row_offset;
  int dst_offset = col_offset*width + row_offset;
  int src_col = ((height + rotate - col_offset) % height);
  for (
    ;
    dst_offset < n;
    dst_offset += num_threads,
    src_col = (height + src_col - num_cols) % height
  ) {
    dst[dst_offset] = src[src_col*width];
  }
}


// packing used by mst.
__global__ void pack(
  double *dst, double *src, int value, int rank, int i
) {
  // int index = blockIdx.x * blockDim.x + threadIdx.x;
  // int num_threads = blockDim.x * gridDim.x;
  //
  // for (int j0 = 0; j0 < i; ++j0) {
  //
  // }
}

// assumes the device that the buffers belong to is the current device
void MPI_All2All_bucket(
  double** sendbufs, size_t sendcount,
	double** recvbufs, size_t recvcount,
  double** tempbufs,
  int size,
  // just use this for debugging
  double** host_buffers, int ln
) {

  size_t bytes_per_elem = sizeof(double);

  // rotate the data in sendbuf to tempbuf.
  for (int rank = 0; rank < size; rank++) {
    CUDACHECK( cudaSetDevice(rank) );
    size_t threads = min(sendcount, COPY_THREADS);
    size_t blocks = min(MAX_BLOCKS, max(1, sendcount / COPY_THREADS));
    rotate<<<blocks,threads>>>(tempbufs[rank], sendbufs[rank], sendcount, size, rank);
  }

// #if DEBUG
//   printf("after rotate\n");
//   for (int rank = 0; rank < size; rank++) {
//     printf("p%d\t\t", rank+1);
//     CUDACHECK( cudaSetDevice(rank) );
//     CUDACHECK( cudaMemcpy(host_buffers[rank], tempbufs[rank], ln * sizeof(double), cudaMemcpyDeviceToHost) );
//   }
//   printf("\n");
//
//   for(int i = 0; i != ln; ++i) {
//     for (int rank = 0; rank < size; rank++) {
//       printf("%f\t", host_buffers[rank][i]);
//     }
//     printf("\n");
//   }
//   printf("\n");
// #endif

  for (int i = 1; i < size; i++) {
    for (int rank = 0; rank < size; rank++) {
      CUDACHECK( cudaSetDevice(rank) );
      CUDACHECK( cudaDeviceSynchronize() );
    }

    // task(?) size
    int ts = size - i;

    if (i % 2 == 1) {
      // send to right,
      for (int rank = 0; rank < size; rank++) {
        int rank_right = (rank + 1) % size;
        CUDACHECK( cudaSetDevice(rank) );
        CUDACHECK( cudaMemcpyPeerAsync(
          recvbufs[rank_right] + i*recvcount, // dst ptr
          rank_right,                         // dst device
          tempbufs[rank] + i*sendcount,       // src ptr
          rank,                               // src device
          sendcount*ts*bytes_per_elem         // count
        ));
      }
      // copy my chunk of received buffer into tempbuf.
      for (int rank = 0; rank < size; rank++) {
        CUDACHECK( cudaSetDevice(rank) );
        CUDACHECK( cudaMemcpyAsync(
          tempbufs[rank] + i*recvcount,       // dst ptr
          recvbufs[rank] + i*recvcount,       // src ptr
          recvcount*bytes_per_elem,           // count
          cudaMemcpyDeviceToDevice            // kind
        ) );
      }
    } else {
      // send to right.
      for (int rank = 0; rank < size; rank++) {
        int rank_right = (rank + 1) % size;
        CUDACHECK( cudaSetDevice(rank) );
        CUDACHECK( cudaMemcpyPeerAsync(
          tempbufs[rank_right] + i*recvcount, // dst ptr
          rank_right,                         // dst device
          recvbufs[rank] + i*sendcount,       // src ptr
          rank,                               // src device
          sendcount*ts*bytes_per_elem         // count
        ));
      }
    }
  }
  // rotate the data in tempbuf to recvbuf.
  for (int rank = 0; rank < size; rank++) {
    CUDACHECK( cudaSetDevice(rank) );
    size_t threads = min(sendcount, COPY_THREADS);
    size_t blocks = max(1, sendcount / COPY_THREADS);
    rotate_rev<<<blocks,threads>>>(recvbufs[rank], tempbufs[rank], sendcount, size, rank);
  }

}


void MPI_All2All_mst(
  double** sendbufs, size_t sendcount,
	double** recvbufs, size_t recvcount,
  double** tempbufs,
  int size,
  // just use this for debugging
  double** host_buffers, int ln
) {
  size_t bytes_per_elem = sizeof(double);


  // copy data to the destination buffer
  for (int rank = 0; rank < size; rank++) {
    CUDACHECK( cudaSetDevice(rank) );
    CUDACHECK( cudaMemcpy(
      recvbufs[rank],                 // dst ptr
      sendbufs[rank],                 // src ptr
      size*sendcount*bytes_per_elem,  // count
      cudaMemcpyDeviceToDevice        // kind
    ) );
  }

  int hsize = size >> 1;
  int value = hsize;
  for (int i = 1; i < size; i *= 2) {
    for (int rank = 0; rank < size; rank++) {
      CUDACHECK( cudaSetDevice(rank) );

      int offset = ((rank & value) == 0);

      // pack the data
      for (int j0 = 0; j0 < i; ++j0) {
        CUDACHECK( cudaMemcpy(
          tempbufs[rank],                           // dst ptr
          recvbufs[rank] + (j0*(value << 1) + value*offset) * sendcount,  // src ptr
          value*sendcount*bytes_per_elem,         // count
          cudaMemcpyDeviceToDevice                  // kind
        ) );
      }
    }

    // point to point
    for (int rank = 0; rank < size; rank++) {
      CUDACHECK( cudaSetDevice(rank) );

      int srcdest = rank ^ value;

      CUDACHECK( cudaMemcpyPeer(
        tempbufs[rank] + 1 * size/2 * sendcount,    // dst ptr
        rank,                                       // dst device
        tempbufs[srcdest] + 0 * size/2 * sendcount, // src ptr
        srcdest,                                    // src device
        hsize * sendcount*bytes_per_elem            // count
      ));

      CUDACHECK( cudaMemcpyPeer(
        tempbufs[srcdest] + 1 * size/2 * sendcount, // dst ptr
        srcdest,                                    // dst device
        tempbufs[rank] + 0 * size/2 * sendcount,    // src ptr
        rank,                                       // src device
        hsize * sendcount*bytes_per_elem            // count
      ));

    }


    for (int rank = 0; rank < size; rank++) {
      CUDACHECK( cudaSetDevice(rank) );

      int offset = ((rank & value) == 0);

      // unpack the data.
      for (int j0 = 0; j0 < i; ++j0) {
        CUDACHECK( cudaMemcpy(
          recvbufs[rank] + (j0*(value << 1) + value*offset) * sendcount,  // dst ptr
          tempbufs[rank] + 1 * size/2 * sendcount,                        // src ptr
          value*sendcount*bytes_per_elem,                                 // count
          cudaMemcpyDeviceToDevice                                        // kind
        ) );
      }
    }

    value = (value >> 1);
  }
}


int main(int argc, char **argv) {
  int n = atoi(argv[1]);
  int p = atoi(argv[2]);

  // min of p and device count
  int device_count = 0;
  CUDACHECK( cudaGetDeviceCount(&device_count) );
  p = p > device_count ? device_count : p;

  int ln = n / p;
  int sn = ln / p;


  // allocate shared memory buffers to be used for communicating between
  // GPUs. Host mem buffers too??
  // should I not be using malloc managed?
  // "The pointer is valid on the CPU and on all GPUs in the system that support
  // managed memory. All accesses to this pointer must obey the Unified Memory
  // programming model."
  //The application can also guide the driver about memory usage patterns via
  // cudaMemAdvise. The application can also explicitly migrate memory to a
  // desired processor's memory via cudaMemPrefetchAsync.

  double **send_buffers;
  double **recv_buffers;
  double **temp_buffers;
  double **host_buffers;

  CUDACHECK( cudaMallocHost(&send_buffers, p * sizeof(double *)) );
  CUDACHECK( cudaMallocHost(&recv_buffers, p * sizeof(double *)) );
  CUDACHECK( cudaMallocHost(&temp_buffers, p * sizeof(double *)) );
  CUDACHECK( cudaMallocHost(&host_buffers, p * sizeof(double *)) );

  for (int device = 0; device < p; device++) {
    CUDACHECK( cudaSetDevice(device) );
    for (int peer = 0; peer < p; peer++) {
      int canAccessPeer = 0;
      cudaDeviceCanAccessPeer(&canAccessPeer, device, peer);
      if (canAccessPeer) {
        cudaDeviceEnablePeerAccess(peer, 0);
        printf("%d can access %d\n", device, peer);
      }
    }
    CUDACHECK( cudaMalloc(send_buffers + device, ln * sizeof(double)) );
    CUDACHECK( cudaMalloc(recv_buffers + device, ln * sizeof(double)) );
    CUDACHECK( cudaMalloc(temp_buffers + device, ln * sizeof(double)) );

    CUDACHECK( cudaMallocHost(host_buffers + device, ln * sizeof(double)) );
  }

  /*
  // using managed memory or set device and allocate buffers for each?
  double *send_buffer, *recv_buffer, *temp_buffer;
  // ln if allocating per device, n if allocating for all devices.
  CUDACHECK( cudaMallocManaged(&send_buffer, n * sizeof(double)) );
  CUDACHECK( cudaMallocManaged(&recv_buffer, n * sizeof(double)) );
  CUDACHECK( cudaMallocManaged(&temp_buffer, n * sizeof(double)) );
  */


  // if we're using peer communication, make sure all the devices we're using
  // can actually perform communicate with the nodes it needs to.

  // initialize data
  for (int device = 0; device < p; device++) {
    CUDACHECK( cudaSetDevice(device) );
    // initialize host buffer to be transferred to device
    for(int i = 0; i != ln; ++i) {
      host_buffers[device][i] = (device+1) * 1.0 + (i+1)*0.01;
    }
    // copy from host to device
    CUDACHECK( cudaMemcpy(send_buffers[device], host_buffers[device], ln * sizeof(double), cudaMemcpyHostToDevice) );
    CUDACHECK( cudaMemset(recv_buffers[device], 0, ln * sizeof(double)) );
    CUDACHECK( cudaMemset(temp_buffers[device], 0, ln * sizeof(double)) );
  }

#if DEBUG
  for (int device = 0; device < p; device++) {
    printf("p%d\t\t", device+1);
  }
  printf("\n");

  for(int i = 0; i != ln; ++i) {
    for (int device = 0; device < p; device++) {
      printf("%f\t", host_buffers[device][i]);
    }
    printf("\n");
  }
  printf("\n");
#endif

// device events for timing
cudaEvent_t *start = (cudaEvent_t *) malloc(p*sizeof(cudaEvent_t));
cudaEvent_t *stop = (cudaEvent_t *) malloc(p*sizeof(cudaEvent_t));
for (int device = 0; device < p; device++) {
  cudaSetDevice(device);
  CUDACHECK( cudaEventCreate(start + device) );
  CUDACHECK( cudaEventCreate(stop + device) );
}
CUDACHECK( cudaGetLastError() );
// Start
for (int device = 0; device < p; device++) {
  cudaSetDevice(device);
  CUDACHECK( cudaEventRecord(start[device]) );
}

// All to all

  // in the future, replace p with MPI_COMM_WORLD-type configuration for comm.
  MPI_All2All_bucket(
    send_buffers, sn,
    recv_buffers, sn,
    temp_buffers,
    p,
    host_buffers, ln
  );

  // MPI_All2All_mst(
  //   send_buffers, sn,
  //   recv_buffers, sn,
  //   temp_buffers,
  //   p,
  //   host_buffers, ln
  // );

  // Stop
  for (int device = 0; device < p; device++) {
    cudaSetDevice(device);
    CUDACHECK( cudaEventRecord(stop[device]) );
  }

  for (int device = 0; device < p; device++) {
    cudaSetDevice(device);
    cudaDeviceSynchronize();
    float time_ms;
    CUDACHECK( cudaEventElapsedTime(&time_ms, start[device], stop[device]) );

    printf("p%d: %f ms\n", device + 1, time_ms);
  }


#if DEBUG
  for (int device = 0; device < p; device++) {
    printf("p%d\t\t", device+1);
    CUDACHECK( cudaSetDevice(device) );
    CUDACHECK( cudaMemcpy(host_buffers[device], recv_buffers[device], ln * sizeof(double), cudaMemcpyDeviceToHost) );
  }
  printf("\n");

  for(int i = 0; i != ln; ++i) {
    for (int device = 0; device < p; device++) {
      printf("%f\t", host_buffers[device][i]);
    }
    printf("\n");
  }
#endif

  // Free buffers
  for (int device = 0; device < p; device++) {
    CUDACHECK( cudaFreeHost(host_buffers[device]) );

    CUDACHECK( cudaFree(send_buffers[device]) );
    CUDACHECK( cudaFree(recv_buffers[device]) );
    CUDACHECK( cudaFree(temp_buffers[device]) );
  }
}

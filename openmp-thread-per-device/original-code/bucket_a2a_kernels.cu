#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DEBUG 1

#define MAX_BLOCKS (32*1024)
// #define MAX_BLOCKS (13)
#define COPY_THREADS 128

#define max(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

#define min(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })


#define CUDACHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// rotates used by bucket algo.
__global__ void rotate(
  double *dst, double *src, size_t width, int height, int rotate
) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  size_t n = (width*height);

  rotate = rotate % height;

  #pragma unroll
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
  #pragma unroll
  for (
    ;
    dst_offset < n;
    dst_offset += num_threads,
    src_col = (height + src_col - num_cols) % height
  ) {
    dst[dst_offset] = src[src_col*width];
  }
}

__global__ void copy(double *dst, double *src, size_t num_elems) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  #pragma unroll
  for (int offset = index; offset < num_elems; offset += num_threads) {
    // might want to use shared memory? looked like that had higher throughtput
    // in the transpose experiments.
    dst[offset] = src[offset];
  }
}

void MPI_All2All_bucket(
  double* sendbuf, size_t sendcount,
	double* recvbuf, size_t recvcount,
  double* tempbuf,
  int size,
  // just use this for debugging
  double* host_buffer
) {
  size_t ln = sendcount * size;

  size_t threads = min(sendcount, COPY_THREADS);
  size_t blocks = min(MAX_BLOCKS, max(1, sendcount / COPY_THREADS));

  for (int rank = 0; rank < size; rank++) {
    CUDACHECK( cudaSetDevice(rank) );
    rotate<<<blocks,threads>>>(tempbuf + rank*ln, sendbuf + rank*ln, sendcount, size, rank);
    CUDACHECK( cudaGetLastError() );
  }

  #if DEBUG
    printf("after rotate\n");
    CUDACHECK( cudaMemset(host_buffer, 0, ln*size*sizeof(double)) );
    CUDACHECK( cudaMemcpy(host_buffer, tempbuf, ln*size*sizeof(double), cudaMemcpyDeviceToHost) );
      for (int i = 0; i < size; i++) {
        printf("p%d\t\t", i+1);
      }
      printf("\n");

      for (size_t j = 0; j < ln; j++) {
        for (int i = 0; i < size; i++) {
          printf("%f\t", host_buffer[i*ln + j]);
        }
        printf("\n");
      }
      printf("\n");
  #endif


  for (int i = 1; i < size; i++) {
    for (int rank = 0; rank < size; rank++) {
      CUDACHECK( cudaSetDevice(rank) );
      CUDACHECK( cudaDeviceSynchronize() );
    }

    // task(?) size
    int ts = size - i;

    if (i % 2 == 1) {
      // send to right.
      for (int rank = 0; rank < size; rank++) {
        int rank_left = (size + rank - 1) % size;
        CUDACHECK( cudaSetDevice(rank) );
        // CUDACHECK( cudaMemPrefetchAsync(
        //   tempbuf + rank_left*ln + i*sendcount,
        //   sendcount*ts*sizeof(double),
        //   rank
        // ) );
        copy<<<blocks, threads>>>(
          recvbuf + rank*ln + i*recvcount,
          tempbuf + rank_left*ln + i*sendcount,
          sendcount*ts
        );
        CUDACHECK( cudaGetLastError() );
      }
      // copy my chunk of received buffer into tempbuf.
      for (int rank = 0; rank < size; rank++) {
        CUDACHECK( cudaSetDevice(rank) );
        copy<<<blocks, threads>>>(
          tempbuf + rank*ln + i*recvcount,
          recvbuf + rank*ln + i*recvcount,
          sendcount
        );
        CUDACHECK( cudaGetLastError() );
      }
    } else {
      // send to right.
      for (int rank = 0; rank < size; rank++) {
        int rank_left = (size + rank - 1) % size;
        CUDACHECK( cudaSetDevice(rank) );
        // CUDACHECK( cudaMemPrefetchAsync(
        //   tempbuf + rank_left*ln + i*sendcount,
        //   sendcount*ts*sizeof(double),
        //   rank
        // ) );
        copy<<<blocks, threads>>>(
          tempbuf + rank*ln + i*recvcount,
          recvbuf + rank_left*ln + i*sendcount,
          sendcount*ts
        );
        CUDACHECK( cudaGetLastError() );
      }
    }

  }
  // rotate the data in tempbuf to recvbuf.
  for (int rank = 0; rank < size; rank++) {
    CUDACHECK( cudaSetDevice(rank) );
    rotate_rev<<<blocks,threads>>>(recvbuf + rank*ln, tempbuf + rank*ln, sendcount, size, rank);
    CUDACHECK( cudaGetLastError() );
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

  // Set peer access.
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
  }

  double *send_buffer;
  double *recv_buffer;
  double *temp_buffer;
  double *host_buffer;

  size_t size = n * sizeof(double);
  CUDACHECK( cudaMallocManaged(&send_buffer, size) );
  CUDACHECK( cudaMallocManaged(&recv_buffer, size) );
  CUDACHECK( cudaMallocManaged(&temp_buffer, size) );

  CUDACHECK( cudaMallocHost(&host_buffer, size) );

  // Initialize
  for (int i = 0; i < p; i++) {
    for (size_t j = 0; j < ln; j++) {
      host_buffer[i*ln + j] = (i+1)*1.0 + (j+1)*0.01;
    }
  }
  CUDACHECK( cudaMemcpy(send_buffer, host_buffer, size, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemset(recv_buffer, 0, size) );
  CUDACHECK( cudaMemset(temp_buffer, 0, size) );

#if DEBUG
  for (int i = 0; i < p; i++) {
    printf("p%d\t\t", i+1);
  }
  printf("\n");

  for (size_t j = 0; j < ln; j++) {
    for (int i = 0; i < p; i++) {
      printf("%f\t", host_buffer[i*ln + j]);
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

  for (int iters = 0; iters < 1; iters++) {
    // Start
    for (int device = 0; device < p; device++) {
      cudaSetDevice(device);
      CUDACHECK( cudaEventRecord(start[device]) );
    }

    // All to all
    MPI_All2All_bucket(send_buffer, sn, recv_buffer, sn, temp_buffer, p, host_buffer);

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
  }

  #if DEBUG
  CUDACHECK( cudaMemcpy(host_buffer, recv_buffer, size, cudaMemcpyDeviceToHost) );
    for (int i = 0; i < p; i++) {
      printf("p%d\t\t", i+1);
    }
    printf("\n");

    for (size_t j = 0; j < ln; j++) {
      for (int i = 0; i < p; i++) {
        printf("%f\t", host_buffer[i*ln + j]);
      }
      printf("\n");
    }
    printf("\n");
  #endif

  // -- Cleanup -------------
  CUDACHECK( cudaFreeHost(host_buffer) );

  CUDACHECK( cudaFree(temp_buffer) );
  CUDACHECK( cudaFree(recv_buffer) );
  CUDACHECK( cudaFree(send_buffer) );
}

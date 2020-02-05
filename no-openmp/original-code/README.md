bucket_all2all.cu:
  - contains implementations (bucket and minimum spanning tree) for all to all communication across CUDA devices.
  - Uses cudaMemcpyPeerAsync() for data movement.
  - Seems to work for 2 GPUs. 
  - uses bucket algorithm by default, MST is commented out in the driver function.
  - USAGE: `./bucket_all2all <# doubles shared by all GPUs> <# GPUs to cooperate in the all-to-all>`
  
bucket_a2a_kernels.cu:
  - similar to bucket_all2all.cu, but replaces cudaMemcpyPeerAsync() with our own copy kernel.
  - this might require unified memory.
  - does not currently appear to be working.
  
  All targets can be made with `make`.

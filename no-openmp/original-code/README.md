bucket_all2all.cu:
  - contains implementations (bucket and minimum spanning tree) for all to all communication across CUDA devices.
  - uses cudaMemcpyPeerAsync() for data movement.
  - seems to work for 2 GPUs.
  - uses bucket algorithm by default, MST is commented out in the driver function.
  - debug information (inputs, outputs) can be enabled or disabled by setting the macro DEBUG to 1 or 0.
  - USAGE: `./bucket_all2all <# doubles shared by all GPUs> <# GPUs to cooperate in the all-to-all>`

bucket_a2a_kernels.cu:
  - similar to bucket_all2all.cu, but replaces cudaMemcpyPeerAsync() with our own copy kernel.
  - this might require unified memory.
  - does not currently appear to be working.

  All targets can be made with `make`.


Someone was using devices 0,3,4,5.

Number of doubles is 2^31-1.

```
CUDA_VISIBLE_DEVICES=1,2,6,7,8,9,10,11,12,13,14,15 ./bucket_all2all 2147483647 12
0 can access 1
0 can access 2
0 can access 3
0 can access 4
0 can access 5
0 can access 6
0 can access 7
0 can access 8
0 can access 9
0 can access 10
0 can access 11
1 can access 0
1 can access 2
1 can access 3
1 can access 4
1 can access 5
1 can access 6
1 can access 7
1 can access 8
1 can access 9
1 can access 10
1 can access 11
2 can access 0
2 can access 1
2 can access 3
2 can access 4
2 can access 5
2 can access 6
2 can access 7
2 can access 8
2 can access 9
2 can access 10
2 can access 11
3 can access 0
3 can access 1
3 can access 2
3 can access 4
3 can access 5
3 can access 6
3 can access 7
3 can access 8
3 can access 9
3 can access 10
3 can access 11
4 can access 0
4 can access 1
4 can access 2
4 can access 3
4 can access 5
4 can access 6
4 can access 7
4 can access 8
4 can access 9
4 can access 10
4 can access 11
5 can access 0
5 can access 1
5 can access 2
5 can access 3
5 can access 4
5 can access 6
5 can access 7
5 can access 8
5 can access 9
5 can access 10
5 can access 11
6 can access 0
6 can access 1
6 can access 2
6 can access 3
6 can access 4
6 can access 5
6 can access 7
6 can access 8
6 can access 9
6 can access 10
6 can access 11
7 can access 0
7 can access 1
7 can access 2
7 can access 3
7 can access 4
7 can access 5
7 can access 6
7 can access 8
7 can access 9
7 can access 10
7 can access 11
8 can access 0
8 can access 1
8 can access 2
8 can access 3
8 can access 4
8 can access 5
8 can access 6
8 can access 7
8 can access 9
8 can access 10
8 can access 11
9 can access 0
9 can access 1
9 can access 2
9 can access 3
9 can access 4
9 can access 5
9 can access 6
9 can access 7
9 can access 8
9 can access 10
9 can access 11
10 can access 0
10 can access 1
10 can access 2
10 can access 3
10 can access 4
10 can access 5
10 can access 6
10 can access 7
10 can access 8
10 can access 9
10 can access 11
11 can access 0
11 can access 1
11 can access 2
11 can access 3
11 can access 4
11 can access 5
11 can access 6
11 can access 7
11 can access 8
11 can access 9
11 can access 10
p1: 915.198975 ms
p2: 903.658508 ms
p3: 905.317383 ms
p4: 904.611816 ms
p5: 909.433838 ms
p6: 907.225098 ms
p7: 909.788147 ms
p8: 929.731567 ms
p9: 914.544617 ms
p10: 915.040283 ms
p11: 915.392517 ms
p12: 920.590332 ms
```
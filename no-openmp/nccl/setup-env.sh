#!/bin/sh

export CFLAGS="-I $HOME/nccl/include -I $HOME/openmpi/include"
export LDFLAGS="-L $HOME/nccl/lib -I $HOME/openmpi/lib"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/nccl/lib:/usr/local/depot/cuda-10.0/lib64

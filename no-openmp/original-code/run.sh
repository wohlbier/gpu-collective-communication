#!/bin/bash
EXE=./bucket_all2all
NGPU=8

min=1
max=31

# subtract 1 for 32 bit overflow

for i in $(seq $min $max)
do
    bytes=$((2**$i-1))
    cmd="${EXE} ${bytes} ${NGPU}"
    echo "running: $cmd"
    eval $cmd
done

#!/bin/bash

N=256
T=8

python3 RandGen.py $N
nvcc life3d.cu -o life3d
./life3d $N $T data/data.in data/data.out

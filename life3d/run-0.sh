#!/bin/bash

N=256
T=8

python3 RandGen.py $N
nvcc life.cu -o life
./life $N $T data/data.in data/data.out

#!/usr/bin/env bash

set -e

mkdir -p build
nvcc src/solver_cuda.cu -Iinclude/ -c -o build/solver.o -arch=native -O3
nvc++ -L"$CUDA_PATH/lib" -lcublas -lcudart -Iinclude/ -I"$CUDA_PATH/include" src/main_solution.cpp src/test_solution.cpp build/solver.o -O3 -march=native -o build/cuda

#!/usr/bin/env bash

set -e

mkdir -p build
g++ src/main.cpp src/test.cpp src/solver.cpp -fopenmp -O3 -march=native -Iinclude -o build/cpu

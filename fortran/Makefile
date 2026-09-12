GPU_COMPILE:=nvfortran -mp=gpu -Minfo=mp -g -O3
# GPU_COMPILE:=flang -O3 -fopenmp -march=native --offload-arch=sm_80 -fopenmp-version=50 # Uncomment for flang

.PHONY: cpu gpu cublas cublas_sym clean

all: cpu gpu cublas cublas_sym

SOURCES = test.f90 main.f90 precision.f90

cpu: solver.f90 $(SOURCES)  | build
	gfortran -O3 -march=native $^ -Jbuild -o build/cpu

gpu: solver_gpu.f90 $(SOURCES) | build
	$(GPU_COMPILE) $^ -o build/gpu

cublas: cublas.f90 solver_cublas.f90 $(SOURCES) | build
	$(GPU_COMPILE) $^ -L$(CUDA_PATH)/lib64 -lcublas -o build/cublas

cublas_sym: cublas.f90 solver_cublas_sym.f90 $(SOURCES) | build
	$(GPU_COMPILE) $^ -L$(CUDA_PATH)/lib64 -lcublas -o build/cublas_sym

build:
	mkdir -p build

clean:
	rm -rf build *.mod

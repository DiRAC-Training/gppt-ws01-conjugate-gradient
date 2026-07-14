CXX = nvc++

HEADERS = $(wildcard include/*.hpp)

.PHONY: cpu cuda cuda_optimised cublas cublas_sym clean fmt

BLOCK_SIZE ?= 128
EXTRA_FLAGS="-DBLOCK_SIZE=$(BLOCK_SIZE)"

all: cpu fixme

build:
	mkdir build

# Regular C++ .cpp compilation
build/%.o: src/%.cpp $(HEADERS) build
	$(CXX) -Wall -O3 -march=native -fopenmp -DNDEBUG -Iinclude/ -I"$(CUDA_PATH)/include" $(EXTRA_FLAGS) -c $< -o $@

cpu: build/main.o build/test.o build/solver.o | build
	$(CXX) -fopenmp $^ -o build/$@

# .cu compilation
build/%.o: src/%.cu $(HEADERS) build
	nvcc -O3 -arch=native -Iinclude/ $(EXTRA_FLAGS) -c $< -o $@

fixme: build/solver_FIXME.o build/main_solution.o build/test_solution.o | build
	$(CXX) -L"$(CUDA_PATH)/lib" -lcublas -lcudart $^ -o build/$@

cuda: build/solver_cuda.o build/main_solution.o build/test_solution.o | build
	$(CXX) -L"$(CUDA_PATH)/lib" -lcublas -lcudart $^ -o build/$@

cuda_optimised: build/solver_cuda_opt.o build/main_solution.o build/test_solution.o | build
	$(CXX) -L"$(CUDA_PATH)/lib" -lcublas -lcudart $^ -o build/$@

cublas: build/solver_cublas.o build/main_solution.o build/test_solution.o | build
	$(CXX) -L"$(CUDA_PATH)/lib" -lcublas -lcudart $^ -o build/$@

cublas_sym: EXTRA_FLAGS+=-DDISABLE_MATVEC_SIMPLE_TEST
cublas_sym: build/solver_cublas_sym.o build/main_solution.o build/test_solution.o | build
	$(CXX) -L"$(CUDA_PATH)/lib" -lcublas -lcudart $^ -o build/$@

fmt:
	clang-format --style=llvm -i src/*.cpp src/*.cu $(HEADERS)

clean:
	rm -rf build

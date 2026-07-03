CXX = nvc++
CXXFLAGS = -std=c++17 -Wall -O3 -fopenmp -march=native -DNDEBUG -I$(INCLUDE_DIR)
LDFLAGS = -fopenmp

# GPU compute capability: 70 = Volta (TitanV), 80 = Ampere (A100).
ARCH = 80
# Host compiler for nvcc (e.g. g++, gcc-14). Empty = nvcc's default.
NVCC_CCBIN =
# Unified (managed) memory for the OpenMP-offload builds; set empty to disable.
GPU_MEM = # -gpu=mem:unified

NVCC = nvcc
# cuBLAS ships in the NVHPC SDK math_libs, alongside cuda under CUDA_PATH (on COSMA it is NVHPC_ROOT).
CUBLAS_INC ?= $(CUDA_PATH)/math_libs/include
CUBLAS_LIB ?= $(CUDA_PATH)/math_libs/lib64
CUDA_LIB   ?= $(CUDA_PATH)/cuda/lib64
NVCC_CXXFLAGS = -std=c++17 -O3 -DNDEBUG -arch=sm_$(ARCH) \
                $(if $(NVCC_CCBIN),-ccbin=$(NVCC_CCBIN)) -I$(INCLUDE_DIR) -I$(CUBLAS_INC)
CUDA_LDFLAGS = -L$(CUBLAS_LIB) -Wl,-rpath,$(CUBLAS_LIB) -lcublas \
               -L$(CUDA_LIB) -Wl,-rpath,$(CUDA_LIB) -lcudart

OMP_TARGET_FLAGS = -fopenmp -mp=gpu -gpu=cc$(ARCH) $(GPU_MEM)

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

HEADERS = $(wildcard $(INCLUDE_DIR)/*.hpp)
# main.o and test.o are each compiled per backend: host / offload / offload+interop.
MAIN     = $(BUILD_DIR)/main.o       # host (cpu)
MAIN_OMP = $(BUILD_DIR)/main_omp.o   # OpenMP offload (gpu, gpu_managed)
MAIN_IOP = $(BUILD_DIR)/main_iop.o   # offload + device-addr interop (cuda, cublas)
TEST     = $(BUILD_DIR)/test.o       # host (cpu)
TEST_OMP = $(BUILD_DIR)/test_omp.o   # OpenMP offload (gpu, gpu_managed)
TEST_IOP = $(BUILD_DIR)/test_iop.o   # offload + device-addr interop (cuda, cublas)

# Variants (each solver source is linked with main + test):
#   cpu         solver.cpp                  host OpenMP
#   gpu         solver_gpu.cpp              OpenMP target offload
#   gpu_managed solver_openmp_solution.cpp  OpenMP target, unified memory
#   cuda        solver_cuda.cu              CUDA one-thread-per-row + cuBLAS dot
#   cuda_opt    solver_cuda_opt.cu          CUDA warp-per-row (coalesced) + cuBLAS dot
#   cublas      solver_cublas.cu            cuBLAS Sgemv
#   cublas_sym  solver_cublas_sym.cu        cuBLAS Ssymv
.PHONY: cpu gpu gpu_managed cuda cuda_opt cublas cublas_sym clean fmt
.DEFAULT_GOAL := cpu

cpu:         $(BUILD_DIR)/conj_grad
gpu:         $(BUILD_DIR)/conj_grad_gpu
gpu_managed: $(BUILD_DIR)/conj_grad_gpu_managed
cuda:        $(BUILD_DIR)/conj_grad_cuda
cuda_opt:    $(BUILD_DIR)/conj_grad_cuda_opt
cublas:      $(BUILD_DIR)/conj_grad_cublas
cublas_sym:  $(BUILD_DIR)/conj_grad_cublas_sym

$(BUILD_DIR):
	mkdir -p $@

# Host objects (main, test, cpu solver)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA objects
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_CXXFLAGS) -c $< -o $@

# OpenMP-offload solver objects need the target flags
$(BUILD_DIR)/solver_openmp_solution.o: $(SRC_DIR)/solver_openmp_solution.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_TARGET_FLAGS) -c $< -o $@
$(BUILD_DIR)/solver_gpu.o: $(SRC_DIR)/solver_gpu.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_TARGET_FLAGS) -c $< -o $@

# main for the OpenMP-offload backends, and for the CUDA interop backends
$(BUILD_DIR)/main_omp.o: $(SRC_DIR)/main.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_TARGET_FLAGS) -c $< -o $@
$(BUILD_DIR)/main_iop.o: $(SRC_DIR)/main.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_TARGET_FLAGS) -DUSE_DEVICE_ADDR -c $< -o $@

# test for the OpenMP-offload backends, and for the CUDA interop backends
$(BUILD_DIR)/test_omp.o: $(SRC_DIR)/test.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_TARGET_FLAGS) -c $< -o $@
$(BUILD_DIR)/test_iop.o: $(SRC_DIR)/test.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_TARGET_FLAGS) -DUSE_DEVICE_ADDR -c $< -o $@

$(BUILD_DIR)/conj_grad: $(BUILD_DIR)/solver.o $(MAIN) $(TEST)
	$(CXX) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/conj_grad_gpu: $(BUILD_DIR)/solver_gpu.o $(MAIN_OMP) $(TEST_OMP)
	$(CXX) $(LDFLAGS) $(OMP_TARGET_FLAGS) $^ -o $@

$(BUILD_DIR)/conj_grad_gpu_managed: $(BUILD_DIR)/solver_openmp_solution.o $(MAIN_OMP) $(TEST_OMP)
	$(CXX) $(LDFLAGS) $(OMP_TARGET_FLAGS) $^ -o $@

$(BUILD_DIR)/conj_grad_cuda: $(BUILD_DIR)/solver_cuda.o $(MAIN_IOP) $(TEST_IOP)
	$(CXX) $(CXXFLAGS) $(OMP_TARGET_FLAGS) $(CUDA_LDFLAGS) $^ -o $@

$(BUILD_DIR)/conj_grad_cuda_opt: $(BUILD_DIR)/solver_cuda_opt.o $(MAIN_IOP) $(TEST_IOP)
	$(CXX) $(CXXFLAGS) $(OMP_TARGET_FLAGS) $(CUDA_LDFLAGS) $^ -o $@

$(BUILD_DIR)/conj_grad_cublas: $(BUILD_DIR)/solver_cublas.o $(MAIN_IOP) $(TEST_IOP)
	$(CXX) $(CXXFLAGS) $(OMP_TARGET_FLAGS) $(CUDA_LDFLAGS) $^ -o $@

$(BUILD_DIR)/conj_grad_cublas_sym: $(BUILD_DIR)/solver_cublas_sym.o $(MAIN_IOP) $(TEST_IOP)
	$(CXX) $(CXXFLAGS) $(OMP_TARGET_FLAGS) $(CUDA_LDFLAGS) $^ -o $@

fmt:
	clang-format --style=llvm -i $(SRC_DIR)/*.cpp $(SRC_DIR)/*.cu $(HEADERS)

clean:
	rm -rf $(BUILD_DIR)

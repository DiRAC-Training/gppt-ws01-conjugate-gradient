CXX=nvc++
CXXFLAGS = -std=c++17 -Wall -O3 -fopenmp -march=native -DNDEBUG -I$(INCLUDE_DIR)
LDFLAGS = -fopenmp

NVCC=nvcc
NVCC_CXXFLAGS = -std=c++17 -O3 -DNDEBUG -I$(INCLUDE_DIR)
NVCC_LDFLAGS = -lcublas

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

SOURCES=src/main.cpp src/test.cpp
HEADERS=$(wildcard $(INCLUDE_DIR)/*.hpp)
OBJECTS=$(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

CUDA_SOURCES=$(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJECTS=$(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SOURCES)) 

OMP_TARGET_FLAGS = -fopenmp -mp=gpu -gpu=mem:unified

.PHONY: all clean fmt omp cuda omp_managed_soln omp_unmanaged_soln

all: $(BUILD_DIR)/conj_grad

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/conj_grad: build/solver.o $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) $^ -o $@

omp: $(BUILD_DIR)/conj_grad_omp

$(BUILD_DIR)/conj_grad_omp: build/solver_offload.o $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) $(OMP_TARGET_FLAGS) $^ -o $@


omp_managed_soln: $(BUILD_DIR)/omp_managed_soln

$(BUILD_DIR)/omp_managed_soln: build/solver_openmp_solution_offload.o $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) $(OMP_TARGET_FLAGS) $^ -o $@


omp_unmanaged_soln: $(BUILD_DIR)/omp_unmanaged_soln

$(BUILD_DIR)/omp_unmanaged_soln: build/solver_openmp_unmanaged_solution_offload.o $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) $(OMP_TARGET_FLAGS) $^ -o $@

# When object filename has _offload at the end, make sure to compile with openmp target flags
$(BUILD_DIR)/%_offload.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_TARGET_FLAGS) -c $< -o $@


cuda_soln: $(BUILD_DIR)/cuda_soln

$(BUILD_DIR)/cuda_soln: build/solver_cuda_solution.o $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -lcublas -lcudart -L$(CUDA_PATH)/lib $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(BUILD_DIR) $(HEADERS)
	$(NVCC) $(NVCC_CXXFLAGS) -c $< -o $@


fmt:
	clang-format --style=llvm -i $(SOURCES) $(HEADERS) $(wildcard $(SRC_DIR)/*.cu)

clean:
	rm -rf $(BUILD_DIR)

FC ?= flang
FCFLAGS = -O3 -fopenmp -Wall -march=native
OFFLOAD = --offload-arch=sm_80

BUILD_DIR = build

# solver.f90 (cpu) and solver_gpu.f90 (gpu) are alternative implementations of
# solver_mod, each built with test.f90 and main.f90. gpu adds nvptx offloading.
SOLVER = solver.f90
BIN    = conj_grad
cpu:        SOLVER = solver.f90
cpu:        BIN    = conj_grad
gpu:        SOLVER = solver_gpu.f90
gpu:        BIN    = conj_grad_gpu
gpu:        FCFLAGS += $(OFFLOAD)

.DEFAULT_GOAL := cpu
.PHONY: cpu gpu program clean

cpu gpu: program

program: | $(BUILD_DIR)
	$(FC) $(FCFLAGS) -J$(BUILD_DIR) -c $(SOLVER) -o $(BUILD_DIR)/solver.o
	$(FC) $(FCFLAGS) -J$(BUILD_DIR) -c test.f90  -o $(BUILD_DIR)/test.o
	$(FC) $(FCFLAGS) -J$(BUILD_DIR) -c main.f90  -o $(BUILD_DIR)/main.o
	$(FC) $(FCFLAGS) $(BUILD_DIR)/solver.o $(BUILD_DIR)/test.o $(BUILD_DIR)/main.o -o $(BUILD_DIR)/$(BIN)

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) *.mod

FC = flang
FCFLAGS = -O3 -fopenmp -march=native
OFFLOAD = --offload-arch=sm_80
OMP50   = -fopenmp-version=50

# cuBLAS ships in the NVHPC SDK; override on the command line if paths differ.
CUBLAS_LIB ?= $(CUDA_PATH)/math_libs/lib64
CUDA_LIB   ?= $(CUDA_PATH)/cuda/lib64

BUILD_DIR = build

# solver.f90 (cpu), solver_gpu.f90 (gpu, nvptx offload), solver_cublas.f90
# (cublas, NVIDIA-only) and solver_cublas_sym.f90 (cublas_sym, Ssymv matvec) are
# alternative implementations of solver_mod, each built with test.f90 and main.f90.
SOLVER = solver.f90
BIN    = conj_grad
EXTRA_OBJ  =
EXTRA_LINK =

cpu:        SOLVER = solver.f90
cpu:        BIN    = conj_grad
gpu:        SOLVER = solver_gpu.f90
gpu:        BIN    = conj_grad_gpu
gpu:        FCFLAGS += $(OFFLOAD)
gpu_cm:     SOLVER = solver_gpu_cm.f90
gpu_cm:     BIN    = conj_grad_gpu_cm
gpu_cm:     FCFLAGS += $(OFFLOAD)
cublas:     SOLVER = solver_cublas.f90
cublas:     BIN    = conj_grad_cublas
cublas:     FCFLAGS += $(OFFLOAD) $(OMP50)
cublas:     EXTRA_OBJ  = $(BUILD_DIR)/cublas.o
cublas:     EXTRA_LINK = -L$(CUBLAS_LIB) -Wl,-rpath,$(CUBLAS_LIB) -lcublas \
                         -L$(CUDA_LIB) -Wl,-rpath,$(CUDA_LIB) -lcudart
cublas_sym: SOLVER = solver_cublas_sym.f90
cublas_sym: BIN    = conj_grad_cublas_sym
cublas_sym: FCFLAGS += $(OFFLOAD) $(OMP50)
cublas_sym: EXTRA_OBJ  = $(BUILD_DIR)/cublas.o
cublas_sym: EXTRA_LINK = -L$(CUBLAS_LIB) -Wl,-rpath,$(CUBLAS_LIB) -lcublas \
                         -L$(CUDA_LIB) -Wl,-rpath,$(CUDA_LIB) -lcudart

.DEFAULT_GOAL := cpu
.PHONY: cpu gpu cublas cublas_sym program clean

cpu gpu cublas cublas_sym: program

program: | $(BUILD_DIR)
	$(FC) $(FCFLAGS) -J$(BUILD_DIR) -c precision.f90 -o $(BUILD_DIR)/precision.o
	$(if $(EXTRA_OBJ),$(FC) $(FCFLAGS) -J$(BUILD_DIR) -c cublas.f90 -o $(BUILD_DIR)/cublas.o)
	$(FC) $(FCFLAGS) -J$(BUILD_DIR) -c $(SOLVER) -o $(BUILD_DIR)/solver.o
	$(FC) $(FCFLAGS) -J$(BUILD_DIR) -c test.f90  -o $(BUILD_DIR)/test.o
	$(FC) $(FCFLAGS) -J$(BUILD_DIR) -c main.f90  -o $(BUILD_DIR)/main.o
	$(FC) $(FCFLAGS) $(BUILD_DIR)/precision.o $(EXTRA_OBJ) $(BUILD_DIR)/solver.o $(BUILD_DIR)/test.o $(BUILD_DIR)/main.o -o $(BUILD_DIR)/$(BIN) $(EXTRA_LINK)

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) *.mod

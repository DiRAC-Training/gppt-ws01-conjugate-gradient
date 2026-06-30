FC = gfortran
FCFLAGS = -O3 -fopenmp -Wall -march=native
LDFLAGS = -fopenmp

SRC_DIR = .
BUILD_DIR = build

SOURCES = solver.f90 test.f90 main.f90
OBJECTS = $(patsubst %.f90,$(BUILD_DIR)/%.o,$(SOURCES))

.PHONY: all clean

all: $(BUILD_DIR)/conj_grad

$(BUILD_DIR):
	mkdir -p $@

# Dependencies
$(BUILD_DIR)/test.o: $(BUILD_DIR)/solver.o
$(BUILD_DIR)/main.o: $(BUILD_DIR)/solver.o $(BUILD_DIR)/test.o

$(BUILD_DIR)/%.o: %.f90 | $(BUILD_DIR)
	$(FC) $(FCFLAGS) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/conj_grad: $(OBJECTS) | $(BUILD_DIR)
	$(FC) $(LDFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR)

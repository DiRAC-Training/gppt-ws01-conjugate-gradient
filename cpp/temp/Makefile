LLVM_DIR = /opt/homebrew/opt/llvm
CXX = $(LLVM_DIR)/bin/clang++
CXXFLAGS = -std=c++17 -O3 -fopenmp -march=native -DNDEBUG -I$(LLVM_DIR)/include
LDFLAGS = -fopenmp -L$(LLVM_DIR)/lib

BUILD_DIR = build

.PHONY: all clean

all: $(BUILD_DIR)/fem_heat

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/solver.o: solver.cpp solver.hpp utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/utils.o: utils.cpp utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test.o: test.cpp test.hpp utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: main.cpp solver.hpp test.hpp utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/fem_heat: $(BUILD_DIR)/solver.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/test.o $(BUILD_DIR)/main.o
	$(CXX) $(LDFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR)

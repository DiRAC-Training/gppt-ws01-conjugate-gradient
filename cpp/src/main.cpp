#include <cstdlib>
#include <chrono>
#include <iostream>
#include <random>

#include "idx.hpp"
#include "solver.hpp"
#include "test.hpp"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

const bool RANDOMISE_SEED = true;

/// Generate b from Ax = b
void calc_b(float *b, const float *A, const float *x, const int n) {
// This is just a naive matrix-vector multiply.
// You could use matvec in solver.cpp if you trust it!
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    b[i] = 0.0;

    for (int j = 0; j < n; ++j) {
      b[i] += A[idx(i, j, n)] * x[j];
    }
  }
}

/// Fill x with random values
void fill_rand_vec(float *x, int size, float min, float max) {
  int seed = 42;

  if (RANDOMISE_SEED) {
    seed = std::chrono::system_clock::now().time_since_epoch().count();
  }
  srand(seed);

#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    x[i] = min + (static_cast<float> (rand()) / ( static_cast <float> (RAND_MAX/(max-min))));
  }
}

/// Create a random, positive-definite matrix
void generate_positive_definite(float *A, int n) {
  std::cout << "Generating matrix\n";
  // Create a totally random matrix
  auto B = std::vector<float>(n * n);
  fill_rand_vec(B.data(), n * n, 0.0, 1.0);

// Make a symmetric matrix
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[idx(i, j, n)] = (B[idx(i, j, n)] + B[idx(j, i, n)]) / 2.0;
    }
  }

  // Add n * identity to make diagonally dominant => positive definite
  for (int i = 0; i < n; ++i) {
    A[idx(i, i, n)] += fmax(float(n) / 32, 1.0);
  }
}

bool run_tests() {
  bool all_passed = true;
  if (!test_matvec_identity()) {
    all_passed = false;
    std::cout << "matvec identity failed!\n";
  }

  if (!test_matvec_simple()) {
    all_passed = false;
    std::cout << "matvec simple failed!\n";
  }

  if (!test_dot()) {
    all_passed = false;
    std::cout << "dot failed!\n";
  }

  return all_passed;
}

int main() {
  if (!run_tests()) {
    std::cout << "A test failed!\n";
    return -1;
  }

  const int n = 8192;
  const int cg_max_iter = 32;

  float *A = new float[n * n]; // note n*n
  float *x_soln = new float[n];
  float *b = new float[n];
  float *x = new float[n];

  // Initial conditions
  generate_positive_definite(A, n); // Generate positive def matrix
  std::cout << "Generating random solution\n";
  fill_rand_vec(x_soln, n, -1.0, 1.0); // Generate a random solution vector
  std::cout << "Generating right hand side\n";
  calc_b(b, A, x_soln,
         n); // Multiple matrix with solution to get RHS of equation, b
  std::fill(x, x + n, 0.0);

  // Solve
  auto start = high_resolution_clock::now();
  int iters = cg_solve(x, A, b, n, cg_max_iter);
  auto stop = high_resolution_clock::now();
  auto duration =
      duration_cast<std::chrono::microseconds>(stop - start).count();

  std::cout << "Performed " << iters << " iterations" << std::endl;
  std::cout << "Solve time: " << duration << " us" << std::endl;
  std::cout << "Time per iteration: " << duration / iters << " us" << std::endl;

  float av_error2 = 0.0;
  for (int i = 0; i < n; ++i) {
    av_error2 += std::fabs(x_soln[i] - x[i]);
  }
  av_error2 /= n;

  std::cout << "Average error = " << std::sqrt(av_error2) << "\n";

  delete[] A;
  delete[] x_soln;
  delete[] b;
  delete[] x;

  return 0;
}

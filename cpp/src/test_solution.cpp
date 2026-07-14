#include <cmath>
#include <iostream>

#include <cuda_runtime.h>

#include "idx.hpp"
#include "solver.hpp"
#include "test.hpp"

bool nearly_eql(real x, real y, real eps = 1e-6) {
  if(std::fabs(x - y) < eps) {
    return true;
  } else {
    std::cout << x << " != " << y << "\n";
    return false;
  }
}

/// Test matvec with an identity matrix
bool test_matvec_identity() {
  const int n = 3;

  real *A = nullptr;
  cudaMallocManaged(&A, n * n * sizeof(real));
  for (int i = 0; i < n; ++i) {
    A[idx(i, i, n)] = 1.0;
  }

  real *x = nullptr;
  cudaMallocManaged(&x, n * sizeof(real));
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = 3.0;

  real *y = nullptr;
  cudaMallocManaged(&y, n * sizeof(real));

  matvec(y, A, x, n);
  cudaDeviceSynchronize();

  // Because A is the identity matrix, y and x should be identical
  bool passed = true;
  for (int i = 0; i < n; ++i) {
    passed &= nearly_eql(x[i], y[i], 1e-6);
  }

  cudaFree(y);
  cudaFree(x);
  cudaFree(A);

  return passed;
}

bool test_matvec_simple() {
  const int n = 3;

  real *A = nullptr;
  cudaMallocManaged(&A, n * n * sizeof(real));

  // Create a matrix with known values
  real A_in[9] = {-1, -6, 2, 4, 3, 10, 0, -100, 1};
  std::copy(A_in, A_in + 9, A);

  // x and y_soln are calculated solutions to y = Ax.
  real *x = nullptr;
  cudaMallocManaged(&x, n * sizeof(real));
  real x_in[3] = {-1.0, 2.0, 0.0};
  std::copy(x_in, x_in + 3, x);

  // A * x =
  // -1*-1 + -6*2 +  2*0 =  -11
  //  4*-1 +  3*2 + 10*0 =    2
  //  0*-1 + -100*2 + 1*0 = -200
  real y_soln[3] = {-11, 2, -200};

  // Calculate y with our matvec test
  real *y = nullptr;
  cudaMallocManaged(&y, n * sizeof(real));
  matvec(y, A, x, n);
  cudaDeviceSynchronize();

  // Compare y to y_soln
  bool passed = true;
  for (int i = 0; i < n; ++i) {
    passed &= nearly_eql(y_soln[i], y[i], 1e-6);
  }

  cudaFree(y);
  cudaFree(x);
  cudaFree(A);

  return passed;
}

bool test_dot() {
  const int n = 1024;

  real *x = nullptr;
  cudaMallocManaged(&x, n * sizeof(real));
  real *y = nullptr;
  cudaMallocManaged(&y, n * sizeof(real));
  // Generate x and y and calculate their dot product in this loop
  real soln = 0.0;
  for (int i = 0; i < n; ++i) {
    const real p = real(i) / real(n);
    x[i] = p;
    y[i] = p;
    soln += p * p;
  }

  // Calculate dot with the function and test against above value
  real res = dot(x, y, n);
  cudaDeviceSynchronize();

  cudaFree(y);
  cudaFree(x);

  return nearly_eql(res, soln, 1e-3);
}

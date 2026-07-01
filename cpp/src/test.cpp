#include <cmath>
#include <vector>

#include "idx.hpp"
#include "solver.hpp"
#include "test.hpp"

bool nearly_eql(float x, float y, float epsilon) {
  return std::fabs(x - y) < epsilon;
}

/// Test matvec with an identity matrix
bool test_matvec_identity() {
  const int n = 3;

  float A[n * n] = {0};
  for (int i = 0; i < n; ++i) {
    A[idx(i, i, n)] = 1.0;
  }
  float x[3] = {1.0, 2.0, 3.0};
  float y[3];

  matvec(y, A, x, n);

  // Because A is the identity matrix, y and x should be identical
  bool passed = true;
  for (int i = 0; i < n; ++i) {
    passed &= nearly_eql(x[i], y[i], 1e-6);
  }

  return passed;
}

bool test_matvec_simple() {
  const int n = 3;

  // Create a matrix with known values
  float mat[9] = {-1, -6, 2, 4, 3, 10, 0, -100, 1};

  // x and y_soln are calculated solutions to y = Ax.
  float x[3] = {-1.0, 2.0, 0.0};

  // mat * x =
  // -1*-1 + -6*2 +  2*0 =  -11
  //  4*-1 +  3*2 + 10*0 =    2
  //  0*-1 + -100*2 + 1*0 = -200
  float y_soln[3] = {-11, 2, -200};

  // Calculate y with our matvec test
  float y[3];
  matvec(y, mat, x, n);

  // Compare y to y_soln
  bool passed = true;
  for (int i = 0; i < n; ++i) {
    passed &= nearly_eql(y_soln[i], y[i], 1e-6);
  }

  return passed;
}

bool test_dot() {
  const int n = 1024;

  // Generate x and y and calculate their dot product in this loop
  auto x = std::vector<float>(n);
  auto y = std::vector<float>(n);
  float soln = 0.0;
  for (int i = 0; i < n; ++i) {
    const float p = float(i) / float(n);
    x[i] = p;
    y[i] = p;
    soln += p * p;
  }

  // Calculate dot with the function and test against above value
  const float res = dot(x.data(), y.data(), n);
  return nearly_eql(res, soln, 1e-3);
}

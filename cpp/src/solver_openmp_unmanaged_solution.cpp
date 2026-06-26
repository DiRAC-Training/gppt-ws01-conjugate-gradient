#include <cstdio>
#include <cstring>
#include <limits>

#include "solver.hpp"
#include "idx.hpp"

/// Dense matrix-vector product: y = A * x.
void matvec(float *y, const float* A, const float *x, const int n) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < n; i++) {
    float sum = 0.0;
    for (int j = 0; j < n; j++)
      sum += A[idx(i,j,n)] * x[j];
    y[i] = sum;
  }
}

/// Dot product: result = sum(a[i] * b[i]).
float dot(const float *a, const float *b, const int n) {
  float sum = 0.0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (int i = 0; i < n; i++)
    sum += a[i] * b[i];
  return sum;
}

/// AXPBY operation: y = alpha * x + beta * y.
void axpby(float *y, const float *x, const float alpha, const float beta,
           const int n) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < n; i++)
    y[i] = alpha * x[i] + beta * y[i];
}

/// Solve A*x = b using the conjugate gradient method.
int cg_solve(float *x, const float *A, const float *b, const int n, const int max_iter) {
  float *r = new float[n];
  float *p = new float[n];
  float *A_times_p = new float[n];

  // Step 1: r_0 = f - K*x_0
  matvec(r, A, x, n);
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    r[i] = b[i] - r[i];

  // Step 2: p_0 = r_0
  std::memcpy(p, r, n * sizeof(float));

  float residual_sq_old = dot(r, r, n);
  int n_iter;

#pragma omp target data map(to : A[0 : n * n])                              \
    map(tofrom : x[0 : n], r[0 : n], p[0 : n]) map(alloc : A_times_p[0 : n])
  for (n_iter = 0; n_iter < max_iter; n_iter++) {
    // Step 3a: alpha_k = (r_k . r_k) / (p_k . K*p_k)
    matvec(A_times_p, A, p, n);
    float alpha = residual_sq_old / dot(p, A_times_p, n);

    // Step 3b: x_{k+1} = x_k + alpha_k * p_k
    axpby(x, p, alpha, 1.0, n);

    // Step 3c: r_{k+1} = r_k - alpha_k * K*p_k
    axpby(r, A_times_p, -alpha, 1.0, n);

    // Step 3d: beta_k = (r_{k+1} . r_{k+1}) / (r_k . r_k)
    //          p_{k+1} = r_{k+1} + beta_k * p_k
    float residual_sq_new = dot(r, r, n);

    if (residual_sq_new < std::numeric_limits<float>().epsilon())
      break;

    float beta = residual_sq_new / residual_sq_old;
    axpby(p, r, 1.0, beta, n);
    residual_sq_old = residual_sq_new;

    std::printf("%d: r = %.6e\n", n_iter, residual_sq_new / n);
  }

  delete[] r;
  delete[] p;
  delete[] A_times_p;

  return n_iter;
}

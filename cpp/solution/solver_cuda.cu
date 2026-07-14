#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "solver.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

// Wrapper struct providing a cuBLAS handle with automatic setup and teardown.
struct CublasHandle {
  cublasHandle_t handle;
  CublasHandle() {
    cublasCreate(&handle);
    cublasSetStream(handle, 0); // share the default stream with our kernels
  }
  ~CublasHandle() { cublasDestroy(handle); }
};
static CublasHandle cublas;

// Dense matrix-vector product kernel: y = A * x. One thread per row.
__global__ void matvec_kernel(float *y, const float *A, const float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0.0;
    for (int j = 0; j < n; j++)
      sum += A[i * n + j] * x[j];
    y[i] = sum;
  }
}

// AXPBY kernel: y = alpha * x + beta * y. One thread per element.
__global__ void axpby_kernel(float *y, const float *x, float alpha, float beta,
                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = alpha * x[i] + beta * y[i];
}

void matvec(float *y, const float *A, const float *x, const int n) {
  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // one thread per row
  matvec_kernel<<<grid, BLOCK_SIZE>>>(y, A, x, n);
}

void axpby(float *y, const float *x, const float alpha, const float beta,
           const int n) {
  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  axpby_kernel<<<grid, BLOCK_SIZE>>>(y, x, alpha, beta, n);
}

// Dot product: result = sum(a[i] * b[i]).
float dot(const float *a, const float *b, const int n) {
  float result = 0.0f;
  cublasSdot(cublas.handle, n, a, 1, b, 1, &result);
  return result;
}

// Solve A*x = b using the conjugate gradient method.
int cg_solve(float *x, const float *A, const float *b, const int n,
             const int max_iter) {
  float *r, *p, *A_times_p;
  cudaMalloc(&r, n * sizeof(float));
  cudaMalloc(&p, n * sizeof(float));
  cudaMalloc(&A_times_p, n * sizeof(float));

  // Step 1: r_0 = f - K*x_0
  matvec(r, A, x, n);
  axpby(r, b, 1.0, -1.0,
        n); // replacing a hand-written kernel for r = b - r with axpby
  // for (int i = 0; i < n; i++) {
  //   r[i] = b[i] - r[i];
  // }

  // Step 2: p_0 = r_0
  cudaMemcpy(p, r, n * sizeof(float), cudaMemcpyDeviceToDevice);

  float residual_sq_old = dot(r, r, n);
  int n_iter;

  for (n_iter = 0; n_iter < max_iter; n_iter++) {
    // Step 3a: alpha_k = (r_k . r_k) / (p_k . K*p_k)
    matvec(A_times_p, A, p, n);
    float alpha = residual_sq_old / dot(p, A_times_p, n);

    // Step 3b: x_{k+1} = x_k + alpha_k * p_k
    axpby(x, p, alpha, 1.0, n);

    // Step 3c: r_{k+1} = r_k - alpha_k * K*p_k
    axpby(r, A_times_p, -alpha, 1.0, n);

    float residual_sq_new = dot(r, r, n);

    // This method is so good it crashes if the residual gets too small!
    if (residual_sq_new < std::numeric_limits<float>().epsilon())
      break;

    // Step 3e: beta_k = (r_{k+1} . r_{k+1}) / (r_k . r_k)
    //          p_{k+1} = r_{k+1} + beta_k * p_k
    float beta = residual_sq_new / residual_sq_old;
    axpby(p, r, 1.0, beta, n);

    std::printf("%d: r = %.6e\n", n_iter, residual_sq_new / n);

    residual_sq_old = residual_sq_new;
  }

  cudaFree(r);
  cudaFree(p);
  cudaFree(A_times_p);

  return n_iter;
}

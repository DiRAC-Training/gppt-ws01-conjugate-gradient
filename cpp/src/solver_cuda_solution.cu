#include <cstdio>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "solver.hpp"

// Wrapper struct providing a cuBLAS handle with automatic setup and teardown.
struct CublasHandle {
  cublasHandle_t handle;
  CublasHandle() { cublasCreate(&handle); }
  ~CublasHandle() { cublasDestroy(handle); }
};
static CublasHandle cublas;

/// Dense matrix-vector product kernel: y = A * x. One thread per row.
__global__ void matvec_kernel(float *y, const float *A, const float *x,
                              int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  while (tid < n) {
    float sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[tid * n + j] * x[j];
    }
    y[tid] = sum;
    tid += num_threads;
  }
}

/// AXPBY kernel: y = alpha * x + beta * y. One thread per element.
__global__ void axpby_kernel(float *y, const float *x, float alpha, float beta,
                             int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  while (tid < n) {
    y[tid] = alpha * x[tid] + beta * y[tid];
    tid += num_threads;
  }
}

static const int BLOCK_SIZE = 1024;

void matvec(float *y, const float *A, const float *x, const int n) {
  // int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  matvec_kernel<<<32, BLOCK_SIZE>>>(y, A, x, n);
  cudaDeviceSynchronize();
}

void axpby(float *y, const float *x, const float alpha, const float beta,
           const int n) {
  // int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  axpby_kernel<<<32, BLOCK_SIZE>>>(y, x, alpha, beta, n);
  cudaDeviceSynchronize();
}

// Provided function - no need to port
float dot(const float *a, const float *b, const int n) {
  float result = 0.0f;
  cublasSdot(cublas.handle, n, a, 1, b, 1, &result);
  return result;
}

/**
 * @brief Solve A*x = b using the unpreconditioned conjugate gradient method.
 *
 * The CG iteration runs on the host. Each linear algebra operation is
 * a CUDA kernel launch. All vectors use managed memory so no explicit
 * data movement is required.
 *
 * @return Number of iterations performed.
 */
int cg_solve(float *x, const float *A, const float *b, const int n,
             const int max_iter) {
  float *r, *p, *A_times_p;
  cudaMallocManaged(&r, n * sizeof(float));
  cudaMallocManaged(&p, n * sizeof(float));
  cudaMallocManaged(&A_times_p, n * sizeof(float));

  // Step 1: r_0 = f - K*x_0
  matvec(r, A, x, n);
  axpby(r, b, 1.0, -1.0, n);
  // for (int i = 0; i < n; i++) {
  //   r[i] = b[i] - r[i];
  // }

  // Step 2: p_0 = r_0
  std::memcpy(p, r, n * sizeof(float));

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

    std::printf("%d: r = %.6e\n", n_iter, residual_sq_new);

    residual_sq_old = residual_sq_new;
  }

  cudaFree(r);
  cudaFree(p);
  cudaFree(A_times_p);

  return n_iter;
}

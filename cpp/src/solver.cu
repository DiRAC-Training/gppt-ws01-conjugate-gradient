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

// CUDA kernels

__global__ void matvec_kernel(float *y, const float *matrix_data,
                              const float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // TODO (exercise step 2):
    //     compute y[i] = sum over j of vals[i * n + j] * x[j]
  }
}

__global__ void axpby_kernel(float *y, const float *x, float alpha, float beta,
                             int n) {
  // TODO (exercise step 3):
  //     one thread per element; compute y[i] = alpha * x[i] + beta * y[i]
}

static const int BLOCK_SIZE = 256;

// Once matvec_kernel has been implemented above, this launcher
// will run on the GPU with no further changes needed.
void matvec(float *y, const float *A, const float *x, const int n) {
  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  matvec_kernel<<<grid, BLOCK_SIZE>>>(y, A, x, n);
  cudaDeviceSynchronize();
}

// TODO (exercise step 3): replace the CPU fallback below with a launch of
// axpby_kernel, following the same pattern as matvec() above. The fallback
// is only here so that intermediate builds (after porting matvec in step 2
// but before porting axpby) still produce correct answers; it relies on x
// and y being in managed memory, which you arranged in step 1.
void axpby(float *y, const float *x, const float alpha, const float beta,
           const int n) {
  for (int i = 0; i < n; i++)
    y[i] = alpha * x[i] + beta * y[i];
}

// Provided: dot product via cuBLAS. Do not modify.
float dot(const float *a, const float *b, const int n) {
  float result = 0.0f;
  cublasSdot(cublas.handle, n, a, 1, b, 1, &result);
  return result;
}

/**
 * @brief Solve A*x = b using the unpreconditioned conjugate gradient method.
 *
 * @return Number of iterations performed.
 */
int cg_solve(float *x, const float *A, const float *b, const int n,
             const int max_iter, const float tol) {
  // TODO (exercise step 1): convert these allocations to cudaMallocManaged
  float *r = new float[n];
  float *p = new float[n];
  float *A_times_p = new float[n];

  // Step 1: r_0 = f - K*x_0
  matvec(r, A, x, n);
  for (int i = 0; i < n; i++)
    r[i] = b[i] - r[i];

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

    // Step 3d: convergence check -- stop if ||r_{k+1}|| < tol
    float residual_sq_new = dot(r, r, n);

    // This method is so good it crashes if the residual gets too small!
    if (residual_sq_new < std::numeric_limits<float>().epsilon())
      break;

    // Step 3e: beta_k = (r_{k+1} . r_{k+1}) / (r_k . r_k)
    //          p_{k+1} = r_{k+1} + beta_k * p_k
    float beta = residual_sq_new / residual_sq_old;
    axpby(p, r, 1.0, beta, n);

    if (n_iter % 50 == 0)
      std::printf("%d: r = %.6e\n", n_iter, residual_sq_new);

    residual_sq_old = residual_sq_new;
  }

  // TODO (exercise step 1): convert these deallocations to cudaFree
  delete[] r;
  delete[] p;
  delete[] A_times_p;

  return n_iter;
}

# Recap exercise: porting a conjugate gradient algorithm

In this exercise you will be guided through the porting of a conjugate gradient algorithm from an existing CPU implementation to a GPU implementation. You do not need to understand the details of the algorithm but the components you will be porting include a matrix-vector multiplication, a dot product and a type of vector addition. Feel free to use the provided solutions to help.

## The problem

The conjugate gradient finds the solution $x$ to the matrix equation $A\vec{x} = \vec{b}$ for a given matrix $A$ and vector $b$[^pos_def]. In our code we will be randomly generating both $A$ and $x$, calculating $b$, then using our algorithm to come up with the (known) solution $x$.

[^pos_def]: For the CG algorithm to work it must be *positive-definite* and *diagonally dominant* which we ensure with the particular way the matrix is randomly generated.

The **conjugate gradient (CG)** method is found in `solver.cpp`. This is the part of the code you will port. The CG loop itself runs on the host; each iteration calls three linear algebra routines, and these are what we will move onto the GPU:

- `matvec` — dense matrix–vector product $\mathbf{y} = K \mathbf{x}$
- `axpby`  — vector update $\mathbf{y} = \alpha \mathbf{x} + \beta \mathbf{y}$
- `dot`    — inner product $\mathbf{a} \cdot \mathbf{b}$

## Project layout

```bash
Makefile
include          // You will probably not touch these
├── idx.hpp
├── solver.hpp
└── test.hpp
src
├── main.cpp
├── solver.cpp   // START HERE FOR OPENMP (also sample CPU version)
├── solver.cu    // START HERE FOR CUDA
├── test.cpp
├── solver_cuda_solution.cu
├── solver_openmp_solution.cpp
└── solver_openmp_unmanaged_solution.cpp
```

## Running the baseline

Build and run the CPU version first so you have a reference for correctness and timing:

```bash
make && build/conj_grad
```

You should see output like:

```
Generating matrix
Generating random solution
Generating right hand side
0: r = 2.397803e+02
1: r = 5.865565e+01
2: r = 5.038668e-01
3: r = 5.082916e-03
4: r = 5.817979e-05
5: r = 2.122006e-04
6: r = 5.293751e-07
7: r = 5.628715e-09
8: r = 5.867202e-11
Performed 9 iterations
Solve time: 34303 us
Time per iteration: 3811 us
Average error = 0.000652638
```

Take note of the **time per iteration** for this CPU version. This is ultimately what you are trying to improve in the GPU port.

## Choose your porting route

You now have a choice of two porting routes. Pick **one** and work through it end-to-end:

- **Route A — OpenMP target offload.** You stay in `solver.cpp` and add directives to each kernel to offload it to the GPU. This is a good starting point if you are already comfortable with OpenMP and less interested in CUDA.
- **Route B — CUDA.** You rewrite each kernel as an explicit CUDA `__global__` function and manage memory through the CUDA runtime. More verbose, but gives you finer control and is closer to what a CUDA port looks like. Suitable for thoes already familiar with OpenMP offloading and wanting more of a challenge.

Both routes assume **managed memory** throughout (`#pragma omp requires unified_shared_memory` / `cudaMallocManaged`). This avoids explicit host/device copies at this stage; the consequences of that choice, and how to manage memory explicitly, are discussed in later modules.

In both routes you will port the kernels one at a time and test after each. An important thing to understand up front is that **you should not expect to see a speedup until all three kernels are on the GPU**. With managed memory, if even one kernel still runs on the host, the data will migrate back and forth between host and device on every CG iteration, and that migration cost will dominate the solve time. Early intermediate builds will often be *slower* than the CPU baseline. This is expected.

### Before you start: pick the right Makefile

The default `Makefile` builds the CPU baseline. Two alternatives are provided for the GPU routes. Copy the one matching your chosen route over the top of `Makefile`:

```bash
# Route A (OpenMP target offload)
cp Makefile_omp Makefile

# Route B (CUDA)
cp Makefile_cuda Makefile
```

We will learn how to adapt the build configurations to compile the code for either CPU or GPU in the [Architecture 1 module](TODO: link to module). For now, the important things to note are:

---

## Route A — OpenMP target offload

You will work entirely inside `solver.cpp`. No new files.

### Step 1: Declare unified shared memory

At the top of `solver.cpp`, after the `#include`s, add:

```cpp
#pragma omp requires unified_shared_memory
```

This tells the OpenMP runtime that host and device share a single address space, so pointers allocated on the host with `new` can be dereferenced from inside `target` regions without explicit `map` clauses.

### Step 2: Offload `matvec`

Find the `matvec` function. Replace the `#pragma omp parallel for` with a target-offload directive that distributes the outer loop across teams and threads on the GPU.

**Hint:** the directive you need is `#pragma omp target teams distribute parallel for`. You may also need a local pointer to the raw matrix storage (`A.vals`) captured by the target region — accessing `A(i, j)` through the `DenseMatrix` struct from inside a target region can be awkward.

**Test.** Rebuild and run. The correctness check should still pass. The solve time will likely be **much worse** than the CPU baseline at this point, because `dot` and `axpby` still run on the host and force data to migrate back every iteration.

### Step 3: Offload `axpby`

Apply the same pattern to `axpby`. It is a straightforward element-wise update with no reduction. Test again — correctness should still hold, performance is still expected to be poor.

### Step 4: Offload `dot`

`dot` is different: it contains a **reduction**. The OpenMP `reduction` clause that you use on the CPU (`reduction(+ : sum)`) also works with `target teams distribute parallel for`. Add it to the directive.

**Test.** Now that all three kernels run on the GPU, managed memory should keep the vectors resident on the device across iterations. You should see a meaningful speedup over the CPU baseline.

---

## Route B — CUDA

A starter `solver.cu` is provided alongside `solver.cpp`. You will work in `solver.cu` for the ported linear algebra routines, and switch a few allocations to `cudaMallocManaged` throughout the code so that host-assembled data is visible to the GPU kernels.

The starter file already contains the scaffold for you:

- `cg_solve` with the full CG iteration, using `new[]` allocations (you convert these in Step 1).
- A `matvec_kernel` with an empty body (See Step 2) and a complete launcher that calls it.
- An `axpby_kernel` with an empty body (See Step 3) and a temporary host-fallback launcher marked for replacement in Step 3.
- A provided `dot` function using cuBLAS, including handle setup — do not modify.

### Step 1: Switch allocations to managed memory

CUDA kernels need their input pointers to refer to device-accessible memory. The simplest way to arrange this, without rewriting the assembly code that runs on the host, is to allocate everything with `cudaMallocManaged`. Managed memory is accessible from both host and device through the same pointer, and the CUDA runtime migrates pages between the two as needed.

The pattern is:

```cpp
// Before:
float *v = new float[n];
// ...
delete[] v;

// After:
float *v;
cudaMallocManaged(&v, n * sizeof(float));
// ...
cudaFree(v);
```

The following allocations need to be converted. They are the buffers that are touched by **both** the host (during assembly / the CG loop control flow) and the GPU kernels:

| File         | Allocation(s) |
|--------------|---------------|
| `main.cpp`   | `rhs`, `temperature` |
| `solver.cu`  | `r`, `p`, `A_times_p` (inside `cg_solve`) |
| `utils.hpp`  | `DenseMatrix::allocate` — the `vals` array |

Remember to switch the corresponding `delete[]`s to `cudaFree` (including in `DenseMatrix::free`).

**Leave `bc_vals` in `utils.cpp` alone.** It is a host-only temporary used during boundary condition application and is never touched by a GPU kernel.

### Step 2: Port `matvec`

Open `solver.cu` and fill in the body of `matvec_kernel`. The natural parallelisation is one thread per row of $K$. Each thread computes its row's dot product against $\mathbf{x}$ and writes the result into $\mathbf{y}$.

```cpp
__global__ void matvec_kernel(const float *vals, const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // TODO: compute y[i] = sum over j of vals[i*n + j] * x[j]
    }
}
```

The launcher is provided for you:

```cpp
void matvec(const DenseMatrix &A, const float *x, float *y) {
    int n    = A.size;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matvec_kernel<<<grid, BLOCK_SIZE>>>(A.vals, x, y, n);
    cudaDeviceSynchronize();
}
```

`BLOCK_SIZE` is defined at the top of the file (256 is a sensible default for this kernel); the grid is sized so there are at least `n` threads in total. The `cudaDeviceSynchronize` ensures the kernel has finished before the host touches the output — you will want it in every launcher while porting, so that any error or correctness problem shows up immediately at the call site rather than somewhere later.

**Test.** Rebuild and run the validator. At this stage `axpby` is still running as a CPU fallback, so managed memory will migrate data back and forth on every CG iteration. The result should be correct, but the solve time will still be poor.

### Step 3: Port `axpby`

Two sub-tasks in `solver.cu`:

1. Fill in the body of `axpby_kernel`. `axpby` is purely element-wise: one thread per element, compute `y[i] = alpha * x[i] + beta * y[i]`.
2. Replace the CPU fallback inside the `axpby` launcher with a kernel launch, following the same pattern as `matvec`.

**Test.** With all three routines running on the GPU, you should now see a meaningful speedup over the CPU baseline.

### Step 4: `dot` — provided

Writing an efficient parallel reduction is non-trivial, and is covered in detail in the [Algorithms module](TODO: link to module) later in the course. For now, a working `dot` is **provided** at the top of `solver.cu`, using a single call to cuBLAS's `cublasSdot` routine. You do not need to understand the library call in detail — interfacing with vendor libraries is the subject of the [Using libraries module](TODO: link to module). `Makefile_cuda` already links against cuBLAS for you.

---

## Wrapping up

Whichever route you took, you should now have:

- A working GPU port of the dense FEM heat solver.
- The analytic validation still passing.
- A measurable speedup over the CPU baseline (once every kernel is on the GPU).

**Stretch tasks**, if you have time:

- Try the *other* porting route as well and compare the effort and end performance.
- Increase `elements_per_side` in `main.cpp` and observe how the CPU/GPU crossover and speedup change with problem size.
- Profile the GPU version (e.g. with Nsight Systems or `rocprof`) and identify which kernel dominates the runtime. Is it what you expected?

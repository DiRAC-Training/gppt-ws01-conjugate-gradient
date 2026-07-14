# Recap exercise: porting a conjugate gradient algorithm

In this exercise you will be guided through the porting of a conjugate gradient algorithm from an existing CPU implementation to a GPU implementation. You do not need to understand the details of the algorithm but the components you will be porting include a matrix-vector multiplication, a dot product and a type of vector addition. Feel free to use the provided solutions to help.

## The problem

The conjugate gradient (CG) finds the solution $x$ to the matrix equation $A\vec{x} = \vec{b}$ for a given matrix $A$ and vector $b$[^pos_def]. In our code we will be randomly generating both $A$ and $x$, calculating $b$, then using our algorithm to come up with the (known) solution $x$.

[^pos_def]: For the CG algorithm to work it must be *positive-definite* and *diagonally dominant*. In this code, the matrix is generated so that it has both these properties.

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
├── precision.hpp
└── test.hpp
src
├── main.cpp
├── solver_FIXME.cu    // START HERE FOR CUDA
└── test.cpp
```

Also in `src` is a number of solutions 

## Running the baseline

Build and run the CPU version first so you have a reference for correctness and timing:

```bash
make && build/cpu
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

Take note of the **time per iteration** for this CPU version reported here in microseconds. This is ultimately what you are trying to improve in the GPU port.

## The porting task

You will rewrite each operation as a CUDA kernel (i.e. a function marked with `__global__`) and manage memory through the CUDA runtime. You will port the kernels one at a time and test after each. Even though the performance of each operation will increase during this process, **you should not expect to see a speedup until all three kernels are on the GPU**. With managed memory, if even one kernel still runs on the host, the data will migrate back and forth between host and device on every CG iteration, and that migration cost will dominate the solve time. Early intermediate builds will often be *slower* than the CPU baseline. This is expected.

A starter `solver_FIXME.cu` is provided alongside `solver.cpp`. You will work in `solver_FIXME.cu` for the ported linear algebra routines, and switch a few allocations to `cudaMallocManaged` throughout the code so that host-assembled data is visible to the GPU kernels.

The starter file already contains the scaffold for you:

- `cg_solve` with the full CG iteration, using `new[]` allocations (you convert these in Step 1).
- A `matvec_kernel` with an empty body (See Step 2) and a complete launcher that calls it.
- An `axpby_kernel` with an empty body (See Step 3) and a temporary host-fallback launcher marked for replacement in Step 3.
- A provided `dot` function using cuBLAS, including handle setup — do not modify.

### Step 1: Switch allocations to managed memory

CUDA kernels act only on GPU memory but we can skip the manual process of allocating and transferring *GPU* memory by instead allocating *host* memory as *managed memory*. In doing this we potentially sacrifice some performance. Managed memory is allocated with `cudaMallocManaged` and is accessible from both host and device through the same pointer. The CUDA runtime migrates data between the two when the data is accessed.

Regular pointers allocated and deallocated through `new` and `delete` are converted as:

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

The allocations that need converted are noted in the code with the comment `TODO (exercise step 1)`. For this step, look where the relevant arrays are defined near the start of `main` in `main.cpp` and at the start of `cg_solve` in `solver_todo.cu`

The following allocations need to be converted. They are the buffers that are touched by **both** the host (during assembly / the CG loop control flow) and the GPU kernels:

| File         | Allocation(s) |
|--------------|---------------|
| `main.cpp`   | `rhs`, `temperature` |
| `solver_FIXME.cu`  | `r`, `p`, `A_times_p` (inside `cg_solve`) |

Remember to switch the corresponding `delete[]`s to `cudaFree`.

### Step 2: Port `matvec`

Open `solver_FIXME.cu` and fill in the body of `matvec_kernel`. The natural parallelisation is one thread per row of $K$. Each thread should compute its row's dot product against the vector `x` and write the result into `y`.

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

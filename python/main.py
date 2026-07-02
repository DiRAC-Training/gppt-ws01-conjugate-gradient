import time
import math
import numpy as np
from idx import idx
from solver import cg_solve
from test import test_matvec_identity, test_matvec_simple, test_dot

RANDOMISE_SEED = True

rng = np.random.default_rng()

def init_rng(randomise):
    """
    Create a rng
    """
    global rng
    seed = 42
    if randomise:
        seed = time.time_ns()
    rng = np.random.default_rng(seed)

def calc_b(b, A, x, n):
    """
    Generate b from Ax = b
    """
    for i in range(n):
        b[i] = 0.0
        for j in range(n):
            b[i] += A[idx(i, j, n)] * x[j]

def fill_rand_vec(x, size, minimum, maximum):
    """
    Fill x with random values in [min, max)
    """
    for i in range(size):
        x[i] = rng.uniform(minimum, maximum)

def generate_positive_definite(A, n):
    """
    Create a random, positive-definite matrix
    """
    print("Generating matrix")
    # Create a totally random matrix
    B = np.zeros(n * n, dtype=np.float32)
    fill_rand_vec(B, n * n, 0.0, 1.0)

    # Make a symmetric matrix
    for i in range(n):
        for j in range(n):
            A[idx(i, j, n)] = ( B[idx(i, j, n)] + B[idx(j, i, n)]) / 2.0
    
    # Add n * identity to make diagonally dominant => positive definite
    for i in range(n):
        A[idx(i, i, n)] += max(float(n) / 32.0, 1.0)

def run_tests():

    all_passed = True

    if not test_matvec_identity():
        print("matvec identity failed!")
        all_passed = False

    if not test_matvec_simple():
        print("matvec simple failed!")
        all_passed = False

    if not test_dot():
        print("dot failed!")
        all_passed = False

    return all_passed

def main():

    if not run_tests():
        print("A test failed!")
        return

    n = 8192
    cg_max_iter = 32

    A = np.zeros(n * n, dtype=np.float32) # note n*n
    x_soln = np.zeros(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)
    x = np.zeros(n, dtype=np.float32)

    init_rng(RANDOMISE_SEED)

    # Initial conditions
    generate_positive_definite(A, n) # Generate positive def matrix
    print("Generating random solution")
    fill_rand_vec(x_soln, n, -1.0, 1.0) # Generate a random solution vector
    print("Generating right hand side")

    calc_b(b, A, x_soln, n) # Multiple matrix with solution to get RHS of equation, b
    x.fill(0.0)
    
    # solve
    start = time.perf_counter()
    iters = cg_solve(x, A, b, n, cg_max_iter)
    stop = time.perf_counter()
    duration = (stop - start) * 1e6

    print("Performed", iters, "iterations")
    print("Solve time:", duration, "us")
    print("Time per iteration:", duration / iters, "us")

    av_error2 = 0.0
    for i in range(n):
        av_error2 += abs(x_soln[i] - x[i])

    av_error2 /= n
    
    print("Average error =", np.sqrt(av_error2))

if __name__ == "__main__":
    main()
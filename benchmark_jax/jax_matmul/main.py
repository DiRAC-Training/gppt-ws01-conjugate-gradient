import time
import numpy as np
import jax
import jax.numpy as jnp
from matvec import matvec
from cg import conjugate_gradient

jax.config.update("jax_enable_x64", True)

def benchmark_matvec(n):

    print(f"\nMatrix size = {n}")

    rng = np.random.default_rng(1234)
    A = rng.random((n, n))
    x = rng.random(n)
    A_jax = jnp.asarray(A)
    x_jax = jnp.asarray(x)

    y = matvec(A_jax, x_jax)
    y.block_until_ready()

    start = time.perf_counter()

    y = matvec(A_jax, x_jax)
    y.block_until_ready()

    end = time.perf_counter()

    print(f"MatVec time : {(end-start)*1000:.3f} ms")

    return np.asarray(y)


def benchmark_cg(n):

    rng = np.random.default_rng(42)
    A = rng.random((n, n))
    A = A.T @ A + n * np.eye(n)
    b = rng.random(n)
    A_jax = jnp.asarray(A)
    b_jax = jnp.asarray(b)

    conjugate_gradient(A_jax, b_jax, matvec)

    start = time.perf_counter()

    x = conjugate_gradient(A_jax, b_jax, matvec)
    x.block_until_ready()

    end = time.perf_counter()

    print(f"CG time     : {(end-start):.4f} s")

if __name__ == "__main__":

    for n in [128, 256, 512, 1024]:
        benchmark_matvec(n)
        benchmark_cg(n)

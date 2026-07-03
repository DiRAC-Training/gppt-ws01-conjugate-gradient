import numpy as np
import jax
import jax.numpy as jnp
from matvec import matvec

jax.config.update("jax_enable_x64", True)

def check_matvec(n, seed):

    rng = np.random.default_rng(seed)

    A = rng.random((n, n), dtype=np.float64)
    x = rng.random(n, dtype=np.float64)

    reference = A @ x

    A_jax = jnp.asarray(A, dtype=jnp.float64)
    x_jax = jnp.asarray(x, dtype=jnp.float64)

    result = np.asarray(matvec(A_jax, x_jax))

    np.testing.assert_allclose(result, reference, rtol=1e-9, atol=1e-9,)

def test_small():
    check_matvec(8, 1)

def test_medium():
    check_matvec(128, 2)

def test_large():
    check_matvec(1024, 3)

if __name__ == "__main__":
    test_small()
    test_medium()
    test_large()
    print("All tests passed.")
    
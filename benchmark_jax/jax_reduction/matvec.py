import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

@jax.jit
def matvec(A, x):
    return jnp.sum(A * x[None, :], axis=1)
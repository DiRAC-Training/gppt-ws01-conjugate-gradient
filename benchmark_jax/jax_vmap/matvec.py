import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

def matvec(A, x):
    return jax.vmap(lambda row: jnp.dot(row, x))(A)
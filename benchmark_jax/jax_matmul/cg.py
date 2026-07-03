import jax.numpy as jnp

def conjugate_gradient(A, b, matvec, tol=1e-8, maxiter=1000):
    n = A.shape[0]
    x = jnp.zeros_like(b)
    r = b - matvec(A, x)
    p = r
    rsold = jnp.dot(r, r)

    for _ in range(maxiter):
        Ap = matvec(A, p)
        alpha = rsold / jnp.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = jnp.dot(r, r)
        if jnp.sqrt(rsnew) < tol:
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    return x
    
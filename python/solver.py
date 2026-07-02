import numpy as np
from idx import idx

def matvec(y, A, x, n):
    """
    Dense matrix-vector product: y = A*x
    """
    for i in range(n):
        sum = 0.0
        for j in range(n):
            sum += A[idx(i, j, n)] * x[j]
        y[i] = sum

def dot(a, b, n):
    """
    Dot product: result = sum(a[i] * b[i])
    """
    sum = 0.0
    for i in range(n):
        sum += a[i] * b[i]
    return sum

def axpby(y, x, alpha, beta, n):
    """
    AXPBY operation: y = alpha * x + beta * y
    """
    for i in range(n):
        y[i] = alpha * x[i] + beta * y[i]

def cg_solve(x, A, b, n, max_iter):
    """
    Solve A*x = b using the conjugate gradient method
    """
    r = np.zeros(n, dtype=np.float32)
    p = np.zeros(n, dtype=np.float32)
    A_times_p = np.zeros(n, dtype=np.float32)

    # Step 1: r_0 = f - K*x_0
    matvec(r, A, x, n)
    for i in range(n):
        r[i] = b[i] - r[i]

    # Step 2: p_0 = r_0
    p[:] = r[:]

    residual_sq_old = dot(r, r, n)
    n_iter = 0

    for n_iter in range(max_iter):
        # Step 3a: alpha_k = (r_k . r_k) / (p_k . K*p_k)
        matvec(A_times_p, A, p, n)
        alpha = residual_sq_old / dot(p, A_times_p, n)

        # Step 3b: x_{k+1} = x_k + alpha_k * p_k
        axpby(x, p, alpha, 1.0, n)

        # Step 3c: r_{k+1} = r_k - alpha_k * K*p_k
        axpby(r, A_times_p, -alpha, 1.0, n)

        # Step 3d: beta_k = (r_{k+1} . r_{k+1}) / (r_k . r_k)
        #          p_{k+1} = r_{k+1} + beta_k * p_k
        residual_sq_new = dot(r, r, n)
        # // This method is so good it crashes if the residual gets too small!
        if residual_sq_new < np.finfo(np.float32).eps:
            break

        beta = residual_sq_new / residual_sq_old
        axpby(p, r, 1.0, beta, n)
        residual_sq_old = residual_sq_new
        
        print(f"{n_iter}: r = {residual_sq_new / n:.6e}")

    return n_iter
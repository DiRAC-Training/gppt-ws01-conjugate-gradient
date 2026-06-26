# Stolen from Stam's! This is matrix free though & just an example...

def solve(A, x, b, ng,
          apply_bcs=lambda x: x,
          max_iterations=100,
          conv_epsilon=1e-8):
    """
    Solve Ax = b via the conjugate gradient method TODO find source
    A is a function which, given a vector, returns the matrix-vector product
    x is the initial guess and also the final output
    b is the right hand side of the matrix equation to be solved
    ng is the number of ghost cells
    apply_bcs is a function which applies boundary conditions
    """

    Ap = np.zeros_like(x)
    r = np.zeros_like(x)
    r[ng:-ng, ng:-ng] = b[ng:-ng, ng:-ng] - A(x)
    r_norm = np.sum(r*r)
    p = r.copy()
    for i in range(max_iterations):
        apply_bcs(p)
        alpha = r_norm/np.sum(p[ng:-ng, ng:-ng]*A(p))
        x += alpha*p
        apply_bcs(x)
        r[ng:-ng, ng:-ng] = b[ng:-ng, ng:-ng] - A(x)
        r_norm_prev = r_norm
        r_norm = np.sum(r*r)
        if r_norm < conv_epsilon:
            return
        beta = r_norm/r_norm_prev
        # beta = 0 # Restart from x
        p = r + beta*p

import numpy as np
from idx import idx
from solver import matvec, dot

def nearly_eql(x, y, epsilon):
    return abs(x - y) < epsilon

def test_matvec_identity():
    """
    Test matvec with an identity matrix
    """
    n = 3

    A = np.zeros(n * n, dtype=np.float32)
    for i in range(n):
        A[idx(i, i, n)] = 1.0
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.zeros(3, dtype=np.float32)

    matvec(y, A, x, n)

    ## Because A is the identity matrix, y and x should be identical
    passed = True
    for i in range(n):
        passed &= nearly_eql(x[i], y[i], 1e-6)

    return passed

def test_matvec_simple():
    n = 3
    
    # Create a matrix with known values
    mat = np.array([-1, -6, 2, 4, 3, 10, 0, -100, 1], dtype=np.float32)

    # x and y_soln are calculated solutions to y = Ax.
    x = np.array([-1.0, 2.0, 0.0], dtype=np.float32)

    # mat * x =
    # -1*-1 + -6*2 +  2*0 =  -11
    # 4*-1 +  3*2 + 10*0 =    2
    # 0*-1 + -100*2 + 1*0 = -200
    y_soln = np.array([-11, 2, -200], dtype=np.float32)

    # Calculate y with our matvec test
    y = np.zeros(3, dtype=np.float32)
    matvec(y, mat, x, n)
    
    # Compare y to y_soln
    passed = True
    for i in range(n):
        passed &= nearly_eql(y_soln[i], y[i], 1e-6)
    return passed

def test_dot():
    n = 1024

    # Generate x and y and calculate their dot product in this loop
    x = np.zeros(n, dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    soln = 0.0
    for i in range(n):
        p = float(i) / float(n)
        x[i] = p
        y[i] = p
        soln += p * p

    # Calculate dot with the function and test against above value
    res = dot(x, y, n)
    return nearly_eql(res, soln, 1e-3)
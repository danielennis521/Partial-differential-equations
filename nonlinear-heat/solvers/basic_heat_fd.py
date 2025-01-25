import numpy as np


def difference_matrix(nx, c):
    # inputs:
    # nx: number of interior points in spatial discretization
    # c: k*dt / dx**2
    # outputs:
    # A: matrix to be used in solving the heat equation with implicit finite differences
     
    A = np.zeros([nx, nx])
    for i in range(nx):
        A[i][i] = 1 + 2*c
        if i != 0:
            A[i][i-1] = -c
        if i != nx-1:
            A[i][i+1] = -c 
    
    return A


def implicit_step(A, u0, c, bounds=[0, 0], bound_type='Dirichlet'):
    # inputs:
    #
    # outputs:
    # u1: the solution at the next time step

    b = u0
    b[0] -= c*bounds[0]
    b[-1] -= c*bounds[1]

    u1 = np.linalg.solve(A, b)

    return u1
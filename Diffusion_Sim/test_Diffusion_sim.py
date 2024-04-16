import numpy as np
import matplotlib.pyplot as plt
import Diffusion_By_FD as fd
import Diffusion_By_Lines as mol
import sympy as sp


# parameters 
nt = 500    # number of time steps to simulate
nx = 50    # number of points to simulate
dt = 0.0001

# initial data
u0 = np.array([np.sin(np.pi*2.0*xi/(nx + 1.0)) + np.cos(np.pi*3.0*xi/(nx + 1.0)) for xi in range(1, nx+1)])

a = 0.0
b = 1.0
ua = 0.0
ub = 0.0

# define the diffusion coefficient and its derivative
def k(x):
    return 0.25 + x

def dk(x):
    return 1.0


# run the finite difference simulation
fd.diffusion_sim(nt, nx, dt, u0, a, b, ua, ub, k, dk)

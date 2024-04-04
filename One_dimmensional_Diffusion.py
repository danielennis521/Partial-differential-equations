import numpy as np
import matplotlib.pyplot as plt

# parameters youll set
nt = 200    # number of time steps to simulate
nx = 30    # number of points to simulate
dt = 0.0001
def k(u, x):
    return 1.0
def dk(u, x):
    return 0.0

# initial data
u0 = [np.sin(np.pi*2.0*x/(nx + 1.0)) + np.cos(np.pi*3.0*x/(nx + 1.0)) for x in range(1, nx+1)]

a = 0.0
b = 1.0
ua = 0.0
ub = 0.0
u = u0
dx = (b-a)/(nx+1)
x = [a] + [a + m*dx for m in range(1, nx+1)] + [b]
lim = [min([ua, ub]+u0), max([ua, ub]+u0)]
m = -1.0 - 2.0*dx*dx/dt
t = 0

A = np.zeros([2*nx, 2*nx])
v = np.zeros(2*nx)
for i in range(nt):
    # set up the matrix for each step of the solution
    d1 = dk(u[0], t)*dx - k(u[0], t)
    d2 = dk(u[0], t)*dx + k(u[0], t)
    A[0][0:4] = [0.0, 0.0, 0.25*d2, 0.0]
    A[1][0:4] = [-2.0, 0.0, 0.5*d2, 0.25*d2]

    for i in range(1, nx-1):
        d1 = dk(u[i], t)*dx - k(u[i], t)
        d2 = dk(u[i], t)*dx + k(u[i], t)
        A[2*i][2*i-2:2*i+4] = [-0.25*d1, 0.0, 0.0, 0.0, 0.25*d2, 0.0]
        A[2*i+1][2*i-2:2*i+4] = [-0.5*d1, -0.25*d1, -2.0, 0.0, 0.5*d2, 0.25*d2]
        
    d1 = dk(u[-1], t)*dx - k(u[-1], t)
    d2 = dk(u[-1], t)*dx + k(u[-1], t)
    A[-2][-4:] = [-0.25*d1, 0.0, 0.0, 0.0]
    A[-1][-4:] = [-0.5*d1, -0.25*d1, -2.0, 0.0]

    # construct the RHS vector 

    # approximate the solution iteratively
    l = np.zeros(2*nx)
    l_prev = np.zeros(2*nx)
    e = 1.0
    while(e>1e-9):
        l_prev = l
        l = (v - np.dot(A, l))/m
        e = np.linalg.norm(l - l_prev)

    t += dt


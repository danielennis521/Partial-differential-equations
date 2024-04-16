import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def generate_matrix(x, dx, dt, k, nx):

    A = [[],[],[]]
    for i in range(nx):

        if i == 0:
            A[0].append(0.0)
            A[1].append(dx*dx/dt + 2.0*k(x[i]))
            A[2].append(-k(x[i]))
        elif i == nx - 1:
            A[0].append(-k(x[i]))
            A[1].append(dx*dx/dt + 2.0*k(x[i]))
            A[2].append(0.0)
        else:
            A[0].append(-k(x[i]))
            A[1].append(dx*dx/dt + 2.0*k(x[i]))
            A[2].append(-k(x[i]))

    return np.array(A)

def populate_vector(v, u, x, dx, dt, dk, nx, ua, ub):

    for i in range(nx):

        if i == 0:
            v[0] = u[0]*dx*dx/dt + 0.5*dx*dk(x[1])*(u[1] - ua)
        elif i == nx - 1:
            v[-1] = u[-1]*dx*dx/dt + 0.5*dx*dk(x[-1])*(ub - u[-2])
        else:
            v[i] = u[i]*dx*dx/dt + 0.5*dx*dk(x[i])*(u[i+1] - u[i-1])

def diffusion_sim(nt, nx, dt, u0, a, b, ua, ub, k, dk):

    u = u0
    dx = (b-a)/(nx+1)
    x = np.array([a + m*dx for m in range(1, nx+1)])
    lim = [min([ua, ub]+u0.tolist()), max([ua, ub]+u0.tolist())]
    t = 0
    A = generate_matrix(x, dx, dt, k, nx)
    v = np.zeros(nx)

    for i in range(nt):
        # construct the RHS vector 
        populate_vector(v, u, x, dx, dt, dk, nx, ua, ub)
        u = la.solve_banded((1, 1), A, v)

        # plot the solution
        plt.cla()
        plt.plot([a] + x.tolist() + [b], [ua]+u.tolist()+[ub], label='t = {}'.format(t))
        plt.legend()
        plt.ylim(lim)
        plt.pause(0.0001)

        t += dt

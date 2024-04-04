import numpy as np
import matplotlib.pyplot as plt

# parameters 
nt = 500    # number of time steps to simulate
nx = 100    # number of points to simulate
dt = 0.00001

def k(u, x):
    return 1.0

def dk(u, x):
    return 0.0

def populate_matrix(A, u, x, t, dx, nx):
    for i in range(nx):
        d1 = dk(u[i], x[i])*dx - 2.0*k(u[i], x[i])
        d2 = dk(u[i], x[i])*dx + 2.0*k(u[i], x[i])
        
        if i == 0:
            A[0][0:4] = [0.0, 0.0, 0.25*d2, 0.0]
            A[1][0:4] = [-2.0, 0.0, 0.5*d2, 0.25*d2]
        elif i == nx - 1:
            A[-2][-4:] = [-0.25*d1, 0.0, 0.0, 0.0]
            A[-1][-4:] = [-0.5*d1, -0.25*d1, -2.0, 0.0]
        else:
            A[2*i][2*i-2:2*i+4] = [-0.25*d1, 0.0, 0.0, 0.0, 0.25*d2, 0.0]
            A[2*i+1][2*i-2:2*i+4] = [-0.5*d1, -0.25*d1, -2.0, 0.0, 0.5*d2, 0.25*d2]

def populate_vector(v, u, x, t, dx, dt, nx):
    for i in range(nx):
        d1 = dk(u[i], x[i])*dx - 2.0*k(u[i], x[i])
        d2 = dk(u[i], x[i])*dx + 2.0*k(u[i], x[i])
        
        if i == 0:
            v[0] = v[1] = (4.0*u[0] - d2*u[1])/dt
        elif i == nx - 1:
            v[-2] = v[-1] = (4.0*u[-1] + d1*u[-2])/dt
        else:
            v[2*i] = v[2*i+1] = (4.0*u[i] + d1*u[i-1] - d2*u[i+1])/dt
    


# initial data
u0 = np.array([np.sin(np.pi*2.0*xi/(nx + 1.0)) + np.cos(np.pi*3.0*xi/(nx + 1.0)) for xi in range(1, nx+1)])

a = 0.0
b = 1.0
ua = 0.0
ub = 0.0
u = u0
dx = (b-a)/(nx+1)
x = np.array([a + m*dx for m in range(1, nx+1)])
lim = [min([ua, ub]+u0.tolist()), max([ua, ub]+u0.tolist())]
m = -1.0 - 2.0*dx*dx/dt
t = 0


A = np.zeros([2*nx, 2*nx])
v = np.zeros(2*nx)
for i in range(nt):
    # set up the matrix for each step of the solution
    populate_matrix(A, u, x, t, dx, nx)

    # construct the RHS vector 
    populate_vector(v, u, x, t, dx, dt, nx)

    # approximate the slopes for the RK step
    l = np.zeros(2*nx)
    l_prev = np.zeros(2*nx)
    e = 1.0
    while(e>1e-9):
        l_prev = l
        l = (v - np.dot(A, l))/m
        e = np.linalg.norm(l - l_prev)

    # update the solution
    for j in range(nx):
        u[j] += dt*0.5*(l[2*j] + l[2*j+1])
    

    # plot the solution
    if i % 50 == 0:
        plt.plot([a] + x.tolist() + [b], [ua]+u.tolist()+[ub], label='t = {}'.format(t))
        plt.legend()
        plt.ylim(lim)
        plt.show()

    t += dt

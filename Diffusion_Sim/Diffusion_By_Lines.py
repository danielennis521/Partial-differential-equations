import numpy as np
import matplotlib.pyplot as plt


def populate_matrix(A, x, dx, nx, k, dk):
    for i in range(nx):
        d1 = dk(x[i])*dx - 2.0*k(x[i])
        d2 = dk(x[i])*dx + 2.0*k(x[i])
        d = k(x[i])
        
        if i == 0:
            A[0][0:4] = [0.0, 0.0, 0.25*d2, 0.0]
            A[1][0:4] = [-2.0*d, 0.0, 0.5*d2, 0.25*d2]
        elif i == nx - 1:
            A[-2][-4:] = [-0.25*d1, 0.0, 0.0, 0.0]
            A[-1][-4:] = [-0.5*d1, -0.25*d1, -2.0*d, 0.0]
        else:
            A[2*i][2*i-2:2*i+4] = [-0.25*d1, 0.0, 0.0, 0.0, 0.25*d2, 0.0]
            A[2*i+1][2*i-2:2*i+4] = [-0.5*d1, -0.25*d1, -2.0*d, 0.0, 0.5*d2, 0.25*d2]

def populate_vector(v, u, x, dx, dt, nx, k, dk):
    for i in range(nx):
        d1 = dk(x[i])*dx - 2.0*k(x[i])
        d2 = dk(x[i])*dx + 2.0*k(x[i])
        d = k(x[i])
        
        if i == 0:
            v[0] = v[1] = (4.0*d*u[0] - d2*u[1])/dt
        elif i == nx - 1:
            v[-2] = v[-1] = (4.0*d*u[-1] + d1*u[-2])/dt
        else:
            v[2*i] = v[2*i+1] = (4.0*d*u[i] + d1*u[i-1] - d2*u[i+1])/dt
    
def diffusion_sim(nt, nx, dt, u0, a, b, ua, ub, k, dk):

    u = u0
    dx = (b-a)/(nx+1)
    x = np.array([a + m*dx for m in range(1, nx+1)])
    lim = [min([ua, ub]+u0.tolist()), max([ua, ub]+u0.tolist())]
    t = 0


    A = np.zeros([2*nx, 2*nx])
    v = np.zeros(2*nx)
    for i in range(nt):
        # set up the matrix for each step of the solution
        populate_matrix(A, x, dx, nx, k, dk)

        # construct the RHS vector 
        populate_vector(v, u, x, dx, dt, nx, k, dk)

        # approximate the slopes for the RK step
        l = np.zeros(2*nx)
        l_prev = np.zeros(2*nx)
        e = 1.0
        m = [-k(x[j]) - 2.0*dx*dx/dt for j in range(nx)]
        count = 0
        while(e>1e-2):
            count += 1
            l_prev = l
            l = v + np.dot(A, l)
            for j in range(nx):
                l[2*j] /= m[j]
                l[2*j+1] /= m[j]
            e = np.linalg.norm(l - l_prev)

        # update the solution
        for j in range(nx):
            u[j] += dt*0.5*(l[2*j] + l[2*j+1])
        

        # plot the solution
        plt.cla()
        plt.plot([a] + x.tolist() + [b], [ua]+u.tolist()+[ub], label='t = {}'.format(t))
        plt.legend()
        plt.ylim(lim)
        plt.pause(0.0001)

        t += dt



import numpy as np
import matplotlib.pyplot as plt


def f(x, u, v, dx, s=1):

    f = np.zeros(len(x))

    for i in range(len(x)):
        if i==0:
            f[i] = (1/dx**2) * v(x[i]) * (-2*u[i] + u[i+1]) - s*(0.5/dx) * u[i] * u[i+1]
        elif i==len(x)-1:
            f[i] = (1/dx**2) * v(x[i]) * (-2*u[i] + u[i-1]) + s*(0.5/dx) * u[i] * u[i-1]
        else:
            f[i] = (1/dx**2) * v(x[i]) * (u[i-1] - 2*u[i] + u[i+1]) \
                    - s*(0.5/dx) * u[i] * (u[i+1] - u[i-1])
    return f


def rk4_step(u, v, x, dx, dt, s):
    k1 = f(x, u, v, dx, s)
    k2 = f(x, u + dt*k1/2, v, dx, s)
    k3 = f(x, u + dt*k2/2, v, dx, s)
    k4 = f(x, u + dt*k3, v, dx, s)

    return u + dt*(k1 + k2 + k3 + k4)/6


# WARNING: This method is currently very unreliable and is likely to diverge
def newton_implicit_step(u, v, x, dx, dt, s):
    n = len(u)
    J = np.zeros([n, n])
    for i in range(n):
        a = -2*dt*v(x[i])/dx**2
        
        if i==0:
            J[i][i] = a - u[i+1]*(0.5*dt/dx) - 1
        elif i==n-1:
            J[i][i] = a + u[i-1]*(0.5*dt/dx) - 1
        else:
            J[i][i] = a - (u[i+1] - u[i-1])*(0.5*dt/dx) - 1


        if i != 0:
            J[i][i-1] = a + u[i-1]*(0.5*dt/dx)
        if i!= n-1:
            J[i][i+1] = a - u[i+1]*(0.5*dt/dx)

    u_next = u
    u_prev = u
    for i in range(100):
        F = f(x, u_next, v, dx, s) - u
        u_prev = u_next
        u_next += np.linalg.solve(J, -F)

        if np.linalg.norm(u_next-u_prev) <= 1e-5*n:
            break

    return u_next


def burger_sim(v, u0, a, b, nx, dt, nt, s=1):
    
    x = np.linspace(a, b, nx, endpoint=True)
    dx = x[1] - x[0]
    u = [u0(t) for t in x]

    for i in range(nt):

        u = rk4_step(u, v, f, x, dx, dt, s)

        if i%10 == 0:
            plt.plot(x, u)
            plt.ylim(top=1, bottom=-1)
            plt.title('t={}'.format(str(np.round(i*dt, 4))))
            plt.pause(0.01)
            plt.cla()
    return


def compare_sim(v0, u00, s0, v1, u01, s1, nx, nt, dt, a, b):

    x = np.linspace(a, b, nx, endpoint=True)
    dx = x[1] - x[0]
    u1 = [u00(t) for t in x]
    u2 = [u01(t) for t in x]

    for i in range(nt):

        u1 = rk4_step(u1, v0, x, dx, dt, s0)
        u2 = rk4_step(u2, v1, x, dx, dt, s1)

        # u1 = newton_implicit_step(u1, v0, x, dx, dt, s0)
        # u2 = newton_implicit_step(u2, v1, x, dx, dt, s1)

        if i%10 == 0:
            plt.plot(x, u1, 'b')
            plt.plot(x, u2, 'g')
            plt.ylim(top=1, bottom=-1)
            plt.title('t={}'.format(str(np.round(i*dt, 4))))
            plt.pause(0.01)
            plt.cla()

    return



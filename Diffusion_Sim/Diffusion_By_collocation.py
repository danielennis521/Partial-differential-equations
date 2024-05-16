import numpy as np
import numpy.polynomial.polynomial as p
import scipy.linalg as la
import matplotlib.pyplot as plt


def rescale_polynomial(p, a0, b0, a1, b1):
    # Rescale the polynomial p from the interval [a0, b0] to [a1, b1]
    s = (b1 - a1) / (b0 - a0)
    new = []
    for i in range(len(p)):
        t = [1]
        for j in range(i):
            t = p.polymul(t, [a1 - a0*s, s])

        new = p.polyadd(np, p.polymul(p[i], t))

    return new


def Chebyshev_Polynomials(n, a=-1, b=1):
    # Generate the n Chebyshev polynomials
    for i in range(n):
        if i == 0:
            L = [np.array([1])]
        elif i == 1:
            L.append(np.array([0, 1]))
        else:
            L.append(p.polysub(p.polymul([0, 2], L[i-1]), L[i-2]))

    if a != -1 or b != 1:
        for i in range(n):
            L[i] = rescale_polynomial(L[i], -1, 1, a, b)

    return L


def generate_matrix(n, x, L, dL, dt, ua, ub):
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i==0:
                B[i][j] = ua
            elif i==n-1:
                B[i][j] = ub
            else:
                B[i, j] = p.polyval(x[i], dL[j])

            A[i, j] = p.polyval(x[i], L[j])
    
    return A, B


def rk4_step(A, c, dt):
    k1 = A.dot(c)
    k2 = A.dot(c + 0.5*dt*k1)
    k3 = A.dot(c + 0.5*dt*k2)
    k4 = A.dot(c + dt*k3)
    return c + dt*(k1 + 2*k2 + 2*k3 + k4)/6 


def diffusion_sim(nt, nx, dt, u0, a=-1, b=1, ua=0, ub=0):

    x = [b] + [np.cos((2*i+1)*np.pi/(2*(nx-2))) for i in range(nx-2)] + [a]
    x.reverse()

    u = [ua] + [u0(x[i]) for i in range(1, nx-1)] + [ub]

    T = Chebyshev_Polynomials(nx, a, b)
    ddT = [p.polyder(p.polyder(T[i])) for i in range(nx)]

    # Generate the matrices
    A, B = generate_matrix(nx, x, T, ddT, dt, ua, ub)
    c = la.solve(A, u)

    B = np.dot(la.inv(A),B)
    lim = [min(u), max(u)]

    for i in range(nt):
        if i % 250 == 0:
            y = np.linspace(a, b, 100)
            z = [np.sum([c[j]*p.polyval(y[i], T[j]) for j in range(nx)]) for i in range(100)]
            plt.cla()
            plt.ylim(lim)
            plt.plot(y, z, 'b')
            plt.plot(x, u, 'ro')
            plt.plot(x, u, 'r')
            plt.pause(0.1)

        c = rk4_step(B, c, dt)
        u = A.dot(c)
        #u = rk4_step(B, u, dt)

        

def u0(x):
    return x

def u01(x):
    if np.abs(x) >= 1.0:
        return 0
    else:
        return np.exp(-1/(1-x**2))

diffusion_sim(int(1e7), 9, 1e-6, u0, -1, 1, 0, 0)

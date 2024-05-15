import numpy as np
import numpy.linalg as la
import numpy.polynomial.polynomial as p
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


def generate_matrix(n, x, L, dL):
    # Populate the matrix A
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = p.polyval(x[i], L[j])
            B[i, j] = p.polyval(x[i], dL[j])

    A = np.dot(la.inv(B), A)

    return A


def RK4_step(u, A, dt):

    return u


def transport_sim(nx, nt, dt, a, b, u0):
    # Simulate the transport equation
    dx = (b - a) / nx
    x = np.linspace(a + 0.5*dx, b - 0.5*dx, nx)
    L = Chebyshev_Polynomials(nx, a, b)
    dL = [p.polyder(L[i]) for i in range(nx)]
    A = generate_matrix(nx, x, L, dL)

    u = np.zeros(nx)
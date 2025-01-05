import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as p
import numpy.linalg as la


def rescale_polynomial(q, a0, b0, a1, b1):
    # Rescale the polynomial q from the interval [a0, b0] to [a1, b1]
    s = (b0 - a0)/(b1 - a1) 
    new = [0]
    for i in range(len(q)):
        t = [1]
        for j in range(i):
            t = p.polymul(t, [a0 - a1*s, s])

        new = p.polyadd(new, p.polymul(q[i], t))

    return new


def Chebyshev_Polynomials(n, a=-1, b=1):
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


class Diffusion_sim():
    def __init__(self, f, k, dk, u0, n, dt, a, b,
                 bound_type='Dirichlet', stepping='foreward'):
        
        self.f = f
        self.n = n
        self.dt = dt
        self.bound_type = bound_type
        self.stepping = stepping
        self.k = k
        self.dk = dk

        self.L = Chebyshev_Polynomials(self.n, a, b)
        self.dL = [p.polyder(g) for g in self.L]
        self.ddL = [p.polyder(g) for g in self.dL]

        self.generate_nodes(a, b)
        self.u = np.array([u0(t) for t in self.x])
        self.generate_matrix()


    def generate_nodes(self, a, b):
        x = [np.cos((2*i+1)*np.pi/(2*(self.n-2))) for i in range(self.n-2)]
        x.reverse()
        self.x = [a] + x + [b]

    
    def generate_matrix(self):
        A = np.zeros((self.n, self.n))
        B = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if bool(i%(self.n-1)) != bool(j%(self.n-1)):
                    B[i, j] = 0
                else:
                    t1 = k(self.x[i])*p.polyval(self.x[i], self.ddL[j])
                    t2 =  dk(self.x[i])*p.polyval(self.x[i], self.dL[j])
                    B[i, j] = t1 + t2
                A[i, j] = p.polyval(self.x[i], self.L[j])

        self.c = la.solve(A, self.u)
        self.A = np.dot(la.inv(A),B)


    def efd_step(self):
        self.c += self.dt * (self.A.dot(self.c))


    def rk4_step(self):
        k1 = self.A.dot(self.c)
        k2 = self.A.dot(self.c + self.dt*k1/2)
        k3 = self.A.dot(self.c + self.dt*k2/2)
        k4 = self.A.dot(self.c + self.dt*k3)

        self.c += self.dt*(k1 + 2*k2 + 2*k3 + k4)/6


    def eval(self, x):
        return np.sum([self.c[i] * p.polyval(x, self.L[i]) for i in range(self.n)])

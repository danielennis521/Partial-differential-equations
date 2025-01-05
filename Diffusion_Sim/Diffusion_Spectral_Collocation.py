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
    def __init__(self, f, k, dk, u0, n, dt, a, b, ua, ub,
                 bound_type='Dirichlet', order=2, stepping='foreward'):
        
        self.f = f
        self.n = n
        self.dt = dt
        self.boundary = [ua, ub]
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
        self.M = np.zeros([self.n, self.n])
        self.N = np.zeros([self.n, self.n])
        
        # require the approximation to solve the pde at the interior nodes
        for i in range(self.n):
            for j in range(self.n):

                self.N[i, j] = p.polyval(self.x[i], self.L[j])

                if i==0 or i==self.n-1:
                    continue
                else:
                    t1 = self.dk(self.x[i]) * p.polyval(self.x[i], self.dL[j])
                    t2 = self.k(self.x[i]) * p.polyval(self.x[i], self.ddL[j])
                    self.M[i, j] = t1 + t2

        # enforce the boundary conditions
        self.b = np.zeros(self.n)
        self.b[0] = self.boundary[0]
        self.b[-1] = self.boundary[1]

        self.c = la.solve(self.N, self.u)
        self.A = np.dot(la.inv(self.N), self.M)
        self.b = np.dot(la.inv(self.N), self.b)


    def efd_step(self):
        self.c += self.dt * (self.A.dot(self.c) + self.b)


    def eval(self, x):
        return np.sum([self.c[i] * p.polyval(x, self.L[i]) for i in range(self.n)])




def f(x):
    return 0

def k(x):
    return 1

def dk(x):
    return 0

def u0(x):
    return np.sin(np.pi*x)



np.set_printoptions(precision=2, suppress=True)
test = Diffusion_sim(f=f, k=k, dk=dk, u0=u0, n=6,
                     dt=0.001, a=-1, b=1, ua=0, ub=1)

#lim = [min(test.u), max(test.u)]
lim = [-1, 1]
x = np.linspace(-1, 1.01, 100)

for i in range(200):
    plt.plot(x, [test.eval(t) for t in x])
    plt.ylim(lim)
    plt.pause(0.1)
    plt.cla()
    test.efd_step()

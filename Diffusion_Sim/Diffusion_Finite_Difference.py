import numpy as np
import scipy.linalg as la
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


class Diffusion_sim():
    def __init__(self, f, k, dk, u0, dx, dt, a, b, ua, ub,
                 bound_type='Dirichlet', order=2, stepping='foreward'):
        
        self.f = f
        self.dx = dx
        self.dt = dt
        self.a = a
        self.b = b
        self.boundary = [ua, ub]
        self.bound_type = bound_type
        self.stepping = stepping
        self.k = k
        self.dk = dk

        self.x = np.arange(a, b+dt, dx)
        self.n = len(self.x)
        self.u = np.array([u0(t) for t in self.x])
        self.generate_matrix()

    
    def generate_matrix(self):
        
        A = []
        B = []
        for t in self.x:
            if t == self.x[0]:
                A.append([0.0, 0.0, 0.0])
                B.append([0.0, 0.0, 0.0])
            elif t == self.x[-1]:
                A.append([0.0,0.0, 0.0])
                B.append([0.0, 0.0, 0.0])
            else:
                A.append([self.k(t), -2.0*self.k(t), self.k(t)])
                B.append([self.dk(t), 0.0, -self.dk(t)])
        
        A = np.array(A).T
        B = np.array(B).T
        self.M = A/(self.dx*self.dx) + B/(self.dx*0.5)

        self.set_boundary_conditions()

        self.M = dia_matrix((self.M, [-1, 0, 1]), shape=(self.n, self.n))
        self.M = self.M.T


    def efd_step(self):
        self.u += self.dt*self.M.dot(self.u)
        self.u += self.B.dot(self.u) + self.C


    def ifd_step(self):
        I = dia_matrix((np.ones(self.n), 0), shape=(self.n, self.n))
        self.u = spsolve(I - self.dt*self.M, self.u)
        self.u += self.B.dot(self.u) + self.C


    def rk4_step(self):
        k1 = self.M.dot(self.u)
        k2 = self.M.dot(self.u + self.dt*k1/2)
        k3 = self.M.dot(self.u + self.dt*k2/2)
        k4 = self.M.dot(self.u + self.dt*k3)

        self.u += self.dt*(k1 + 2*k2 + 2*k3 + k4)/6
        self.u += self.B.dot(self.u) + self.C


    def set_boundary_conditions(self):
        bounds = {'Dirichlet': self.set_dirichlet,
                  'Neumann': self.set_neumann,
                  'Mixed Left': self.set_mixleft,
                  'Mixed Right': self.set_mixright}
        bounds[self.bound_type]()


    def set_dirichlet(self):
        self.u[0] = self.boundary[0]
        self.u[-1] = self.boundary[1]
    

    def set_neumann(self):
        A = np.zeros((3, self.n))
        A[2, 1] = 1
        A[0, -2] = 1
        A[1, 0] = -1
        A[1, -1] = -1
        self.B = dia_matrix((A , [-1, 0, 1]), shape=(self.n, self.n))
        self.C = np.zeros(self.n)
        self.C[0] = -self.boundary[0]*self.dt
        self.C[-1] = self.boundary[1]*self.dt
    

    def set_mixleft(self):
        self.u[0] = self.boundary[0]

        A = np.zeros((3, self.n))
        A[0, -2] = 1
        A[1, -1] = -1
        self.B = dia_matrix((A , [-1, 0, 1]), shape=(self.n, self.n))
        self.C = np.zeros(self.n)
        self.C[-1] = self.boundary[1]*self.dt
    

    def set_mixright(self):
        self.u[-1] = self.boundary[1]

        A = np.zeros((3, self.n))
        A[2, 1] = 1
        A[1, 0] = -1
        self.B = dia_matrix((A , [-1, 0, 1]), shape=(self.n, self.n))
        self.C = np.zeros(self.n)
        self.C[0] = -self.boundary[0]*self.dt


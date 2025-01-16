# The Burger_Sim class is based on the finite difference solution of Burgers equation
# implimented in Burger_FD.py. The main purpose of this class is to provide an easy way 
# to compare and graph the behavior for different initial data and coefficient functions.


import numpy as np
import matplotlib.pyplot as plt
import Burger_FD as fd


class Burger_Sim():

    def __init__(self, a, b, nx, dt, t=0):
        
        self.a = a
        self.b = b
        self.x = np.linspace(a, b, nx, endpoint=True)
        self.dx = self.x[1] - self.x[0]
        self.dt = dt

        self.u = []
        self.v = []
        self.s = []

        self.t = t


    def add_sim(self, v, s, u0):

        self.u.append([u0(t) for t in self.x])
        self.v.append(v)
        self.s.append(s)


    def step(self):

        for i in range(len(self.u)):
            self.u[i] = fd.rk4_step(self.u[i], self.v[i], self.x,
                                    self.dx, self.dt, self.s[i])
            
            self.t += self.dt


    def plot_n(self, n, color='b'):

        plt.plot(self.x, self.u[n], color)
        plt.ylim(top=np.max(self.u[n]) * 1.1, bottom=np.min(self.u[n]) * 1.1)
        plt.title('t={}'.format(str(self.t)))
        plt.show()
    

    def run(self, n):

        for i in range(n):
            self.step()


    def play(self, n, colors=[]):
        
        upper = max(self.u[0])
        lower = min(self.u[0])

        for i in range(n):
            plt.title('t={}'.format(str(self.t)))
            plt.ylim(top=upper, bottom=lower)
            
            for j in range(len(self.u)):
                plt.plot(self.x, self.u[j])

            
            plt.pause(0.005)
            plt.cla()

            self.step()
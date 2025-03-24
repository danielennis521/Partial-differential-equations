# The Burger_Sim class is based on the finite difference solution of Burgers equation
# implimented in Burger_FD.py. The main purpose of this class is to provide an easy way 
# to compare and graph the behavior for different initial data and coefficient functions.


import numpy as np
import matplotlib.pyplot as plt
import Burger_FD as fd
from matplotlib.animation import FuncAnimation


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

        self.colors = []
        self.labels = []


    def add_sim(self, v, s, u0, label='', color=''):

        self.u.append([u0(t) for t in self.x])
        self.v.append(v)
        self.s.append(s)
        self.labels.append(label)
        self.colors.append(color)


    def step(self):
        # explicit first order finite difference in time

        for i in range(len(self.u)):
            self.u[i] = fd.rk4_step(self.u[i], self.v[i], self.x,
                                    self.dx, self.dt, self.s[i])
            
            self.t += self.dt

    
    def implicit_step(self):
        # implicit first order finite difference in time

        for i in range(len(self.u)):
            self.u[i] = fd.newton_implicit_step(self.u[i], self.v[i], self.x,
                                    self.dx, self.dt, self.s[i])
            
            self.t += self.dt


    def plot_n(self, n, color='b'):

        plt.plot(self.x, self.u[n], color)
        plt.ylim(top=np.max(self.u[n]) * 1.1, bottom=np.min(self.u[n]) * 1.1)
        plt.title('t={}'.format(str(self.t)))
        plt.show()
    

    def run(self, n):
        # inputs:
        # n: number of steps to simulate
        # outputs:
        # none, advances the simulation n steps

        for i in range(n):
            self.step()


    def play(self, n):
        # inputs:
        # n: number of steps to simulate
        # output:
        # animates the next n steps of the solutions using matplotlib

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

    
    def create_gif(self, n, label=False, filename='heat_sim_animation.gif'):
        # inputs:
        # n: number of time steps to simulate
        # filename: name for the gif to be saved under, should end in .gif
        # outputs:
        # Saves the animation of the solutions as a gif in the same directory as the Burger_Sim.py file
    
        upper_bound = np.max(self.u) * 1.1
        lower_bound= np.min(self.u) * 1.1
        fig, ax = plt.subplots()

        def update_plot(frame):
            nonlocal self

            ax.clear()
            ax.set_facecolor('lightgray')
            fig.patch.set_facecolor('lightgray')
            # plot each solution curve
            for i in range(len(self.u)):
                ax.plot(self.x, self.u[i], label=self.labels[i], color=self.colors[i])

            
            # figure formating
            if label:
                ax.legend()   
            plt.xlabel("Position")
            plt.ylabel("velocity")             
            time_label = str(np.round(self.t, 5))
            ax.set_ylim(top=upper_bound, bottom=lower_bound)
            ax.set_title('Time={}'.format(time_label))
            
            
            self.step()
            self.step()
            self.step()


        ani = FuncAnimation(fig, update_plot, frames=n, repeat=False)
        ani.save(filename=filename, writer='pillow', fps=75)

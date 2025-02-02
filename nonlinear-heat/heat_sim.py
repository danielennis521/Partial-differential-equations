import numpy as np
import solvers.nonlinear_heat_fd as nh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class heat_sim():
    def __init__(self, boundary, dt, boundary_conditions=[0, 0], bound_type='Dirichlet'):
        # inputs:
        # boundary: list/array/tuple of length 2 defining the boundary of the simulation
        # boundary_conditions: list/array/tuple of length 2 specifying values imposed on
        #                       the function or its derivatives at the endpoints
        # bound_type: string taking one of the values dirichlet, neumann, dirichlet_left
        #              or, dirichlet_right. dirichlet_left meaning the first entry is a 
        #              dirichlet conditions and the second is a neumann condition, likewise
        #              for dirichlet_right
        # dt: time step to be used in all simulations

        self.boundary = boundary
        self.boundary_conditions = boundary_conditions
        self.bound_type = bound_type
        self.dt = dt
        self.t = 0
        self.x = []
        self.u = []
        self.k = []
        self.dk = []
        self.ddk = []
        self.dx = []
        self.labels = []
        self.colors = []


    def add_sim(self, u0, k, dk, ddk, nx, label='', color='random'):
        # inputs: 
        # u0: a function specifying the initial distribution of the data
        # k: the function specifying the thermal diffusivity
        # dk: derivative of k
        # ddk: derivative of dk
        # nx: number of *INTERIOR* points to use for spatial discretization.
        #     nx = 100 will simulate 100 points inside the interval and the 2 boundary points

        domain = np.linspace(self.boundary[0], self.boundary[1], nx+2, endpoint=True)
        self.dx.append(domain[1] - domain[0])

        self.x.append(domain)
        self.u.append([u0(t) for t in domain[1:-1]])
        self.k.append(k)
        self.dk.append(dk)
        self.ddk.append(ddk)
        self.labels.append(label)

        if color=='random':
            self.colors.append(tuple(np.random.rand(3)))
        else:
            self.colors.append(color)


    def step(self):
        # advances all simulations foreward in time by one step od length dt
        
        for i in range(len(self.u)):
            c = self.dt/self.dx[i]**2
            J = nh.generate_jacobian(self.u[i], c, self.k[i], self.dk[i], self.ddk[i])
            self.u[i] = nh.newton_step(self.u[i], J, c, self.k[i], self.dk[i])
        
        self.t += self.dt

    
    def run(self, n, display=False, labels=False):
        # advances all simulations a specified number of steps
        #
        # inputs:
        # n: number of time steps to simulate
        # display: if True then the evolution of the simulation will be shown with matplotlib

        upper_bound = np.max(self.u) * 1.1
        lower_bound= np.min(self.u) * 1.1
        fig, ax = plt.subplots()
        
        for i in range(n):
            if display:
                
                # plot each solution curve
                for i in range(len(self.u)):
                    solution = [self.boundary_conditions[0]] + list(self.u[i]) \
                                + [self.boundary_conditions[1]]

                    ax.plot(self.x[i], solution, label=self.labels[i], color=self.colors[i])

                
                # figure formating
                if labels:
                    ax.legend()            
                time_label = str(np.round(self.t, 5))
                ax.set_ylim(top=upper_bound, bottom=lower_bound)
                ax.set_title('Time={}'.format(time_label))
                plt.pause(0.01)
                plt.cla()
        
            self.step()

    
    def create_gif(self, n, label=False, filename='heat_sim_animation.gif'):
        # inputs:
        # n: number of time steps to simulate
        # filename: name for the gif to be saved under, should end in .gif
        # outputs:
        # Saves the animation of the solutions as a gif in the same directory as the heat_sim.py file
    
        upper_bound = np.max(self.u) * 1.1
        lower_bound= np.min(self.u) * 1.1
        fig, ax = plt.subplots()

        def update_plot(frame):
            nonlocal self

            ax.clear()
            # plot each solution curve
            for i in range(len(self.u)):
                solution = [self.boundary_conditions[0]] + list(self.u[i]) \
                            + [self.boundary_conditions[1]]

                ax.plot(self.x[i], solution, label=self.labels[i], color=self.colors[i])

            
            # figure formating
            if label:
                ax.legend()            
            time_label = str(np.round(self.t, 5))
            ax.set_ylim(top=upper_bound, bottom=lower_bound)
            ax.set_title('Time={}'.format(time_label))
            
        
            for i in range(3):
                self.step()

        ani = FuncAnimation(fig, update_plot, frames=n, repeat=False)
        ani.save(filename=filename, writer='pillow', fps=60)

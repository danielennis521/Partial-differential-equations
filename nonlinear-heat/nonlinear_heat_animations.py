import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import solvers.basic_heat_fd as bh
import solvers.nonlinear_heat_fd as nh



def run_heat_sim(u0, a, b, dt, nx, nt, k, dk, ddk):
    # inputs:
    # u0: function defining the initial heat distribution
    # a: left endpoint
    # b: right endpoint
    # dt: timestep 
    # nx: number of points for spatial discretization
    # nt: number of timesteps to run for
    # k: the function specifying the thermal diffusivity
    # dk: derivative of k
    # ddk: derivative of dk
    #
    # output:
    # displays the evolution of the solution using matplotlib


    x = np.linspace(a, b, nx+2, endpoint=True)
    u = np.array([u0(t) for t in x[1:-1]])
    t = 0.0  
    dx = x[1] - x[0]
    c = dt/dx**2

    upper_bound = max(u) * 1.1
    lower_bound= min(u) * 1.1
    fig, ax = plt.subplots()

    for i in range(nt):
        J = nh.generate_jacobian(u, c, k, dk, ddk)
        u = nh.newton_step(u, J, c, k, dk)
        t += dt

        time_label = str(np.round(t, 5))
        ax.clear()
        ax.set_ylim(bottom=lower_bound, top=upper_bound)
        ax.plot(x, [0] + list(u) + [0], color='b', label='nonlinear: k=0.5/(1 + u^2)')
        ax.legend()
        ax.set_title('Time={}'.format(time_label))
        plt.pause(0.01)



def heat_compare_gif(u0, a, b, dt, nx, nt, k, dk, ddk, filename='heat_sim.gif'):
    # inputs:
    # u0: function defining the initial heat distribution
    # a: left endpoint
    # b: right endpoint
    # dt: timestep 
    # nx: number of points for spatial discretization
    # nt: number of timesteps to run for
    # k: the function specifying the thermal diffusivity
    # dk: derivative of k
    # ddk: derivative of dk
    #
    # output:
    # creates a gif of the evolution of the normal heat equation and the given nonlinear variant

    x = np.linspace(a, b, nx+2, endpoint=True)
    u = np.array([u0(t) for t in x[1:-1]])
    t = 0.0
    baseline = np.array([u0(t) for t in x[1:-1]])
    
    dx = x[1] - x[0]
    c = dt/dx**2

    A = bh.difference_matrix(nx, c)

    upper_bound = max(u) * 1.1
    lower_bound= min(u) * 1.1

    fig, ax = plt.subplots()

    def update_plot(frame):
        nonlocal u, t, baseline
        for i in range(3):
            J = nh.generate_jacobian(u, c, k, dk, ddk)
            u = nh.newton_step(u, J, c, k, dk)
            baseline = bh.implicit_step(A, baseline, c)
            t += dt

        time_label = str(np.round(t, 5))

        ax.clear()
        ax.set_ylim(bottom=lower_bound, top=upper_bound)
        ax.plot(x, [0] + list(u) + [0], color='b', label='nonlinear: k=0.5/(1 + u^2)')
        ax.plot(x, [0] + list(baseline) + [0], color='r', label='linear: k=1')
        ax.legend()
        ax.set_title('Time={}'.format(time_label))

    ani = FuncAnimation(fig, update_plot, frames=nt, repeat=False)

    ani.save(filename=filename, writer='pillow', fps=60)



def nonlinear_heat_gif():
    return


def heat_compare_with_curvature_gif(u0, a, b, dt, nx, nt, k, dk, ddk, filename='heat_sim.gif'):
    # inputs:
    # u0: function defining the initial heat distribution
    # a: left endpoint
    # b: right endpoint
    # dt: timestep 
    # nx: number of points for spatial discretization
    # nt: number of timesteps to run for
    # k: the function specifying the thermal diffusivity
    # dk: derivative of k
    # ddk: derivative of dk
    #
    # output:
    # creates a gif with two plots one of the evolution of the two heat equations
    # as well as one with the curvatures of the two heat equations
    return

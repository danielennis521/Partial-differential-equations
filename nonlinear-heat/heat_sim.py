import numpy as np
import nonlinear_heat_fd as nh


class heat_sim():
    def __init__(self, boundary, boundary_conditions, bound_type, dt):
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
        self.x = []
        self.u = []
        self.k = []
        self.dk = []
        self.ddk = []
        self.dx = []


    def add_sim(self, u0, k, dk, ddk, nx):
        # inputs: 
        # u0: a function specifying the initial distribution of the data
        # k: the function specifying the thermal diffusivity
        # dk: derivative of k
        # ddk: derivative of dk
        # nx: number of points to use for spatial discretization

        domain = np.linspace(self.boundary[0], self.boundary[1], nx, endpoint=True)
        self.dx.append(domain[1] - domain[0])

        self.x.append(domain)
        self.u.append([u0(t) for t in domain])
        self.k.append(k)
        self.dk.append(dk)
        self.ddk.append(ddk)


    def step(self):

        for i in range(len(self.u)):
            c = self.dt/self.dx[i]**2
            J = nh.generate_jacobian(self.u[i], c, self.k[i], self.dk[i], self.ddk[i])
            self.u[i] = nh.newton_step(self.u[i], J, c, self.k[i], self.dk[i])
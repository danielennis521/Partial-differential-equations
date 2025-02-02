import numpy as np

def first_deriv(u, dx):
    du = np.zeros(len(u))

    for i in range(len(u)):

        if i==0:
            du[i] = (-3*u[i] + 4*u[i+1] - u[i+2]) / (2*dx)
        elif i==len(u)-1:
            du[i] = (3*u[i] - 4*u[i-1] + u[i-2]) / (2*dx)
        else:
            du[i] = (u[i+1] - u[i-1]) / (2*dx)
    
    return du


def second_deriv(u, dx):
    ddu = np.zeros(len(u))

    for i in range(len(u)):

        if i==0:
            ddu[i] = (2*u[i] - 5*u[i+1] + 4*u[i+2] - u[i+3]) / dx**2
        elif i==len(u)-1:
            ddu[i] = (2*u[i] - 5*u[i-1] + 4*u[i-2] - u[i-3]) / dx**2
        else:
            ddu[i] = (u[i-1] - 2*u[i] + u[i+1]) / dx**2

    return ddu


def curvature(u, dx):

    du = first_deriv(u, dx)
    ddu = second_deriv(u, dx)

    k = ddu / np.sqrt( (1 + du**2)**3 )

    return k
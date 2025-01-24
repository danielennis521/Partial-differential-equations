import nonlinear_quadratic__heat as qh
import nonlinear_heat as nh
import numpy as np


def f(x):
    return (1-x)*np.sin(np.pi*x)

def k(u):
    return 0.5*(1 + u**2)

def dk(u):
    return u

def ddk(u):
    return 0

def u0(x):
    return np.sin(np.pi*x)

#nh.heat_compare_gif(u0=u0, a=-1, b=1, dt=1e-3, nx=50, nt=int(2*1e2), k=k, dk=dk, ddk=ddk)
nh.run_heat_sim(u0=u0, a=-1, b=1, dt=1e-3, nx=50, nt=int(2*1e2), k=k, dk=dk, ddk=ddk)

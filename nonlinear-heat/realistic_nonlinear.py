import numpy as np
import numpy.polynomial.polynomial as p
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib.animation import FuncAnimation



############################################################
# functions for discretization
############################################################

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



def chebyshev_polynomials(n, a=-1, b=1):
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



def generate_matricies(x, L, dL, ddL):
    n = len(L)
    m = len(x)
    T = np.zeros([n, m])
    dT = np.zeros([n, m])
    ddT = np.zeros([n, m])

    for i in range(n):
        T[i] = p.polyval(x, L[i])
        dT[i] = p.polyval(x, dL[i])
        ddT[i] = p.polyval(x, ddL[i])
    
    return T.T, dT.T, ddT.T 




############################################################
# set up the simulation with ODE solver
############################################################

n = 15
x = np.linspace(-1, 1, n, endpoint=True)
L = chebyshev_polynomials(n)
dL = [p.polyder(l) for l in L]
ddL = [p.polyder(l) for l in dL]
T, dT, ddT = generate_matricies(x, L, dL, ddL)

B = np.linalg.inv(T)

# enforce boundary conditions
C = ddT.copy()
C[0] = np.zeros(n)
C[-1] = np.zeros(n)

# matrix for ODE system
A = np.dot(np.linalg.inv(T), C)

def f2(t, x):
    return 4*A.dot(x)


def f(t, a):
    
    # spatial discretization of PDE
    b = ( (6 - 1.5*T.dot(a)) * ddT.dot(a) - 1.5*dT.dot(a) * dT.dot(a))
    
    # enforce boundary conditions
    b[0] = 0
    b[-1] = 0

    return np.dot(B, b)


u0 = 2*np.sin(np.pi*x)
a0 = np.linalg.solve(T, u0)
y = np.linspace(-1, 1, 10*n, endpoint=True)
S, dS, ddS = generate_matricies(y, L, dL, ddL)

sim = integrate.Radau(f, 0, a0, t_bound=10, max_step=2*1e-4)
sim2 = integrate.Radau(f2, 0, a0, t_bound=10, max_step=2*1e-4)



############################################################
# create a gif of the results
############################################################

fig, ax = plt.subplots()

def update_plot(frame):

    ax.clear()

    ax.set_title(label='time = '+str(round(sim.t, 5)))
    ax.set_ylim(bottom=-2.1, top=2.1)
    ax.plot(y, S.dot(sim.y), label='nonlinear')
    ax.plot(y, S.dot(sim2.y), label='linear')
    ax.legend()
    sim.step()
    sim2.step()


ani = FuncAnimation(fig, update_plot, frames=150, repeat=False)
ani.save(filename='realistic_nonlinear.gif', writer='pillow', fps=60)
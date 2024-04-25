import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import matplotlib.pyplot as plt


def Gram_Schmidt_Orthonormalization(n, a=-1, b=1):
    # Generate the n orthogonal polynomials using the Gram-Schmidt orthonormalization process

    L = []
    # generate the orthogonal polynomials   
    for i in range(2, n):
        if i==0:
            p = np.polynomial.Polynomial([1])

        elif i==1:
            p = 
        else:
            num = (p*L[i-1]*L[i-1]).integ()
            den = (L[i-1]*L[i-1]).integ()
            p = np.polynomial.Polynomial([(num(b) - num(a))/(den(b) - den(a)), 1])*L[i-1]
            
            den = (L[i-2]*L[i-2]).integ()
            p -= ((den(b) - den(a))/(num(b) - num(a)))*L[i-2]    
        
        L.append(p)

    # find the roots of the highest order polynomial (these will be the collocation points)
    x = L[-1].roots()

    # return the Legendre polynomials and the collocation points
    return L, x


def populate_matrix(A, x, dt, nx, k, dk, L, dL, ddL):

    for i in range(nx):
        for j in range(nx):
            A[i][j] = np.polyval(L[j], x[i]) - dt*[np.polyval(dL[j], x[i])*dk[i] + np.polyval(ddL[j], x[i])*k[i]]
            

def diffusion_sim(nt, nx, dt, u0, a, b, ua, ub, k, dk):
    # Simulate the diffusion equation using the collocation method
    # note that we use the legendra polynomials as basis functions and the roots of the largest one as collocation points
    # be careful about the number of collocation points used, collocation generally requires much fewer points than finite differences

    A = np.zeros([nx, nx])
    lim = [min([ua, ub]+u0.tolist()), max([ua, ub]+u0.tolist())]
    u = u0
    t = 0

    L, x = Gram_Schmidt_Orthonormalization(nx)
    populate_matrix(A, x, dt, nx, k, dk, L, dL, ddL)

    for i in range(nt):

        u = sp.linalg.gmres(A, u)

        # plot the solution
        plt.cla()
        plt.plot([a] + x.tolist() + [b], [ua]+u.tolist()+[ub], label='t = {}'.format(t))
        plt.legend()
        plt.ylim(lim)
        plt.pause(0.0001)

        t += dt

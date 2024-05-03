import numpy as np
import numpy.polynomial.polynomial as p
import scipy.linalg as la
import scipy.sparse as sp
import matplotlib.pyplot as plt


def Gram_Schmidt_Orthonormalization(n, a=-1, b=1):
    # Generate the n orthogonal polynomials using the Gram-Schmidt orthonormalization process
    
    L = [[1]]
    # generate the orthogonal polynomials   
    for i in range(1, n):
        num = p.polyint(p.polymul([0, 1], p.polymul(L[i-1], L[i-1])))
        den = p.polyint(p.polymul(L[i-1],L[i-1]))
        c = (p.polyval(a, num) - p.polyval(b, num))/(p.polyval(b, den) - p.polyval(a, den))
        L.append(p.polymul([c, 1], L[i-1]))

        if i > 1:
            num = den
            den = p.polyint(p.polymul(L[i-2],L[i-2]))
            c = (p.polyval(a, num) - p.polyval(b, num))/(p.polyval(b, den) - p.polyval(a, den))
            L[-1] = p.polyadd(L[-1], p.polymul([c], L[i-2]))    
 
    # find the roots of the highest order polynomial (these will be the collocation points)
    x = p.polyroots(L[-1])
    # return the Legendre polynomials and the collocation points
    return L, x


def populate_matrix(A, x, dt, nx, k, dk, L, dL, ddL):

    for i in range(nx):
        for j in range(nx):
            A[i][j] = p.polyval(x[i], L[j]) - dt*(p.polyval(x[i], dL[j])*dk(x[i]) + p.polyval(x[i], ddL[j])*k(x[i]))
            

def diffusion_sim(nt, nx, dt, u0, a, b, ua, ub, k, dk):
    # Simulate the diffusion equation using the collocation method
    # note that we use the legendra polynomials as basis functions and the roots of the largest one as collocation points
    # be careful about the number of collocation points used, collocation generally requires much fewer points than finite differences

    A = np.zeros([nx, nx])
    lim = [min([ua, ub]+u0.tolist()), max([ua, ub]+u0.tolist())]
    t = 0

    L, x = Gram_Schmidt_Orthonormalization(nx+1)
    u = [np.sin(i) for i in x]
    dL = [[0]] + [np.polyder(poly) for poly in L[1:]]
    ddL = [[0], [0]] + [np.polyder(poly) for poly in dL[2:]]

    populate_matrix(A, x, dt, nx, k, dk, L, dL, ddL)

    for i in range(nt):

        u = sp.linalg.gmres(A, u)[0]

        # plot the solution
        plt.cla()
        print(u)
        plt.plot([a] + list(x) + [b], [ua] + u.tolist() + [ub], label='t = {}'.format(t))
        plt.legend()
        plt.ylim(lim)
        plt.pause(0.01)

        t += dt
        

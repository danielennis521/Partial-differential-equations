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
 
    return L


def Chebyshev_Polynomials(n, a=-1, b=1):
    # Generate the n Chebyshev polynomials
    
    L = [[1]]
    L.append([0, 1])
    for i in range(2, n):
        L.append(p.polysub(p.polymul([0, 2], L[i-1]), L[i-2]))
    
    return L


def populate_matrix(M, N, x, ua, ub, nx, k, dk, L, dL, ddL):

    for i in range(nx):
        for j in range(nx):
            if i == 0:
                N[i][j] = ua
            elif i == nx-1:
                N[i][j] = ub
            else:
                N[i][j] = (p.polyval(x[i], dL[j])*dk(x[i]) + p.polyval(x[i], ddL[j])*k(x[i]))

            M[i][j] = p.polyval(x[i], L[j])
                        

def diffusion_sim(nt, nx, dt, u0, a, b, k, dk, ua=0, ub=0):
    # Simulate the diffusion equation using the collocation method
    # note that we use the legendra polynomials as basis functions and the roots of the largest one as collocation points
    # be careful about the number of collocation points used, collocation generally requires much fewer points than finite differences

    M = np.zeros([nx, nx])
    N = np.zeros([nx, nx])
    lim = [-1, 1]
    t = 0
    
    #L = Gram_Schmidt_Orthonormalization(nx)
    L = Chebyshev_Polynomials(nx)
    x = [a + (b-a)/(nx-1)*i for i in range(nx)]
    w = np.array([[p.polyval(x[i], L[j]) for j in range(nx)] for i in range(nx)])

    u = [ua] + [np.sin(np.pi*i) for i in x[1:-1]] + [ub]
    dL = [[0]] + [np.polyder(poly) for poly in L[1:]]
    ddL = [[0], [0]] + [np.polyder(poly) for poly in dL[2:]]

    populate_matrix(M, N, x, ua, ub, nx, k, dk, L, dL, ddL)

    for i in range(nt):

        # solve for the collocation coefficients
        a = sp.linalg.gmres(M+dt*N, u)[0]
        # reconstruct the solution
        u = np.dot(M, a)
        
        # plot the solution
        #if i % 100 == 0:
        if True:
            plt.cla()
            plt.plot(list(x), u.tolist(), label='t = {}'.format(t))
            plt.legend()
            plt.ylim(lim)
            plt.pause(0.03)

        t += dt
        

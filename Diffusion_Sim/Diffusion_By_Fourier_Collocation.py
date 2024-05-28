import numpy as np
import numpy.polynomial.polynomial as p
import scipy.linalg as la
import matplotlib.pyplot as plt




def diffusion_sim(c, dc, nt, nx, dt, u0, a=-1, b=1, ua=0, ub=0):

    x = np.linspace(a, b, nx, endpoint=True)
    u = [ua] + [u0(x[i]) for i in range(1, nx-1)] + [ub]
    v = [c(x[i]) for i in range(nx)]
    dv = [dc(x[i]) for i in range(nx)]

    # construct the frequency domain
    k=np.zeros(nx)

    if ((nx%2)==0):
        #-even number                                                                                   
        for i in range(1,nx//2):
            k[i]=i
            k[nx-i]=-i
    else:
        #-odd number                                                                                    
        for i in range(1,(nx-1)//2):
            k[i]=i
            k[nx-i]=-i

    lim = [min(u), max(u)]

    for i in range(nt):

        f = np.fft.fft(u)
        du = np.fft.ifft(1j*k*f).real
        ddu = np.fft.ifft(-k**2*f).real

        # slopes and their transforms for RK4
        l1 = dv*du + v*ddu
        fl1 = np.fft.fft(l1)

        l2 = dv*(u + np.fft.ifft(1j*k*fl1).real*dt/2) + v*(du*np.fft.ifft(-k**2*fl1).real*dt/2)
        fl2 = np.fft.fft(l2)

        l3 = dv*(u + np.fft.ifft(1j*k*fl2).real*dt/2) + v*(du*np.fft.ifft(-k**2*fl2).real*dt/2)
        fl3 = np.fft.fft(l3)
        
        l4 = dv*(u + np.fft.ifft(1j*k*fl3).real*dt) + v*(du*np.fft.ifft(-k**2*fl3).real*dt)

        u = u + dt*(l1 + 2*l2 + 2*l3 + l4)/6


        if i % 250 == 0:
            y = np.linspace(a, b, 100)
            
            plt.cla()
            plt.ylim(lim)
            #plt.plot(y, z, 'b')
            plt.plot(x, u, 'ro')
            plt.plot(x, u, 'r')
            plt.pause(0.1)


        

def u0(x):
    return np.sin(np.pi*x)

def u01(x):
    if np.abs(x) >= 1.0:
        return 0
    else:
        return np.exp(-1/(1-x**2))

def k(x):
    return 1.0

def dk(x):
    return 0.0


diffusion_sim(k, dk, int(1e7), 9, 1e-4, u0, -1, 1, 0, 0)
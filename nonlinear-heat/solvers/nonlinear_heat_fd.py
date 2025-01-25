import numpy as np



def F(u, prev, c, k, dk):

    f = np.zeros(len(u))

    for i in range(len(u)):
        # du and ddu represent the differences used to approximate derivatives
        if i==0:
            du = u[i+1]
            ddu = ( - 2*u[i] + u[i+1])

            f[i] = prev[i] - u[i] - c*( 0.25*dk(u[i]) * du**2 + k(u[i]) * ddu )
            
        elif i==len(u)-1:
            du = -u[i-1]
            ddu = (u[i-1] - 2*u[i])

            f[i] = prev[i] - u[i] - c*( 0.25*dk(u[i]) * du**2 + k(u[i]) * ddu )
            
        else:
            du = (u[i+1] - u[i-1])
            ddu = (u[i-1] - 2*u[i] + u[i+1])

            f[i] = prev[i] - u[i] - c*( 0.25*dk(u[i]) * du**2 + k(u[i]) * ddu )
    
    return f



def generate_jacobian(u, c, k, dk, ddk):

    n = len(u)
    J = np.zeros([n, n])

    for i in range(n):
        # du and ddu represent the differences used to approximate derivatives
        if i==0:
            du = u[i+1]
            ddu = ( - 2*u[i] + u[i+1])

            J[i][i] = 1 - c*(0.25*ddk(u[i])*du + dk(u[i])*ddu - 2*k(u[i]))
            
            J[i][i+1] = -c*(0.5*dk(u[i])*du + k(u[i]))        

        elif i==n-1:
            du = -u[i-1]
            ddu = (u[i-1] - 2*u[i])

            J[i][i-1] = c*(0.5*dk(u[i])*du - k(u[i]))

            J[i][i] = 1 - c*(0.25*ddk(u[i])*du + dk(u[i])*ddu - 2*k(u[i]))
            
        else:
            du = (u[i+1] - u[i-1])
            ddu = (u[i-1] - 2*u[i] + u[i+1])

            J[i][i-1] = c*(0.5*dk(u[i])*du - k(u[i]))

            J[i][i] = 1 - c*(0.25*ddk(u[i])*du + dk(u[i])*ddu - 2*k(u[i]))
            
            J[i][i+1] = -c*(0.5*dk(u[i])*du + k(u[i]))

    return J



def newton_step(u, J, c, k, dk):

    n = len(u)
    u_prev = u
    u_next = u

    for i in range(100):
        u_prev = u_next
        u_next -= np.linalg.solve(J, F(u, u_next, c, k, dk))

        if np.linalg.norm(u_prev - u_next) < 1e-9 * n:
            break

    return u_next




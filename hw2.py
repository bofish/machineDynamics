import numpy as np
from math import pi, radians, degrees
import matplotlib.pyplot as plt
from hw1 import Fourbar
from numerical.calculus import trapezoidal_integ

if __name__ == "__main__":

    N = 720
    L = np.array([0.3, 0.1, 0.18, 0.25, 0.36, 0.18])
    m = np.array([0, 1.0, 2.0, 0.2])
    I = np.array([0, 0.02, 0.06, 0.005])
    theta = np.array([0.0, 0.0, 1, 0])
    omega2 = 120 / 60 * 2 * pi
    alpha2 = 0
    ic = theta[2:4]
    x = []
    y = []
    Theta = list([])
    Omega = list([])
    Alpha = list([])
    Staticsforce = list([])
    Dynamicsforce = list([])
    Shakingforce = list([])
    Shakingforce_abs = []
    Shakingmoment_abs = []
    fourbar = Fourbar(L, m, I)

    for theta2 in np.linspace(0, 2*pi, N):
        # print(degrees(theta2) )
        if theta2 >= radians(150) and theta2 <= radians(240):
            M4 = 20
        else:
            M4 = 0

        theta[1] = theta2
        theta, ic, plot_position = fourbar.get_position(theta, ic)
        dtheta = fourbar.get_angularVelocity(omega2)
        ddtheta = fourbar.get_angularAcceleration(alpha2)
        staticsforce = fourbar.get_staticsForce(M4)
        ddx, ddy = fourbar.get_acceleration()
        dynamicsforce = fourbar.get_dynamicsForce(M4)
        shakingRes = fourbar.get_shakingForce()
        
        x.append(plot_position[0])
        y.append(plot_position[1])
        Theta.append(theta.tolist())
        Omega.append(dtheta.tolist())
        Alpha.append(ddtheta.tolist())
        Staticsforce.append(staticsforce.tolist())
        Dynamicsforce.append(dynamicsforce.tolist())
        Shakingforce.append(shakingRes['shakingForce'].tolist())
        Shakingforce_abs.append(shakingRes['shakingForce_abs'])
        Shakingmoment_abs.append(shakingRes['shakingMoment_abs'])

    Theta = np.array(Theta)
    Omega = np.array(Omega)
    Alpha = np.array(Alpha)
    Dynamicsforce = np.array(Dynamicsforce)
    Shakingforce = np.array(Shakingforce)
    ls = ['-', '--', '-.', ':']

    TL = Dynamicsforce[:,8]
    TLavg = np.sum(TL)/TL.shape
    # root_i = [i for i in range(len(TL)) if TL[i]-TLavg<0.19 and TL[i]-TLavg>-0.05]
    # root_i.insert(0, 0)
    # root_i.insert(5, N*240//360)
    # root_i.append(N-1)
    root_i = [0, 12, 130, 217, 352, 428, 479, 480, 719]
    # print(root_i)
    # print(len(root_i),Theta[root_i,1], TL[root_i])
    
    delta_E = []
    for i in range(len(root_i)-1):
        a = root_i[i]
        b = root_i[i+1]
        delta_E.append( trapezoidal_integ(Theta[a:b,1], TL[a:b]-TLavg))   
    E_accum = np.cumsum(delta_E)
    print(delta_E, E_accum)
    print(np.argmax(E_accum), np.argmin(E_accum))
# Ploting 
    
    # Input torque
    plt.figure()
    plt.plot(Theta[:,1], TL, 'x', label='TL(T2)')
    plt.plot(Theta[:,1], np.ones(N)*TLavg, '--r', label='TLavg')
    plt.plot(Theta[root_i,1], TL[root_i], 'o', label='root')
    plt.xlabel(r'$\theta_2 (rad)$')
    plt.ylabel(r'$Input\ Torque (N \cdot m)$')
    plt.legend()
    plt.show()
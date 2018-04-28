from math import pi, radians, degrees
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from hw1 import Fourbar
from numerical.calculus import trapezoidal_integ

def objFun(x, fourbar, T_motor, I_shaft, M4):
    '''
    x[0]=omega2, x[1]=alpha2
    '''
    fourbar.get_angularVelocity(x[0])
    fourbar.get_angularAcceleration(x[1])
    fourbar.get_velocity()
    fourbar.get_acceleration()
    T2 = fourbar.get_T_by_energy_method(M4)

    return T2-T_avg, 0

if __name__ == "__main__":

    N = 360
    L = np.array([0.3, 0.1, 0.18, 0.25, 0.36, 0.18])
    m = np.array([0, 1.0, 2.0, 0.2])
    theta = np.array([0.0, 0.0, 1, 0])
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

    # hw2 new parameter
    I_shaft = 1.119 # Magic I: 5.7081
    I = np.array([0, I_shaft, 0.06, 0.005])
    T_avg = 1.99900449
    omega2_avg = 120 / 60 * 2 * pi
    omega2 = []
    alpha2 = []
    fs_res = {'x':[12.56, 0]}  # Initial guess: omega2 = omega2_avg
    fourbar = Fourbar(L, m, I)

    for theta2 in  np.linspace(0, 2*pi, N):
        if theta2 >= radians(150) and theta2 <= radians(240):
            M4 = 20
        else:
            M4 = 0

        theta[1] = theta2
        theta, ic, plot_position = fourbar.get_position(theta, ic)
        
        # Inverse Dynamic to find out the omega2 and alpha2
        fs_res = root(objFun, fs_res['x'],args=(fourbar, T_avg, I_shaft, M4), method='lm')

        # Analysis the Kinematic and Dynamic of the system 
        dtheta = fourbar.get_angularVelocity(fs_res.x[0])
        ddtheta = fourbar.get_angularAcceleration(fs_res.x[1])
        dx, dy = fourbar.get_velocity()
        ddx, ddy = fourbar.get_acceleration()
        dynamicsforce = fourbar.get_dynamicsForce(M4)
        shakingRes = fourbar.get_shakingForce(M4)
    
        # Store the result
        Theta.append(theta.tolist())
        Omega.append(dtheta.tolist())
        Alpha.append(ddtheta.tolist())
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

#Plotting
    # Angular velocity
    fig, ax = plt.subplots()
    ax.plot(Theta[:,1], Omega[:,1], ls=ls[0], label=r'$\omega_2$')
    ax.plot(Theta[:,1], Omega[:,2], ls=ls[1], label=r'$\omega_3$')
    ax.plot(Theta[:,1], Omega[:,3], ls=ls[2], label=r'$\omega_4$')
    ax.set_xlabel(r'$\theta_2 (rad)$')
    ax.set_ylabel(r'$\omega (rad/s)$')
    ax.legend(loc=1) 

    # Angular Acceleration
    fig, ax = plt.subplots()
    ax.plot(Theta[:,1], Alpha[:,1], ls=ls[0], label=r'$\alpha_2$')
    ax.plot(Theta[:,1], Alpha[:,2], ls=ls[1], label=r'$\alpha_3$')
    ax.plot(Theta[:,1], Alpha[:,3], ls=ls[2], label=r'$\alpha_4$')
    ax.set_xlabel(r'$\theta_2 (rad)$')
    ax.set_ylabel(r'$\alpha (rad/s^2)$')
    ax.legend(loc=1)

    # Input torque
    fig, ax = plt.subplots()
    ax.plot(Theta[:,1], Dynamicsforce[:,8], ls=ls[0], label=r'$T_2$')
    ax.set_xlabel(r'$\theta_2 (rad)$')
    ax.set_ylabel(r'$Input\ Torque (N \cdot m)$')
    ax.legend(loc=1)

    # Dynamic reaction force
    fig, ax = plt.subplots()
    reactionForce12 = np.sqrt(Dynamicsforce[:,0]**2 + Dynamicsforce[:,1]**2)
    reactionForce32 = np.sqrt(Dynamicsforce[:,2]**2 + Dynamicsforce[:,3]**2)
    ax.plot(Theta[:,1], reactionForce12, ls=ls[0], label=r'$Force_{12}$')
    ax.plot(Theta[:,1], reactionForce32, ls=ls[1], label=r'$Force_{32}$')
    ax.set_xlabel(r'$\theta_2 (rad)$')
    ax.set_ylabel(r'$Reaction\ force (N)$')
    ax.legend(loc=1)

    # Shaking force
    fig, ax = plt.subplots()
    ax.plot(Theta[:,1], Shakingforce_abs, ls=ls[0], label=r'$|F_{s}|$')
    ax.plot(Theta[:,1], Shakingforce[:,0], ls=ls[1], label=r'$F_{sx}$')
    ax.plot(Theta[:,1], Shakingforce[:,1], ls=ls[2], label=r'$F_{sy}$')
    ax.set_xlabel(r'$\theta_2 (rad)$')
    ax.set_ylabel(r'$Shaking\ Force (N)$')
    ax.legend(loc=1)

    # Shaking Moment
    fig, ax = plt.subplots()
    ax.plot(Theta[:,1], Shakingmoment_abs, label=r'$M_{s}$')
    ax.set_xlabel(r'$\theta_2 (rad)$')
    ax.set_ylabel(r'$Shaking\ Moment (N \cdot m)$')
    ax.legend(loc=1)

    # # Shaking force and moment in polar plot
    # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    # ax.plot(Theta[:,1], Shakingforce_abs, label=r'$|F_s|$')
    # ax.legend()
    # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    # ax.plot(Theta[:,1], Shakingmoment_abs, label=r'$|M_s|$')
    # ax.legend()

    plt.show()

# # Without flywheel, calculate Tavg, root and delta E
#     N = 3600
#     TL = Dynamicsforce[:,8]
#     TLavg = np.sum(TL)/TL.shape
#     # root_i = [i for i in range(len(TL)) if TL[i]-TLavg<0.05 and TL[i]-TLavg>-0.05]
#     # root_i.insert(0, 0)
#     # root_i.insert(5, N*240//360)
#     # root_i.append(N-1)
#     root_i = [0, 61, 652, 1088, 1764, 2143, 2399, 2400, N-1]
#     # print(root_i)
#     # # print(len(root_i),Theta[root_i,1], TL[root_i])
#     print(TLavg)
#     delta_E = []
#     for i in range(len(root_i)-1):
#         a = root_i[i]
#         b = root_i[i+1]
#         delta_E.append( trapezoidal_integ(Theta[a:b,1], TL[a:b]-TLavg))   
#     E_accum = np.cumsum(delta_E)
#     print(delta_E, E_accum)
#     print(np.argmin(E_accum), np.argmax(E_accum))

#     # Ploting input torque 
#     plt.figure()
#     plt.plot(Theta[:,1], TL, '-b', label='TL(T2)')
#     plt.plot(Theta[:,1], np.ones(N)*TLavg, '--r', label='TLavg')
#     plt.plot(Theta[root_i,1], TL[root_i], 'o', label='root')
#     plt.xlabel(r'$\theta_2 (rad)$')
#     plt.ylabel(r'$Input\ Torque (N \cdot m)$')
#     plt.legend()
#     plt.show()


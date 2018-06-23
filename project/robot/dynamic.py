from math import sin, cos, pi
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def cal_force(theta, alpha, ddq_CG, m, I, l_CG, DH, N):
    g = 9.81
    d1 = DH[0]
    a2 = DH[1]
    a3 = DH[2]
    In1 = []
    In2 = []
    In3 = []
    for i in range(N):
        t2 = theta[i, 1]
        t3 = theta[i, 2]
        x2 = a2*(1+l_CG[1])*cos(t2)
        y2 = a2*(1+l_CG[1])*sin(t2)
        x3 = a2*cos(t2) + a3*(1+l_CG[2])*cos(t2 + t3)
        y3 = a2*sin(t2) + a3*(1+l_CG[2])*sin(t2 + t3)
        p2 = np.array([x2, y2])
        p3 = np.array([x3, y3])
        In1.append( I[0] + I[1] + m[1]*x2**2 + I[2] + m[2]*x3**2 )
        In2.append( I[1] + I[2] + m[2]*np.linalg.norm(p2-p3)**2 )
        In3.append( I[2] )
    force = np.zeros((N, 9))
    for i in range(N):
        t1 = theta[i, 0]
        t2 = theta[i, 1]
        t3 = theta[i, 2]
        ddt1 = alpha[i, 0]
        ddt2 = alpha[i, 1]
        ddt3 = alpha[i, 2]
        a2x = ddq_CG[i, 1, 0]
        a2y = ddq_CG[i, 1, 1]
        a3x = ddq_CG[i, 2, 0]
        a3y = ddq_CG[i, 2, 1]
        A1 = [1, 0, -1,  0,  0,  0, 0, 0, 0]
        A2 = [0, 1,  0, -1,  0,  0, 0, 0, 0]
        A3 = [0, 0,  1,  0, -1,  0, 0, 0, 0]
        A4 = [0, 0,  0,  1,  0, -1, 0, 0, 0]
        A5 = [0, 0,  0,  0,  1,  0, 0, 0, 0]
        A6 = [0, 0,  0,  0,  0,  1, 0, 0, 0]
        A7 = [0, 0,  0,  0,  0,  0, 1, 0, 0]
        A8 = [0, 0, a2*(1+l_CG[1])*cos(t2), -a2*(1+l_CG[1])*sin(t2), a2*(-l_CG[1])*cos(t2), -a2*(-l_CG[1])*sin(t2), 0, 1, 0]
        A9 = [0, 0, 0, 0, a3*(1+l_CG[2])*cos(t2+t3), -a3*(1+l_CG[2])*cos(t2+t3), 0, 0, 1]
        Ad = np.array([A1, A2, A3, A4, A5, A6, A7, A8, A9])
        b1 = [0.0]
        b2 = [m[0]*g]
        b3 = [m[1]*a2x]
        b4 = [m[1]*(a2y+g)]
        b5 = [m[2]*a3x]
        b6 = [m[2]*(a3y+g)]
        b7 = [In1[i]*ddt1]
        b8 = [In2[i]*ddt2]
        b9 = [In3[i]*(ddt2+ddt3)]
        Bd = np.array([b1, b2, b3, b4, b5, b6, b7, b8, b9])
        Xd = np.linalg.lstsq(Ad, Bd, rcond=-1)
        force[i, :] = Xd[0].T
    return force

def cal_energy(omega, dq_CG, force, m):
    g = 9.81
    M1 = force[:, 6]
    M2 = force[:, 7]
    M3 = force[:, 8]
    dt1 = omega[:, 0]
    dt2 = omega[:, 1]
    dt3 = omega[:, 2]
    v2y = dq_CG[:, 1, 1]
    v3y = dq_CG[:, 2, 1]
    power = M1*dt1 + M2*dt2 + M3*dt3 + m[1]*g*v2y + m[2]*g*v3y
    energy = scipy.integrate.simps(power)
    return energy

if __name__ == '__main__':
    #----- Acceleration Plan -----#
    from acceleration_plan import acceleration_plan
    P_tar = [[4, -5], [4, 5]]
    T_acc = 0.83479259
    V_tar = 2.99252796
    N = 100
    P, dP, ddP, time = acceleration_plan(P_tar, T_acc, V_tar, N)

    #----- Kinematic Analysis -----#
    import kinematic as km
    # Angular displacement, velocity, acceleration
    DH = [0.5, 3.59435767, 3.54154706]
    init_theta = [0.0, 0.0, 0.0]
    theta = km.cal_angular_displacement(P, init_theta, DH, N)
    omega = np.gradient(theta, axis=0)
    alpha = np.gradient(omega, axis=0)

    # Linear displacement, velocity, acceleration of CG
    l_CG = [-0.5, -2.0, -1.0]
    q_CG = km.cal_linear_displacement(l_CG, theta, DH, N)
    dq_CG = np.gradient(q_CG, axis=0)
    ddq_CG = np.gradient(dq_CG, axis=0)

    #----- Dynamic Analysis -----#
    m = [1.0, 1.0, 1.0]
    I = [1.0, 1.0, 1.0]
    force = cal_force(theta, alpha, ddq_CG, m, I, l_CG, DH, N)
    energy = cal_energy(omega, dq_CG, force, m)

    #----- Plot -----#
    # plt.figure(figsize=[8, 8])
    # plt.subplot(3,1,1)
    # plt.plot(time, force[:, 6], label='torque 1')
    # plt.title('l_CG = {}, {}, {}'.format(*l_CG))
    # plt.legend()
    # plt.subplot(3,1,2)
    # plt.plot(time, force[:, 7], label='torque 2')
    # plt.legend()
    # plt.subplot(3,1,3)
    # plt.plot(time, force[:, 8], label='torque 3')
    # plt.legend()

    # plt.figure()
    # plt.title('l_CG = {}, {}, {}'.format(*l_CG))
    # plt.plot(time, theta[:, 0], label='theta 1')
    # plt.plot(time, theta[:, 1], label='theta 2')
    # plt.plot(time, theta[:, 2], label='theta 3')
    # plt.legend()
    # plt.show()

    # Animations
    from general import make_ani
    l = [0.0, 0.0, 0.0]
    q = km.cal_linear_displacement(l, theta, DH, N)
    x = np.concatenate((np.zeros((N,1)), q[:, :, 0]), axis=1)
    y = np.concatenate((np.zeros((N,1)), q[:, :, 1]), axis=1)
    z = np.concatenate((np.zeros((N,1)), q[:, :, 2]), axis=1)
    x_locus = x[:,3]
    y_locus = y[:,3]
    z_locus = z[:,3]
    ani = make_ani(x, y, z, x_locus, y_locus, z_locus, q_CG, N, save_gif = False)
    plt.show()

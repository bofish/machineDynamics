from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def position(x, P, DH):
    theta1 = float(x[0])
    theta2 = float(x[1])
    theta3 = float(x[2])
    P_x = P[0]
    P_y = P[1]
    P_z = P[2]
    d1 = DH[0]
    a2 = DH[1]
    a3 = DH[2]
    return [
    a2*cos(theta1)*cos(theta2) + a3*cos(theta1)*cos(theta2 + theta3) - P_x,
    a2*sin(theta1)*cos(theta2) + a3*sin(theta1)*cos(theta2 + theta3) - P_y,
    a2*sin(theta2) + a3*sin(theta2 + theta3) + d1 - P_z]

def cal_angular_displacement(P, init_theta, DH, N):
    theta = np.zeros((N, 3))
    for i in range(N):
        theta[i, :] = fsolve(position, init_theta, args=(P[i, :], DH))
        init_theta = theta[i, :]
    return theta

def cal_linear_displacement(P, theta, DH, N):
    d1 = DH[0]
    a2 = DH[1]
    a3 = DH[2]
    q_plot = np.array([np.zeros((3, 4)) for i in range(N)])
    q_CG = np.array([np.zeros((3, 3)) for i in range(N)])
    for i in range(N):
        t1 = theta[i, 0]
        t2 = theta[i, 1]
        t3 = theta[i, 2]
        x_plot = [0, 0, a2*cos(t1)*cos(t2), P[i, 0]]
        y_plot = [0, 0, a2*sin(t1)*cos(t2), P[i, 1]]
        z_plot = [0, d1, a2*sin(t2)+d1, P[i, 2]]
        q_plot[i, :] = [x_plot, y_plot, z_plot]
        q1_CG = [0.0, 0.0, d1/2]
        q2_CG = [0.0+a2*cos(t1)*cos(t2)/2, 0.0+a2*sin(t1)*cos(t2)/2, d1+a2*sin(t2)/2]
        q3_CG = [0.0+a2*cos(t1)*cos(t2)+a3*cos(t1)*cos(t2 + t3)/2, 0.0+a2*sin(t1)*cos(t2)+a3*sin(t1)*cos(t2 + t3)/2, d1+a2*sin(t2)+a3*sin(t2 + t3)/2]
        q_CG[i, :] = [q1_CG, q2_CG, q3_CG]
    return q_CG, q_plot

# By analytical method

def cal_angular_velocity(dP, theta, DH, N):
    d1 = DH[0]
    a2 = DH[1]
    a3 = DH[2]
    omega = np.zeros((N, 3))
    for i in range(N):
        t1 = theta[i, 0]
        t2 = theta[i, 1]
        t3 = theta[i, 2]
        Av = np.array([
            [-a2*sin(t1)*cos(t2)-a3*sin(t1)*cos(t2+t3), -a2*cos(t1)*sin(t2)-a3*cos(t1)*sin(t2+t3), -a3*cos(t1)*sin(t2+t3)], 
            [a2*cos(t1)*cos(t2)+a3*cos(t1)*cos(t2+t3), -a2*sin(t1)*sin(t2)-a3*sin(t1)*sin(t2+t3), -a3*sin(t1)*sin(t2+t3)], 
            [0.0, a2*cos(t2)+a3*cos(t2+t3), a3*cos(t2+t3)]])
        Bv = np.array([[dP[i, 0]],[dP[i, 1]],[dP[i, 2]]])
        Xv = np.linalg.lstsq(Av, Bv, rcond=-1)
        omega[i, :] = Xv[0].T
    return omega

def cal_linear_velocity(dP, theta, omega, DH, N):
    d1 = DH[0]
    a2 = DH[1]
    a3 = DH[2]
    dq_CG = np.array([np.zeros((3, 3)) for i in range(N)])
    for i in range(N):
        t1 = theta[i, 0]
        t2 = theta[i, 1]
        t3 = theta[i, 2]
        dt1 = omega[i, 0]
        dt2 = omega[i, 1]
        dt3 = omega[i, 2]
        # TODO: dx1, dy1, dz1
        dq1_CG = [1*i, 2*i, 3*i]
        dq2_CG = [4*i, 5*i, 6*i]
        dq3_CG = [7*i, 8*i, 9*i]
        dq_CG[i, :] = [dq1_CG, dq2_CG, dq3_CG]
    return dq_CG

def cal_angular_acceleration(dP, theta, omega, DH, N):
    d1 = DH[0]
    a2 = DH[1]
    a3 = DH[2]
    alpha = np.zeros((N, 3))
    for i in range(N):
        t1 = theta[i, 0]
        t2 = theta[i, 1]
        t3 = theta[i, 2]
        dt1 = omega[i, 0]
        dt2 = omega[i, 1]
        dt3 = omega[i, 2]
        # A matrix
        A11 = -a2*sin(t1)*cos(t2)-a3*sin(t1)*cos(t2+t3)
        A12 = -a2*cos(t1)*sin(t2)-a3*cos(t1)*sin(t2+t3)
        A13 = -a3*cos(t1)*sin(t2+t3)
        A21 =  a2*cos(t1)*cos(t2)+a3*cos(t1)*cos(t2+t3)
        A22 = -a2*sin(t1)*sin(t2)-a3*sin(t1)*sin(t2+t3)
        A23 = -a3*sin(t1)*sin(t2+t3)
        A31 =  0.0
        A32 =  a2*cos(t2)
        A33 =  a3*cos(t2+t3)
        Aa = np.array([[A11, A12, A13],[A21, A22, A23],[A31, A32, A33]])
        # b vector
        B1 = ddP[i, 0] - a2*(-dt1**2*cos(t1)*cos(t2) + 2*dt1*dt2*sin(t1)*sin(t2) - dt2**2*cos(t1)*cos(t2)) - a3*(-dt1**2*cos(t1)*cos(t2+t3) + 2*dt1*(dt2+dt3)*sin(t1)*sin(t2+t3) - (dt2+dt3)**2*cos(t1)*cos(t2+t3))
        B2 = ddP[i, 1] - a2*(-dt1**2*sin(t1)*cos(t2) - 2*dt1*dt2*cos(t1)*sin(t2) - dt2**2*sin(t1)*cos(t2)) - a3*(-dt1**2*sin(t1)*cos(t2+t3) - 2*dt1*(dt2+dt3)*cos(t1)*sin(t2+t3) - (dt2+dt3)**2*sin(t1)*cos(t2+t3))
        B3 = ddP[i, 2] + a2*dt2**2*sin(t2) + a3*(dt2+dt3)**2*sin(t2+t3)
        Ba = np.array([[B1],[B2],[B3]])
        Xa = np.linalg.lstsq(Aa, Ba, rcond=-1)
        alpha[i, :] = Xa[0].T
    return alpha

def cal_linear_acceleration(ddP, theta, omega, alpha, DH, N):
    d1 = DH[0]
    a2 = DH[1]
    a3 = DH[2]
    ddq_CG = np.array([np.zeros((3, 3)) for i in range(N)])
    for i in range(N):
        t1 = theta[i, 0]
        t2 = theta[i, 1]
        t3 = theta[i, 2]
        dt1 = omega[i, 0]
        dt2 = omega[i, 1]
        dt3 = omega[i, 2]
        ddt1 = alpha[i, 0]
        ddt2 = alpha[i, 1]
        ddt3 = alpha[i, 2]
        # TODO: [ddx1, ddy1, ddz1]
        ddq1_CG = [1, 1, 1]
        ddq2_CG = [1, 1, 1]
        ddq3_CG = [1, 1, 1]
        ddq_CG[i, :] = [ddq1_CG, ddq2_CG, ddq3_CG]
    return ddq_CG

def cal_force(theta, alpha, ddq_CG, m, I,  DH, N):
    g = 9.81
    d1 = DH[0]
    a2 = DH[1]
    a3 = DH[2]
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
        A3 = [0, 0,  0,  0,  0,  0, 1, 0, 0]
        A4 = [0, 0,  1,  0, -1,  0, 0, 0, 0]
        A5 = [0, 0,  1,  0, -1,  0, 0, 0, 0]
        A6 = [0, 0, a2/2*sin(t2), -a2/2*cos(t2), a2/2*sin(t2), -a2/2*cos(t2), 0, 1, 0]
        A7 = [0, 0,  0,  0,  1,  0, 0, 0, 0]
        A8 = [0, 0,  0,  0,  0,  1, 0, 0, 0]
        A9 = [0, 0, 0, 0, a3/2*sin(t2+t3), -a3/2*cos(t2+t3), 0, 0, 1]
        Ad = np.array([A1, A2, A3, A4, A5, A6, A7, A8, A9])
        b1 = [0.0]
        b2 = [m[0]*g]
        b3 = [I[0]*ddt1]
        b4 = [m[1]*a2x]
        b5 = [m[1]*(a2y+g)]
        b6 = [I[1]*ddt2]
        b7 = [m[2]*a3x]
        b8 = [m[2]*(a3y+g)]
        b9 = [I[2]*(ddt2+ddt3)]
        Bd = np.array([b1, b2, b3, b4, b5, b6, b7, b8, b9])
        Xd = np.linalg.lstsq(Ad, Bd, rcond=-1)
        force[i, :] = Xd[0].T
    return force

# By numerical method
def get_angular_velocity(dP, theta, DH, N):
    omega = np.gradient(theta, axis=0)
    return omega

if __name__ == '__main__':
    from acceleration_plan import acceleration_plan
    #----- Acceleration Plan -----#
    P_tar = [[4, -5], [2, 5]]
    T_acc = 0.83479259
    V_tar = 2.99252796
    N = 1000
    P, dP, ddP, time = acceleration_plan(P_tar, T_acc, V_tar, N)

    # Angular displacement, velocity, acceleration
    DH = [0.5, 3.59435767, 3.54154706]
    init_theta = [0.0, 0.0, 0.0]
    theta = cal_angular_displacement(P, init_theta, DH, N)
    omega = np.gradient(theta, axis=0)
    alpha = np.gradient(omega, axis=0)

    # Linear displacement, velocity, acceleration
    q_CG, q_plot = cal_linear_displacement(P, theta, DH, N)
    dq_CG = cal_linear_velocity(dP, theta, omega, DH, N)
    ddq_CG = cal_linear_acceleration(ddP, theta, omega, alpha, DH, N)

    # Dynamic analysis
    m = [1.0, 1.0, 1.0]
    I = [1.0, 1.0, 1.0]
    force = cal_force(theta, alpha, ddq_CG, m, I,  DH, N)


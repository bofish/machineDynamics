from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def DH_matrix(a, alpha, theta, d):
    return np.array([
        [cos(theta), -cos(alpha)*sin(theta), sin(alpha)*sin(theta), a*cos(theta)],
        [sin(theta), cos(alpha)*cos(theta), -sin(alpha)*cos(theta), a*sin(theta)],
        [0.0, sin(alpha), cos(alpha), d],
        [0.0, 0.0, 0.0, 1.0]])

def DH_transform(P_local, theta, DH, N, ord=3):
    d1 = DH[0]
    a2 = DH[1]
    a3 = DH[2]
    P_global = np.zeros((N, 4))
    P = P_local + [1]
    if ord == 3:
        for i  in range(N):
            t1 = theta[i, 0]
            t2 = theta[i, 1]
            t3 = theta[i, 2]
            A1 = DH_matrix(0.0, pi/2, t1, d1)
            A2 = DH_matrix(a2, 0.0, t2, 0.0)
            A3 = DH_matrix(a3, 0.0, t3, 0.0)
            A = np.matmul(A1, np.matmul(A2, A3))
            P_global[i, :] = np.matmul(A, P)
    elif ord == 2:
        for i  in range(N):
            t1 = theta[i, 0]
            t2 = theta[i, 1]
            A1 = DH_matrix(0.0, pi/2, t1, d1)
            A2 = DH_matrix(a2, 0.0, t2, 0.0)
            A = np.matmul(A1, A2)
            P_global[i, :] = np.matmul(A, P)
    elif ord == 1:
        for i  in range(N):
            t1 = theta[i, 0]
            A = DH_matrix(0.0, pi/2, t1, d1)
            P_global[i, :] = np.matmul(A, P)
    return P_global[:, 0:3]

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

def cal_linear_displacement(l_CG, theta, DH, N):
    d1 = DH[0]
    a2 = DH[1]
    a3 = DH[2]
    P_CG_local = [[0.0, l_CG[0]*d1, 0.0], [l_CG[1]*a2, 0.0, 0.0], [l_CG[2]*a3, 0.0, 0.0]]
    P_CG = np.zeros((N, 3, 3))
    for i in range(3):
        P_CG[:, i, :] = DH_transform(P_CG_local[i], theta, DH, N, ord=i+1)
    return P_CG

if __name__ == '__main__':
    from acceleration_plan import acceleration_plan
    #----- Acceleration Plan -----#
    P_tar = [[4, -5], [2, 5]]
    T_acc = 0.83479259
    V_tar = 2.99252796
    N = 1000
    P, dP, ddP, time = acceleration_plan(P_tar, T_acc, V_tar, N)

    #----- Kinematic Analysis -----#
    # Angular displacement, velocity, acceleration
    DH = [0.5, 3.59435767, 3.54154706]
    init_theta = [0.0, 0.0, 0.0]
    theta = cal_angular_displacement(P, init_theta, DH, N)
    omega = np.gradient(theta, axis=0)
    alpha = np.gradient(omega, axis=0)

    # Linear displacement, velocity, acceleration of CG
    l_CG = [-0.5, -0.5, -0.5]
    q_CG = cal_linear_displacement(l_CG, theta, DH, N)
    dq_CG = np.gradient(q_CG, axis=2)
    ddq_CG = np.gradient(dq_CG, axis=2)

    # Dynamic analysis
    m = [1.0, 1.0, 1.0]
    I = [1.0, 1.0, 1.0]
    force = cal_force(theta, alpha, ddq_CG, m, I,  DH, N)


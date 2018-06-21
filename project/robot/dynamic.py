from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':

    # Dynamic analysis
    m = [1.0, 1.0, 1.0]
    I = [1.0, 1.0, 1.0]
    force = cal_force(theta, alpha, ddq_CG, m, I,  DH, N)
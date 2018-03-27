import numpy as np
from math import sin, cos, pi
from scipy.optimize import fsolve
from numpy.linalg import lstsq, norm
import matplotlib.pyplot as plt
from matplotlib import animation
'''
L[0:5]: L1, L2, L3, L4, L_BE, L_CE 
'''


def fourbarLoop(x, L, theta):

    return [
        L[0]*cos(theta[0]) + L[1]*cos(theta[1]) + L[2]*cos(x[0]) - L[3]*cos(x[1]),
        L[0]*sin(theta[0]) + L[1]*sin(theta[1]) + L[2]*sin(x[0]) - L[3]*sin(x[1])
    ]

class Fourbar():

    g = 9.81

    def __init__(self, L, m, I):
        self.L = L
        self.m = m
        self.I = I

    def get_position(self, theta, ic):
        res = fsolve(fourbarLoop, ic, args=(self.L, theta))
        self.theta = np.array([
            theta[0], theta[1], res[0] %(2*pi), res[1] %(2*pi)
        ])
        plot_position = np.array([
            [0, self.L[0]*cos(self.theta[0]), self.L[0]*cos(self.theta[0])+self.L[1]*cos(self.theta[1]), self.L[0]*cos(self.theta[0])+self.L[1]*cos(self.theta[1])+self.L[2]*cos(self.theta[2]), self.L[3]*cos(self.theta[3]), 0],
            [0, self.L[0]*sin(self.theta[0]), self.L[0]*sin(self.theta[0])+self.L[1]*sin(self.theta[1]), self.L[0]*sin(self.theta[0])+self.L[1]*sin(self.theta[1])+self.L[2]*sin(self.theta[2]), self.L[3]*sin(self.theta[3]), 0]
        ])
        ic = res % (2*pi)
        return self.theta, ic, plot_position

    def get_angularVelocity(self, omega2):
        Av = np.matrix([
            [-self.L[2]*sin(self.theta[2]), self.L[3]*sin(self.theta[3])],
            [self.L[2]*cos(self.theta[2]), -self.L[3]*cos(self.theta[3])]
        ])
        Bv = np.matrix([
            [self.L[1]*sin(self.theta[1])*omega2],
            [-self.L[1]*cos(self.theta[1])*omega2]
        ])
        X = lstsq(Av, Bv)
        self.dtheta = np.array([
            0, omega2, X[0][0], X[0][1]
        ])
        return self.dtheta

    def get_angularAcceleration(self, alpha2):
        Aa = np.matrix([
            [-self.L[2]*sin(self.theta[2]), self.L[3]*sin(self.theta[3])],
            [self.L[2]*cos(self.theta[2]), -self.L[3]*cos(self.theta[3])]
        ])
        Ba = np.matrix([
            [self.L[1]*sin(self.theta[1])*alpha2 + self.L[1]*cos(self.theta[1])*self.dtheta[1]**2 + self.L[2]*cos(self.theta[2])*self.dtheta[2]**2 - self.L[3]*cos(self.theta[3])*self.dtheta[3]**2],
            [-self.L[1]*cos(self.theta[1])*alpha2 + self.L[1]*sin(self.theta[1])*self.dtheta[1]**2 + self.L[2]*sin(self.theta[2])*self.dtheta[2]**2 - self.L[3]*sin(self.theta[3])*self.dtheta[3]**2]
        ])
        X = lstsq(Aa, Ba)
        self.ddtheta = np.array([
            0, alpha2, X[0][0], X[0][1]
        ])
        return self.ddtheta

    def get_acceleration(self):
        r_BA = np.array([
            self.L[1]*cos(self.theta[1]), self.L[1]*sin(self.theta[1]), 0
        ])
        r_EB = np.array([
            self.L[4]*cos(self.theta[2]), self.L[4]*sin(self.theta[2]), 0
        ])
        r_G4D = np.array([
            self.L[3]/2*cos(self.theta[3]), self.L[3]/2*sin(self.theta[3]), 0
        ])
        
        a_B = 0.0 + np.cross([0.0, 0.0, self.ddtheta[1]], r_BA) - self.dtheta[1]**2*r_BA
        a_E = a_B + np.cross([0, 0, self.ddtheta[2]], r_EB) - self.dtheta[2]**2*r_EB
        a_G4 = 0 + np.cross([0, 0, self.ddtheta[3]], r_G4D) - self.dtheta[3]**2*r_G4D
        
        r_CD = np.array([
            self.L[3]*cos(self.theta[3]), self.L[3]*sin(self.theta[3]), 0
        ])
        r_EC = np.array([
            self.L[5]*cos(self.theta[2]), self.L[5]*sin(self.theta[2]), 0
        ])
        a_C = 0.0 + np.cross([0.0, 0.0, self.ddtheta[3]], r_CD) - self.dtheta[3]**2*r_CD
        a_E2 = a_C + np.cross([0, 0, self.ddtheta[2]], r_EC) - self.dtheta[2]**2*r_EC

        self.ddx = np.array([
            0, 0, a_E[0], a_G4[0]
        ])
        self.ddy = np.array([
            0, 0, a_E[1], a_G4[1]
        ])
        self.ddz = np.array([
            0, 0, a_E[2], a_G4[2]
        ])
        # print(self.ddx, self.ddy, self.ddz)

        return self.ddx, self.ddy

    def get_staticsForce(self):
        # X[0:8]: f12x, f12y, f23x, f23y, f34x, f34y, f14x, f14y, M2
        Afs = np.matrix([
            [1, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, self.L[1]*sin(self.theta[1]), -self.L[1]*cos(self.theta[1]), 0, 0, 0, 0, 1],
            [0, 0, 1, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, -1, 0, 0, 0],
            [0, 0, self.L[4]*sin(self.theta[2]), -self.L[4]*cos(self.theta[2]), -self.L[5]*sin(self.theta[2]), self.L[5]*cos(self.theta[2]), 0, 0, 0],
            [0, 0, 0, 0, 1, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, -1, 0],
            [0, 0, 0, 0, -self.L[3]/2*sin(self.theta[3]), self.L[3]/2*cos(self.theta[3]), -self.L[3]/2*sin(self.theta[3]), self.L[3]/2*cos(self.theta[3]), 0]
        ])
        Bfs = np.matrix([
            [0],
            [self.m[1]*Fourbar.g],
            [0],
            [0],
            [self.m[2]*Fourbar.g],
            [0],
            [0],
            [self.m[3]*Fourbar.g],
            [0]
        ])
        X = lstsq(Afs, Bfs)
        self.staticsforce = np.array([
            X[0].item(0), X[0].item(1), X[0].item(2), X[0].item(3), X[0].item(4), X[0].item(5), X[0].item(6), X[0].item(7), X[0].item(8)
        ])
        return self.staticsforce

    def get_dynamicsForce(self):
        # X[0:8]: f12x, f12y, f23x, f23y, f34x, f34y, f14x, f14y, M2
        Afs = np.matrix([
            [1, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, self.L[1]*sin(self.theta[1]), -self.L[1]*cos(self.theta[1]), 0, 0, 0, 0, 1],
            [0, 0, 1, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, -1, 0, 0, 0],
            [0, 0, self.L[4]*sin(self.theta[2]), -self.L[4]*cos(self.theta[2]), -self.L[5]*sin(self.theta[2]), self.L[5]*cos(self.theta[2]), 0, 0, 0],
            [0, 0, 0, 0, 1, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, -1, 0],
            [0, 0, 0, 0, -self.L[3]/2*sin(self.theta[3]), self.L[3]/2*cos(self.theta[3]), -self.L[3]/2*sin(self.theta[3]), self.L[3]/2*cos(self.theta[3]), 0]
        ])
        Bfs = np.matrix([
            [self.m[1]*self.ddx[1]],
            [self.m[1]*self.ddy[1] + self.m[1]*Fourbar.g],
            [self.I[1]*self.ddtheta[1]],
            [self.m[2]*self.ddx[2]],
            [self.m[2]*self.ddy[2] + self.m[2]*Fourbar.g],
            [self.I[2]*self.ddtheta[2]],            
            [self.m[3]*self.ddx[3]],
            [self.m[3]*self.ddy[3] + self.m[3]*Fourbar.g],
            [self.I[3]*self.ddtheta[3]]
        ])
        X = lstsq(Afs, Bfs)
        self.dynamicsforce = np.array([
            X[0].item(0), X[0].item(1), X[0].item(2), X[0].item(3), X[0].item(4), X[0].item(5), X[0].item(6), X[0].item(7), X[0].item(8)
        ])
        return self.dynamicsforce

    def get_shakingForce(self):
        self.Fs = np.array([
            self.dynamicsforce[0]+self.dynamicsforce[6], self.dynamicsforce[1]+self.dynamicsforce[7]
        ])
        self.Fs_abs = norm(self.Fs)

        r_DA = np.array([
            self.L[0]*cos(self.theta[0]), self.L[0]*sin(self.theta[0]), 0
        ])
        f_14 = np.array([
            self.dynamicsforce[6], self.dynamicsforce[7], 0
        ])
        self.Ms = np.cross(r_DA, f_14)
        self.Ms_abs = norm(self.Ms)

        shakingOutput = {
            'shakingForce': self.Fs,
            'shakingForce_abs': self.Fs_abs,
            'shakingMoment': self.Ms,
            'shakingMoment_abs': self.Ms_abs
        }
        return shakingOutput

    def plot_animation(self, x, y, save_as_gif):
        fig, ax = plt.subplots()
        line, = ax.plot(x[0], y[0], '-o')
        ax.axis('equal')
        # ax.axis([-20,40,-20,30])

        def init():
            return line,

        def animate(i):
            line.set_xdata(x[i])
            line.set_ydata(y[i])
            return line

        ani = animation.FuncAnimation(fig=fig,
                                    func=animate,
                                    frames=len(x),
                                    init_func=init,
                                    interval=30,
                                    blit=False)
        if save_as_gif:
            ani.save('animation.gif', writer='imagemagick', fps=60)
        plt.show()

N = 360
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
    theta[1] = theta2
    theta, ic, plot_position,  = fourbar.get_position(theta, ic)
    dtheta = fourbar.get_angularVelocity(omega2)
    ddtheta = fourbar.get_angularAcceleration(alpha2)
    staticsforce = fourbar.get_staticsForce()
    ddx, ddy = fourbar.get_acceleration()
    dynamicsforce = fourbar.get_dynamicsForce()
    shakingRes = fourbar.get_shakingForce()
    
    x.append(plot_position[0])
    y.append(plot_position[1])
    Theta.append(theta)
    Omega.append(dtheta)
    Alpha.append(ddtheta)
    Staticsforce.append(staticsforce)
    Dynamicsforce.append(dynamicsforce)
    Shakingforce.append(shakingRes['shakingForce'])
    Shakingforce_abs.append(shakingRes['shakingForce_abs'])
    Shakingmoment_abs.append(shakingRes['shakingMoment_abs'])

print(len(Theta), Theta[0:3])
# print(Omega)
# print(Alpha)
# print(Staticsforce)
# print(Dynamicsforce)
# print(Shakingforce)
# print(Shakingforce_abs)
# print(Shakingmoment_abs)


fourbar.plot_animation(x, y, save_as_gif=False)
    

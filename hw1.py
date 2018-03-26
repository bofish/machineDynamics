import numpy as np
from math import sin, cos, pi
from scipy.optimize import fsolve
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from matplotlib import animation

def fourbarLoop(x, L, theta):

    return [
        L[0]*cos(theta[0]) + L[1]*cos(theta[1]) + L[2]*cos(x[0]) - L[3]*cos(x[1]),
        L[0]*sin(theta[0]) + L[1]*sin(theta[1]) + L[2]*sin(x[0]) - L[3]*sin(x[1])
    ]

class Fourbar():

    g = 9.81

    def __init__(self, L, m, I, G):
        self.L = L
        self.m = m
        self.I = I
        self.G = G

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

    def get_velocity(self, omega2):
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

    def get_acceleration(self, alpha2):
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
        print(X)
        return self.ddtheta

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
        print(X)
        self.staticsforce = np.array([
            0, X[0][0], X[0][1], X[0][2], X[0][3], X[0][4], X[0][5], X[0][6], X[0][7], X[0][8]
        ])
        return self.staticsforce

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
# G = np.array([[], [], [], []])
G = 0
theta = np.array([0.0, 0.0, 1, 0])
omega2 = 120 / 60 * 2 * pi
alpha2 = 0
ic = theta[2:4]
x = []
y = []

fourbar = Fourbar(L, m, I, G)

for theta2 in np.linspace(0, 2*pi, N):
    theta[1] = theta2
    theta, ic, plot_position,  = fourbar.get_position(theta, ic)
    dtheta = fourbar.get_velocity(omega2)
    ddtheta = fourbar.get_acceleration(alpha2)
    staticsforce = fourbar.get_staticsForce()
    x.append(plot_position[0])
    y.append(plot_position[1])

print(staticsforce)

fourbar.plot_animation(x, y, save_as_gif=False)
    

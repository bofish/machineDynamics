import numpy as np
from math import sin, cos, pi
from scipy.optimize import fsolve
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

    # def get_velocity(self, omega2):
    #     Av = np.matrix([
    #         [],
    #         []
    #     ])


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
L = np.array([0.3, 0.1, 0.18, 0.25])
m = np.array([0, 1.0, 2.0, 0.2])
I = np.array([0, 0.02, 0.06, 0.005])
# G = np.array([[], [], [], []])
G = 0
theta = np.array([0.0, 0.0, 1, 0])
ic = theta[2:4]
x = []
y = []

fourbar = Fourbar(L, m, I, G)

for theta2 in np.linspace(0, 2*pi, N):
    theta[1] = theta2
    theta, ic, plot_position,  = fourbar.get_position(theta, ic)
    x.append(plot_position[0])
    y.append(plot_position[1])

fourbar.plot_animation(x, y, save_as_gif=False)
    

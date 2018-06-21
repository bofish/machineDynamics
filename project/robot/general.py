import matplotlib as mpl
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import matplotlib.pyplot as plt
mpl.rcParams['legend.fontsize'] = 10
from matplotlib import animation

def make_ani(x, y, z, locus_x, locus_y, locus_z, N, save_gif = False):
    #----- Manipulator Animation -----#
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    line, = ax.plot(x[0], y[0], z[0],'b-o', label='parametric curve')
    tracer, = ax.plot([0,0],[0,0],[0,0], label='parametric curve')

    ax.set_xlim3d([-6, 6])
    ax.set_xlabel('X')
    ax.set_ylim3d([-6, 6])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0, 6])
    ax.set_zlabel('Z')
    ax.set_title('RRR Manipulator')

    def update_lines(i):
        line.set_data(x[i], y[i])
        line.set_3d_properties(z[i])
        tracer.set_data(locus_x[0:i], locus_y[0:i])
        tracer.set_3d_properties(locus_z[0:i])
        return line,

    ani = animation.FuncAnimation(fig, update_lines, frames=N, interval=N, blit=False)
    if save_gif:
        ani.save('robot.gif', writer='imagemagick', fps=10)

    return ani
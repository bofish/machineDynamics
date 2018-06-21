from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt

def acceleration_plan(target_point, t_acc, V_tar, N, is_plot = False):
    p_start = target_point[0]
    p_end = target_point[1]
    
    # Pre-calculate parameter
    A = 0.5*V_tar*pi/t_acc
    s_total = np.sqrt((p_end[0] - p_start[0])**2 + (p_end[1] - p_start[1])**2)
    direct_vector = np.array([(p_end[0] - p_start[0])/s_total, (p_end[1] - p_start[1])/s_total])
    t_c =  (s_total - 2*A*t_acc**2/pi)/V_tar
    endtime = 2*t_acc + t_c
    times = np.linspace(0.0, endtime, N)

    # Initial empty list
    P = np.zeros((N, 3))
    dP = np.zeros((N, 3))
    ddP = np.zeros((N, 3))
    s = np.zeros(N)
    v = np.zeros(N)
    a = np.zeros(N)

    for ind, t in enumerate(times):
        if 0<=t<t_acc:
            a_i = A*sin(pi/t_acc*t)
            v_i = -A*(t_acc/pi)*cos(pi/t_acc*t) + A*t_acc/pi
            s_i = -A*(t_acc/pi)**2*sin(pi/t_acc*t) + A*t_acc/pi*t
        elif t_acc<=t<(t_acc+t_c):
            a_i = 0.0
            v_i = V_tar
            s_i = V_tar*(t-t_acc) + A*t_acc**2/pi
        elif (t_acc+t_c)<=t<=(2*t_acc+t_c):
            a_i = - A*sin(pi/t_acc*(t-t_acc-t_c))
            v_i = A*(t_acc/pi)*cos(pi/t_acc*(t-t_acc-t_c)) + V_tar - A*t_acc/pi
            s_i = A*(t_acc/pi)**2*sin(pi/t_acc*(t-t_acc-t_c)) + (V_tar-A*t_acc/pi)*(t-t_acc-t_c) + (V_tar*t_c + A*t_acc**2/pi)
        P[ind, :] = [p_start[0] + s_i*direct_vector[0], p_start[1] + s_i*direct_vector[1], 0.0]
        dP[ind, :] = [p_start[0] + v_i*direct_vector[0], p_start[1] + v_i*direct_vector[1], 0.0]
        ddP[ind, :] = [p_start[0] + a_i*direct_vector[0], p_start[1] + a_i*direct_vector[1], 0.0]
        s[ind] = s_i
        v[ind] = v_i
        a[ind] = a_i

    if is_plot:
        plt.figure()
        plt.plot(times, s)
        plt.xlabel('time')
        plt.ylabel('displacement')
        plt.title('time v.s. displacement')

        plt.figure()
        plt.plot(times, v)
        plt.xlabel('time')
        plt.ylabel('velocity')
        plt.title('time v.s. velocity')

        plt.figure()
        plt.plot(times, a)
        plt.xlabel('time')
        plt.ylabel('acceleration')
        plt.title('time v.s. acceleration')
        plt.show()
    return P, dP, ddP, times

if __name__ == '__main__':
    target_point = [[4, -5], [2, 5]]
    t_acc = 0.83479259
    V_tar = 2.99252796
    N = 50
    P, dP, ddP, times = acceleration_plan(target_point, t_acc, V_tar, N, is_plot = True)
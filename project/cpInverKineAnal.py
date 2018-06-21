from math import cos, sin, pi, sqrt
import numpy as np
from numpy.linalg import inv, solve, lstsq
from scipy.optimize import fsolve

import robot.general as general
from robot.acceleration_plan import acceleration_plan

import matplotlib as mpl
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import matplotlib.pyplot as plt
mpl.rcParams['legend.fontsize'] = 10
from matplotlib import animation

#----- Acceleration Plan -----#
P_tar = [[4, -5], [2, 5]]
T_acc = 0.83479259
V_tar = 2.99252796
N = 50
P, dP, ddP, time = acceleration_plan(P_tar, T_acc, V_tar, N)

P_x = P[:, 0]
P_y = P[:, 1]
P_z = P[:, 2]
dP_x = dP[:, 0]
dP_y = dP[:, 1]
dP_z = dP[:, 2]
ddP_x = ddP[:, 0]
ddP_y = ddP[:, 1]
ddP_z = ddP[:, 2]

#----- Basic Parameter for manipulator model -----#
# DH-parameter
d1 = 0.5
# a2 = 3
# a3 = 4
a2 = 3.59435767
a3 = 3.54154706
# Mass and inertia of each arm
m1 = 1
# m2 = 1
# m3 = 1
m2 =  7.8282
m3 =  3.2448
I1 = m1*d1**2/3
I3 = m2*a2**2/3
I2 = m3*a3**2/3
g = 9.81
# Manipulator likage coordinate
x = list([])
y = list([])
z = list([])
# Velocity of center of mass of arm 2
v2mx = []
v2my = []
v2mz = []
# Acceleration of end of arm 2
a2x = []
a2y = []
a2z = []
# Acceleration of end effecter
a3x = ddP_x
a3y = ddP_y
a3z = ddP_z
# Driving motor theta
t1 = []
t2 = []
t3 = []
# Driving motor angular velocity
dt1 = []
dt2 = []
dt3 = []
# Driving motor angular velocity
ddt1 = []
ddt2 = []
ddt3 = []
# Driving motor torque
f1x = []
f1y = []
f2x = []
f2y = []
f3x = []
f3y = []
tau1 = []
tau2 = []
tau3 = []
# Rate of change of the work done by the manupulator
P = []
energy = 0
initialPt = [0, 0, 0]

def position(x, P_x, P_y, P_z, d1, a2, a3):
    t1 = float(x[0])
    t2 = float(x[1])
    t3 = float(x[2])

    return [
    a2*cos(t1)*cos(t2) + a3*cos(t1)*cos(t2+t3) - P_x,
    a2*sin(t1)*cos(t2) + a3*sin(t1)*cos(t2+t3) - P_y,
    a2*sin(t2) + a3*sin(t2+t3) + d1 - P_z
    ]

for i in range(0,N):
    #----- Solve the inverse kinematic for theta1~3 -----#
    result = fsolve(position,initialPt,args=(P_x[i], P_y[i], P_z[i], d1, a2, a3))
    t1.append(result[0])
    t2.append(result[1])
    t3.append(result[2])
    x.append([0, 0, a2*cos(t1[-1])*cos(t2[-1]), P_x[i]])
    y.append([0, 0, a2*sin(t1[-1])*cos(t2[-1]), P_y[i]])
    z.append([0, d1, a2*sin(t2[-1])+d1, P_z[i]])

    initialPt = [result[0], result[1], result[2]]

    # #----- Check the thetas -----#
    # theta1 = np.rad2deg(result[0] % (2*pi))
    # theta2 = np.rad2deg(result[1] % (2*pi))
    # theta3 = np.rad2deg(result[2] % (2*pi))
    # print(theta1,theta2,theta3)

    # #----- Check the linkage point -----#
    # P0 = [0,0,0]
    # P1 = [0,0,d1]
    # P2 = [a2*cos(result[0])*cos(result[1]), a2*sin(result[0])*cos(result[1]), a2*sin(result[1])+d1]
    # P3 = [P_x[i],P_y[i],P_z[i]]
    # print(P0,P1,P2,P3)

    # #----- Forward Kinematic check -----#
    # th1 = 0
    # th2 = 0.988205423
    # th3 = 4.563686928
    # Px = a2*cos(th1)*cos(th2) + a3*cos(th1)*cos(th2+th3)
    # Py = a2*sin(th1)*cos(th2) + a3*sin(th1)*cos(th2+th3)
    # Pz = a2*sin(th2) + a3*sin(th2+th3) + d1
    # print(Px,Py,Pz)

    #----- Solve angular velocity -----#
    Av = np.matrix([[-a2*sin(t1[-1])*cos(t2[-1])-a3*sin(t1[-1])*cos(t2[-1]+t3[-1]), -a2*cos(t1[-1])*sin(t2[-1])-a3*cos(t1[-1])*sin(t2[-1]+t3[-1]), -a3*cos(t1[-1])*sin(t2[-1]+t3[-1])], [a2*cos(t1[-1])*cos(t2[-1])+a3*cos(t1[-1])*cos(t2[-1]+t3[-1]), -a2*sin(t1[-1])*sin(t2[-1])-a3*sin(t1[-1])*sin(t2[-1]+t3[-1]), -a3*sin(t1[-1])*sin(t2[-1]+t3[-1])], [0, a2*cos(t2[-1])+a3*cos(t2[-1]+t3[-1]), a3*cos(t2[-1]+t3[-1])]])
    Bv = np.matrix([[dP_x[i]],[dP_y[i]],[dP_z[i]]])
    Xv = inv(Av)*Bv
    dt1.append(Xv.item(0))
    dt2.append(Xv.item(1))
    dt3.append(Xv.item(2))

    #----- Solve angular acceleration -----#
    A11 = -a2*sin(t1[-1])*cos(t2[-1])-a3*sin(t1[-1])*cos(t2[-1]+t3[-1])
    A12 = -a2*cos(t1[-1])*sin(t2[-1])-a3*cos(t1[-1])*sin(t2[-1]+t3[-1])
    A13 = -a3*cos(t1[-1])*sin(t2[-1]+t3[-1])
    A21 =  a2*cos(t1[-1])*cos(t2[-1])+a3*cos(t1[-1])*cos(t2[-1]+t3[-1])
    A22 = -a2*sin(t1[-1])*sin(t2[-1])-a3*sin(t1[-1])*sin(t2[-1]+t3[-1])
    A23 = -a3*sin(t1[-1])*sin(t2[-1]+t3[-1])
    A31 =  0
    A32 =  a2*cos(t2[-1])
    A33 =  a3*cos(t2[-1]+t3[-1])
    Aa = np.matrix([[A11, A12, A13],[A21, A22, A23],[A31, A32, A33]])
    
    B1 = ddP_x[i] - a2*(-dt1[-1]**2*cos(t1[-1])*cos(t2[-1]) + 2*dt1[-1]*dt2[-1]*sin(t1[-1])*sin(t2[-1]) - dt2[-1]**2*cos(t1[-1])*cos(t2[-1])) - a3*(-dt1[-1]**2*cos(t1[-1])*cos(t2[-1]+t3[-1]) + 2*dt1[-1]*(dt2[-1]+dt3[-1])*sin(t1[-1])*sin(t2[-1]+t3[-1]) - (dt2[-1]+dt3[-1])**2*cos(t1[-1])*cos(t2[-1]+t3[-1]))
    B2 = ddP_y[i] - a2*(-dt1[-1]**2*sin(t1[-1])*cos(t2[-1]) - 2*dt1[-1]*dt2[-1]*cos(t1[-1])*sin(t2[-1]) - dt2[-1]**2*sin(t1[-1])*cos(t2[-1])) - a3*(-dt1[-1]**2*sin(t1[-1])*cos(t2[-1]+t3[-1]) - 2*dt1[-1]*(dt2[-1]+dt3[-1])*cos(t1[-1])*sin(t2[-1]+t3[-1]) - (dt2[-1]+dt3[-1])**2*sin(t1[-1])*cos(t2[-1]+t3[-1]))
    B3 = ddP_z[i] + a2*dt2[-1]**2*sin(t2[-1]) + a3*(dt2[-1]+dt3[-1])**2*sin(t2[-1]+t3[-1])
    Ba = np.matrix([[B1],[B2],[B3]])
    Xa = inv(Aa)*Ba
    ddt1.append(Xa.item(0))
    ddt2.append(Xa.item(1))
    ddt3.append(Xa.item(2))

    #----- Drive a2x, a2y, a2z of endpoint of arm 2 -----#
    a2x.append(a2*(-ddt1[-1]*sin(t1[-1])*cos(t2[-1]) - ddt2[-1]*cos(t1[-1])*sin(t2[-1])) + a2*(-dt1[-1]**2*cos(t1[-1])*cos(t2[-1]) + 2*dt1[-1]*dt2[-1]*sin(t1[-1])*sin(t2[-1]) - dt2[-1]**2*cos(t1[-1])*cos(t2[-1])))

    a2y.append(a2*(ddt1[-1]*cos(t1[-1])*cos(t2[-1]) - ddt2[-1]*sin(t1[-1])*sin(t2[-1])) + a2*(-dt1[-1]**2*sin(t1[-1])*cos(t2[-1]) - 2*dt1[-1]*dt2[-1]*cos(t1[-1])*sin(t2[-1]) - dt2[-1]**2*sin(t1[-1])*cos(t2[-1])))

    a2z.append(a2*(ddt2[-1]*cos(t2[-1]) - dt2[-1]**2*sin(t2[-1])))

    #----- Drive v2mx, v2my, v2mz of center of mass of arm 2 -----#
    v2mx.append(0.5*a2*(-dt1[-1]*sin(t1[-1])*cos(t2[-1]) - dt2[-1]*cos(t1[-1])*sin(t2[-1])))
    v2my.append(0.5*a2*(dt1[-1]*cos(t1[-1])*cos(t2[-1]) - dt2[-1]*sin(t1[-1])*sin(t2[-1])))
    v2mz.append(0.5*a2*dt2[-1]*cos(t2[-1]))

    #----- Solve kinetostatic (inverse dynamic) -----#
    A1 = [1, 0, -1,  0,  0,  0, 0, 0, 0]
    A2 = [0, 1,  0, -1,  0,  0, 0, 0, 0]
    A3 = [0, 0,  0,  0,  0,  0, 1, 0, 0]
    A4 = [0, 0,  1,  0, -1,  0, 0, 0, 0]
    A5 = [0, 0,  1,  0, -1,  0, 0, 0, 0]
    A6 = [0, 0, a2/2*sin(t2[-1]), -a2/2*cos(t2[-1]), a2/2*sin(t2[-1]), -a2/2*cos(t2[-1]), 0, 1, 0]
    A7 = [0, 0,  0,  0,  1,  0, 0, 0, 0]
    A8 = [0, 0,  0,  0,  0,  1, 0, 0, 0]
    A9 = [0, 0, 0, 0, a3/2*sin(t2[-1]+t3[-1]), -a3/2*cos(t2[-1]+t3[-1]), 0, 0, 1]
    Ad = np.matrix([A1, A2, A3, A4, A5, A6, A7, A8, A9])

    b1 = [0]
    b2 = [m1*g]
    b3 = [I1*ddt1[-1]]
    b4 = [m2*a2x[-1]]
    b5 = [m2*(a2y[-1]+g)]
    b6 = [I2*ddt2[-1]]
    b7 = [m3*a3x[i]]
    b8 = [m3*(a3y[i]+g)]
    b9 = [I3*(ddt2[-1]+ddt3[i])]
    Bd = np.matrix([b1, b2, b3, b4, b5, b6, b7, b8, b9])
    Xd = lstsq(Ad,Bd)
    f1x.append(Xd[0].item(0))
    f1y.append(Xd[0].item(1))
    f2x.append(Xd[0].item(2))
    f2y.append(Xd[0].item(3))
    f3x.append(Xd[0].item(4))
    f3y.append(Xd[0].item(5))
    tau1.append(Xd[0].item(6))
    tau2.append(Xd[0].item(7))
    tau3.append(Xd[0].item(8))

    #----- Drive the required power -----#
    P.append(abs(tau1[-1]*dt1[-1]) + abs(tau2[-1]*dt2[-1]) + abs(tau3[-1]*dt3[-1]) + abs(m2*g*v2my[-1]) + abs(m3*g*v2my[-1]))

for p in P:
    dt = 3/N
    energy = energy + p*dt

#----- Pick out the end effecter point for locus ploting -----#
X = [item for sublist in x for item in sublist]
X = X[3::4]
Y = [item for sublist in y for item in sublist]
Y = Y[3::4]
Z = [item for sublist in z for item in sublist]
Z = Z[3::4]

#----- Acceleration, velocity and displacement of end effecter -----#
# plt.figure()
# # plt.scatter(time,a)
# # # plt.figure()
# # plt.scatter(time,v)
# # # plt.figure()
# # plt.scatter(time,s)
# # plt.legend(['a','v','s'])
# dotsize = 10
# plt.title('Acceleration plan')
# plt.subplot(2,2,1)
# plt.scatter(time,a,s=dotsize)
# plt.title('Time-Acceleration')

# plt.subplot(2,2,2)
# plt.scatter(time,v,s=dotsize)
# plt.title('Time-Velocity')

# plt.subplot(2,2,3)
# plt.scatter(time,s,s=dotsize)
# plt.title('Time-Displacement')

#----- Locus of end effecter -----#
plt.subplot(2,2,4)
plt.scatter(P_x,P_y)
plt.scatter(0,0,color='red')
plt.xlim([-6, 6])
plt.xlabel('X')
plt.ylim([-6, 6])
plt.ylabel('Y')
plt.title('Locus of end effecter')

#----- Time v.s. Theta plot -----#
plt.figure()
plt.plot(time,t1,time,t2,time,t3)
plt.legend(['theta1','theta2','theta3'])
plt.title('Motor rotate angle')

#----- Joint and motor dynamic plot -----#
plt.figure()
plt.plot(time,tau1,time,tau2,time,tau3)
plt.legend(['tau1','tau2','tau3'])
plt.title('Motor driving torque')
# plt.figure()
# plt.plot(time,f1x,time,f2x,time,f3x)
# plt.legend(['f1x','f2x','f3x'])
# plt.title('Joint react force in x direct')
# plt.figure()
# plt.plot(time,f1y,time,f2y,time,f3y)
# plt.legend(['f1y','f2y','f3y'])
# plt.title('Joint react force in y direct')

#----- Time v.s. total power -----#

# plt.figure()
# plt.plot(time,P)
# plt.title('The rate of change of the work done by the manipulator')
# print('Energy:'+str(energy))
# print('Endtime:'+str(endtime))
# #----- Manipulator Animation -----#

ani = general.make_ani(x, y, z, X, Y, Z, N)

plt.show()


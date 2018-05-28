from math import pi, cos, sin
import numpy as np
import matplotlib.pyplot as plt
'''
ISSUE: the y-label of the shaking force plot is displaced
'''
N = 7200
theta2 = np.linspace(0.0, 720.0, N)
gasForce = []
for t2 in theta2:
    if t2 <= 10:
        gasForce.append(13000.0 + t2*2000.0/10)
    elif t2 <= 160:
        gasForce.append(15000.0 - (t2 - 10)*15000.0/150)
    elif t2 <= 650:
        gasForce.append(0.0)
    elif t2 <= 710:
        gasForce.append((t2 - 650)*2000.0/60)
    else:
        gasForce.append(2000.0 + (t2 - 710)*11000.0/10)

# theta2 = np.radians(theta2)
omega2 = 3600/60*2*pi
t_end = 4*pi/omega2
times = np.linspace(0.0, t_end, N)
Theta2 = times*omega2

# Geometry parameter
r = 0.038
l = 0.133
m2 = 5.0
m3 = 0.5
m4 = 0.3
rG2 = 0.3*r
la = 0.36*l
lb = 0.64*l
m2a = m2*rG2/r
m3a = m3*lb/(lb+la)
m3b = m3*la/(lb+la)
mA = m2a + m3a
mB = m3b + m4
I2 = 0.05
I3 = 0.002

# Gas force only 
Tg21 = []
Fgsx = []
Fgsy = []
Mgs = []

# inertia force only 
Ti21 = []
Fisx = []
Fisy = []
Mis = []

for index_theta, theta2 in enumerate(Theta2):
    Fg = gasForce[index_theta]

    tanPhi = r*sin(theta2)/(l*np.sqrt(1-(r/l*sin(theta2))**2))
    Tg21.append(-Fg*r*sin(theta2)*(1+r/l*cos(theta2)))
    Fgsx.append(-2*Fg)
    Fgsy.append(2*Fg*tanPhi)
    Mgs.append(0.0)

    Fisx.append(- mA*(-r*omega2**2*cos(theta2)) - mB*(-r*omega2**2*(cos(theta2)+r/l*cos(2*theta2))))
    Fisy.append(- mA*(-r*omega2**2*sin(theta2)))
    Ti21.append(-0.5*mB*r**2*omega2**2*(r/(2*l)*sin(theta2) - sin(2*theta2) - 3*r/(2*l)*sin(3*theta2)))
    Mis.append(-0.5*mB*r**2*omega2**2*(r/(2*l)*sin(theta2) - sin(2*theta2) - 3*r/(2*l)*sin(3*theta2)))

Theta2 = np.degrees(Theta2)

Fgsx = np.array(Fgsx)
Fgsy = np.array(Fgsy)
Fgs = np.sqrt(Fgsx**2 + Fgsy**2)

Fisx = np.array(Fisx)
Fisy = np.array(Fisy)
Fis = np.sqrt(Fisx**2 + Fisy**2)

ls = ['-', '--', ':']
color = ['k', 'r', 'b']
# ---- Gas force only ---- #
plt.figure()
plt.plot(Theta2, Tg21, c=color[0], ls=ls[0], label='Tg21')
plt.xlabel('theta2 (degree)')
plt.ylabel('Output Torque (N*m)')
plt.title('Gas force only')
plt.legend()

plt.figure()
plt.plot(Theta2, Fgs, c=color[0], ls=ls[0], label='Fgs_abs')
plt.plot(Theta2, Fgsx, c=color[1], ls=ls[1], label='Fgs_x')
plt.plot(Theta2, Fgsy, c=color[2], ls=ls[2], label='Fgs_y')
plt.xlabel('theta2 (degree)')
plt.ylabel('Shaking Force (N)')
plt.title('Gas force only')
plt.legend()

plt.figure()
plt.plot(Theta2, Mgs, c=color[0], ls=ls[0], label='Mgs')
plt.xlabel('theta2 (degree)')
plt.ylabel('Shaking Moment (N)')
plt.title('Gas force only')
plt.legend()

# ---- Inertia force only ---- #
plt.figure()
plt.plot(Theta2, Ti21, c=color[0], ls=ls[0], label='Ti21')
plt.xlabel('theta2 (degree)')
plt.ylabel('Output Torque (N*m)')
plt.title('Inertia force only')
plt.legend()

plt.figure()
plt.plot(Theta2, Fis, c=color[0], ls=ls[0], label='Fis_abs')
plt.plot(Theta2, Fisx, c=color[1], ls=ls[1], label='Fis_x')
plt.plot(Theta2, Fisy, c=color[2], ls=ls[2], label='Fis_y')
plt.xlabel('theta2 (degree)')
plt.ylabel('Shaking Force (N)')
plt.title('Inertia force only')
plt.legend()

plt.figure()
plt.plot(Theta2, Mis, c=color[0], ls=ls[0], label='Mis')
plt.xlabel('theta2 (degree)')
plt.ylabel('Shaking Moment (N*m)')
plt.title('Inertia force only')
plt.legend()

plt.show()
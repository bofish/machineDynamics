from math import sin, cos, pi, sqrt, degrees, radians
import numpy as np
import matplotlib.pyplot as plt
from hw1 import Fourbar
        
'''
L[0:5]: L1, L2, L3, L4, L_BE, L_CE
'''

class BerkofFourbar(Fourbar):
    
    def stp1_design_coupler(self, linktype='simple'):
        if linktype == 'simple':
            # Design coupler, make A3=0
            # Assumption: Link3 is symmetry
            l3 = self.L[2]
            m3 = self.m[2]
            lce = self.L[5]
            self.L[4] = l3/2
            self.L[5] = -l3/2
            self.m[2] = m3*2
            self.I[2] = 2*m3*(l3/2+lce)**2
        else:
            print('Not found link type')

    def stp2_force_balance(self, linktype='simple'):
        if linktype == 'simple':
            # Force balancing, make Rt=constant
            # Assumption: I2, I4, m2, m4 is not change
            m2 = self.m[1]
            m3 = self.m[2]
            m4 = self.m[3]
            l2 = self.L[1]
            l3 = self.L[2]
            l4 = self.L[3]
            b3 = l3/2
            b2 = m3*l2*(b3/l3 - 1)/m2
            b4 = -m3*l4*(b3/l3)/m4
            self.b = [0, b2, b3, b4]
        else:
            print('Not found link type')

    def stp3_add_counterweight(self, linktype='simple'):
        if linktype == 'simple':
            A = []
            for i in [1, 3]:                    
                Ksq = self.I[i]/self.m[i]
                A_i = -self.m[i]*(Ksq + self.b[i]**2 + self.L[i]*self.b[i])
                A.append(A_i)
 
            self.A = [0, A[0], 0, A[1]]
        else:
            print('Not found link type')

    def update_by_berkof(self):
        self.stp1_design_coupler()
        self.stp2_force_balance()
        self.stp3_add_counterweight()

    def get_velocity_berkof(self):
        r_BA = np.array([
            self.L[1]*cos(self.theta[1]), self.L[1]*sin(self.theta[1]), 0])
        r_G2 = np.array([
            self.b[1]*cos(self.theta[1]), self.b[1]*sin(self.theta[1]), 0])
        r_G3 = np.array([
            self.b[2]*cos(self.theta[2]), self.b[2]*sin(self.theta[2]), 0])
        r_G4 = np.array([
            self.b[3]*cos(self.theta[3]), self.b[3]*sin(self.theta[3]), 0])
        
        v_B = 0.0 + np.cross([0.0, 0.0, self.dtheta[1]], r_BA)
        v_G2 = 0.0 + np.cross([0.0, 0.0, self.dtheta[1]], r_G2)
        v_G3 = v_B + np.cross([0.0, 0.0, self.dtheta[2]], r_G3)
        v_G4 = 0.0 + np.cross([0.0, 0.0, self.dtheta[3]], r_G4)

        # # Verification
        # r_CD = np.array([
        #     self.L[3]*cos(self.theta[3]), self.L[3]*sin(self.theta[3]), 0])
        # r_CG3 = np.array([
        #     self.L[5]*cos(self.theta[2]), self.L[5]*sin(self.theta[2]), 0])
        # v_C = 0.0 + np.cross([0.0, 0.0, self.dtheta[3]], r_CD)
        # v_G32 = v_C + np.cross([0.0, 0.0, self.dtheta[2]], r_CG3)
        # print('v_E={}, v_E2={}, error={}'.format(v_G3, v_G32, v_G3-v_G32))

        self.dx = np.array([0, v_G2[0], v_G3[0], v_G4[0]])
        self.dy = np.array([0, v_G2[1], v_G3[1], v_G4[1]])
        self.dz = np.array([0, v_G2[2], v_G3[2], v_G4[2]])
        # print(self.dx, self.dy, self.dz)

        return self.dx, self.dy

    def get_acceleration_berkof(self):
        r_BA = np.array([
            self.L[1]*cos(self.theta[1]), self.L[1]*sin(self.theta[1]), 0])
        r_G2 = np.array([
            self.b[1]*cos(self.theta[1]), self.b[1]*sin(self.theta[1]), 0])
        r_G3 = np.array([
            self.b[2]*cos(self.theta[2]), self.b[2]*sin(self.theta[2]), 0])
        r_G4 = np.array([
            self.b[3]*cos(self.theta[3]), self.b[3]*sin(self.theta[3]), 0])

        a_B = 0.0 + np.cross([0.0, 0.0, self.ddtheta[1]], r_BA) - self.dtheta[1]**2*r_BA
        a_G2 = 0.0 + np.cross([0.0, 0.0, self.ddtheta[1]], r_G2) - self.dtheta[1]**2*r_G2
        a_G3 = a_B + np.cross([0, 0, self.ddtheta[2]], r_G3) - self.dtheta[2]**2*r_G3
        a_G4 = 0.0 + np.cross([0, 0, self.ddtheta[3]], r_G4) - self.dtheta[3]**2*r_G4
        
        # Verification
        # r_CD = np.array([
        #     self.L[3]*cos(self.theta[3]), self.L[3]*sin(self.theta[3]), 0])
        # r_CG3 = np.array([
        #     self.L[5]*cos(self.theta[2]), self.L[5]*sin(self.theta[2]), 0])
        # a_C = 0.0 + np.cross([0.0, 0.0, self.ddtheta[3]], r_CD) - self.dtheta[3]**2*r_CD
        # a_G32 = a_C + np.cross([0, 0, self.ddtheta[2]], r_CG3) - self.dtheta[2]**2*r_CG3
        # print('a_E={}, a_E2={}, error={}'.format(a_G3, a_G32, a_G3-a_G32))

        self.ddx = np.array([0, a_G2[0], a_G3[0], a_G4[0]])
        self.ddy = np.array([0, a_G2[1], a_G3[1], a_G4[1]])
        self.ddz = np.array([0, a_G2[2], a_G3[2], a_G4[2]])
        # print(self.ddx, self.ddy, self.ddz)

        return self.ddx, self.ddy

    def get_M_ctrwt(self, M4=0):
            self.M_ctrwt = -self.A[1]*self.ddtheta[1] + -self.A[3]*self.ddtheta[3] + M4
            return self.M_ctrwt

    def get_shakingForce_berkof(self, M4=0):
        self.Fs = np.array([
            -self.dynamicsforce[0] - self.dynamicsforce[6], -self.dynamicsforce[1] - self.dynamicsforce[7]
        ])
        # self.Fs_abs = norm(self.Fs)
        self.Fs_abs = sqrt(self.Fs[0]**2 + self.Fs[1]**2)

        self.Ms = self.I[1]*self.ddtheta[1] + self.I[3]*self.ddtheta[3] + self.A[1]*self.ddtheta[1] + self.A[3]*self.ddtheta[3]
        self.Ms_abs = self.Ms

        shakingOutput = {
            'shakingForce': self.Fs,
            'shakingForce_abs': self.Fs_abs,
            'shakingMoment': self.Ms,
            'shakingMoment_abs': self.Ms_abs
        }
        return shakingOutput



if __name__ == '__main__':

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
    T2 = []
    Shakingforce = list([])
    Shakingforce_abs = []
    Shakingmoment_abs = []
    M_ctrwt = []
    fourbar = BerkofFourbar(L, m, I)
    fourbar.update_by_berkof()



    for theta2 in np.linspace(0, 2*pi, N):
        # print(degrees(theta2) )
        if theta2 >= radians(150) and theta2 <= radians(240):
            M4 = 0
        else:
            M4 = 0

        theta[1] = theta2
        theta, ic, plot_position = fourbar.get_position(theta, ic)
        dtheta = fourbar.get_angularVelocity(omega2)
        ddtheta = fourbar.get_angularAcceleration(alpha2)
        staticsforce = fourbar.get_staticsForce(M4)
        dx, dy = fourbar.get_velocity_berkof()
        ddx, ddy = fourbar.get_acceleration_berkof()
        t2 = fourbar.get_T_by_energy_method(M4)
        dynamicsforce = fourbar.get_dynamicsForce(M4)
        shakingRes = fourbar.get_shakingForce_berkof(M4)
        m_ctrwt = fourbar.get_M_ctrwt(M4)

        x.append(plot_position[0])
        y.append(plot_position[1])
        Theta.append(theta.tolist())
        Omega.append(dtheta.tolist())
        Alpha.append(ddtheta.tolist())
        Staticsforce.append(staticsforce.tolist())
        Dynamicsforce.append(dynamicsforce.tolist())
        T2.append(t2)
        Shakingforce.append(shakingRes['shakingForce'].tolist())
        Shakingforce_abs.append(shakingRes['shakingForce_abs'])
        Shakingmoment_abs.append(shakingRes['shakingMoment_abs'])
        M_ctrwt.append(m_ctrwt)

    Theta = np.array(Theta)
    Omega = np.array(Omega)
    Alpha = np.array(Alpha)
    Dynamicsforce = np.array(Dynamicsforce)
    Shakingforce = np.array(Shakingforce)
    ls = ['-', '--', '-.', ':']

# Plotting
    # Angular velocity
    plt.figure()
    plt.plot(Theta[:,1], Omega[:,1], ls=ls[0])
    plt.plot(Theta[:,1], Omega[:,2], ls=ls[1])
    plt.plot(Theta[:,1], Omega[:,3], ls=ls[2])
    plt.legend([r'$\omega_2$', r'$\omega_3$', r'$\omega_4$'], loc=1)
    plt.xlabel(r'$\theta_2 (rad)$')
    plt.ylabel(r'$\omega_3, \omega_4 (rad/s)$')
    
    # Angular Acceleration
    plt.figure()
    plt.plot(Theta[:,1], Alpha[:,1], ls=ls[0])
    plt.plot(Theta[:,1], Alpha[:,2], ls=ls[1])
    plt.plot(Theta[:,1], Alpha[:,3], ls=ls[2])
    plt.legend([r'$\alpha_2$', r'$\alpha_3$', r'$\alpha_2$'], loc=1)
    plt.xlabel(r'$\theta_2 (rad)$')
    plt.ylabel(r'$\alpha_3, \alpha_4 (rad/s^2)$')

    # Statics force when theta 2 is equal to 0, 90, 180, 270
    forceTag = ['f12x', 'f12y', 'f23x', 'f23y', 'f34x', 'f34y', 'f14x', 'f14y', 'M2']
    for i in np.linspace(0, 3*N/4, 4):
        print(degrees(Theta[int(i),1]))
        print('\nStatics force for theta2 = 0, 90, 180, 270:')
        print(', '.join('{0}: {1:4.4f}'.format(forceTag[c], k) for c, k in enumerate(Staticsforce[int(i)])))

    # Dynamic reaction force
    plt.figure()
    for i in range(0,4):
        reactionForce = np.sqrt(Dynamicsforce[:,i*2]**2 + Dynamicsforce[:,i*2+1]**2)
        plt.plot(Theta[:,1], reactionForce, linestyle=ls[i])
    plt.legend([r'$Force_{12}$', r'$Force_{23}$', r'$Force_{34}$', r'$Force_{14}$'], loc=1)
    plt.xlabel(r'$\theta_2 (rad)$')
    plt.ylabel(r'$Dynamic Reaction Force (N)$')

    # Dynamic reaction force
    plt.figure()
    for i in range(0,2):
        reactionForce = np.sqrt(Dynamicsforce[:,i*2]**2 + Dynamicsforce[:,i*2+1]**2)
        plt.plot(Theta[:,1], reactionForce, linestyle=ls[i])
    plt.legend([r'$Force_{12}$', r'$Force_{32}$'], loc=1)
    plt.xlabel(r'$\theta_2 (rad)$')
    plt.ylabel(r'$Reaction\ Force (N)$')

    # Input torque
    plt.figure()
    plt.plot(Theta[:,1], Dynamicsforce[:,8], ls=ls[0])
    plt.plot(Theta[:,1],np.ones(N)*1.999, ls=ls[1])
    plt.xlabel(r'$\theta_2 (rad)$')
    plt.ylabel(r'$Input\ Torque (N \cdot m)$')
    plt.legend([r'$Without\ flywheel$', r'$With\ flywheel$'], loc=1)

    # Shaking force
    plt.figure()
    plt.plot(Theta[:,1], Shakingforce_abs)
    plt.plot(Theta[:,1], Shakingforce[:,0], linestyle=ls[1])
    plt.plot(Theta[:,1], Shakingforce[:,1], linestyle=ls[2])
    plt.legend([r'$|F_{s}|$', r'$F_{sx}$', r'$F_{sy}$'], loc=1)
    plt.xlabel(r'$\theta_2 (rad)$')
    plt.ylabel(r'$Shaking\ Force (N)$')

    # Shaking Moment
    plt.figure()
    plt.plot(Theta[:,1], Shakingmoment_abs, ls=ls[0], label=r'$M_s$')
    plt.plot(Theta[:,1], M_ctrwt, ls=ls[1], label=r'$M_ctrwt$')
    plt.xlabel(r'$\theta_2 (rad)$')
    plt.ylabel(r'$Shaking\ Moment (N \cdot m)$')
    plt.legend(loc=1)

    # Shaking force and moment in polar plot
    plt.figure()
    plt.polar(Theta[:,1], Shakingforce_abs)
    plt.legend(r'$|F_s|$')
    plt.figure()
    plt.polar(Theta[:,1], Shakingmoment_abs, linestyle=ls[1])
    plt.legend(r'$|M_s|$')

    plt.show() 
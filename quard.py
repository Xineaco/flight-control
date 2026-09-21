import numpy as np
import casadi as ca

class quard:

    def __init__(self,q):
        self.x = q[0]
        self.y = q[1]
        self.z = q[2]
        self.phi = q[3]
        self.theta = q[4]
        self.psi = q[5]
        self.d_x = q[6]
        self.d_y = q[7]
        self.d_z = q[8]
        self.d_phi = q[9]
        self.d_theta = q[10]
        self.d_psi = q[11]

    def model(self, q, u):
        m = 1.0  # 质量
        g = 9.8  # 重力
        I_x, I_y, I_z = 0.0075, 0.0075, 0.0135

        phi, theta, psi = q[3], q[4], q[5]
        d_phi, d_theta, d_psi = q[9], q[10], q[11]

        dd_x = (u[0] / m) * (ca.cos(phi) * ca.sin(theta) * ca.cos(psi)
                             + ca.sin(phi) * ca.sin(psi))
        dd_y = (u[0] / m) * (ca.cos(phi) * ca.sin(theta) * ca.sin(psi)
                             - ca.sin(phi) * ca.cos(psi))
        dd_z = (u[0] / m) * ca.cos(phi) * ca.cos(theta) - g

        dd_phi = (I_y - I_z) * d_theta * d_psi / I_x + u[1] / I_x
        dd_theta = (I_z - I_x) * d_phi * d_psi / I_y + u[2] / I_y
        dd_psi = (I_x - I_y) * d_phi * d_theta / I_z + u[3] / I_z

        d_q = ca.vertcat(q[6], q[7], q[8],  # vx, vy, vz
                         d_phi, d_theta, d_psi,  # 角速度
                         dd_x, dd_y, dd_z,  # 线加速度
                         dd_phi, dd_theta, dd_psi)  # 角加速度

        return d_q




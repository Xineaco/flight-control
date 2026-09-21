"""
QuadNMPC: 四旋翼非线性 MPC (CasADi Opti, 多步打靶)

状态 q = [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]   (12,)
控制 u = [T, tau_x, tau_y, tau_z]                          (4,)

    mpc = QuadNMPC(dynamics, dt=0.1, N=10)
    u0  = mpc.solve(q_now, ref)             # ref: (12,) 参考值, 整段预测都用它
"""
import numpy as np
import casadi as ca


class QuadNMPC:
    def __init__(self, dynamics, dt=0.1, N=5, nx=12, nu=4, m=1.0, g=9.8, pai=3.1416,
                 Q=None, R=None, u_lb=None, u_ub=None,
                 state_lb=None, state_ub=None, slack_weight=1e4):

        # ---------------- 1. 参数配置 ----------------
        self.dt, self.N, self.nx, self.nu = dt, N, nx, nu
        self.m, self.g = m, g
        self.u_trim = ca.DM([m * g, 0, 0, 0])            # 悬停推力 = mg
        Q = ca.diag([50, 50, 50, 1, 1, 1, 10, 10, 10, 1, 1, 1]) if Q is None else Q
        R = ca.diag([0.1, 0.05, 0.05, 0.05]) if R is None else R
        u_lb = np.array([0., -1., -1., -1.]) if u_lb is None else np.asarray(u_lb, float)
        u_ub = np.array([30., 1., 1., 1.]) if u_ub is None else np.asarray(u_ub, float)
        if state_lb is None:                             # 姿态限幅, 位置/速度不限
            state_lb = np.array([-np.inf, -np.inf, -np.inf,
                                 -0.2 * pai, -0.2 * pai, -1.0 * pai,
                                 -np.inf, -np.inf, -np.inf,
                                 -np.inf, -np.inf, -np.inf])
        else:
            state_lb = np.asarray(state_lb, float)
        if state_ub is None:
            state_ub = np.array([np.inf, np.inf, np.inf,
                                  0.2 * pai,  0.2 * pai,  1.0 * pai,
                                  np.inf, np.inf, np.inf,
                                  np.inf, np.inf, np.inf])
        else:
            state_ub = np.asarray(state_ub, float)

        # 连续动力学 -> RK4 单步
        q_s = ca.MX.sym('q', nx)
        u_s = ca.MX.sym('u', nu)
        self.f = ca.Function('f', [q_s, u_s], [dynamics(q_s, u_s)])

        def rk4(x, u):
            h = dt
            k1 = self.f(x, u)
            k2 = self.f(x + h / 2 * k1, u)
            k3 = self.f(x + h / 2 * k2, u)
            k4 = self.f(x + h * k3, u)
            return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # ---------------- 2. 搭建 Opti 问题 ----------------
        opti = ca.Opti()
        X = opti.variable(N + 1, nx)                     # 状态轨迹
        U = opti.variable(N, nu)                         # 控制序列
        S = opti.variable(N + 1, nx)                     # 状态软约束松弛量
        p_x0 = opti.parameter(nx)                        # 当前状态
        p_ref = opti.parameter(nx)                       # 期望状态 (参考值)

        # ---------------- 3. 代价函数 ----------------
        # J = Σ[(X-ref)'Q(X-ref) + (U-u_trim)'R(U-u_trim)] + 终端项 + 越界惩罚
        cost = 0
        for k in range(N):
            e = X[k, :].T - p_ref
            cost += e.T @ Q @ e
            du = U[k, :].T - self.u_trim
            cost += du.T @ R @ du
        eN = X[N, :].T - p_ref
        cost += eN.T @ Q @ eN
        cost += slack_weight * ca.sum1(ca.vec(S))
        opti.minimize(cost)

        # ---------------- 4. 约束 ----------------
        opti.subject_to(X[0, :].T == p_x0)               # 初值
        for k in range(N):                               # 动力学 (多步打靶)
            opti.subject_to(X[k + 1, :].T == rk4(X[k, :].T, U[k, :].T))
        opti.subject_to(opti.bounded(np.tile(u_lb, (N, 1)), U,      # 控制硬约束
                                     np.tile(u_ub, (N, 1))))
        opti.subject_to(opti.bounded(0, S, np.inf))      # 状态软约束: s>=0,
        for j in range(nx):                              # lb-s <= X <= ub+s
            if np.isfinite(state_lb[j]):
                opti.subject_to(X[:, j] - (state_lb[j] - S[:, j]) >= 0)
            if np.isfinite(state_ub[j]):
                opti.subject_to(X[:, j] - (state_ub[j] + S[:, j]) <= 0)

        opti.solver('ipopt', {'ipopt': {'print_level': 0}, 'print_time': False})

        self.opti, self.X, self.U = opti, X, U
        self.p_x0, self.p_ref = p_x0, p_ref
        self._U_prev = None
        self._status = None

    # ---------------- 5. 求解 Opti 并返回 u ----------------
    def solve(self, x0, *args):
        """solve(x0, ref) 或 solve(x0, u_prev, ref); ref 为参考值 (nx,); 返回 u0 (nu,)"""
        ref = np.asarray(args[-1], float)
        if ref.ndim > 1:                                 # 传了 (N+1,nx) 轨迹时取当前时刻
            ref = ref[0]
        self.opti.set_value(self.p_x0, np.asarray(x0, float).reshape(-1))
        self.opti.set_value(self.p_ref, ref.reshape(-1))
        if self._U_prev is not None:                     # 热启动: 上一步解平移
            self.opti.set_initial(self.U,
                                  np.vstack([self._U_prev[1:], self._U_prev[-1:]]))
        try:
            self.opti.solve()
            self._status = self.opti.return_status()
            self._U_prev = np.array(self.opti.value(self.U))
            return self._U_prev[0, :]                    # 第一个控制量
        except RuntimeError:                             # 求解失败 -> 悬停推力
            self._status = 'FAILED: ' + self.opti.return_status()
            return np.array(self.u_trim).ravel()

    def status(self):
        return self._status

"""
QuadNMPC: 四旋翼非线性 MPC (CasADi Opti, 多步打靶)

状态 q = [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]   (12,)
控制 u = [T, tau_x, tau_y, tau_z]                          (4,)

    mpc = QuadNMPC(dynamics, dt=0.1, N=10)

    # ① 单点参考(整段时域都用它)
    u0 = mpc.solve(q_now, q_ref)

    # ② 时变参考轨迹 —— A1 修改点
    #    ref_traj: (N+1, nx), 第 k 行是时刻 k 的参考
    u0 = mpc.solve(q_now, ref_traj)

    # ③ 兼容旧写法(u_prev 参数仍可传, 但热启动已内部自动处理)
    u0 = mpc.solve(q_now, u_prev, q_ref)

改动说明(A1):
    代价函数原来对 k=0..N 都用同一个常量 p_ref, 等于告诉优化器
    "未来整个时域内目标不动"。现在参考是 (N+1, nx) 的 parameter,
    逐点惩罚 X[k] - ref[k], 因此可以跟踪时变参考(例如领机的预测轨迹)。
    单点参考仍然支持, 内部会自动扩展成 (N+1, nx)。
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
        # ---- A1: 参考改为 (N+1, nx) 的时变轨迹 ----
        p_ref = opti.parameter(N + 1, nx)

        # ---------------- 3. 代价函数 ----------------
        # J = Σ_k [(X[k]-ref[k])'Q(X[k]-ref[k]) + (U[k]-u_trim)'R(U[k]-u_trim)]
        #     + 终端项 + 越界惩罚
        cost = 0
        for k in range(N):
            e = X[k, :].T - p_ref[k, :].T
            cost += e.T @ Q @ e
            du = U[k, :].T - self.u_trim
            cost += du.T @ R @ du
        eN = X[N, :].T - p_ref[N, :].T
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

        self.opti, self.X, self.U, self.S = opti, X, U, S
        self.p_x0, self.p_ref = p_x0, p_ref
        self._U_prev = None
        self._X_prev = None
        self._status = None
        self._n_calls = 0
        self._n_fail = 0

    # ---------------- 5. 参考整理 ----------------
    @staticmethod
    def _as_ref_array(ref, N, nx):
        """把参考统一整理成 (N+1, nx)。

        接受:
          (nx,)            单点参考     -> 扩展成 (N+1, nx)
          (N+1, nx)        时变参考轨迹  -> 直接使用
          (M, nx), M>N+1               -> 截断到 N+1
          (M, nx), M<N+1               -> 用最后一行补齐
        """
        ref = np.asarray(ref, dtype=float)
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        if ref.shape[1] != nx:
            raise ValueError('ref 的第二维应为 nx=%d, 实际 %d' % (nx, ref.shape[1]))
        if ref.shape[0] == N + 1:
            return ref
        if ref.shape[0] > N + 1:
            return ref[:N + 1, :].copy()
        out = np.empty((N + 1, nx), dtype=float)
        m = ref.shape[0]
        out[:m, :] = ref
        out[m:, :] = ref[-1, :]
        return out

    # ---------------- 6. 求解 Opti 并返回 u ----------------
    def solve(self, x0, *args):
        """solve(x0, ref) 或 solve(x0, u_prev, ref)

        ref 可以是 (nx,) 单点参考, 也可以是 (N+1, nx) 时变参考轨迹。
        返回 u0 (nu,)。
        """
        ref_arr = self._as_ref_array(args[-1], self.N, self.nx)

        self.opti.set_value(self.p_x0, np.asarray(x0, float).reshape(-1))
        self.opti.set_value(self.p_ref, ref_arr)
        # 热启动: 控制序列平移一步, 状态轨迹也一并平移(原来只平移了 U)
        if self._U_prev is not None:
            self.opti.set_initial(self.U,
                                  np.vstack([self._U_prev[1:], self._U_prev[-1:]]))
        if self._X_prev is not None:
            self.opti.set_initial(self.X,
                                  np.vstack([self._X_prev[1:], self._X_prev[-1:]]))
        self._n_calls += 1
        try:
            self.opti.solve()
            self._status = self.opti.return_status()
            self._U_prev = np.array(self.opti.value(self.U))
            self._X_prev = np.array(self.opti.value(self.X))
            return self._U_prev[0, :]                    # 第一个控制量
        except RuntimeError:                             # 求解失败 -> 悬停推力
            self._n_fail += 1
            self._status = 'FAILED: ' + self.opti.return_status()
            return np.array(self.u_trim).ravel()

    # ---------------- 7. 供编队使用的前馈接口 ----------------
    def predict_states(self):
        """返回上一步求解得到的预测状态轨迹, 形状 (N+1, nx)。

        这是 A1 的核心接口: 领机求解后把这条轨迹交给僚机, 僚机用它构造
        时变参考, 从而获得前馈预告 —— 而不是只跟踪领机的当前状态。
        """
        if self._X_prev is None:
            return None
        return np.array(self._X_prev, dtype=float).copy()

    def predict_controls(self):
        """返回上一步求解得到的预测控制序列, 形状 (N, nu)。"""
        if self._U_prev is None:
            return None
        return np.array(self._U_prev, dtype=float).copy()

    def status(self):
        return self._status

    def stats(self):
        """求解调用次数与失败次数, 便于统计失败率(原来从未被调用)。"""
        return {'calls': self._n_calls, 'failures': self._n_fail,
                'success_rate': (self._n_calls - self._n_fail) / max(self._n_calls, 1)}

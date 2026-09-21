import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import quard
import mpc as mpc

if __name__ == '__main__':
    dt, N, nx, nu = 0.1, 5, 12, 4
    q_s = ca.SX.sym('q', nx)
    u_s = ca.SX.sym('u', nu)

    # ---- 模型 ----
    def model(q, u):
        p = quard.quard(q_s)
        return p.model(q,u)

    # 真实对象: rk 积分器
    dq = model(q_s, u_s)
    plant = ca.integrator('plant', 'rk',
                          {'x': q_s, 'p': u_s, 'ode': dq},
                          0.0, dt)

    #---- 参考轨迹  ----
    def refence(t):
        q_ref = np.array([np.sin(t) , 0.0,  5.0,            # x, y, z
                         0.0, 0.0, 0.0,                         # phi, theta, psi
                         np.cos(t), 0.0, 0.0,               # v_x,v_y,v_z
                         0.0, 0.0, 0.0],                        # v_phi,v_theta,v_psi
                         dtype=float)
        return q_ref

    # 2) MPC 控制器
    mpc = mpc.QuadNMPC(dynamics=model, dt=dt, N=N, nx=nx, nu=nu)

    # 4) 闭环仿真
    qk_1 = np.array([0.0, 0.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    uk_1 = np.array([1.0 * 9.8, 0, 0, 0])
    qk_2 = np.array([0.0, 0.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    uk_2 = np.array([1.0 * 9.8, 0, 0, 0])


    # ---- 记录容器 ----
    t_hist_1, q_hist_1, u_hist_1, q_ref_hist_1 = [], [], [], []
    t_hist_2, q_hist_2, u_hist_2, q_ref_hist_2 = [], [], [], []
    t = 0.0

    for i in range(100):
        #扰动生成器：
        rng = np.random.default_rng(seed=i)
        cov = (np.array([0.0, 0.0, 0.0,  # x, y, z
                         0.0, 0.0, 0.0,  # phi, theta, psi
                         0.5, 0.5, 0.5,  # v_x,v_y,v_z
                         0.5, 0.5, 0.5])  # v_phi,v_theta,v_psi
               * np.eye(nx))
        turb = rng.multivariate_normal(np.zeros(nx), 0.2 * cov).reshape(-1)
        uk_1 = mpc.solve(qk_1, uk_1, refence(i * dt))
        uk_2 = mpc.solve(qk_2, uk_2, qk_1)

        q_hist_1.append(np.array(qk_1, dtype=float).copy())                     # 记录当前状态
        u_hist_1.append(np.array(uk_1, dtype=float).copy())                     # 记录当前控制
        t_hist_1.append(t)                                                      # 记录当前时间
        q_ref_hist_1.append(refence(t))

        q_hist_2.append(np.array(qk_1, dtype=float).copy())                     # 记录当前状态
        u_hist_2.append(np.array(uk_1, dtype=float).copy())                     # 记录当前控制
        t_hist_2.append(t)                                                      # 记录当前时间
        q_ref_hist_2.append(refence(t))

        qk_1 = np.array(plant(x0=qk_1, p = uk_1)['xf']).flatten() # 对象推进
        qk_2 = np.array(plant(x0=qk_2, p=uk_2)['xf']).flatten()

        t += dt

        if i % 1 == 0:
            print(f'步{i:3d}: x_1={qk_1[0]} y_1={qk_1[1]} z_1={qk_1[2]} '
                  f'phi_1={qk_1[3]} theta_1={qk_1[4]} psi_1={qk_1[5]}  '
                  f'T_1={uk_1[0]} '
                  f'步{i:3d}: x_2={qk_2[0]} y_2={qk_2[1]} z_2={qk_2[2]} '
                  f'phi_2={qk_2[3]} theta_2={qk_2[4]} psi_2={qk_2[5]}  '
                  f'T_2={uk_2[0]} '
                  )


    # 末态补上
    q_hist_1.append(np.array(qk_1, dtype=float).copy())
    t_hist_1.append(t)
    q_ref_hist_1.append(refence(t))
    q_hist_2.append(np.array(qk_1, dtype=float).copy())
    t_hist_2.append(t)
    q_ref_hist_2.append(refence(t))
    # ---- 转成 numpy ----
    t_arr_1 = np.array(t_hist_1)      # (101,)
    q_arr_1 = np.array(q_hist_1)      # (101, 12)
    u_arr_1 = np.array(u_hist_1)      # (100, 4)
    q_ref_arr_1 = np.array(q_ref_hist_1, dtype=float)   # (101, 12)
    t_arr_2 = np.array(t_hist_2)  # (101,)
    q_arr_2 = np.array(q_hist_2)  # (101, 12)
    u_arr_2 = np.array(u_hist_2)  # (100, 4)
    q_ref_arr_2 = np.array(q_ref_hist_2, dtype=float)  # (101, 12)


    state_names = ['x_1', 'y_1', 'z_1', 'phi_1', 'theta_1', 'psi_1',
                   'vx_1', 'vy_1', 'vz_1', 'p_1', 'q_1', 'r_1']
    groups_1 = [
        ('Position_1 [m]',         [0, 1, 2]),
        ('Attitude_1 [deg]',       [3, 4, 5]),
        ('Velocity_1 [m/s]',       [6, 7, 8]),
        ('Angular rate_1 [deg/s]', [9, 10, 11]),
    ]
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.flatten()

    for g, (title, idxs) in enumerate(groups_1):
        ax = axs[g]
        for j in idxs:
            ax.plot(t_arr_1, q_arr_1[:, j], label=state_names[j])
        if g == 1:
            ax.set_ylabel('deg_1')
        if g == 3:
            ax.set_ylabel('deg/s_1')
        ax.set_title(title)
        ax.set_xlabel('t [s]_1')
        ax.legend()
        ax.grid(True)

    # x 参考随时间变化 -> 画曲线; y/z 参考是常数 -> 画水平线。
    axs[0].plot(t_arr_1, q_ref_arr_1[:, 0], ls='--', color='gray', alpha=0.7,
                label='x_ref_1')
    for j in (1, 2):
        axs[0].axhline(float(q_ref_arr_1[0, j]), ls='--', color='gray', alpha=0.7)
    axs[0].legend()

    # 控制量
    ctrl_names = ['T [N]_1', 'tau_x_1', 'tau_y_1', 'tau_z_1']
    for j in range(4):
        axs[4].plot(t_arr_1[:-1], u_arr_1[:, j], label=ctrl_names[j])
    axs[4].set_title('Control input')
    axs[4].set_xlabel('t [s]_1')
    axs[4].legend()
    axs[4].grid(True)

    fig.suptitle('QuadNMPC closed-loop: state & control vs time', fontsize=14)
    plt.tight_layout()
    plt.show()


    state_names = ['x_2', 'y_2', 'z_2', 'phi_2', 'theta_2', 'psi_2',
                   'vx_2', 'vy_2', 'vz_2', 'p_2', 'q_2', 'r_2']
    groups_2 = [
        ('Position_2 [m]', [0, 1, 2]),
        ('Attitude_2 [deg]', [3, 4, 5]),
        ('Velocity_2 [m/s]', [6, 7, 8]),
        ('Angular rate_2 [deg/s]', [9, 10, 11]),
    ]
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.flatten()

    for g, (title, idxs) in enumerate(groups_2):
        ax = axs[g]
        for j in idxs:
            ax.plot(t_arr_2, q_arr_2[:, j], label=state_names[j])
        if g == 1:
            ax.set_ylabel('deg_2')
        if g == 3:
            ax.set_ylabel('deg/s_2')
        ax.set_title(title)
        ax.set_xlabel('t [s]_2')
        ax.legend()
        ax.grid(True)

    # x 参考随时间变化 -> 画曲线; y/z 参考是常数 -> 画水平线。
    axs[0].plot(t_arr_2, q_ref_arr_2[:, 0], ls='--', color='gray', alpha=0.7,
                label='x_ref_2')
    for j in (1, 2):
        axs[0].axhline(float(q_ref_arr_2[0, j]), ls='--', color='gray', alpha=0.7)
    axs[0].legend()

    # 控制量
    ctrl_names = ['T [N]_2', 'tau_x', 'tau_y', 'tau_z']
    for j in range(4):
        axs[4].plot(t_arr_2[:-1], u_arr_2[:, j], label=ctrl_names[j])
    axs[4].set_title('Control input')
    axs[4].set_xlabel('t [s]_2')
    axs[4].legend()
    axs[4].grid(True)

    fig.suptitle('QuadNMPC closed-loop: state & control vs time', fontsize=14)
    plt.tight_layout()
    plt.show()
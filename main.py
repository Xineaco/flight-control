"""
双机编队 NMPC 仿真 —— 领航/跟随结构

=====================================================================
本次改动(仅针对 A1: 僚机跟踪的是领机"当前状态"还是"预测轨迹")
=====================================================================
原文 (main.py:54):
    uk_2 = mpc.solve(qk_2, uk_2, qk_1 + OFFSET)
僚机的参考是领机的 **当前** 状态, 因此:
    * 结构性稳态滞后: 僚机永远在追上一时刻的目标;
    * 迎头撞机风险: 领机反向运动时, 僚机正奔向"领机刚才所在的位置"。

现在:
    * 领机求解后取自己的预测状态轨迹 traj_L = ctrl1.predict_states()
      (形状 (N+1, nx));
    * 僚机的参考 = traj_L 逐行加编队偏移 d, 即一条 **时变参考轨迹**
      (形状 (N+1, nx)), 交给 MPC 逐点惩罚 —— 僚机获得了对领机未来
      N 步运动的前馈预告。

对照开关:
    MODE = 'feedforward'  -> 新方案(领机预测轨迹 + 时变参考)
    MODE = 'chase'        -> 原方案(领机当前状态 + 单点参考)
    MODE = 'both'         -> 依次跑两种并输出改善幅度
两种模式除参考构造外完全一致, 因此差异可归因于 A1 本身。
"""

import os
import time

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import quard
import mpc as mpcmod

# ===================== 可调配置 =====================
MODE = 'both'             # 'feedforward' | 'chase' | 'both'
STEPS = 100               # 仿真步数
DT, N, NX, NU = 0.1, 5, 12, 4
FIGURE_SHOW = True        # True: 跑完弹出绘图窗口; False: 只存文件(results\ 目录)

# 编队偏移: 领机在前, 僚机跟在它后方 1 m。
# 期望的队形: x_F = x_L + 1  =>  dx = x_L - x_F = -1。
# 参考构造: r2 = 领机预测轨迹 + OFFSET, 因此 dx 的目标值就等于 OFFSET[0] = -1。
#
# 【待解决的已知问题】在正弦参考 + feedforward 下, 实测 dx 稳定收敛到 -1.0(符合期望),
# 但如果在 OFFSET=-1 下观察到 dx 收敛到 +1(阵形沿 x 镜像), 说明参考符号或度量口径
# 仍有不一致, 尚未定位。相关对账数据见 E:\deepseek workssapce\a1_*.txt
OFFSET = np.zeros(NX)
OFFSET[0] = -1.0
# ===================================================


# ---- 参考轨迹 ----
def reference(t):
    """领机的时变参考状态 (12,)"""
    return np.array([5 * np.sin(t), 0.0, 5.0,            # x, y, z
                     0.0, 0.0, 0.0,                      # phi, theta, psi
                     5 * np.cos(t), 0.0, 0.0,            # vx, vy, vz
                     0.0, 0.0, 0.0],                     # p, q, r
                    dtype=float)


def reference_traj(t0, dt, n):
    """参考轨迹 (n+1, nx): 第 k 行是 t0+k*dt 的参考"""
    return np.array([reference(t0 + k * dt) for k in range(n + 1)], dtype=float)


def build_plant():
    """真实对象: 以 RK 积分器推进, 控制量作为参数 p"""
    q_s = ca.SX.sym('q', NX)
    u_s = ca.SX.sym('u', NU)
    dq = quard.quard(q_s).model(q_s, u_s)
    return ca.integrator('plant', 'rk',
                         {'x': q_s, 'p': u_s, 'ode': dq},
                         0.0, DT)


def run(mode):
    """跑一次闭环, 返回记录字典与两个控制器"""
    plant = build_plant()

    def model(q, u):
        """MPC 内部模型: 与 plant 同一动力学"""
        return quard.quard(q).model(q, u)

    ctrl1 = mpcmod.QuadNMPC(dynamics=model, dt=DT, N=N, nx=NX, nu=NU)
    ctrl2 = mpcmod.QuadNMPC(dynamics=model, dt=DT, N=N, nx=NX, nu=NU)

    q1 = np.array([0.0, 0.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    u1 = np.array([1.0 * 9.8, 0, 0, 0])
    q2 = np.array([2.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    u2 = np.array([1.0 * 9.8, 0, 0, 0])

    rec = {k: [] for k in ['t', 'q1', 'q2', 'u1', 'u2',
                           'ref1', 'ref2', 'dt1', 'dt2', 's1', 's2',
                           'xL', 'refF_x']}
    t = 0.0

    for i in range(STEPS):
        # ---------- ① 领机: 跟踪时变参考轨迹 ----------
        r1 = reference_traj(t, DT, N)
        t0 = time.perf_counter()
        u1 = ctrl1.solve(q1, r1)
        dt1 = time.perf_counter() - t0

        # ---------- ② 僚机参考 ----------
        if mode == 'feedforward':
            # A1: 用领机的预测轨迹构造时变参考(逐行加编队偏移)
            traj_l = ctrl1.predict_states()
            if traj_l is None:                 # 第一步尚未解出, 退化为当前状态
                traj_l = np.tile(q1, (N + 1, 1))
            assert traj_l.shape == (N + 1, NX), traj_l.shape
            r2 = traj_l + OFFSET               # (N+1, nx) 时变参考
        else:
            # 原方案: 单点参考 = 领机当前状态 + 偏移
            r2 = q1 + OFFSET

        t0 = time.perf_counter()
        u2 = ctrl2.solve(q2, r2)
        dt2 = time.perf_counter() - t0

        # ---------- ③ 记录 ----------
        rec['t'].append(t)
        rec['q1'].append(q1.copy())
        rec['q2'].append(q2.copy())
        rec['u1'].append(np.array(u1, dtype=float).copy())
        rec['u2'].append(np.array(u2, dtype=float).copy())
        rec['ref1'].append(r1[0])
        rec['ref2'].append(np.atleast_2d(np.asarray(r2, float))[0])
        rec['xL'].append(float(q1[0]))
        rec['refF_x'].append(float(np.atleast_2d(np.asarray(r2, float))[0, 0]))
        rec['dt1'].append(dt1)
        rec['dt2'].append(dt2)
        rec['s1'].append(ctrl1.status())
        rec['s2'].append(ctrl2.status())

        if i % 10 == 0:
            print('步%3d: x_1=%7.3f z_1=%7.3f | x_2=%7.3f z_2=%7.3f | '
                  'T_1=%6.2f T_2=%6.2f | solve %.0f/%.0f ms' %
                  (i, q1[0], q1[2], q2[0], q2[2], u1[0], u2[0],
                   1e3 * dt1, 1e3 * dt2))

        # ---------- ④ 推进对象 ----------
        q1 = np.array(plant(x0=q1, p=u1)['xf']).flatten()
        q2 = np.array(plant(x0=q2, p=u2)['xf']).flatten()
        t += DT

    # 末态补一条(所有 key 都要补, 否则各数组长度不一致)
    #
    # 长度不变式(两种模式一致):
    #   t / q1 / q2 / ref1 / ref2 / xL / refF_x  ->  STEPS + 1  (含末态)
    #   u1 / u2 / dt1 / dt2 / s1 / s2           ->  STEPS      (每步一个, 无末态)
    # 控制量等"每步一个"的量不补末态行, 否则绘图时 t[:-1] 与 u 的维度对不上。
    rec['t'].append(t)
    rec['q1'].append(q1.copy())
    rec['q2'].append(q2.copy())
    rec['ref1'].append(reference(t))
    rec['ref2'].append(q1 + OFFSET)
    rec['xL'].append(float(q1[0]))
    rec['refF_x'].append(float(q1[0] + OFFSET[0]))

    # 自检: 保证上面两条不变式成立
    n_state = len(rec['t'])
    for key in ('q1', 'q2', 'ref1', 'ref2', 'xL', 'refF_x'):
        assert len(rec[key]) == n_state, (key, len(rec[key]), n_state)
    for key in ('u1', 'u2', 'dt1', 'dt2', 's1', 's2'):
        assert len(rec[key]) == n_state - 1, (key, len(rec[key]), n_state - 1)

    return rec, ctrl1, ctrl2


def summarize(rec, mode):
    """计算并打印编队指标"""
    t = np.array(rec['t'])
    q1 = np.array(rec['q1'], dtype=float)
    q2 = np.array(rec['q2'], dtype=float)
    ref1 = np.array(rec['ref1'], dtype=float)
    dt1 = np.array(rec['dt1'], dtype=float)
    dt2 = np.array(rec['dt2'], dtype=float)

    rel = q1[:, 0:3] - q2[:, 0:3]
    dist = np.linalg.norm(rel, axis=1)
    form_err = np.linalg.norm(rel - OFFSET[0:3], axis=1)      # 队形保持误差
    lead_e = np.linalg.norm(q1[:, 0:3] - ref1[:, 0:3], axis=1)

    print('\n' + '=' * 66)
    print('MODE = %s' % mode)
    print('=' * 66)
    print('领机位置跟踪 RMSE    = %.4f m   (max %.4f)' % (
        np.sqrt((lead_e ** 2).mean()), lead_e.max()))
    print('两机间距 |p1-p2|     = mean %.4f  min %.4f  max %.4f  final %.4f  (目标 %.2f)'
          % (dist.mean(), dist.min(), dist.max(), dist[-1], np.linalg.norm(OFFSET)))
    print('队形误差 |(p1-p2)-d| = mean %.4f  max %.4f  final %.4f'
          % (form_err.mean(), form_err.max(), form_err[-1]))
    print('求解时间 领机 mean %.1f max %.1f ms | 僚机 mean %.1f max %.1f ms'
          % (1e3 * dt1.mean(), 1e3 * dt1.max(), 1e3 * dt2.mean(), 1e3 * dt2.max()))
    out = {'lead_rmse': float(np.sqrt((lead_e ** 2).mean())),
           'dist_min': float(dist.min()),
           'dist_final': float(dist[-1]),
           'form_mean': float(form_err.mean()),
           'form_max': float(form_err.max()),
           'form_final': float(form_err[-1]),
           'solve_max': float(max(dt1.max(), dt2.max()))}
    return out, rec


def plot(rec, mode):
    """绘图: 4x3 = 12 格。

    第 0-3 格: 位置 / 姿态 / 速度 / 角速率  (leader 实线, follower 虚线, leader 参考点线)
    第 4 格  : 控制量(4 通道 x 2 架)
    第 5 格  : x-z 轨迹
    第 6 格  : 两机间距 |p1-p2|
    第 7 格  : 位置跟踪误差 |p - p_ref|
    第 8 格  : 相对位置 dx/dy/dz 及期望值
    第 9 格  : 队形误差
    第 10 格 : 间距放大(看是否逼近安全距离)
    第 11 格 : 求解耗时
    """
    t = np.array(rec['t'])
    q1 = np.array(rec['q1'], dtype=float)
    q2 = np.array(rec['q2'], dtype=float)
    ref1 = np.array(rec['ref1'], dtype=float)
    u1 = np.array(rec['u1'], dtype=float)
    u2 = np.array(rec['u2'], dtype=float)
    dt1 = np.array(rec['dt1'], dtype=float)
    dt2 = np.array(rec['dt2'], dtype=float)

    rel = q1[:, 0:3] - q2[:, 0:3]
    dist = np.linalg.norm(rel, axis=1)
    form_err = np.linalg.norm(rel - OFFSET[0:3], axis=1)
    lead_e = np.linalg.norm(q1[:, 0:3] - ref1[:, 0:3], axis=1)
    foll_e = np.linalg.norm(q2[:, 0:3] - (q1[:, 0:3] + OFFSET[0:3]), axis=1)

    state_names = ['x', 'y', 'z', 'phi', 'theta', 'psi',
                   'vx', 'vy', 'vz', 'p', 'q', 'r']
    ctrl_names = ['T [N]', 'tau_x', 'tau_y', 'tau_z']
    groups = [
        ('Position [m]',         [0, 1, 2]),
        ('Attitude [rad]',       [3, 4, 5]),
        ('Velocity [m/s]',       [6, 7, 8]),
        ('Angular rate [rad/s]', [9, 10, 11]),
    ]

    fig, axs = plt.subplots(4, 3, figsize=(17, 18))
    axs = axs.flatten()

    # ---- 第 0-3 格: 四组状态(恢复原图) ----
    for g, (title, idxs) in enumerate(groups):
        ax = axs[g]
        for j in idxs:
            ax.plot(t, q1[:, j], '-', lw=1.3, label='%s_leader' % state_names[j])
            ax.plot(t, q2[:, j], '--', lw=1.3, label='%s_follower' % state_names[j])
        if g == 0:                       # 位置: 叠加领机参考
            ax.plot(t, ref1[:, 0], ':', color='gray', label='x_ref_leader')
            for j in (1, 2):
                ax.axhline(float(ref1[0, j]), ls=':', color='gray', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True)

    # ---- 第 4 格: 控制量 ----
    ax = axs[4]
    for j in range(4):
        ax.plot(t[:-1], u1[:, j], '-', lw=1.2, label='%s_leader' % ctrl_names[j])
        ax.plot(t[:-1], u2[:, j], '--', lw=1.2, label='%s_follower' % ctrl_names[j])
    ax.set_title('Control input')
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)

    # ---- 第 5 格: x-z 轨迹 ----
    ax = axs[5]
    ax.plot(q1[:, 0], q1[:, 2], '-', label='leader')
    ax.plot(q2[:, 0], q2[:, 2], '--', label='follower')
    ax.plot(ref1[:, 0], ref1[:, 2], ':', color='gray', label='leader ref')
    ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
    ax.set_title('x-z trajectory'); ax.legend(fontsize=8); ax.grid(True)

    # ---- 第 6 格: 编队间距 ----
    ax = axs[6]
    ax.plot(t, dist, 'k-', label='|p1 - p2|')
    ax.axhline(np.linalg.norm(OFFSET), ls='--', color='gray',
               label='target %.2f' % np.linalg.norm(OFFSET))
    ax.set_xlabel('t [s]'); ax.set_ylabel('distance [m]')
    ax.set_title('formation distance'); ax.legend(fontsize=8); ax.grid(True)

    # ---- 第 7 格: 位置跟踪误差 ----
    ax = axs[7]
    ax.plot(t, lead_e, '-', label='leader |p - p_ref|')
    ax.plot(t, foll_e, '--', label='follower |p - (p_L + d)|')
    ax.set_xlabel('t [s]'); ax.set_ylabel('|p - p_ref| [m]')
    ax.set_title('position tracking error'); ax.legend(fontsize=8); ax.grid(True)

    # ---- 第 8 格: 相对位置 ----
    ax = axs[8]
    ax.plot(t, rel[:, 0], label='dx')
    ax.plot(t, rel[:, 1], label='dy')
    ax.plot(t, rel[:, 2], label='dz')
    ax.axhline(OFFSET[0], ls='--', color='gray',
               label='desired dx=%.1f' % OFFSET[0])
    ax.set_xlabel('t [s]'); ax.set_title('relative position p1-p2')
    ax.legend(fontsize=8); ax.grid(True)

    # ---- 第 9 格: 队形误差 ----
    ax = axs[9]
    ax.plot(t, form_err, label='|(p1-p2) - d|')
    ax.set_xlabel('t [s]'); ax.set_ylabel('[m]')
    ax.set_title('formation error'); ax.legend(fontsize=8); ax.grid(True)

    # ---- 第 10 格: 间距放大(安全距离检查) ----
    ax = axs[10]
    ax.plot(t, dist, label='|p1-p2|')
    ax.axhline(np.linalg.norm(OFFSET), ls='--', color='gray')
    ax.set_ylim(0, max(2.0, float(dist.max()) * 1.1))
    ax.set_xlabel('t [s]'); ax.set_title('distance (zoom, collision check)')
    ax.legend(fontsize=8); ax.grid(True)

    # ---- 第 11 格: 求解耗时 ----
    ax = axs[11]
    ax.plot(t[:-1], 1e3 * dt1, label='leader solve')
    ax.plot(t[:-1], 1e3 * dt2, '--', label='follower solve')
    ax.axhline(1e3 * DT, ls=':', color='r', label='dt = %.0f ms' % (1e3 * DT))
    ax.set_xlabel('t [s]'); ax.set_ylabel('ms')
    ax.set_title('solve time'); ax.legend(fontsize=8); ax.grid(True)

    fig.suptitle('MODE = %s   (leader solid / follower dashed)' % mode,
                 fontsize=14, y=0.995)
    # 用带 rect 的 tight_layout 给总标题留出顶部空间, 否则会压住第一行子图的图例
    plt.tight_layout(rect=(0, 0, 1, 0.975))

    # 把图存到工程目录下的 results\ , 方便直接打开查看
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, 'a1_%s.png' % mode)
    plt.savefig(fn, dpi=110)

    # 立刻显示本张图, 然后释放, 避免多张图窗堆积(堆积时部分窗口会是空白)
    if FIGURE_SHOW:
        plt.show()
    plt.close(fig)
    return fn


if __name__ == '__main__':
    results = {}
    modes = ['chase', 'feedforward'] if MODE == 'both' else [MODE]
    for mode in modes:
        rec, c1, c2 = run(mode)
        stats, rec = summarize(rec, mode)
        print('求解统计: 领机 %s | 僚机 %s' % (c1.stats(), c2.stats()))
        print('图 -> %s' % plot(rec, mode))
        results[mode] = stats

    if len(results) > 1:
        print('\n' + '=' * 66)
        print('A1 前后对照 (chase = 原方案, feedforward = 本文修改)')
        print('=' * 66)
        keys = ['lead_rmse', 'dist_min', 'dist_final',
                'form_mean', 'form_max', 'form_final', 'solve_max']
        print('%-12s %14s %14s %12s' % ('指标', 'chase(原)', 'feedforward', '变化'))
        for k in keys:
            a, b = results['chase'][k], results['feedforward'][k]
            imp = (a - b) / a * 100 if a != 0 else float('nan')
            print('%-12s %14.4f %14.4f %11.1f%%' % (k, a, b, imp))

    # 图窗已在 plot() 内逐张显示并关闭(避免多图窗堆积导致空白),
    # 图片文件保存在工程目录的 results\ 下。

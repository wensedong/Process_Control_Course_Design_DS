#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习 PID 参数整定模块 — Q-Learning 表格法

原理:
  - 状态: 将超调量、振荡程度、稳态误差离散化为 3×3×3 = 27 个状态
  - 动作: 7 种参数调整方向 (Kp/Ki/Kd ±, keep)
  - 奖励: -IAE + 衰减比奖励 - 超调/静差惩罚
  - 算法: Q-Learning (ε-greedy 探索, Q-table 迭代更新)

用法:
  from rl_tuner import rl_tune
  result = rl_tune(K=1.0, T=1.0, tau=0.5, setpoint=1.0, episodes=500)
  # result = {'Kp': ..., 'Ki': ..., 'Kd': ..., 'method_name': ..., ...}
"""

import csv
import os
import numpy as np
from datetime import datetime
from simulation import simulate, PlantConfig, PIDConfig, SimulationConfig
from metrics import compute_metrics


# ============================================================
# 状态离散化参数
# ============================================================

N_STATES = 27     # 3(超调) × 3(振荡) × 3(静差)
N_ACTIONS = 7     # Kp+ Kp- Ki+ Ki- Kd+ Kd- keep

ACTION_NAMES = ['Kp▲', 'Kp▼', 'Ki▲', 'Ki▼', 'Kd▲', 'Kd▼', '—']

# 离散化阈值
OVERSHOOT_BINS   = [10.0, 30.0]      # <10%=0, 10-30%=1, >30%=2
OSCILLATION_BINS = [0.6, 0.85]       # 衰减率: <0.6=severe, 0.6-0.85=mild, >0.85=overdamped
ERROR_BINS       = [3.0, 10.0]       # 稳态误差%: <3%=0, 3-10%=1, >10%=2

# 参数调整步长
KP_FACTOR = 1.3      # Kp 乘/除因子
KI_FACTOR = 1.5      # Ki 乘/除因子
KD_FACTOR = 1.5      # Kd 乘/除因子
KI_INIT  = 0.05      # Ki 从 0 启动时的初值
KD_INIT  = 0.02      # Kd 从 0 启动时的初值

# 参数范围
# 注意: KI_MIN/KD_MIN 设为小正数而非 0 — 若为 0，参数被除到 0 后无法通过乘因子恢复
KP_MIN, KP_MAX = 0.01, 20.0
KI_MIN, KI_MAX = 0.001, 5.0
KD_MIN, KD_MAX = 0.001, 5.0

# Q-Learning 超参数
ALPHA   = 0.1         # 学习率
GAMMA   = 0.9         # 折扣因子
EPS_START = 1.0       # 初始探索率
EPS_END   = 0.05      # 最终探索率
EPS_DECAY = 0.98      # 每 episode 探索率衰减


# ============================================================
# 仿真辅助 (与 tuning_methods 一致的接口)
# ============================================================

def _sim(K, T, tau, Kp, Ki, Kd, setpoint, sim_time, dt=0.01):
    """运行仿真并返回 (t, sp, pv, co, err) 元组"""
    r = simulate(
        PlantConfig(K=K, T=T, tau=tau),
        PIDConfig(Kp=Kp, Ki=Ki, Kd=Kd,
                  output_min=-1e6, output_max=1e6),
        SimulationConfig(setpoint=setpoint, sim_time=sim_time, dt=dt))
    return r['t'], r['sp'], r['pv'], r['co'], r['error']


# ============================================================
# 状态离散化
# ============================================================

def _discretize_overshoot(overshoot):
    """超调量 → 0/1/2"""
    if overshoot < OVERSHOOT_BINS[0]:
        return 0       # 低超调
    elif overshoot < OVERSHOOT_BINS[1]:
        return 1       # 中超调
    else:
        return 2       # 高超调 / 发散


def _discretize_oscillation(decay_rate):
    """衰减率 → 0/1/2"""
    if decay_rate is None:
        return 0       # 无振荡 → 过阻尼
    if decay_rate < OSCILLATION_BINS[0]:
        return 2       # 严重振荡 (衰减率 < 0.6)
    elif decay_rate < OSCILLATION_BINS[1]:
        return 1       # 轻度振荡 (0.6~0.85)
    else:
        return 0       # 过阻尼 / 衰减太快


def _discretize_error(steady_error):
    """稳态误差 → 0/1/2"""
    if steady_error < ERROR_BINS[0]:
        return 0       # 小静差
    elif steady_error < ERROR_BINS[1]:
        return 1       # 中静差
    else:
        return 2       # 大静差


def state_index(metrics):
    """
    将性能指标映射到离散状态编号 (0~26)。

    编码: s = overshoot_lv * 9 + oscillation_lv * 3 + error_lv
    """
    ov_lv = _discretize_overshoot(metrics['overshoot'])
    os_lv = _discretize_oscillation(metrics.get('decay_rate'))
    er_lv = _discretize_error(metrics['steady_state_error'])
    return ov_lv * 9 + os_lv * 3 + er_lv


# ============================================================
# 动作
# ============================================================

def apply_action(Kp, Ki, Kd, action_idx):
    """
    执行动作, 返回调整后的 (Kp, Ki, Kd)。

    动作映射:
      0: Kp▲ (×KP_FACTOR)
      1: Kp▼ (÷KP_FACTOR)
      2: Ki▲ (×KI_FACTOR, 从0启动则设为 KI_INIT)
      3: Ki▼ (÷KI_FACTOR)
      4: Kd▲ (×KD_FACTOR, 从0启动则设为 KD_INIT)
      5: Kd▼ (÷KD_FACTOR)
      6: keep (不变)
    """
    if action_idx == 0:    # Kp▲
        Kp = min(KP_MAX, Kp * KP_FACTOR)
    elif action_idx == 1:  # Kp▼
        Kp = max(KP_MIN, Kp / KP_FACTOR)
    elif action_idx == 2:  # Ki▲
        if Ki < 1e-9:
            Ki = KI_INIT
        else:
            Ki = min(KI_MAX, Ki * KI_FACTOR)
    elif action_idx == 3:  # Ki▼
        Ki = max(KI_MIN, Ki / KI_FACTOR)
    elif action_idx == 4:  # Kd▲
        if Kd < 1e-9:
            Kd = KD_INIT
        else:
            Kd = min(KD_MAX, Kd * KD_FACTOR)
    elif action_idx == 5:  # Kd▼
        Kd = max(KD_MIN, Kd / KD_FACTOR)
    # action_idx == 6: keep

    # 精度保留
    return round(Kp, 4), round(Ki, 4), round(Kd, 4)


# ============================================================
# 奖励函数
# ============================================================

def compute_reward(metrics, stable, setpoint=1.0):
    """
    计算奖励值。

    设计原则:
      - 主目标: 最小化 IAE (负值, 越小越好 → 越大奖励)
      - 奖励接近 4:1 衰减比 (工程最佳)
      - 惩罚发散和过大超调
    """
    if not stable:
        return -50.0     # 发散: 严重惩罚

    ov   = metrics['overshoot']
    dr   = metrics.get('decay_rate')
    se   = metrics['steady_state_error']
    iae  = metrics['iae']
    rise = metrics.get('rise_time')

    # 归一化 IAE — 除以 (setpoint × 仿真时间) 得到相对指标
    # IAE ≈ 0 表示完美跟踪, IAE ≈ 5 表示很差
    reward = -iae

    # 衰减比奖励 (4:1 衰减为最佳, 即衰减率 ≈ 0.75)
    if dr is not None and 0.65 <= dr <= 0.85:
        if ov < 15 and se < 3:
            reward += 15.0    # 接近最优: 大幅奖励
        else:
            reward += 5.0     # 衰减比好但超调/静差稍差

    # 超调惩罚
    if ov > 50:
        reward -= 10.0
    elif ov > 30:
        reward -= 5.0
    elif ov > 15:
        reward -= 1.0

    # 稳态误差惩罚
    if se > 10:
        reward -= 8.0
    elif se > 5:
        reward -= 3.0
    elif se > 3:
        reward -= 0.5

    # 过阻尼惩罚 (响应太慢)
    if rise is not None and rise > 5.0:
        reward -= 2.0
    if dr is not None and dr > 0.95:
        reward -= 1.0     # 几乎无振荡, 可能响应过慢

    return reward


def is_terminal_state(metrics, stable):
    """
    判断是否终止当前 episode。

    终止条件:
      1. 发散 (stable=False)
      2. 达到良好性能 (衰减比接近 4:1, 低超调, 低静差)
    """
    if not stable:
        return True

    ov = metrics['overshoot']
    dr = metrics.get('decay_rate')
    se = metrics['steady_state_error']

    # 达标: 超调 < 10%, 衰减率 0.65~0.85, 静差 < 2%, 且衰减率有效
    if dr is not None and ov < 10 and 0.65 <= dr <= 0.85 and se < 2:
        return True

    return False


# ============================================================
# Q-Learning 整定器
# ============================================================

class QLearningTuner:
    """
    Q-Learning PID 参数整定器。

    参数:
        K, T, tau: 被控对象参数 G(s) = K·e^(-τs)/(Ts+1)
        setpoint:  设定值
        sim_time:  每次仿真时长
        dt:        仿真步长
        alpha:     学习率
        gamma:     折扣因子
        eps_start, eps_end, eps_decay: ε-greedy 探索参数
    """

    def __init__(self, K=1.0, T=1.0, tau=0.0,
                 setpoint=1.0, sim_time=25.0, dt=0.01,
                 alpha=ALPHA, gamma=GAMMA,
                 eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY):
        self.K = K
        self.T = T
        self.tau = tau
        self.setpoint = setpoint
        self.sim_time = sim_time
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # 初始化 Q 表: 27 states × 7 actions
        self.Q = np.zeros((N_STATES, N_ACTIONS))

        # 训练记录
        self.best_params = None
        self.best_reward = -float('inf')
        self.best_metrics = None
        self.history = []       # 每 episode 的最佳奖励 (向后兼容)
        self.episode_log = []   # 每 episode 的详细记录 (供 CSV 导出)

    def _random_params(self):
        """
        生成随机初始 PID 参数 (稳定性感知)。

        对于有迟延系统: 根据 FOPDT 近似稳定边界约束 Kp 上限。
        Kp_crit ≈ T / (K · τ)，初始化在 ~60% 临界值以内，
        避免 Agent 在必然发散的参数区域浪费探索步数。
        """
        if self.tau > 1e-6:
            # 一阶+迟延: 粗略稳定边界估计
            kp_crit_est = self.T / (self.K * self.tau * 1.2)
            kp_max_init = min(5.0, kp_crit_est * 0.6)
            kp_min_init = 0.1
        else:
            # 无迟延: 一阶惯性系统无条件稳定 (纯比例下)
            kp_max_init = 8.0
            kp_min_init = 0.3

        # 对数均匀采样 Kp (PID 参数通常在对数尺度上更均匀)
        Kp = 10 ** np.random.uniform(
            np.log10(max(KP_MIN, kp_min_init)),
            np.log10(max(kp_max_init, kp_min_init * 2)))
        # Ki/Kd 上限与 Kp 联动, 避免积分/微分过强
        Ki = np.random.uniform(KI_MIN, min(2.0, Kp * 0.6))
        Kd = np.random.uniform(KD_MIN, min(1.0, Kp * 0.25))
        return round(Kp, 4), round(Ki, 4), round(Kd, 4)

    def _choose_action(self, s, eps):
        """ε-greedy 动作选择"""
        if np.random.random() < eps:
            return np.random.randint(N_ACTIONS)
        else:
            # 贪婪选择 (平局时随机打破)
            best = np.argmax(self.Q[s, :])
            # 如果有多个相同最大值, 随机选一个
            max_val = self.Q[s, best]
            candidates = np.where(self.Q[s, :] == max_val)[0]
            return np.random.choice(candidates)

    def train(self, episodes=500, steps_per_episode=8,
              progress_callback=None):
        """
        训练 Q-Learning 整定器。

        注意 — 状态转移的特殊性:
          本问题中, 每个 RL step 需要运行一次**完整仿真** (sim_time 秒)
          才能获得新状态 s'。这不是标准的逐时间步 MDP, 而是:
            "当前 PID 参数的性能" —动作→ "新 PID 参数的性能"
          每一步都是独立的阶跃响应, 从 t=0 重新启动。
          因此:
            - steps_per_episode 不宜过大 (避免在同一条轨迹上绕圈)
            - episodes 应足够大 (靠多轮重启充分探索参数空间)
            - 本质更接近 episodic parameter optimization

        参数:
            episodes:          训练轮数 (默认 500)
            steps_per_episode: 每轮最大调整步数 (默认 8)
            progress_callback: 可选, 每 episode 调用 callback(ep, eps, best_r, best_params)

        返回:
            dict: 最佳参数及性能指标
        """
        eps = self.eps_start

        for ep in range(episodes):
            # 随机初始参数 (稳定性感知)
            Kp, Ki, Kd = self._random_params()

            # 初始仿真 → 初始状态
            t, sp, pv, co, err = _sim(
                self.K, self.T, self.tau,
                Kp, Ki, Kd, self.setpoint, self.sim_time, self.dt)
            metrics = compute_metrics(t, sp, pv, err)
            stable = np.isfinite(pv[-1]) and abs(pv[-1]) < 1e6
            s = state_index(metrics)

            episode_best_reward = -float('inf')
            episode_best_params = None
            episode_terminal = False
            episode_steps = 0
            episode_stable = stable  # 初始仿真是否收敛

            for step in range(steps_per_episode):
                # 选择动作
                a = self._choose_action(s, eps)

                # 执行动作 → 新参数
                new_Kp, new_Ki, new_Kd = apply_action(Kp, Ki, Kd, a)

                # 运行仿真 → 新状态 & 奖励
                t, sp, pv, co, err = _sim(
                    self.K, self.T, self.tau,
                    new_Kp, new_Ki, new_Kd,
                    self.setpoint, self.sim_time, self.dt)
                new_metrics = compute_metrics(t, sp, pv, err)
                new_stable = np.isfinite(pv[-1]) and abs(pv[-1]) < 1e6
                s_next = state_index(new_metrics)
                r = compute_reward(new_metrics, new_stable, self.setpoint)

                # Q-Learning 更新
                best_next = np.max(self.Q[s_next, :])
                self.Q[s, a] += self.alpha * (r + self.gamma * best_next - self.Q[s, a])

                # 跟踪最佳
                if r > episode_best_reward:
                    episode_best_reward = r
                    episode_best_params = (new_Kp, new_Ki, new_Kd)

                if r > self.best_reward:
                    self.best_reward = r
                    self.best_params = (new_Kp, new_Ki, new_Kd)
                    self.best_metrics = new_metrics

                # 状态转移
                Kp, Ki, Kd = new_Kp, new_Ki, new_Kd
                s = s_next

                # 终止检查
                if is_terminal_state(new_metrics, new_stable):
                    episode_terminal = True
                    break

            episode_steps = step + 1  # step 从 0 开始, +1 为实际步数
            self.history.append(episode_best_reward)

            # 记录每轮详细日志 (供 CSV 导出)
            bp = episode_best_params or (Kp, Ki, Kd)
            self.episode_log.append({
                'episode':      ep + 1,
                'epsilon':      round(eps, 4),
                'Kp':           bp[0],
                'Ki':           bp[1],
                'Kd':           bp[2],
                'reward':       round(episode_best_reward, 4),
                'steps':        episode_steps,
                'is_terminal':  episode_terminal,
                'stable':       episode_stable,
            })

            # 衰减探索率
            eps = max(self.eps_end, eps * self.eps_decay)

            # 进度回调
            if progress_callback is not None:
                progress_callback(ep + 1, episodes, eps,
                                  self.best_reward, self.best_params)

        return self.get_result()

    def get_result(self):
        """返回最佳整定结果 (与 tuning_methods 兼容的 dict 格式)"""
        if self.best_params is None:
            return None

        Kp, Ki, Kd = self.best_params
        # 最终仿真确认
        t, sp, pv, co, err = _sim(
            self.K, self.T, self.tau,
            Kp, Ki, Kd, self.setpoint, self.sim_time, self.dt)
        final_metrics = compute_metrics(t, sp, pv, err)

        return {
            'Kp': round(Kp, 4),
            'Ki': round(Ki, 4),
            'Kd': round(Kd, 4),
            'delta': None,
            'TI': None,
            'TD': None,
            'method_name': '强化学习整定 (Q-Learning)',
            'metrics': final_metrics,
            'best_reward': round(self.best_reward, 2),
            'episodes': len(self.history),
            'history': self.history,
            'q_table': self.Q.copy(),
        }

    def get_q_table(self):
        """返回 Q 表副本 (用于可视化)"""
        return self.Q.copy()

    def get_q_insights(self):
        """
        返回 Q 表的关键洞察: 每个状态下推荐的动作及其 Q 值。
        """
        insights = []
        for s in range(N_STATES):
            best_a = np.argmax(self.Q[s, :])
            q_val = self.Q[s, best_a]
            ov_lv = s // 9
            os_lv = (s % 9) // 3
            er_lv = s % 3
            insights.append({
                'state': s,
                'overshoot_level': ov_lv,
                'oscillation_level': os_lv,
                'error_level': er_lv,
                'best_action': ACTION_NAMES[best_a],
                'best_action_idx': int(best_a),
                'q_value': round(q_val, 4),
            })
        return insights

    def export_training_csv(self, path):
        """
        导出训练日志 CSV。

        列: episode, epsilon, Kp, Ki, Kd, reward, steps, is_terminal, stable
        """
        if not self.episode_log:
            return False

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            # 元数据头
            w.writerow(['# Q-Learning PID 整定 — 训练日志'])
            w.writerow(['# 时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            w.writerow(['# 对象', f'K={self.K}, T={self.T}, tau={self.tau}'])
            w.writerow(['# 设定值', str(self.setpoint)])
            w.writerow(['# 最佳参数',
                        f'Kp={self.best_params[0]:.4f} Ki={self.best_params[1]:.4f} Kd={self.best_params[2]:.4f}'
                        if self.best_params else 'N/A'])
            w.writerow(['# 最佳奖励', str(round(self.best_reward, 4))])
            w.writerow(['#'])
            # 数据头
            w.writerow(['episode', 'epsilon', 'Kp', 'Ki', 'Kd', 'reward',
                        'steps', 'is_terminal', 'stable'])
            # 数据行
            for row in self.episode_log:
                w.writerow([
                    row['episode'], row['epsilon'],
                    f'{row["Kp"]:.4f}', f'{row["Ki"]:.4f}', f'{row["Kd"]:.4f}',
                    f'{row["reward"]:.4f}', row['steps'],
                    int(row['is_terminal']), int(row['stable']),
                ])
        return True


# ============================================================
# 便利函数 (供 GUI 调用)
# ============================================================

def rl_tune(K, T, tau, setpoint=1.0, sim_time=25.0,
            episodes=500, steps_per_episode=8,
            progress_callback=None):
    """
    强化学习 PID 整定的便捷入口。

    参数:
        K, T, tau:     被控对象参数
        setpoint:       设定值
        sim_time:       仿真时长
        episodes:       训练轮数
        progress_callback: 进度回调 callback(ep, total, eps, best_r, best_params)

    返回:
        dict: {'Kp', 'Ki', 'Kd', 'method_name', 'metrics', 'history', ...}
        或 None (训练失败)
    """
    tuner = QLearningTuner(
        K=K, T=T, tau=tau,
        setpoint=setpoint, sim_time=sim_time,
    )
    result = tuner.train(
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        progress_callback=progress_callback,
    )

    # 自动保存训练日志 CSV
    if result is not None:
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'saved_results', 'rl_training')
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(
            save_dir,
            f'rl_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        tuner.export_training_csv(csv_path)
        result['training_csv'] = csv_path

    return result


# ============================================================
# 演示 & 测试
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  强化学习 PID 整定 — 独立测试")
    print("=" * 60)

    # 测试对象: 有迟延的一阶系统
    test_K = 1.0
    test_T = 2.0
    test_tau = 0.5

    print(f"\n被控对象: K={test_K}, T={test_T}, τ={test_tau}")
    print(f"训练中...")

    def show_progress(ep, total, eps, best_r, best_params):
        if ep % 20 == 0 or ep == total:
            n = ep * 30 // total
            bar = '#' * n + '-' * (30 - n)
            print(f"  [{bar}] {ep:3d}/{total}  eps={eps:.3f}  "
                  f"best_R={best_r:+.2f}  Kp={best_params[0]:.2f} "
                  f"Ki={best_params[1]:.2f} Kd={best_params[2]:.2f}")

    result = rl_tune(test_K, test_T, test_tau,
                     setpoint=1.0, sim_time=30.0,
                     episodes=500, progress_callback=show_progress)

    if result:
        m = result['metrics']
        print(f"\n{'='*60}")
        print(f"  整定结果")
        print(f"{'='*60}")
        print(f"  Kp = {result['Kp']:.4f}")
        print(f"  Ki = {result['Ki']:.4f}")
        print(f"  Kd = {result['Kd']:.4f}")
        print(f"  ---")
        print(f"  超调量:   {m['overshoot']:.2f}%")
        print(f"  衰减率:   {m['decay_rate']}")
        if m['decay_rate'] is not None:
            print(f"            ({m['decay_rate']*100:.1f}%)")
        print(f"  稳态误差: {m['steady_state_error']:.2f}%")
        print(f"  IAE:      {m['iae']:.4f}")
        print(f"  ISE:      {m['ise']:.4f}")
        print(f"  上升时间: {m['rise_time']}")
        print(f"  调节时间: {m['settling_time']}")
        print(f"  ---")
        print(f"  最佳奖励: {result['best_reward']}")
    else:
        print("训练失败!")

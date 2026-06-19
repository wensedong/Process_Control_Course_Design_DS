#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN (Deep Q-Network) PID 参数整定模块

将 Q-Learning 表格法升级为基于 PyTorch 的深度强化学习:
  - 状态: 连续特征向量 (不再离散化为 0/1/2)
  - 网络: MLP (2×64 隐藏层, ReLU) → 7 维 Q 值输出
  - 机制: Double DQN (策略网络 + 目标网络, 软更新)
  - 经验回放: Replay Buffer (容量 10000, 批量采样)

与 rl_tuner.py 共享动作空间、奖励函数和仿真接口。
用法:
  from dqn_tuner import dqn_tune
  result = dqn_tune(K=1.0, T=2.0, tau=0.5, setpoint=1.0, episodes=500)
"""

import csv
import os
import numpy as np
import random
from collections import deque
from datetime import datetime

from simulation import simulate, PlantConfig, PIDConfig, SimulationConfig
from metrics import compute_metrics

# 复用 rl_tuner 的动作/奖励/终止逻辑 (与表格法完全一致)
from rl_tuner import (
    apply_action, compute_reward, is_terminal_state,
    KP_MIN, KP_MAX, KI_MIN, KI_MAX, KD_MIN, KD_MAX,
    N_ACTIONS,
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    raise ImportError(
        "DQN tuner 需要 PyTorch。请执行: pip install torch"
    )


# ============================================================
# 仿真辅助
# ============================================================

def _sim(K, T, tau, Kp, Ki, Kd, setpoint, sim_time, dt=0.01):
    """运行仿真, 返回 (t, sp, pv, co, err) 元组"""
    r = simulate(
        PlantConfig(K=K, T=T, tau=tau),
        PIDConfig(Kp=Kp, Ki=Ki, Kd=Kd,
                  output_min=-1e6, output_max=1e6),
        SimulationConfig(setpoint=setpoint, sim_time=sim_time, dt=dt))
    return r['t'], r['sp'], r['pv'], r['co'], r['error']


# ============================================================
# 连续状态向量 — 替代表格法的 0/1/2 离散化
# ============================================================

STATE_DIM = 8  # 5 个响应特征 + 3 个 PID 参数

def get_state(metrics, stable, Kp, Ki, Kd, setpoint=1.0, sim_time=25.0):
    """
    将仿真性能指标映射为连续状态向量 (8 维), 供神经网络输入。

    各维度的工程归一化:
      [0] overshoot_norm  = overshoot / 100          ∈ [0, ~5], >1 表示超调>100%
      [1] decay_rate_val  = decay_rate or 1.0        ∈ [0, 1],  1.0=无振荡(最优)
      [2] error_norm      = steady_error / 100        ∈ [0, 1],  >0.1 表示静差>10%
      [3] iae_norm        = iae / 5                   ∈ [0, ~5], 越小越好
      [4] stable_val      = 1.0 if stable else 0.0    二值信号
      [5] Kp_norm         = Kp / KP_MAX               ∈ [0, 1]
      [6] Ki_norm         = Ki / KI_MAX               ∈ [0, 1]
      [7] Kd_norm         = Kd / KD_MAX               ∈ [0, 1]

    归一化目的: 神经网络对输入尺度敏感, 统一到 [0,1] 附近利于梯度稳定传播。
    """
    ov = metrics['overshoot']
    dr = metrics.get('decay_rate')
    se = metrics['steady_state_error']
    iae = metrics['iae']

    # [0] 超调归一化 (上限截断防异常值炸梯度)
    ov_norm = min(ov / 100.0, 5.0)

    # [1] 衰减率: None → 1.0 (无振荡=最佳阻尼), 否则取实际值
    dr_val = dr if dr is not None else 1.0

    # [2] 稳态误差归一化
    se_norm = min(se / 100.0, 1.0)

    # [3] IAE 归一化 (5 作为一个 "差" 的基准, 截断防极端值)
    iae_norm = min(iae / 5.0, 5.0)

    # [4] 稳定性标志
    stable_val = 1.0 if stable else 0.0

    # [5-7] PID 参数归一化 (让网络知道当前参数背景)
    kp_norm = Kp / KP_MAX
    ki_norm = Ki / KI_MAX
    kd_norm = Kd / KD_MAX

    return np.array([
        ov_norm, dr_val, se_norm, iae_norm, stable_val,
        kp_norm, ki_norm, kd_norm,
    ], dtype=np.float32)


# ============================================================
# Q-Network (MLP)
# ============================================================

class QNetwork(nn.Module):
    """
    Deep Q-Network: 5 层全连接 MLP。

    输入: 状态向量 (STATE_DIM=8)
    输出: 各动作的 Q 值 (N_ACTIONS=7)

    架构:
      Linear(8, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 7)

    梯度流动:
      前向: 状态张量 [B, 8] → 逐层矩阵乘 + ReLU → Q 值 [B, 7]
      反向: MSELoss 梯度从 Q(s,a) 经 gather 操作回传到各层权重
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=N_ACTIONS, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        """
        前向传播。

        参数:
          state: [batch_size, state_dim] 归一化后的状态张量

        返回:
          q_values: [batch_size, action_dim] 每个动作的预估 Q 值
        """
        return self.net(state)


# ============================================================
# 经验回放缓冲区
# ============================================================

class ReplayBuffer:
    """
    经验回放缓冲区 (FIFO, 定长 deque)。

    打破样本时序相关性: 不从连续仿真步中顺序学习,
    而是随机采样历史经验, 满足 DQN 的 i.i.d. 假设。
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存入一条 transition: (s, a, r, s', done)"""
        self.buffer.append((
            state.astype(np.float32),
            action,
            reward,
            next_state.astype(np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        """
        随机采样一个 mini-batch。

        返回:
          states:      [B, state_dim]  FloatTensor
          actions:     [B]             LongTensor (索引)
          rewards:     [B]             FloatTensor
          next_states: [B, state_dim]  FloatTensor
          dones:       [B]             FloatTensor
        """
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            torch.FloatTensor(np.stack(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.stack(s_)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN PID 整定器
# ============================================================

class DQNTuner:
    """
    Double DQN PID 参数整定器。

    DQN 核心机制:
      - 策略网络 (policy_net): 用于选择动作 + 梯度更新
      - 目标网络 (target_net):  用于计算 TD target, 定期软更新
      - 软更新: θ_target ← τ·θ_policy + (1-τ)·θ_target
        避免 Q 值估计的目标震荡, 提升训练稳定性

    参数:
        K, T, tau: 被控对象 G(s) = K·e^(-τs)/(Ts+1)
        setpoint, sim_time: 仿真设置
        hidden_dim: Q 网络隐藏层维度
        lr: Adam 学习率
        gamma: 折扣因子
        tau_soft: 目标网络软更新系数
        buffer_capacity: 回放缓冲区容量
        batch_size: 每步采样数
        eps_start/end/decay: ε-greedy 探索策略
    """

    def __init__(self, K=1.0, T=1.0, tau=0.0,
                 setpoint=1.0, sim_time=25.0, dt=0.01,
                 hidden_dim=64, lr=3e-4, gamma=0.9, tau_soft=0.005,
                 buffer_capacity=10000, batch_size=64,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.98):
        self.K = K
        self.T = T
        self.tau = tau
        self.setpoint = setpoint
        self.sim_time = sim_time
        self.dt = dt
        self.gamma = gamma
        self.tau_soft = tau_soft
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 双网络
        self.policy_net = QNetwork(STATE_DIM, N_ACTIONS, hidden_dim).to(self.device)
        self.target_net = QNetwork(STATE_DIM, N_ACTIONS, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 初始化同步
        self.target_net.eval()  # 目标网络不训练, 仅前向

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 经验回放
        self.buffer = ReplayBuffer(buffer_capacity)

        # 训练记录
        self.best_params = None
        self.best_reward = -float('inf')
        self.best_metrics = None
        self.history = []        # 每 episode 的最佳奖励 (向后兼容)
        self.episode_log = []    # 每 episode 的详细记录 (供 CSV 导出)
        self._ep_losses = []     # 当前 episode 的 loss 列表

        # 步数计数器 (用于打印训练统计)
        self._train_steps = 0

    def _random_params(self):
        """
        稳定性感知的随机初始 PID 参数 (与 rl_tuner 一致)。

        对 FOPDT 系统基于粗略稳定边界约束 Kp 上限,
        避免 Agent 在必然发散的参数区域浪费探索。
        """
        if self.tau > 1e-6:
            kp_crit_est = self.T / (self.K * self.tau * 1.2)
            kp_max_init = min(KP_MAX * 0.25, kp_crit_est * 0.6)
            kp_min_init = KP_MIN
        else:
            kp_max_init = 8.0
            kp_min_init = 0.3

        Kp = 10 ** np.random.uniform(
            np.log10(max(KP_MIN, kp_min_init)),
            np.log10(max(kp_max_init, kp_min_init * 2)))
        Ki = np.random.uniform(KI_MIN, min(2.0, Kp * 0.6))
        Kd = np.random.uniform(KD_MIN, min(1.0, Kp * 0.25))
        return round(Kp, 4), round(Ki, 4), round(Kd, 4)

    def choose_action(self, state, epsilon):
        """
        ε-greedy 动作选择 (基于策略网络)。

        梯度说明:
          选择动作时不计算梯度 (torch.no_grad),
          仅在前向传播获取 Q 值, 不做反向传播。
        """
        if random.random() < epsilon:
            return random.randint(0, N_ACTIONS - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)  # [1, 7]
        return int(q_values.argmax(dim=1).item())

    def update_policy(self):
        """
        单步 DQN 更新: 从回放缓冲区采样 → 计算 TD 误差 → 梯度下降。

        DQN 损失函数:
          L(θ) = E[( r + γ·(1-done)·max_a' Q_target(s',a') - Q_policy(s,a) )²]

        梯度流动路径:
          loss → q_value (gather) → policy_net(s) → ... → Linear 权重
          目标侧的 Q_target(s') 被 torch.no_grad() 阻断, 不参与梯度计算。

        返回:
          loss 值 (float) 或 None (缓冲区不足)
        """
        if len(self.buffer) < self.batch_size:
            return None

        # 采样
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ── 当前 Q 值: Q_policy(s, a) ──
        # gather: 从 [B, 7] 中按 action 索引取出对应的 Q 值 → [B]
        q_values = self.policy_net(states)          # [B, 7]
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        # ── 目标 Q 值: r + γ·(1-done)·max_a' Q_target(s', a') ──
        with torch.no_grad():
            next_q_values = self.target_net(next_states)  # [B, 7]
            max_next_q = next_q_values.max(dim=1)[0]      # [B]
            target_q = rewards + self.gamma * (1.0 - dones) * max_next_q  # [B]

        # ── 损失 & 反向传播 ──
        loss = self.loss_fn(q_value, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪: 防止异常样本导致梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── 目标网络软更新 ──
        # θ_target ← τ·θ_policy + (1-τ)·θ_target
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(),
                              self.policy_net.parameters()):
                tp.data.copy_(self.tau_soft * pp.data +
                              (1.0 - self.tau_soft) * tp.data)

        self._train_steps += 1
        return loss.item()

    def train(self, episodes=500, steps_per_episode=8,
              progress_callback=None):
        """
        DQN 训练主循环。

        双层结构:
          外层 Episode:  随机初始化 PID → 仿真 → 连续状态
          内层 Step:     ε-greedy 选动作 → 改参数 → 仿真 → 存经验 → 网络更新

        参数:
            episodes:          训练轮数
            steps_per_episode: 每轮最大调整步数
            progress_callback: callback(ep, total, eps, best_r, best_params)
        """
        eps = self.eps_start

        for ep in range(episodes):
            # ── Episode 初始化 ──
            Kp, Ki, Kd = self._random_params()
            t, sp, pv, co, err = _sim(
                self.K, self.T, self.tau,
                Kp, Ki, Kd, self.setpoint, self.sim_time, self.dt)
            metrics = compute_metrics(t, sp, pv, err)
            stable = np.isfinite(pv[-1]) and abs(pv[-1]) < 1e6
            state = get_state(metrics, stable, Kp, Ki, Kd,
                              self.setpoint, self.sim_time)

            episode_rewards = []
            episode_stable = stable   # 初始仿真是否收敛
            episode_terminal = False
            self._ep_losses = []

            # ── Episode 内步循环 ──
            for step in range(steps_per_episode):
                # ① 选择动作
                action = self.choose_action(state, eps)

                # ② 执行动作 → 新参数 (带限幅保护)
                new_Kp, new_Ki, new_Kd = apply_action(Kp, Ki, Kd, action)

                # ③ 运行完整仿真 → 新状态 + 奖励
                t, sp, pv, co, err = _sim(
                    self.K, self.T, self.tau,
                    new_Kp, new_Ki, new_Kd,
                    self.setpoint, self.sim_time, self.dt)
                new_metrics = compute_metrics(t, sp, pv, err)
                new_stable = np.isfinite(pv[-1]) and abs(pv[-1]) < 1e6
                next_state = get_state(new_metrics, new_stable,
                                       new_Kp, new_Ki, new_Kd,
                                       self.setpoint, self.sim_time)
                reward = compute_reward(new_metrics, new_stable, self.setpoint)
                done = is_terminal_state(new_metrics, new_stable)

                # ④ 存入经验回放
                self.buffer.push(state, action, reward, next_state, done)

                # ⑤ 网络更新
                loss = self.update_policy()
                if loss is not None:
                    self._ep_losses.append(loss)

                # ⑥ 跟踪最佳参数 (全局)
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_params = (new_Kp, new_Ki, new_Kd)
                    self.best_metrics = new_metrics

                episode_rewards.append(reward)

                # ⑦ 状态转移
                state = next_state
                Kp, Ki, Kd = new_Kp, new_Ki, new_Kd

                if done:
                    episode_terminal = True
                    break

            # ── Episode 收尾 ──
            episode_steps = step + 1  # step 从 0 开始
            best_ep_r = max(episode_rewards) if episode_rewards else -float('inf')
            avg_loss = (sum(self._ep_losses) / len(self._ep_losses)
                        if self._ep_losses else 0.0)
            self.history.append(best_ep_r)

            # 记录每轮详细日志 (供 CSV 导出)
            bp = self.best_params or (Kp, Ki, Kd)
            self.episode_log.append({
                'episode':      ep + 1,
                'epsilon':      round(eps, 4),
                'Kp':           bp[0],
                'Ki':           bp[1],
                'Kd':           bp[2],
                'reward':       round(best_ep_r, 4),
                'avg_loss':     round(avg_loss, 6),
                'steps':        episode_steps,
                'is_terminal':  episode_terminal,
                'stable':       episode_stable,
            })

            # ε 指数衰减
            eps = max(self.eps_end, eps * self.eps_decay)

            # 进度回调
            if progress_callback is not None:
                bp = self.best_params or (0, 0, 0)
                progress_callback(ep + 1, episodes, eps,
                                  self.best_reward, bp)

        return self.get_result()

    def get_result(self):
        """返回最佳整定结果 (与 tuning_methods 兼容的 dict 格式)"""
        if self.best_params is None:
            return None

        Kp, Ki, Kd = self.best_params
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
            'method_name': 'DQN强化学习整定',
            'metrics': final_metrics,
            'best_reward': round(self.best_reward, 2),
            'episodes': len(self.history),
            'history': self.history,
        }

    def get_network_state(self):
        """返回策略网络的状态字典 (供保存/加载)"""
        return {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def export_training_csv(self, path):
        """
        导出训练日志 CSV。

        列: episode, epsilon, Kp, Ki, Kd, reward, avg_loss, steps, is_terminal, stable
        """
        if not self.episode_log:
            return False

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['# DQN PID 整定 — 训练日志'])
            w.writerow(['# 时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            w.writerow(['# 对象', f'K={self.K}, T={self.T}, tau={self.tau}'])
            w.writerow(['# 设定值', str(self.setpoint)])
            w.writerow(['# 最佳参数',
                        f'Kp={self.best_params[0]:.4f} Ki={self.best_params[1]:.4f} Kd={self.best_params[2]:.4f}'
                        if self.best_params else 'N/A'])
            w.writerow(['# 最佳奖励', str(round(self.best_reward, 4))])
            w.writerow(['# 设备', str(self.device)])
            w.writerow(['#'])
            w.writerow(['episode', 'epsilon', 'Kp', 'Ki', 'Kd', 'reward',
                        'avg_loss', 'steps', 'is_terminal', 'stable'])
            for row in self.episode_log:
                w.writerow([
                    row['episode'], row['epsilon'],
                    f'{row["Kp"]:.4f}', f'{row["Ki"]:.4f}', f'{row["Kd"]:.4f}',
                    f'{row["reward"]:.4f}', f'{row["avg_loss"]:.6f}',
                    row['steps'], int(row['is_terminal']), int(row['stable']),
                ])
        return True


# ============================================================
# 便利函数 (供 GUI 调用)
# ============================================================

def dqn_tune(K, T, tau, setpoint=1.0, sim_time=25.0,
             episodes=500, steps_per_episode=8,
             progress_callback=None):
    """
    DQN PID 整定的便捷入口。

    参数与 rl_tune() 对齐, 可直接替换调用。
    """
    tuner = DQNTuner(
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
            'saved_results', 'dqn_training')
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(
            save_dir,
            f'dqn_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        tuner.export_training_csv(csv_path)
        result['training_csv'] = csv_path

    return result


# ============================================================
# 测试主函数
# ============================================================

if __name__ == '__main__':
    import time

    print("=" * 60)
    print("  DQN PID 整定 — 独立测试 (PyTorch)")
    print("=" * 60)
    print(f"  设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()

    # 测试对象: 有迟延的一阶系统 (与 rl_tuner 相同, 便于对比)
    test_K = 1.0
    test_T = 2.0
    test_tau = 0.5

    print(f"被控对象: K={test_K}, T={test_T}, τ={test_tau}")
    print(f"训练中 (500 轮, 每轮最多 8 步)...\n")

    def show_progress(ep, total, eps, best_r, best_params):
        if ep % 25 == 0 or ep == total:
            n = ep * 30 // total
            bar = '#' * n + '-' * (30 - n)
            bp = best_params or (0, 0, 0)
            print(f"  [{bar}] {ep:3d}/{total}  eps={eps:.3f}  "
                  f"R={best_r:+.2f}  Kp={bp[0]:.2f} Ki={bp[1]:.2f} Kd={bp[2]:.2f}")

    t0 = time.time()
    result = dqn_tune(test_K, test_T, test_tau,
                      setpoint=1.0, sim_time=30.0,
                      episodes=500, progress_callback=show_progress)
    elapsed = time.time() - t0

    if result:
        m = result['metrics']
        print(f"\n{'='*60}")
        print(f"  整定结果 (耗时 {elapsed:.1f}s)")
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

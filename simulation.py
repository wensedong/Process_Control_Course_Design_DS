#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真引擎模块
- 一阶惯性 + 纯迟延对象: G(s) = K * e^(-tau*s) / (T*s + 1)
- PID控制器 (对PV微分, 抗积分饱和)
- 闭环仿真 (带发散检测 + 步数上限)
"""

import numpy as np
from collections import deque
from dataclasses import dataclass


# ============================================================
# 配置 dataclass
# ============================================================

@dataclass
class PlantConfig:
    """被控对象配置"""
    K: float = 1.0       # 静态增益
    T: float = 1.0       # 时间常数 (s)
    tau: float = 0.0     # 纯迟延时间 (s)


@dataclass
class PIDConfig:
    """PID控制器配置"""
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    output_min: float = -10.0
    output_max: float = 10.0


@dataclass
class SimulationConfig:
    """仿真运行配置"""
    setpoint: float = 1.0
    sim_time: float = 25.0
    dt: float = 0.01
    max_steps: int = 6000


# ============================================================
# 被控对象
# ============================================================

class Plant:
    """一阶惯性加纯迟延对象 G(s) = K * e^(-tau*s) / (T*s + 1)"""

    def __init__(self, K=1.0, T=1.0, tau=0.0, dt=0.01):
        self.K = K
        self.T = max(T, 1e-6)
        self.tau = tau
        self.dt = dt
        self.y = 0.0
        self._init_delay()

    def _init_delay(self):
        if self.tau > 0:
            self._delay_steps = max(1, int(self.tau / self.dt))
            self._delay_buf = deque([0.0] * self._delay_steps, maxlen=self._delay_steps)
        else:
            self._delay_steps = 0
            self._delay_buf = deque([0.0], maxlen=1)

    def reset(self, y0=0.0):
        self.y = y0
        self._init_delay()

    def step(self, u):
        """单步仿真: 返回当前输出 y"""
        if self._delay_steps > 0:
            self._delay_buf.append(u)
            u_delayed = self._delay_buf[0]
        else:
            u_delayed = u
        self.y += self.dt * (self.K * u_delayed - self.y) / self.T
        return self.y

    @property
    def transfer_function(self):
        """返回传递函数字符串"""
        def fmt(v):
            if abs(v - round(v, 1)) < 1e-9:
                return f"{v:.1f}"
            return f"{v:.2f}"
        parts = []
        if abs(self.K - 1.0) > 1e-9:
            parts.append(fmt(self.K))
        if self.tau > 0:
            parts.append(f"e^(-{fmt(self.tau)}s)")
        if abs(self.T - 1.0) > 1e-9 or (abs(self.K - 1.0) < 1e-9 and self.tau == 0):
            parts.append(f"1/({fmt(self.T)}s+1)")
        elif self.tau == 0 and abs(self.K - 1.0) < 1e-9:
            parts.append(f"1/({fmt(self.T)}s+1)")
        return " · ".join(parts) if parts else "1/(1.0s+1)"


# ============================================================
# PID 控制器 (抗积分饱和)
# ============================================================

class PIDController:
    """离散PID控制器 (对PV微分, 条件积分抗饱和)"""

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, dt=0.01,
                 output_min=-1e6, output_max=1e6):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_min = output_min
        self.output_max = output_max
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_pv = 0.0
        self.first_run = True

    def compute(self, setpoint, pv):
        """计算控制量输出 (带抗积分饱和)"""
        error = setpoint - pv

        # 比例
        P = self.Kp * error

        # 积分 (条件积分 — 抗饱和)
        candidate_integral = self.integral + error * self.dt
        raw_output = (P + self.Ki * candidate_integral
                      + self.Kd * (0.0 if self.first_run
                                   else (-(pv - self.prev_pv) / self.dt)))

        output = max(self.output_min, min(self.output_max, raw_output))

        # 仅在以下情况接受积分增量:
        #  1) 输出未饱和 (output == raw_output)
        #  2) 饱和但误差方向指向脱离饱和
        if output == raw_output or (
            output == self.output_max and error < 0
        ) or (
            output == self.output_min and error > 0
        ):
            self.integral = candidate_integral

        # 微分 (对PV微分, 避免微分冲击)
        if self.first_run:
            D = 0.0
            self.first_run = False
        else:
            D = self.Kd * (-(pv - self.prev_pv) / self.dt)

        self.prev_pv = pv
        output = P + self.Ki * self.integral + D
        return max(self.output_min, min(self.output_max, output))


# ============================================================
# 闭环仿真
# ============================================================

def simulate(plant, pid, sim):
    """
    闭环PID仿真 (使用 dataclass 配置对象)。

    参数:
        plant: PlantConfig — 对象参数 (K, T, tau)
        pid:   PIDConfig   — 控制器参数 (Kp, Ki, Kd, output_min, output_max)
        sim:   SimulationConfig — 仿真设置 (setpoint, sim_time, dt, max_steps)

    返回:
        dict: {'t', 'sp', 'pv', 'co', 'error', 'stable'}
              stable = True/False 表示仿真是否正常收敛
    """
    plant_obj = Plant(K=plant.K, T=plant.T, tau=plant.tau, dt=sim.dt)
    pid_obj = PIDController(Kp=pid.Kp, Ki=pid.Ki, Kd=pid.Kd, dt=sim.dt,
                            output_min=pid.output_min, output_max=pid.output_max)

    n_steps = min(int(sim.sim_time / sim.dt), sim.max_steps)
    t = np.linspace(0, sim.sim_time, n_steps)
    sp = np.full(n_steps, sim.setpoint)
    pv = np.zeros(n_steps)
    co = np.zeros(n_steps)
    error = np.zeros(n_steps)

    pv[0] = 0.0
    error[0] = sim.setpoint
    stable = True

    for i in range(1, n_steps):
        co[i] = pid_obj.compute(sp[i], pv[i - 1])
        pv[i] = plant_obj.step(co[i])
        error[i] = sp[i] - pv[i]

        # 发散检测
        if not np.isfinite(pv[i]) or abs(pv[i]) > 1_000_000:
            # 截断到发散点
            t = t[:i + 1]
            sp = sp[:i + 1]
            pv = pv[:i + 1]
            co = co[:i + 1]
            error = error[:i + 1]
            stable = False
            break

    return {'t': t, 'sp': sp, 'pv': pv, 'co': co, 'error': error, 'stable': stable}


# ============================================================
# 峰值检测 (供 tuning_methods 使用)
# ============================================================

def find_peaks(y):
    """简易峰值检测 (局部极大值), 返回索引列表"""
    peaks = []
    n = len(y)
    if n < 3:
        return peaks
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            peaks.append(i)
    return peaks

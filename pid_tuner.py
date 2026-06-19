#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PID参数手动整定交互程序 v2.0
基于一阶惯性环节 G(s) = K/(Ts+1) 的单回路控制系统
功能:
  - 交互式调整PID参数，实时观察阶跃响应曲线
  - 性能指标: 超调量, 稳态误差, 调节时间, 上升时间, IAE, ISE, 衰减比, 衰减率
  - 响应曲线动画播放
  - 曲线图和数据保存(CSV/PNG/SVG)
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# 配置中文字体
import matplotlib.font_manager as fm
_cn_fonts = [f for f in fm.fontManager.ttflist if 'Microsoft YaHei' in f.name]
if _cn_fonts:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] + matplotlib.rcParams['font.sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import os
from datetime import datetime


class FirstOrderPlant:
    """一阶惯性环节模型 G(s) = K / (T*s + 1)"""

    def __init__(self, K=1.0, T=1.0):
        self.K = K   # 增益
        self.T = T   # 时间常数

    def step_response(self, u, dt):
        """使用欧拉法离散化仿真 dy/dt = (K*u - y) / T"""
        return lambda y, u_in: y + dt * (self.K * u_in - y) / self.T


class PIDController:
    """PID控制器 (对PV微分, 抗积分饱和)"""

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, dt=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_pv = 0.0
        self.first_run = True

    def compute(self, setpoint, pv):
        """计算PID输出"""
        error = setpoint - pv

        # 比例项
        P = self.Kp * error

        # 积分项 (带抗积分饱和)
        self.integral += error * self.dt
        I = self.Ki * self.integral

        # 微分项 (对PV微分, 避免微分冲击)
        if self.first_run:
            D = 0.0
            self.first_run = False
        else:
            dPV = -(pv - self.prev_pv) / self.dt
            D = self.Kd * dPV

        self.prev_error = error
        self.prev_pv = pv

        return P + I + D


def simulate(plant_K, plant_T, Kp, Ki, Kd, setpoint=1.0, sim_time=20.0, dt=0.01):
    """
    运行闭环PID控制仿真
    返回: t, sp, pv, co, error 数组
    """
    plant = FirstOrderPlant(K=plant_K, T=plant_T)
    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt)

    n_steps = int(sim_time / dt)
    t = np.linspace(0, sim_time, n_steps)
    sp = np.ones(n_steps) * setpoint
    pv = np.zeros(n_steps)
    co = np.zeros(n_steps)
    error = np.zeros(n_steps)

    pv[0] = 0.0

    for i in range(1, n_steps):
        error[i] = sp[i] - pv[i-1]
        co[i] = pid.compute(sp[i], pv[i-1])
        pv[i] = pv[i-1] + dt * (plant.K * co[i] - pv[i-1]) / plant.T

    # 补全error[0]
    error[0] = sp[0] - pv[0]

    return t, sp, pv, co, error


# ---------------------------------------------------------------------------
# 性能指标计算 (scipy-free — uses simple peak detection)
# ---------------------------------------------------------------------------

def _find_peaks_simple(y):
    """简单的局部极大值检测 (不需要scipy). 返回峰值的索引列表."""
    peaks = []
    n = len(y)
    if n < 3:
        return peaks
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            peaks.append(i)
    return peaks


def compute_metrics(t, sp, pv, error, tolerance=0.02):
    """
    计算全部性能指标:
      - 上升时间 tr, 超调量 σ%, 调节时间 ts, 稳态误差
      - IAE, ISE
      - 衰减比, 衰减率 (无超调则返回 None)
    """
    setpoint_val = sp[0]
    pv_final = pv[-1]
    dt = t[1] - t[0] if len(t) > 1 else 0.01

    # ---- 稳态误差 ----
    steady_state_error = abs(setpoint_val - pv_final) / setpoint_val * 100 if setpoint_val != 0 else 0

    # ---- 上升时间 (10%→90%稳态值) ----
    y10 = 0.1 * pv_final
    y90 = 0.9 * pv_final
    t10 = t90_idx = None
    t_rise = None
    for i in range(len(t)):
        if t10 is None and pv[i] >= y10:
            t10 = t[i]
        if t10 is not None and pv[i] >= y90:
            t90_idx = t[i]
            break
    if t10 is not None and t90_idx is not None:
        t_rise = t90_idx - t10

    # ---- 超调量 ----
    pv_max = max(pv)
    if pv_final != 0:
        overshoot = max(0, (pv_max - pv_final) / pv_final * 100)
    else:
        overshoot = 0

    # ---- 调节时间 (进入±tolerance后不再超出) ----
    settling_idx = None
    upper = pv_final * (1 + tolerance)
    lower = pv_final * (1 - tolerance)
    for i in range(len(t) - 1, -1, -1):
        if pv[i] > upper or pv[i] < lower:
            settling_idx = i
            break
    if settling_idx is not None and settling_idx < len(t) - 1:
        t_settling = t[settling_idx + 1]
    else:
        t_settling = 0.0

    # ---- IAE & ISE ----
    iae = np.sum(np.abs(error)) * dt
    ise = np.sum(error ** 2) * dt

    # ---- 衰减比 & 衰减率 ----
    # 找PV曲线上超过稳态值的所有峰值
    peaks_idx = _find_peaks_simple(pv)
    # 只保留 > pv_final 的峰（真正的超调峰），且时间要在稳态值首次达到之后
    overshoot_peaks = []
    # 先找出首次进入稳态附近的时间点
    first_settle_idx = None
    for i in range(len(t)):
        if pv[i] >= pv_final * 0.99:
            first_settle_idx = i
            break
    if first_settle_idx is not None:
        for idx in peaks_idx:
            if idx > first_settle_idx and pv[idx] > pv_final:
                overshoot_peaks.append(pv[idx])

    if len(overshoot_peaks) >= 2:
        a1 = overshoot_peaks[0] - pv_final
        a2 = overshoot_peaks[1] - pv_final
        if a1 > 1e-9:
            decay_ratio = a2 / a1
            decay_rate = 1.0 - decay_ratio
        else:
            decay_ratio = None
            decay_rate = None
    else:
        decay_ratio = None
        decay_rate = None

    return {
        'rise_time': t_rise,
        'overshoot': overshoot,
        'settling_time': t_settling,
        'steady_state_error': steady_state_error,
        'final_value': pv_final,
        'iae': iae,
        'ise': ise,
        'decay_ratio': decay_ratio,
        'decay_rate': decay_rate,
    }


# ---------------------------------------------------------------------------
# 主应用
# ---------------------------------------------------------------------------

class PidTunerApp:
    """PID手动整定GUI应用 v2.0"""

    def __init__(self, root):
        self.root = root
        self.root.title("PID参数手动整定 v2.0 - 单回路控制系统")
        self.root.geometry("1200x780")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ---- 默认参数 ----
        self.plant_K    = tk.DoubleVar(value=1.0)
        self.plant_T    = tk.DoubleVar(value=1.0)
        self.Kp         = tk.DoubleVar(value=1.0)
        self.Ki         = tk.DoubleVar(value=0.0)
        self.Kd         = tk.DoubleVar(value=0.0)
        self.setpoint   = tk.DoubleVar(value=1.0)
        self.sim_time   = tk.DoubleVar(value=20.0)

        self._updating = False

        # ---- 仿真数据缓存 ----
        self._t = self._sp = self._pv = self._co = self._error = None
        self._metrics = None

        # ---- 动画状态 ----
        self._animating = False
        self._anim_job = None
        self._anim_frame = 0
        self._anim_speed = tk.IntVar(value=5)   # 1-10, 倍速

        self.build_ui()
        self.update_plot()

    # ==================================================================
    # 界面构建
    # ==================================================================

    def build_ui(self):
        # -------- 顶部控制面板 --------
        control_frame = ttk.LabelFrame(self.root, text="参数设置", padding=8)
        control_frame.pack(fill=tk.X, padx=8, pady=(8, 2))

        # 被控对象
        plant_frame = ttk.LabelFrame(control_frame, text="被控对象 G(s)=K/(Ts+1)", padding=5)
        plant_frame.grid(row=0, column=0, padx=6, pady=3, sticky='n')

        ttk.Label(plant_frame, text="增益 K:").grid(row=0, column=0, sticky='w')
        ttk.Scale(plant_frame, from_=0.1, to=5.0, variable=self.plant_K,
                  orient=tk.HORIZONTAL, length=140, command=lambda e: self.on_param_change()).grid(row=0, column=1)
        ttk.Entry(plant_frame, textvariable=self.plant_K, width=5).grid(row=0, column=2)

        ttk.Label(plant_frame, text="时间常数 T:").grid(row=1, column=0, sticky='w')
        ttk.Scale(plant_frame, from_=0.1, to=10.0, variable=self.plant_T,
                  orient=tk.HORIZONTAL, length=140, command=lambda e: self.on_param_change()).grid(row=1, column=1)
        ttk.Entry(plant_frame, textvariable=self.plant_T, width=5).grid(row=1, column=2)

        # PID参数
        pid_frame = ttk.LabelFrame(control_frame, text="PID控制器参数", padding=5)
        pid_frame.grid(row=0, column=1, padx=6, pady=3, sticky='n')

        ttk.Label(pid_frame, text="Kp:").grid(row=0, column=0, sticky='w')
        ttk.Scale(pid_frame, from_=0.0, to=10.0, variable=self.Kp,
                  orient=tk.HORIZONTAL, length=180, command=lambda e: self.on_param_change()).grid(row=0, column=1)
        ttk.Entry(pid_frame, textvariable=self.Kp, width=5).grid(row=0, column=2)

        ttk.Label(pid_frame, text="Ki:").grid(row=1, column=0, sticky='w')
        ttk.Scale(pid_frame, from_=0.0, to=5.0, variable=self.Ki,
                  orient=tk.HORIZONTAL, length=180, command=lambda e: self.on_param_change()).grid(row=1, column=1)
        ttk.Entry(pid_frame, textvariable=self.Ki, width=5).grid(row=1, column=2)

        ttk.Label(pid_frame, text="Kd:").grid(row=2, column=0, sticky='w')
        ttk.Scale(pid_frame, from_=0.0, to=5.0, variable=self.Kd,
                  orient=tk.HORIZONTAL, length=180, command=lambda e: self.on_param_change()).grid(row=2, column=1)
        ttk.Entry(pid_frame, textvariable=self.Kd, width=5).grid(row=2, column=2)

        # 仿真设置
        sim_frame = ttk.LabelFrame(control_frame, text="仿真设置", padding=5)
        sim_frame.grid(row=0, column=2, padx=6, pady=3, sticky='n')
        ttk.Label(sim_frame, text="设定值 SP:").grid(row=0, column=0, sticky='w')
        ttk.Entry(sim_frame, textvariable=self.setpoint, width=7).grid(row=0, column=1)
        ttk.Label(sim_frame, text="仿真时间 (s):").grid(row=1, column=0, sticky='w')
        ttk.Entry(sim_frame, textvariable=self.sim_time, width=7).grid(row=1, column=1)
        ttk.Button(sim_frame, text="刷新曲线", command=self.update_plot).grid(row=2, column=0, columnspan=2, pady=6)

        # 预设
        preset_frame = ttk.LabelFrame(control_frame, text="预设", padding=5)
        preset_frame.grid(row=0, column=3, padx=6, pady=3, sticky='n')
        for txt, p in [("P", 'P'), ("PI", 'PI'), ("PD", 'PD'), ("PID", 'PID'), ("重置", 'reset')]:
            ttk.Button(preset_frame, text=txt, command=lambda v=p: self.set_preset(v)).pack(fill=tk.X, pady=1)

        # 输入框联动
        for var in [self.plant_K, self.plant_T, self.Kp, self.Ki, self.Kd,
                    self.setpoint, self.sim_time]:
            var.trace_add('write', lambda *args: self.on_entry_change())

        # ============ 性能指标面板 (紧贴控制栏下方) ============
        self.metrics_frame = ttk.LabelFrame(self.root, text="性能指标", padding=6)
        self.metrics_frame.pack(fill=tk.X, padx=8, pady=(2, 2))

        self.metric_labels = {}

        # 第一行: 时域指标 + 衰减指标
        row0_metrics = [
            ('rise_time',           '上升时间 tr'),
            ('overshoot',           '超调量'),
            ('settling_time',       '调节时间 ts'),
            ('steady_state_error',  '稳态误差'),
            ('final_value',         '稳态值'),
            ('decay_ratio',         '衰减比'),
            ('decay_rate',          '衰减率'),
        ]
        for i, (key, label) in enumerate(row0_metrics):
            ttk.Label(self.metrics_frame, text=label + ":").grid(row=0, column=i*2, padx=3, sticky='e')
            lbl = ttk.Label(self.metrics_frame, text="--", font=('Consolas', 10, 'bold'), foreground='blue')
            lbl.grid(row=0, column=i*2+1, padx=3, sticky='w')
            self.metric_labels[key] = lbl

        # 第二行: 积分指标
        row1_metrics = [
            ('iae',  'IAE'),
            ('ise',  'ISE'),
        ]
        for i, (key, label) in enumerate(row1_metrics):
            ttk.Label(self.metrics_frame, text=label + ":").grid(row=1, column=i*2, padx=3, sticky='e')
            lbl = ttk.Label(self.metrics_frame, text="--", font=('Consolas', 10, 'bold'), foreground='blue')
            lbl.grid(row=1, column=i*2+1, padx=3, sticky='w')
            self.metric_labels[key] = lbl

        # -------- 图表区域 --------
        self.fig = Figure(figsize=(10, 4.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=2)

        # Matplotlib工具栏 + 动画 + 保存 (同一行)
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill=tk.X, padx=8, pady=(2, 4))

        # 工具栏
        self.toolbar = NavigationToolbar2Tk(self.canvas, action_frame)
        self.toolbar.update()

        # 动画控制 (放在工具栏右侧)
        anim_group = ttk.LabelFrame(action_frame, text="动画播放", padding=3)
        anim_group.pack(side=tk.LEFT, padx=(20, 4))

        self.btn_play = ttk.Button(anim_group, text="▶ 播放", command=self.toggle_animation)
        self.btn_play.pack(side=tk.LEFT, padx=2)

        ttk.Label(anim_group, text="速度:").pack(side=tk.LEFT, padx=(6, 1))
        ttk.Scale(anim_group, from_=1, to=10, variable=self._anim_speed,
                  orient=tk.HORIZONTAL, length=60).pack(side=tk.LEFT, padx=1)
        ttk.Label(anim_group, textvariable=self._anim_speed, width=2).pack(side=tk.LEFT)
        ttk.Label(anim_group, text="x").pack(side=tk.LEFT)
        self._anim_progress = ttk.Label(anim_group, text="0%", width=4)
        self._anim_progress.pack(side=tk.LEFT, padx=3)

        # 保存功能
        save_group = ttk.LabelFrame(action_frame, text="保存", padding=3)
        save_group.pack(side=tk.RIGHT, padx=4)

        ttk.Button(save_group, text="保存图像 (.png)", command=self.save_figure_png).pack(side=tk.LEFT, padx=2)
        ttk.Button(save_group, text="保存图像 (.svg)", command=self.save_figure_svg).pack(side=tk.LEFT, padx=2)
        ttk.Button(save_group, text="导出数据 (.csv)", command=self.save_data_csv).pack(side=tk.LEFT, padx=2)

    # ==================================================================
    # 预设 & 参数变更
    # ==================================================================

    def _round_pid_params(self):
        """将PID参数保留到小数点后两位"""
        for var in [self.Kp, self.Ki, self.Kd, self.plant_K, self.plant_T]:
            var.set(round(var.get(), 2))

    def set_preset(self, preset_type):
        presets = {
            'P':     {'Kp': 2.00, 'Ki': 0.00, 'Kd': 0.00},
            'PI':    {'Kp': 1.50, 'Ki': 0.50, 'Kd': 0.00},
            'PD':    {'Kp': 2.00, 'Ki': 0.00, 'Kd': 0.50},
            'PID':   {'Kp': 2.00, 'Ki': 1.00, 'Kd': 0.50},
            'reset': {'Kp': 1.00, 'Ki': 0.00, 'Kd': 0.00},
        }
        if preset_type in presets:
            p = presets[preset_type]
            self.Kp.set(p['Kp'])
            self.Ki.set(p['Ki'])
            self.Kd.set(p['Kd'])
            self.stop_animation()
            self.update_plot()

    def on_param_change(self):
        """滑块变动时实时取整并刷新"""
        if self._updating:
            return
        self._updating = True       # 阻止 trace 回调重复调度
        self._round_pid_params()
        self.update_plot()
        self._updating = False

    def on_entry_change(self):
        if self._updating:
            return
        self._updating = True
        self.root.after(300, self._delayed_update)

    def _delayed_update(self):
        self._round_pid_params()
        self.update_plot()
        self._updating = False

    # ==================================================================
    # 仿真 + 绘图
    # ==================================================================

    def update_plot(self):
        """运行仿真并绘制曲线"""
        try:
            K  = self.plant_K.get()
            T  = self.plant_T.get()
            Kp = self.Kp.get()
            Ki = self.Ki.get()
            Kd = self.Kd.get()
            sp_val = self.setpoint.get()
            sim_t  = self.sim_time.get()

            self._t, self._sp, self._pv, self._co, self._error = \
                simulate(K, T, Kp, Ki, Kd, sp_val, sim_t, dt=0.01)
            self._metrics = compute_metrics(self._t, self._sp, self._pv, self._error)

            self._draw_static_plot()
            self._update_metrics_display()

        except Exception as e:
            print(f"更新出错: {e}")

    def _draw_static_plot(self):
        """绘制完整静态曲线"""
        self.fig.clear()

        t, sp, pv, co = self._t, self._sp, self._pv, self._co
        Kp, Ki, Kd = self.Kp.get(), self.Ki.get(), self.Kd.get()
        sp_val = self.setpoint.get()
        sim_t  = self.sim_time.get()

        # 子图1: PV + SP
        ax1 = self.fig.add_subplot(211)
        ax1.plot(t, sp, 'k--', linewidth=1.5, label=f'设定值 SP={sp_val}')
        ax1.plot(t, pv, 'b-', linewidth=2.0, label=f'PV (Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f})')
        ax1.set_ylabel('输出 y(t)')
        ax1.set_title('PID控制阶跃响应曲线 — 被控对象: G(s)=K/(Ts+1)', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, sim_t)

        # 参考线
        if self._metrics['final_value'] is not None:
            ax1.axhline(y=self._metrics['final_value'], color='gray', linewidth=0.8, alpha=0.4, linestyle=':')
        ax1.axhline(y=sp_val, color='green', linewidth=0.8, alpha=0.25, linestyle=':')

        # 子图2: 控制器输出
        ax2 = self.fig.add_subplot(212, sharex=ax1)
        ax2.plot(t, co, 'r-', linewidth=1.5, label='控制器输出 CO(t)')
        ax2.set_xlabel('时间 t (s)')
        ax2.set_ylabel('控制量 u(t)')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, sim_t)

        self.fig.tight_layout()
        self.canvas.draw()

    # ==================================================================
    # 动画播放
    # ==================================================================

    def toggle_animation(self):
        if self._animating:
            self.stop_animation()
        else:
            self.start_animation()

    def start_animation(self):
        if self._t is None:
            return
        self._animating = True
        self.btn_play.config(text="⏸ 暂停")
        self._anim_frame = 0
        self._run_animation_step()

    def stop_animation(self):
        self._animating = False
        self.btn_play.config(text="▶ 播放")
        if self._anim_job is not None:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None
        self._anim_progress.config(text="0 %")
        # 恢复完整静态图
        if self._t is not None:
            self._draw_static_plot()

    def _run_animation_step(self):
        if not self._animating:
            return

        t, sp, pv, co = self._t, self._sp, self._pv, self._co
        n_total = len(t)
        speed = self._anim_speed.get()
        # 每步显示到当前帧
        end_idx = min(self._anim_frame + speed * 10, n_total - 1)

        self.fig.clear()
        sim_t = self.sim_time.get()
        Kp, Ki, Kd = self.Kp.get(), self.Ki.get(), self.Kd.get()

        # 子图1
        ax1 = self.fig.add_subplot(211)
        ax1.plot(t[:end_idx+1], sp[:end_idx+1], 'k--', linewidth=1.5, label=f'SP={self.setpoint.get()}')
        ax1.plot(t[:end_idx+1], pv[:end_idx+1], 'b-', linewidth=2.0,
                 label=f'PV (Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f})')
        ax1.set_ylabel('输出 y(t)')
        ax1.set_title(f'PID控制阶跃响应 — 播放中 ({t[end_idx]:.1f}s / {sim_t:.0f}s)', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, sim_t)

        # 动态ylim跟随
        y_min, y_max = min(np.min(pv), np.min(sp)) - 0.05, max(np.max(pv), np.max(sp)) + 0.1
        ax1.set_ylim(y_min, y_max)

        # 子图2
        ax2 = self.fig.add_subplot(212, sharex=ax1)
        ax2.plot(t[:end_idx+1], co[:end_idx+1], 'r-', linewidth=1.5)
        ax2.set_xlabel('时间 t (s)')
        ax2.set_ylabel('控制量 u(t)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, sim_t)

        self.fig.tight_layout()
        self.canvas.draw()

        progress = int(end_idx / (len(t) - 1) * 100)
        self._anim_progress.config(text=f"{progress} %")

        # 下一帧
        if end_idx < n_total - 1:
            self._anim_frame = end_idx
            interval = max(5, int(50 / speed))
            self._anim_job = self.root.after(interval, self._run_animation_step)
        else:
            # 播放完毕
            self.stop_animation()
            self._draw_static_plot()
            self._anim_progress.config(text="100 %")

    # ==================================================================
    # 保存功能
    # ==================================================================

    def _get_save_dir(self):
        """获取默认保存目录"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        saved_dir = os.path.join(script_dir, 'saved_results')
        os.makedirs(saved_dir, exist_ok=True)
        return saved_dir

    def _make_timestamp(self):
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def save_figure_png(self):
        if self._t is None:
            messagebox.showwarning("提示", "请先生成仿真曲线。")
            return
        default_dir = self._get_save_dir()
        default_name = f"pid_response_{self._make_timestamp()}.png"
        path = filedialog.asksaveasfilename(
            initialdir=default_dir, initialfile=default_name,
            defaultextension=".png", filetypes=[("PNG图像", "*.png"), ("All files", "*.*")])
        if path:
            self._draw_static_plot()
            self.fig.savefig(path, dpi=150, bbox_inches='tight')
            messagebox.showinfo("保存成功", f"图像已保存至:\n{path}")

    def save_figure_svg(self):
        if self._t is None:
            messagebox.showwarning("提示", "请先生成仿真曲线。")
            return
        default_dir = self._get_save_dir()
        default_name = f"pid_response_{self._make_timestamp()}.svg"
        path = filedialog.asksaveasfilename(
            initialdir=default_dir, initialfile=default_name,
            defaultextension=".svg", filetypes=[("SVG矢量图", "*.svg"), ("All files", "*.*")])
        if path:
            self._draw_static_plot()
            self.fig.savefig(path, dpi=150, bbox_inches='tight', format='svg')
            messagebox.showinfo("保存成功", f"矢量图已保存至:\n{path}")

    def save_data_csv(self):
        if self._t is None:
            messagebox.showwarning("提示", "请先生成仿真数据。")
            return
        default_dir = self._get_save_dir()
        default_name = f"pid_data_{self._make_timestamp()}.csv"
        path = filedialog.asksaveasfilename(
            initialdir=default_dir, initialfile=default_name,
            defaultextension=".csv", filetypes=[("CSV文件", "*.csv"), ("All files", "*.*")])
        if path:
            K, T = self.plant_K.get(), self.plant_T.get()
            Kp, Ki, Kd = self.Kp.get(), self.Ki.get(), self.Kd.get()
            m = self._metrics

            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                # 头部: 参数摘要
                writer.writerow(['# PID仿真数据'])
                writer.writerow(['# 时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['# 模型', f'K={K}, T={T}, G(s)=K/(Ts+1)'])
                writer.writerow(['# PID', f'Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}'])
                writer.writerow(['# 设定值', f'{self.setpoint.get()}'])
                writer.writerow(['#'])
                writer.writerow(['# 性能指标'])
                for key, label in [
                    ('rise_time', '上升时间(s)'), ('overshoot', '超调量(%)'),
                    ('settling_time', '调节时间(s)'), ('steady_state_error', '稳态误差(%)'),
                    ('final_value', '稳态值'), ('iae', 'IAE'), ('ise', 'ISE'),
                    ('decay_ratio', '衰减比'), ('decay_rate', '衰减率')]:
                    val = m.get(key)
                    writer.writerow([f'# {label}', f'{val:.6f}' if val is not None else 'N/A'])
                writer.writerow(['#'])
                # 数据列
                writer.writerow(['Time(s)', 'Setpoint', 'ProcessVariable', 'ControlOutput', 'Error'])
                for i in range(len(self._t)):
                    writer.writerow([
                        f'{self._t[i]:.6f}',
                        f'{self._sp[i]:.6f}',
                        f'{self._pv[i]:.6f}',
                        f'{self._co[i]:.6f}',
                        f'{self._error[i]:.6f}',
                    ])

            messagebox.showinfo("保存成功", f"数据已导出至:\n{path}")

    # ==================================================================
    # 指标显示
    # ==================================================================

    def _update_metrics_display(self):
        m = self._metrics

        # --- 上升时间 ---
        if m['rise_time'] is not None:
            self.metric_labels['rise_time'].config(text=f"{m['rise_time']:.3f} s")
        else:
            self.metric_labels['rise_time'].config(text="--")

        # --- 超调量 ---
        self.metric_labels['overshoot'].config(text=f"{m['overshoot']:.2f}%")
        if m['overshoot'] > 30:
            self.metric_labels['overshoot'].config(foreground='red')
        elif m['overshoot'] > 10:
            self.metric_labels['overshoot'].config(foreground='orange')
        else:
            self.metric_labels['overshoot'].config(foreground='green')

        # --- 调节时间 ---
        if m['settling_time'] is not None:
            self.metric_labels['settling_time'].config(text=f"{m['settling_time']:.3f} s")
        else:
            self.metric_labels['settling_time'].config(text="--")

        # --- 稳态误差 ---
        self.metric_labels['steady_state_error'].config(text=f"{m['steady_state_error']:.3f}%")
        if m['steady_state_error'] > 5:
            self.metric_labels['steady_state_error'].config(foreground='red')
        else:
            self.metric_labels['steady_state_error'].config(foreground='green')

        # --- 稳态值 ---
        self.metric_labels['final_value'].config(text=f"{m['final_value']:.4f}")

        # --- IAE ---
        self.metric_labels['iae'].config(text=f"{m['iae']:.4f}")

        # --- ISE ---
        self.metric_labels['ise'].config(text=f"{m['ise']:.4f}")

        # --- 衰减比 ---
        if m['decay_ratio'] is not None:
            self.metric_labels['decay_ratio'].config(
                text=f"{m['decay_ratio']:.4f} ({m['decay_ratio']*100:.1f}%)")
            if 0.2 <= m['decay_ratio'] <= 0.35:
                self.metric_labels['decay_ratio'].config(foreground='green')
            else:
                self.metric_labels['decay_ratio'].config(foreground='orange')
        else:
            self.metric_labels['decay_ratio'].config(text="N/A (无振荡)")
            self.metric_labels['decay_ratio'].config(foreground='gray')

        # --- 衰减率 ---
        if m['decay_rate'] is not None:
            self.metric_labels['decay_rate'].config(
                text=f"{m['decay_rate']:.4f} ({m['decay_rate']*100:.1f}%)")
            if m['decay_rate'] >= 0.65:
                self.metric_labels['decay_rate'].config(foreground='green')
            else:
                self.metric_labels['decay_rate'].config(foreground='orange')
        else:
            self.metric_labels['decay_rate'].config(text="N/A (无振荡)")
            self.metric_labels['decay_rate'].config(foreground='gray')

    # ==================================================================
    # 关闭
    # ==================================================================

    def on_closing(self):
        self.stop_animation()
        plt.close('all')
        self.root.destroy()


def main():
    root = tk.Tk()
    app = PidTunerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()

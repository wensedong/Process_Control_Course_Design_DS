#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PID参数整定程序 v3.1 — 主界面
单回路控制系统, 一阶惯性+纯迟延对象 G(s) = K*e^(-tau*s)/(Ts+1)

功能:
  - 手动调整PID参数, 实时观察阶跃响应曲线
  - 四大教材整定方法: 临界比例带法 / 衰减曲线法 / 响应曲线法 / 经验法
  - 控制规律选择: P / PI / PID, 输出限幅可调
  - 性能指标: 超调量, 稳态误差, 调节时间, 上升时间, IAE, ISE, 衰减比, 衰减率
  - 动画播放, 曲线图/数据保存
  - 抗积分饱和 + 发散检测
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.font_manager as fm
_cn = [f for f in fm.fontManager.ttflist if 'Microsoft YaHei' in f.name]
if _cn:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] + matplotlib.rcParams['font.sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

from simulation import Plant, simulate, PlantConfig, PIDConfig, SimulationConfig
from metrics import compute_metrics
from tuning_methods import (
    critical_ratio_calculate, auto_find_critical, manual_critical_measure,
    decay_curve_calculate, find_decay_curve_params,
    response_curve_calculate, batch_compare,
    Kp_to_delta, delta_to_Kp,
    DECAY_TABLE, CRITICAL_TABLE,
)
from data_export import save_figure, save_csv, _get_default_dir, _timestamp
from fuzzy_pid import FuzzyPIDController
from rl_tuner import rl_tune


# ============================================================
# 配色常量
# ============================================================

COLOR_PRIMARY    = '#2563eb'   # 主色调 蓝
COLOR_SUCCESS    = '#16a34a'   # 正常 绿
COLOR_WARNING    = '#ea580c'   # 警告 橙
COLOR_DANGER     = '#dc2626'   # 异常 红
COLOR_MUTED      = '#6b7280'   # 弱化 灰
COLOR_BG         = '#f8fafc'   # 浅灰背景
COLOR_PLOT_BG    = '#f1f5f9'   # 图表区背景


# ============================================================
# 主应用
# ============================================================

class PidTunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PID参数整定 v3.1 — 单回路控制系统")
        self.root.geometry("1200x920")
        self.root.configure(bg=COLOR_BG)

        # ---- 设置 ttk 主题 ----
        style = ttk.Style()
        available = style.theme_names()
        if 'clam' in available:
            style.theme_use('clam')
        elif 'vista' in available:
            style.theme_use('vista')

        # 自定义样式
        style.configure('Title.TLabel', font=('Microsoft YaHei UI', 11, 'bold'))
        style.configure('Heading.TLabel', font=('Microsoft YaHei UI', 10, 'bold'),
                        foreground='#1e293b')
        style.configure('Metric.TLabel', font=('Consolas', 9),
                        foreground=COLOR_PRIMARY, background='white')
        style.configure('MetricNA.TLabel', font=('Consolas', 9),
                        foreground=COLOR_MUTED, background='white')
        style.configure('MetricGood.TLabel', font=('Consolas', 9),
                        foreground=COLOR_SUCCESS, background='white')
        style.configure('MetricWarn.TLabel', font=('Consolas', 9),
                        foreground=COLOR_WARNING, background='white')
        style.configure('MetricBad.TLabel', font=('Consolas', 9),
                        foreground=COLOR_DANGER, background='white')
        style.configure('Status.TLabel', font=('Microsoft YaHei UI', 9))
        style.configure('TF.TLabel', font=('Consolas', 10, 'bold'),
                        foreground='#334155', background=COLOR_BG)
        style.configure('Apply.TButton', font=('Microsoft YaHei UI', 9, 'bold'))
        style.configure('Section.TLabelframe.Label',
                        font=('Microsoft YaHei UI', 10, 'bold'),
                        foreground='#1e40af')

        # ---- 默认参数 ----
        self.plant_K    = tk.DoubleVar(value=1.0)
        self.plant_T    = tk.DoubleVar(value=1.0)
        self.plant_tau  = tk.DoubleVar(value=0.0)
        self.Kp         = tk.DoubleVar(value=1.0)
        self.Ki         = tk.DoubleVar(value=0.0)
        self.Kd         = tk.DoubleVar(value=0.0)
        self.setpoint   = tk.DoubleVar(value=1.0)
        self.sim_time   = tk.DoubleVar(value=25.0)
        self.output_min = tk.DoubleVar(value=-10.0)
        self.output_max = tk.DoubleVar(value=10.0)

        # 整定方法相关
        self.control_law   = tk.StringVar(value='PID')
        self.tuning_method = tk.StringVar(value='手动经验法')
        self.crit_mode     = tk.StringVar(value='auto')
        self.recommended_params = None

        self._updating = False
        self._animating = False
        self._anim_job = None
        self._anim_frame = 0
        self._anim_speed = tk.IntVar(value=5)

        # 仿真缓存
        self._t = self._sp = self._pv = self._co = self._error = None
        self._metrics = None
        self._stable = True

        self.build_ui()
        self.update_plot()

    # ==================================================================
    # 界面构建
    # ==================================================================

    def build_ui(self):
        # ====== 第一行: 参数设置 (白底) ======
        ctrl = tk.LabelFrame(self.root, text=" 参数设置 ", font=('Microsoft YaHei UI', 10, 'bold'),
                             fg='#1e40af', bg='#ffffff', padx=8, pady=6)
        ctrl.pack(fill=tk.X, padx=8, pady=(8, 2))

        # --- 被控对象 ---
        plant = tk.LabelFrame(ctrl, text=" 被控对象 G(s)=K·e⁻ᵗˢ/(Ts+1) ",
                              font=('Microsoft YaHei UI', 9), fg='#334155',
                              bg='#ffffff', padx=6, pady=4)
        plant.grid(row=0, column=0, padx=5, pady=2, sticky='n')

        for i, (lb, var, rng) in enumerate([
            ('K',  self.plant_K,   (0.1, 5.0)),
            ('T',  self.plant_T,   (0.1, 10.0)),
            ('τ',  self.plant_tau, (0.0, 5.0)),
        ]):
            tk.Label(plant, text=f"{lb}:", bg='#ffffff', font=('Microsoft YaHei UI', 9)).grid(
                row=i, column=0, sticky='e', padx=(0, 4))
            ttk.Scale(plant, from_=rng[0], to=rng[1], variable=var,
                      orient=tk.HORIZONTAL, length=120,
                      command=lambda e: self.on_slider_change()).grid(row=i, column=1)
            ttk.Entry(plant, textvariable=var, width=5).grid(row=i, column=2, padx=(3, 0))

        # --- PID参数 ---
        pidf = tk.LabelFrame(ctrl, text=" PID 参数 ", font=('Microsoft YaHei UI', 9),
                             fg='#334155', bg='#ffffff', padx=6, pady=4)
        pidf.grid(row=0, column=1, padx=5, pady=2, sticky='n')

        for i, (lb, var, rng) in enumerate([
            ('Kp', self.Kp, (0.0, 10.0)),
            ('Ki', self.Ki, (0.0, 5.0)),
            ('Kd', self.Kd, (0.0, 5.0)),
        ]):
            tk.Label(pidf, text=f"{lb}:", bg='#ffffff', font=('Microsoft YaHei UI', 9)).grid(
                row=i, column=0, sticky='e', padx=(0, 4))
            ttk.Scale(pidf, from_=rng[0], to=rng[1], variable=var,
                      orient=tk.HORIZONTAL, length=150,
                      command=lambda e: self.on_slider_change()).grid(row=i, column=1)
            ttk.Entry(pidf, textvariable=var, width=5).grid(row=i, column=2, padx=(3, 0))

        # --- 仿真设置 ---
        simf = tk.LabelFrame(ctrl, text=" 仿真设置 ", font=('Microsoft YaHei UI', 9),
                             fg='#334155', bg='#ffffff', padx=6, pady=4)
        simf.grid(row=0, column=2, padx=5, pady=2, sticky='n')
        settings = [
            ('SP:',   self.setpoint,   6),
            ('Tsim:', self.sim_time,   6),
            ('Umin:', self.output_min, 6),
            ('Umax:', self.output_max, 6),
        ]
        for i, (lb, var, w) in enumerate(settings):
            tk.Label(simf, text=lb, bg='#ffffff', font=('Microsoft YaHei UI', 9)).grid(
                row=i, column=0, sticky='e', padx=(0, 4))
            ttk.Entry(simf, textvariable=var, width=w).grid(row=i, column=1)

        # --- 预设按钮 ---
        presetf = tk.LabelFrame(ctrl, text=" 预设 ", font=('Microsoft YaHei UI', 9),
                                fg='#334155', bg='#ffffff', padx=6, pady=4)
        presetf.grid(row=0, column=3, padx=5, pady=2, sticky='n')
        for t, v in [("P", 'P'), ("PI", 'PI'), ("PD", 'PD'), ("PID", 'PID'), ("重置", 'reset')]:
            tk.Button(presetf, text=t, command=lambda v=v: self.set_preset(v),
                     font=('Microsoft YaHei UI', 9), bg='#f1f5f9', fg='#334155',
                     activebackground=COLOR_PRIMARY, activeforeground='white',
                     relief='flat', padx=8, pady=1, cursor='hand2').pack(fill=tk.X, pady=1)

        # --- 整定方法 ---
        tunef = tk.LabelFrame(ctrl, text=" 整定方法 ", font=('Microsoft YaHei UI', 9),
                              fg='#334155', bg='#ffffff', padx=6, pady=4)
        tunef.grid(row=0, column=4, padx=5, pady=2, sticky='n')

        tk.Label(tunef, text="方法:", bg='#ffffff', font=('Microsoft YaHei UI', 9)).grid(
            row=0, column=0, sticky='e', padx=(0, 4))
        method_cb = ttk.Combobox(tunef, textvariable=self.tuning_method, width=16,
                                 values=['手动经验法', '临界比例带法', '衰减曲线法', '响应曲线法',
                                         '模糊PID控制', '强化学习整定'],
                                 state='readonly')
        method_cb.grid(row=0, column=1, columnspan=2, pady=2)
        method_cb.bind('<<ComboboxSelected>>', lambda e: self.on_method_change())

        tk.Label(tunef, text="规律:", bg='#ffffff', font=('Microsoft YaHei UI', 9)).grid(
            row=1, column=0, sticky='e', padx=(0, 4))
        law_cb = ttk.Combobox(tunef, textvariable=self.control_law, width=6,
                              values=['P', 'PI', 'PID'], state='readonly')
        law_cb.grid(row=1, column=1, pady=2)
        law_cb.bind('<<ComboboxSelected>>', lambda e: self.on_method_change())

        # 临界比例带模式切换
        self.crit_mode_frame = tk.Frame(tunef, bg='#ffffff')
        self.crit_mode_frame.grid(row=2, column=0, columnspan=3, pady=2)
        tk.Label(self.crit_mode_frame, text="模式:", bg='#ffffff',
                font=('Microsoft YaHei UI', 9)).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Radiobutton(self.crit_mode_frame, text="手动", variable=self.crit_mode,
                        value='manual', command=self.on_crit_mode_change).pack(
                            side=tk.LEFT, padx=2)
        ttk.Radiobutton(self.crit_mode_frame, text="自动", variable=self.crit_mode,
                        value='auto', command=self.on_crit_mode_change).pack(
                            side=tk.LEFT, padx=2)

        btn_row = tk.Frame(tunef, bg='#ffffff')
        btn_row.grid(row=3, column=0, columnspan=3, pady=(4, 1))
        self.btn_tune = tk.Button(btn_row, text="执行整定", command=self.run_tuning,
                                  font=('Microsoft YaHei UI', 9),
                                  bg=COLOR_PRIMARY, fg='white', relief='flat',
                                  activebackground='#1d4ed8', padx=10, pady=2,
                                  cursor='hand2')
        self.btn_tune.pack(side=tk.LEFT, padx=(0, 5))
        self.btn_apply = tk.Button(btn_row, text="应用推荐",
                                   command=self.apply_recommended,
                                   font=('Microsoft YaHei UI', 9),
                                   bg='#16a34a', fg='white', relief='flat',
                                   activebackground='#15803d', padx=10, pady=2,
                                   cursor='hand2')
        self.btn_apply.pack(side=tk.LEFT)
        self.btn_apply.config(state='disabled')

        self.lbl_recommend = tk.Label(tunef, text="推荐: --",
                                       fg=COLOR_MUTED, bg='#ffffff',
                                       font=('Microsoft YaHei UI', 9))
        self.lbl_recommend.grid(row=4, column=0, columnspan=3, sticky='w', padx=3)

        # 输入框联动
        for var in [self.plant_K, self.plant_T, self.plant_tau,
                    self.Kp, self.Ki, self.Kd, self.setpoint, self.sim_time,
                    self.output_min, self.output_max]:
            var.trace_add('write', lambda *a: self.on_entry_change())

        # ====== 第二行: 性能指标 (白底单行) ======
        self._build_metrics_panel()

        # ====== 第三行: 操作栏 (动画 | 保存) ======
        action = tk.Frame(self.root, bg='#ffffff',
                          highlightbackground='#e2e8f0', highlightthickness=1)
        action.pack(fill=tk.X, padx=8, pady=(4, 0))

        # 动画控制 (左)
        anim = tk.Frame(action, bg='#ffffff')
        anim.pack(side=tk.LEFT, padx=(12, 0))
        tk.Label(anim, text="曲线回放", font=('Microsoft YaHei UI', 9, 'bold'),
                fg='#334155', bg='#ffffff').pack(side=tk.LEFT, padx=(0, 8))
        self.btn_play = tk.Button(anim, text="▶ 播放",
                                  command=self.toggle_animation,
                                  font=('Microsoft YaHei UI', 10, 'bold'),
                                  bg=COLOR_PRIMARY, fg='white',
                                  activebackground='#1d4ed8', activeforeground='white',
                                  relief='flat', padx=14, pady=4, cursor='hand2')
        self.btn_play.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(anim, text="倍速", font=('Microsoft YaHei UI', 8),
                fg=COLOR_MUTED, bg='#ffffff').pack(side=tk.LEFT)
        ttk.Scale(anim, from_=1, to=10, variable=self._anim_speed,
                  orient=tk.HORIZONTAL, length=60).pack(side=tk.LEFT, padx=(4, 2))
        tk.Label(anim, textvariable=self._anim_speed, width=2,
                font=('Consolas', 9, 'bold'), fg=COLOR_PRIMARY, bg='#ffffff').pack(side=tk.LEFT)
        tk.Label(anim, text="x", font=('Microsoft YaHei UI', 7),
                fg=COLOR_MUTED, bg='#ffffff').pack(side=tk.LEFT)
        self._anim_progress = tk.Label(anim, text="  0%", width=5,
                                        font=('Consolas', 10, 'bold'),
                                        fg=COLOR_PRIMARY, bg='#ffffff')
        self._anim_progress.pack(side=tk.LEFT, padx=(8, 2))

        # 保存按钮 (右)
        sv = tk.Frame(action, bg='#ffffff')
        sv.pack(side=tk.RIGHT, padx=(0, 12))
        tk.Label(sv, text="导出:", font=('Microsoft YaHei UI', 9, 'bold'),
                fg='#334155', bg='#ffffff').pack(side=tk.LEFT, padx=(0, 6))
        for txt, cmd in [("PNG", self.save_png), ("SVG", self.save_svg),
                         ("CSV", self.save_csv_file)]:
            tk.Button(sv, text=txt, command=cmd,
                     font=('Microsoft YaHei UI', 9),
                     bg=COLOR_PRIMARY, fg='white',
                     activebackground='#1d4ed8', activeforeground='white',
                     relief='flat', padx=10, pady=3, cursor='hand2').pack(
                         side=tk.LEFT, padx=2)

        # ====== 第四行: 状态 ======
        self.status_var = tk.StringVar(value="● 就绪 — 调整参数以刷新曲线")
        self.status_label = tk.Label(self.root, textvariable=self.status_var,
                                      font=('Microsoft YaHei UI', 8),
                                      fg=COLOR_SUCCESS, bg=COLOR_BG, anchor='w')
        self.status_label.pack(fill=tk.X, padx=12, pady=(2, 1))

        # ====== 图表 ======
        self.fig = Figure(figsize=(9, 6.5), dpi=100,
                          facecolor=COLOR_PLOT_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=2)

        # ====== 底部: matplotlib 工具栏 ======
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=8, pady=(1, 5))
        self.toolbar = NavigationToolbar2Tk(self.canvas, bar)
        self.toolbar.update()

    # ==================================================================
    # 性能指标面板
    # ==================================================================

    def _build_metrics_panel(self):
        """单行性能指标面板 — 白底, 9个指标均匀排列"""
        outer = tk.Frame(self.root, bg='#ffffff',
                         highlightbackground='#e2e8f0', highlightthickness=1)
        outer.pack(fill=tk.X, padx=8, pady=(2, 0))
        outer.grid_columnconfigure(tuple(range(9)), weight=1)

        self.metric_labels = {}

        metrics = [
            ('rise_time',          '上升时间'),
            ('overshoot',          '超调量'),
            ('settling_time',      '调节时间'),
            ('steady_state_error', '稳态误差'),
            ('final_value',        '稳态值'),
            ('decay_ratio',        '衰减比'),
            ('decay_rate',         '衰减率'),
            ('iae',                'IAE'),
            ('ise',                'ISE'),
        ]
        for i, (k, lb) in enumerate(metrics):
            grp = tk.Frame(outer, bg='#ffffff')
            grp.grid(row=0, column=i, padx=2, pady=(5, 4), sticky='n')
            tk.Label(grp, text=lb, font=('Microsoft YaHei UI', 7),
                    fg=COLOR_MUTED, bg='#ffffff').pack()
            lbl = tk.Label(grp, text='--', font=('Consolas', 10, 'bold'),
                          fg=COLOR_PRIMARY, bg='#ffffff')
            lbl.pack()
            self.metric_labels[k] = lbl

    # ==================================================================
    # 参数变更回调
    # ==================================================================

    def _round_params(self):
        for v in [self.Kp, self.Ki, self.Kd, self.plant_K, self.plant_T, self.plant_tau]:
            v.set(round(v.get(), 2))

    def on_slider_change(self):
        if self._updating:
            return
        self._updating = True
        self._round_params()
        self.update_plot()
        self._updating = False

    def on_entry_change(self):
        if self._updating:
            return
        self._updating = True
        self.root.after(300, self._delayed_update)

    def _delayed_update(self):
        self._round_params()
        self.update_plot()
        self._updating = False

    def on_method_change(self):
        method = self.tuning_method.get()
        if method == '临界比例带法':
            self.crit_mode_frame.grid()
            self._update_crit_ui()
        else:
            self.crit_mode_frame.grid_remove()

        if method == '手动经验法':
            self.btn_tune.config(state='disabled', text='执行整定')
            self.btn_apply.config(state='disabled')
            self.lbl_recommend.config(text='推荐: --', fg=COLOR_MUTED)
            self.recommended_params = None
        else:
            self.btn_tune.config(state='normal')
            self.recommended_params = None
            self.btn_apply.config(state='disabled')
            self._update_crit_ui()

    def on_crit_mode_change(self):
        self._update_crit_ui()

    def _update_crit_ui(self):
        method = self.tuning_method.get()
        if method == '临界比例带法' and self.crit_mode.get() == 'manual':
            self.btn_tune.config(text='记录临界点')
            self.lbl_recommend.config(
                text='先手动调Kp至等幅振荡, 再点"记录临界点"',
                fg=COLOR_MUTED)
        elif method == '临界比例带法':
            self.btn_tune.config(text='自动搜索')
            self.lbl_recommend.config(text='点击"自动搜索"查找临界点',
                                       fg=COLOR_MUTED)
        elif method == '衰减曲线法':
            self.btn_tune.config(text='执行整定')
            self.lbl_recommend.config(text='点击"执行整定"计算 φ=0.75',
                                       fg=COLOR_MUTED)
        elif method == '响应曲线法':
            self.btn_tune.config(text='执行整定')
            self.lbl_recommend.config(text='点击"执行整定"计算',
                                       fg=COLOR_MUTED)
        elif method == '模糊PID控制':
            self.btn_tune.config(text='模糊PID已激活')
            self.lbl_recommend.config(
                text='拖动PID滑块设置基础参数, 模糊控制器会自动在线调整',
                fg=COLOR_PRIMARY)
        elif method == '强化学习整定':
            self.btn_tune.config(text='开始训练')
            self.lbl_recommend.config(
                text='Q-Learning自动探索最优PID参数 (约200轮仿真)',
                fg=COLOR_MUTED)

    def set_preset(self, preset_type):
        presets = {
            'P':     {'Kp': 2.00, 'Ki': 0.00, 'Kd': 0.00, 'law': 'P'},
            'PI':    {'Kp': 1.50, 'Ki': 0.50, 'Kd': 0.00, 'law': 'PI'},
            'PD':    {'Kp': 2.00, 'Ki': 0.00, 'Kd': 0.50, 'law': 'PID'},
            'PID':   {'Kp': 2.00, 'Ki': 1.00, 'Kd': 0.50, 'law': 'PID'},
            'reset': {'Kp': 1.00, 'Ki': 0.00, 'Kd': 0.00, 'law': None},
        }
        if preset_type in presets:
            p = presets[preset_type]
            self.Kp.set(p['Kp']); self.Ki.set(p['Ki']); self.Kd.set(p['Kd'])
            if p['law'] is not None:
                self.control_law.set(p['law'])
            self.stop_animation()
            self.update_plot()

    # ==================================================================
    # 整定方法执行
    # ==================================================================

    def run_tuning(self):
        method = self.tuning_method.get()
        law = self.control_law.get()
        K = self.plant_K.get()
        T = self.plant_T.get()
        tau = self.plant_tau.get()

        if method == '手动经验法':
            return

        if method == '临界比例带法':
            if self.crit_mode.get() == 'manual':
                self._run_critical_manual(K, T, tau, law)
            else:
                self._run_critical_ratio(K, T, tau, law)
        elif method == '衰减曲线法':
            self._run_decay_curve(K, T, tau, law)
        elif method == '响应曲线法':
            self._run_response_curve(K, T, tau, law)
        elif method == '模糊PID控制':
            # 模糊PID不需要"执行整定" — 已经在 update_plot 中自动激活
            self.update_plot()
        elif method == '强化学习整定':
            self._run_rl_tuning(K, T, tau)

    def _run_critical_manual(self, K, T, tau, law):
        from tuning_methods import manual_critical_measure
        Kp_now = self.Kp.get()

        if self.Ki.get() != 0 or self.Kd.get() != 0:
            if not messagebox.askyesno('提示',
                '手动临界比例带法要求纯比例控制 (Ki=0, Kd=0)。\n'
                '是否自动将 Ki/Kd 置零并继续？'):
                return
            self.Ki.set(0.0)
            self.Kd.set(0.0)

        self.lbl_recommend.config(text='正在测量振荡周期...', fg=COLOR_WARNING)
        self.root.update_idletasks()

        result = manual_critical_measure(K, T, tau, Kp_now,
                                          setpoint=self.setpoint.get(), sim_time=80.0)
        if not result['success']:
            messagebox.showwarning('临界比例带法 (手动)', result['message'])
            self.lbl_recommend.config(text=result['message'], fg=COLOR_DANGER)
            return

        params = critical_ratio_calculate(result['delta_cr'], result['T_cr'], law)
        if params is None:
            self.lbl_recommend.config(text='无效的控制规律', fg=COLOR_DANGER)
            return

        self.recommended_params = params
        self.btn_apply.config(state='normal')
        self.lbl_recommend.config(
            text=f"δ_cr={result['delta_cr']:.1f}% T_cr={result['T_cr']:.2f}s → "
                 f"Kp={params['Kp']:.2f} Ki={params['Ki']:.2f} Kd={params['Kd']:.2f}",
            fg=COLOR_SUCCESS)
        self.update_plot()

    def _run_critical_ratio(self, K, T, tau, law):
        self.lbl_recommend.config(text='正在搜索临界振荡点...', fg=COLOR_WARNING)
        self.root.update_idletasks()

        result = auto_find_critical(K, T, tau, setpoint=self.setpoint.get(),
                                     sim_time=80.0)
        if not result['success']:
            messagebox.showwarning('临界比例带法', result['message'])
            self.lbl_recommend.config(text=result['message'], fg=COLOR_DANGER)
            return

        params = critical_ratio_calculate(result['delta_cr'], result['T_cr'], law)
        if params is None:
            self.lbl_recommend.config(text='无效的控制规律', fg=COLOR_DANGER)
            return

        self.recommended_params = params
        self.btn_apply.config(state='normal')
        self.lbl_recommend.config(
            text=f"δ_cr={result['delta_cr']:.1f}% T_cr={result['T_cr']:.2f}s → "
                 f"Kp={params['Kp']:.2f} Ki={params['Ki']:.2f} Kd={params['Kd']:.2f}",
            fg=COLOR_SUCCESS)

    def _run_decay_curve(self, K, T, tau, law):
        phi = 0.75
        self.lbl_recommend.config(text=f'正在搜索 φ={phi} 衰减曲线...',
                                   fg=COLOR_WARNING)
        self.root.update_idletasks()

        result = find_decay_curve_params(K, T, tau, phi=phi,
                                          setpoint=self.setpoint.get(), sim_time=60.0)
        if not result['success']:
            messagebox.showwarning('衰减曲线法', result['message'])
            self.lbl_recommend.config(text=result['message'], fg=COLOR_DANGER)
            return

        if result['T_s_or_tr'] is None:
            messagebox.showwarning('衰减曲线法', '无法测量振荡周期')
            return

        params = decay_curve_calculate(result['delta_s'], result['T_s_or_tr'],
                                        phi=phi, control_law=law)
        if params is None:
            self.lbl_recommend.config(text='无效的控制规律', fg=COLOR_DANGER)
            return

        self.recommended_params = params
        self.btn_apply.config(state='normal')
        self.lbl_recommend.config(
            text=f"δ_s={result['delta_s']:.1f}% Ts={result['T_s_or_tr']:.2f}s → "
                 f"Kp={params['Kp']:.2f} Ki={params['Ki']:.2f} Kd={params['Kd']:.2f}",
            fg=COLOR_SUCCESS)

    def _run_response_curve(self, K, T, tau, law):
        if tau <= 0:
            messagebox.showwarning('响应曲线法', '需要 τ > 0, 请设置纯迟延时间')
            return

        params = response_curve_calculate(K, T, tau, law)
        if params is None:
            self.lbl_recommend.config(text='τ/T 超出教材范围 (需 ≤1.5)',
                                       fg=COLOR_DANGER)
            return

        self.recommended_params = params
        self.btn_apply.config(state='normal')
        self.lbl_recommend.config(
            text=f"ε={params['epsilon']:.4f} τ/T={params['tau_T_ratio']:.2f} → "
                 f"Kp={params['Kp']:.2f} Ki={params['Ki']:.2f} Kd={params['Kd']:.2f}",
            fg=COLOR_SUCCESS)

    def _run_rl_tuning(self, K, T, tau):
        """强化学习整定: Q-Learning 自动探索最优 PID 参数"""
        self.btn_tune.config(state='disabled', text='训练中...')
        self.lbl_recommend.config(text='Q-Learning 正在探索参数空间...', fg=COLOR_WARNING)
        self.root.update_idletasks()

        episodes = 200

        def progress_cb(ep, total, eps, best_r, best_params):
            """每 episode 更新 GUI 进度"""
            if ep % 10 == 0 or ep == total:
                pct = ep * 100 // total
                self.lbl_recommend.config(
                    text=f'训练中 [{pct}%] ep={ep}/{total} ε={eps:.2f} '
                         f'best Kp={best_params[0]:.2f} Ki={best_params[1]:.2f} '
                         f'Kd={best_params[2]:.2f}',
                    fg=COLOR_WARNING)
                self.root.update_idletasks()

        try:
            result = rl_tune(K, T, tau,
                             setpoint=self.setpoint.get(),
                             sim_time=self.sim_time.get(),
                             episodes=episodes,
                             progress_callback=progress_cb)

            if result is None:
                self.lbl_recommend.config(text='训练失败 — 请重试', fg=COLOR_DANGER)
                self.btn_tune.config(state='normal', text='开始训练')
                return

            self.recommended_params = result
            self.btn_apply.config(state='normal')

            m = result.get('metrics', {})
            dr = m.get('decay_rate')
            dr_str = f'{dr*100:.0f}%' if dr is not None else 'N/A'
            self.lbl_recommend.config(
                text=f'RL→Kp={result["Kp"]:.2f} Ki={result["Ki"]:.2f} Kd={result["Kd"]:.2f} '
                     f'| σ={m.get("overshoot",0):.1f}% φ={dr_str} IAE={m.get("iae",0):.3f}',
                fg=COLOR_SUCCESS)

        except Exception as e:
            self.lbl_recommend.config(text=f'RL训练出错: {e}', fg=COLOR_DANGER)
        finally:
            self.btn_tune.config(state='normal', text='开始训练')

    def apply_recommended(self):
        if self.recommended_params is None:
            return
        p = self.recommended_params
        self.Kp.set(p['Kp']); self.Ki.set(p['Ki']); self.Kd.set(p['Kd'])
        self.update_plot()

    # ==================================================================
    # 仿真 + 绘图
    # ==================================================================

    def update_plot(self):
        try:
            K, T, tau = self.plant_K.get(), self.plant_T.get(), self.plant_tau.get()
            Kp, Ki, Kd = self.Kp.get(), self.Ki.get(), self.Kd.get()
            sp_val = self.setpoint.get()
            sim_t = self.sim_time.get()

            plant_cfg = PlantConfig(K=K, T=T, tau=tau)
            sim_cfg = SimulationConfig(setpoint=sp_val, sim_time=sim_t)

            # 模糊 PID 模式: 使用自适应控制器
            if self.tuning_method.get() == '模糊PID控制':
                ctrl = FuzzyPIDController(
                    Kp_base=Kp, Ki_base=Ki, Kd_base=Kd,
                    scale_Kp=0.3, scale_Ki=0.08, scale_Kd=0.04,
                    e_max=1.0, ec_max=0.3, dt=0.01,
                    output_min=self.output_min.get(),
                    output_max=self.output_max.get())
                pid_cfg = PIDConfig(Kp=Kp, Ki=Ki, Kd=Kd)  # 仅用于兼容
                result = simulate(plant_cfg, pid_cfg, sim_cfg, controller=ctrl)
                self._fuzzy_active = True
            else:
                pid_cfg = PIDConfig(Kp=Kp, Ki=Ki, Kd=Kd,
                                    output_min=self.output_min.get(),
                                    output_max=self.output_max.get())
                result = simulate(plant_cfg, pid_cfg, sim_cfg)
                self._fuzzy_active = False

            self._t     = result['t']
            self._sp    = result['sp']
            self._pv    = result['pv']
            self._co    = result['co']
            self._error = result['error']
            self._stable = result['stable']

            # 保存参数历史 (模糊PID模式用于绘图)
            self._kp_hist = result['kp_hist']
            self._ki_hist = result['ki_hist']
            self._kd_hist = result['kd_hist']

            self._metrics = compute_metrics(self._t, self._sp, self._pv, self._error)

            self._draw_static()
            self._update_metrics()
            self._update_status()

        except Exception as e:
            self.status_var.set(f"⚠ 仿真出错: {e}")
            self.status_label.config(fg=COLOR_DANGER)

    def _draw_static(self):
        self.fig.clear()
        t, sp, pv, co = self._t, self._sp, self._pv, self._co
        sim_t = self.sim_time.get()
        Kp, Ki, Kd = self.Kp.get(), self.Ki.get(), self.Kd.get()
        fuzzy = getattr(self, '_fuzzy_active', False)

        self.fig.set_facecolor(COLOR_PLOT_BG)

        if fuzzy:
            # 模糊PID: 三子图 (PV, CO, Kp/Ki/Kd变化)
            ax1 = self.fig.add_subplot(311)
        else:
            ax1 = self.fig.add_subplot(211)
        ax1.set_facecolor('white')
        ax1.plot(t, sp, 'k--', lw=1.5, alpha=0.7, label=f'SP={self.setpoint.get()}')
        ax1.plot(t, pv, COLOR_PRIMARY, lw=2.0,
                 label=f'PV (Kp={Kp:.2f} Ki={Ki:.2f} Kd={Kd:.2f})')
        ax1.set_ylabel('输出 y(t)', fontsize=9)
        title = '模糊PID控制阶跃响应曲线' if fuzzy else 'PID控制阶跃响应曲线'
        ax1.set_title(title, fontsize=12, fontweight='bold', color='#1e293b')
        ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
        ax1.grid(True, alpha=0.3, color='#cbd5e1')
        ax1.set_xlim(0, sim_t)
        if self._metrics['final_value'] is not None:
            ax1.axhline(self._metrics['final_value'], color='gray',
                        lw=0.6, ls=':', alpha=0.5)
        ax1.axhline(sp[0], color=COLOR_SUCCESS, lw=0.6, ls=':', alpha=0.3)

        # 传递函数分式标注
        pk, pT, ptau = self.plant_K.get(), self.plant_T.get(), self.plant_tau.get()
        if ptau > 0:
            tf_str = r'$G(s)=\frac{%.2f\ e^{-%.2fs}}{%.2fs+1}$' % (pk, ptau, pT)
        else:
            tf_str = r'$G(s)=\frac{%.2f}{%.2fs+1}$' % (pk, pT)
        ax1.text(0.02, 0.97, tf_str, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbeb',
                           edgecolor='#d97706', alpha=0.9))

        if fuzzy:
            # 控制量子图
            ax2 = self.fig.add_subplot(312, sharex=ax1)
        else:
            ax2 = self.fig.add_subplot(212, sharex=ax1)
        ax2.set_facecolor('white')
        ax2.plot(t, co, COLOR_DANGER, lw=1.5, alpha=0.85, label='控制量 CO(t)')
        umin, umax = self.output_min.get(), self.output_max.get()
        ax2.axhline(umin, color='gray', lw=0.5, ls=':', alpha=0.4)
        ax2.axhline(umax, color='gray', lw=0.5, ls=':', alpha=0.4)
        ax2.set_ylabel('控制量 u(t)', fontsize=9)
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax2.grid(True, alpha=0.3, color='#cbd5e1')
        ax2.set_xlim(0, sim_t)

        if fuzzy:
            # 参数变化子图 (模糊PID专用)
            ax3 = self.fig.add_subplot(313, sharex=ax1)
            ax3.set_facecolor('white')
            kp_h = getattr(self, '_kp_hist', None)
            ki_h = getattr(self, '_ki_hist', None)
            kd_h = getattr(self, '_kd_hist', None)
            if kp_h is not None and len(kp_h) == len(t):
                ax3.plot(t, kp_h, COLOR_PRIMARY, lw=1.2, alpha=0.8, label='Kp(t)')
                ax3.plot(t, ki_h, COLOR_SUCCESS, lw=1.2, alpha=0.8, label='Ki(t)')
                ax3.plot(t, kd_h, COLOR_WARNING, lw=1.2, alpha=0.8, label='Kd(t)')
            ax3.set_xlabel('时间 t (s)', fontsize=9)
            ax3.set_ylabel('PID参数', fontsize=9)
            ax3.set_title('模糊控制器在线参数调整', fontsize=9, color='#64748b')
            ax3.legend(loc='upper right', fontsize=7, ncol=3, framealpha=0.9)
            ax3.grid(True, alpha=0.3, color='#cbd5e1')
            ax3.set_xlim(0, sim_t)
        else:
            ax2.set_xlabel('时间 t (s)', fontsize=9)

        self.fig.tight_layout()
        self.canvas.draw()

    def _update_status(self):
        if not self._stable:
            self.status_var.set(
                "⚠ 仿真发散 — 请减小 Kd 或 Kp，或增大输出限幅")
            self.status_label.config(fg=COLOR_DANGER)
        else:
            self.status_var.set("● 就绪 — 调整参数以刷新曲线")
            self.status_label.config(fg=COLOR_SUCCESS)

    # ==================================================================
    # 动画
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
        self._run_anim_step()

    def stop_animation(self):
        self._animating = False
        self.btn_play.config(text="▶ 播放")
        if self._anim_job:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None
        self._anim_progress.config(text="0%")
        if self._t is not None:
            self._draw_static()

    def _run_anim_step(self):
        if not self._animating:
            return
        t, sp, pv, co = self._t, self._sp, self._pv, self._co
        n = len(t)
        speed = self._anim_speed.get()
        end = min(self._anim_frame + speed * 10, n - 1)
        sim_t = self.sim_time.get()
        Kp, Ki, Kd = self.Kp.get(), self.Ki.get(), self.Kd.get()
        fuzzy = getattr(self, '_fuzzy_active', False)

        self.fig.clear()
        self.fig.set_facecolor(COLOR_PLOT_BG)

        if fuzzy:
            ax1 = self.fig.add_subplot(311)
        else:
            ax1 = self.fig.add_subplot(211)
        ax1.set_facecolor('white')
        ax1.plot(t[:end+1], sp[:end+1], 'k--', lw=1.5, alpha=0.7,
                 label=f'SP={self.setpoint.get()}')
        ax1.plot(t[:end+1], pv[:end+1], COLOR_PRIMARY, lw=2.0,
                 label=f'PV (Kp={Kp:.2f} Ki={Ki:.2f} Kd={Kd:.2f})')
        ax1.set_ylabel('输出 y(t)', fontsize=9)
        ax1.set_title(f'动画播放中 ({t[end]:.1f}s / {sim_t:.0f}s)',
                      fontsize=12, fontweight='bold', color='#1e293b')
        ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
        ax1.grid(True, alpha=0.3, color='#cbd5e1')
        ax1.set_xlim(0, sim_t)
        y_min = min(np.min(pv[:end+1]), np.min(sp[:end+1])) - 0.05
        y_max = max(np.max(pv[:end+1]), np.max(sp[:end+1])) + 0.1
        ax1.set_ylim(y_min, y_max)

        if fuzzy:
            ax2 = self.fig.add_subplot(312, sharex=ax1)
        else:
            ax2 = self.fig.add_subplot(212, sharex=ax1)
        ax2.set_facecolor('white')
        ax2.plot(t[:end+1], co[:end+1], COLOR_DANGER, lw=1.5, alpha=0.85)
        umin, umax = self.output_min.get(), self.output_max.get()
        ax2.axhline(umin, color='gray', lw=0.5, ls=':', alpha=0.4)
        ax2.axhline(umax, color='gray', lw=0.5, ls=':', alpha=0.4)
        ax2.set_ylabel('控制量 u(t)', fontsize=9)
        ax2.grid(True, alpha=0.3, color='#cbd5e1')
        ax2.set_xlim(0, sim_t)

        if fuzzy:
            ax3 = self.fig.add_subplot(313, sharex=ax1)
            ax3.set_facecolor('white')
            kp_h = getattr(self, '_kp_hist', None)
            ki_h = getattr(self, '_ki_hist', None)
            kd_h = getattr(self, '_kd_hist', None)
            if kp_h is not None and len(kp_h) >= end + 1:
                ax3.plot(t[:end+1], kp_h[:end+1], COLOR_PRIMARY, lw=1.2, label='Kp(t)')
                ax3.plot(t[:end+1], ki_h[:end+1], COLOR_SUCCESS, lw=1.2, label='Ki(t)')
                ax3.plot(t[:end+1], kd_h[:end+1], COLOR_WARNING, lw=1.2, label='Kd(t)')
            ax3.set_xlabel('时间 t (s)', fontsize=9)
            ax3.set_ylabel('PID参数', fontsize=9)
            ax3.set_title('模糊控制器在线参数调整', fontsize=9, color='#64748b')
            ax3.legend(loc='upper right', fontsize=7, ncol=3, framealpha=0.9)
            ax3.grid(True, alpha=0.3, color='#cbd5e1')
            ax3.set_xlim(0, sim_t)
        else:
            ax2.set_xlabel('时间 t (s)', fontsize=9)

        self.fig.tight_layout()
        self.canvas.draw()
        self._anim_progress.config(text=f"{int(end/(n-1)*100)}%")

        if end < n - 1:
            self._anim_frame = end
            self._anim_job = self.root.after(
                max(5, int(50/speed)), self._run_anim_step)
        else:
            self.stop_animation()
            self._draw_static()
            self._anim_progress.config(text="100%")

    # ==================================================================
    # 指标更新
    # ==================================================================

    def _update_metrics(self):
        m = self._metrics

        def _set(k, text, color=COLOR_PRIMARY):
            self.metric_labels[k].config(text=text, fg=color)

        # 上升时间
        if m['rise_time'] is not None:
            _set('rise_time', f"{m['rise_time']:.3f}s")
        else:
            _set('rise_time', '--', COLOR_MUTED)

        # 超调量 (颜色编码)
        ov = m['overshoot']
        if ov > 30:
            _set('overshoot', f"{ov:.2f}%", COLOR_DANGER)
        elif ov > 10:
            _set('overshoot', f"{ov:.2f}%", COLOR_WARNING)
        else:
            _set('overshoot', f"{ov:.2f}%", COLOR_SUCCESS)

        # 调节时间
        if m['settling_time'] is not None:
            _set('settling_time', f"{m['settling_time']:.3f}s")
        else:
            _set('settling_time', '--', COLOR_MUTED)

        # 稳态误差
        se = m['steady_state_error']
        if abs(se) > 5:
            _set('steady_state_error', f"{se:.3f}%", COLOR_DANGER)
        else:
            _set('steady_state_error', f"{se:.3f}%", COLOR_SUCCESS)

        # 稳态值
        _set('final_value', f"{m['final_value']:.4f}")

        # IAE / ISE
        _set('iae', f"{m['iae']:.4f}")
        _set('ise', f"{m['ise']:.4f}")

        # 衰减比
        dr = m['decay_ratio']
        if dr is not None:
            if 0.2 <= dr <= 0.35:
                _set('decay_ratio', f"{dr:.4f} ({dr*100:.1f}%)", COLOR_SUCCESS)
            else:
                _set('decay_ratio', f"{dr:.4f} ({dr*100:.1f}%)", COLOR_WARNING)
        else:
            _set('decay_ratio', 'N/A', COLOR_MUTED)

        # 衰减率
        dr2 = m['decay_rate']
        if dr2 is not None:
            if dr2 >= 0.65:
                _set('decay_rate', f"{dr2:.4f} ({dr2*100:.1f}%)", COLOR_SUCCESS)
            else:
                _set('decay_rate', f"{dr2:.4f} ({dr2*100:.1f}%)", COLOR_WARNING)
        else:
            _set('decay_rate', 'N/A', COLOR_MUTED)

    # ==================================================================
    # 保存
    # ==================================================================

    def save_png(self):
        if self._t is None:
            messagebox.showwarning("提示", "请先生成仿真曲线")
            return
        d = _get_default_dir()
        p = filedialog.asksaveasfilename(
            initialdir=d, initialfile=f"pid_{_timestamp()}.png",
            defaultextension=".png", filetypes=[("PNG", "*.png")])
        if p:
            self._draw_static()
            save_figure(self.fig, p, 'png')
            messagebox.showinfo("OK", f"已保存:\n{p}")

    def save_svg(self):
        if self._t is None:
            messagebox.showwarning("提示", "请先生成仿真曲线")
            return
        d = _get_default_dir()
        p = filedialog.asksaveasfilename(
            initialdir=d, initialfile=f"pid_{_timestamp()}.svg",
            defaultextension=".svg", filetypes=[("SVG", "*.svg")])
        if p:
            self._draw_static()
            save_figure(self.fig, p, 'svg')
            messagebox.showinfo("OK", f"已保存:\n{p}")

    def save_csv_file(self):
        if self._t is None:
            messagebox.showwarning("提示", "请先生成仿真数据")
            return
        d = _get_default_dir()
        p = filedialog.asksaveasfilename(
            initialdir=d, initialfile=f"pid_{_timestamp()}.csv",
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            method_name = self.recommended_params.get('method_name', '手动整定') \
                if self.recommended_params else '手动整定'
            save_csv(p, self.plant_K.get(), self.plant_T.get(), self.plant_tau.get(),
                     self.Kp.get(), self.Ki.get(), self.Kd.get(),
                     self.setpoint.get(), self._metrics,
                     self._t, self._sp, self._pv, self._co, self._error,
                     method_name, stable=self._stable)
            messagebox.showinfo("OK", f"已导出:\n{p}")

    # ==================================================================
    # 关闭
    # ==================================================================

    def on_closing(self):
        self.stop_animation()
        plt.close('all')
        self.root.destroy()


def main():
    root = tk.Tk()
    PidTunerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()

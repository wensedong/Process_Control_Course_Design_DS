#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能指标计算模块
- 上升时间 / 超调量 / 调节时间 / 稳态误差
- IAE / ISE
- 衰减比 / 衰减率 (无振荡时返回 None)
"""

import numpy as np


def compute_metrics(t, sp, pv, error, tolerance=0.02):
    """
    计算全部性能指标。

    参数:
        t, sp, pv, error: 仿真输出数组
        tolerance: 调节时间误差带 (默认±2%)

    返回 dict:
        rise_time, overshoot, settling_time,
        steady_state_error, final_value,
        iae, ise, decay_ratio, decay_rate
    """
    setpoint_val = sp[0]
    pv_final = pv[-1]

    # -- 稳态误差 (%) --
    if abs(setpoint_val) > 1e-9:
        steady_state_error = abs(setpoint_val - pv_final) / abs(setpoint_val) * 100
    else:
        steady_state_error = 0.0

    # -- 上升时间 (10% → 90% 稳态值) --
    t_rise = None
    y10 = 0.1 * pv_final
    y90 = 0.9 * pv_final
    t10 = None
    for i in range(len(t)):
        if t10 is None and pv[i] >= y10:
            t10 = t[i]
        if t10 is not None and pv[i] >= y90:
            t_rise = t[i] - t10
            break

    # -- 超调量 (%) --
    pv_max = np.max(pv)
    if abs(pv_final) > 1e-9:
        overshoot = max(0.0, (pv_max - pv_final) / abs(pv_final) * 100)
    else:
        overshoot = 0.0

    # -- 调节时间 --
    upper = pv_final * (1 + tolerance)
    lower = pv_final * (1 - tolerance)
    settling_idx = None
    for i in range(len(t) - 1, -1, -1):
        if pv[i] > upper or pv[i] < lower:
            settling_idx = i
            break
    if settling_idx is not None and settling_idx < len(t) - 1:
        t_settling = t[settling_idx + 1]
    else:
        t_settling = 0.0

    # -- IAE / ISE --
    dt = t[1] - t[0] if len(t) > 1 else 0.01
    iae = float(np.sum(np.abs(error)) * dt)
    ise = float(np.sum(error ** 2) * dt)

    # -- 衰减比 / 衰减率 --
    decay_ratio, decay_rate = _compute_decay(t, pv, pv_final)

    return {
        'rise_time':          t_rise,
        'overshoot':          overshoot,
        'settling_time':      t_settling,
        'steady_state_error': steady_state_error,
        'final_value':        pv_final,
        'iae':                iae,
        'ise':                ise,
        'decay_ratio':        decay_ratio,
        'decay_rate':         decay_rate,
    }


def _compute_decay(t, pv, pv_final):
    """计算衰减比 / 衰减率"""
    peaks_idx = _find_peaks(pv)
    # 只保留超过稳态值的峰（真正的超调峰）
    overshoot_peaks = []
    first_settle = None
    for i in range(len(pv)):
        if pv[i] >= pv_final * 0.99:
            first_settle = i
            break
    if first_settle is not None:
        for idx in peaks_idx:
            if idx > first_settle and pv[idx] > pv_final:
                overshoot_peaks.append(pv[idx])

    if len(overshoot_peaks) >= 2:
        a1 = overshoot_peaks[0] - pv_final
        a2 = overshoot_peaks[1] - pv_final
        if a1 > 1e-9:
            return a2 / a1, 1.0 - a2 / a1
    return None, None


def _find_peaks(y):
    """局部极大值"""
    peaks = []
    n = len(y)
    if n < 3:
        return peaks
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            peaks.append(i)
    return peaks

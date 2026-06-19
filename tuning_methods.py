#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
教材整定方法模块
- 临界比例带法
- 衰减曲线法 (φ=0.75 / φ=0.9)
- 响应曲线法 / 动态特性参数法
- 经验法辅助: 批量对比

教材参考: 《火电厂热工自动控制技术及应用》
PID形式: 比例带 δ, 积分时间 TI, 微分时间 TD
换算关系: Kp = 100 / δ, Ki = Kp / TI, Kd = Kp * TD
"""

import numpy as np
from simulation import simulate, PlantConfig, PIDConfig, SimulationConfig
from metrics import compute_metrics


# ============================================================
# 内部辅助: 用 dataclass 包装 simulate 调用, 返回元组
# ============================================================

def _sim(K, T, tau, Kp, Ki, Kd, setpoint, sim_time, dt=0.01,
         output_min=-1e6, output_max=1e6):
    """运行仿真并返回 (t, sp, pv, co, err) 元组 (向后兼容)"""
    r = simulate(
        PlantConfig(K=K, T=T, tau=tau),
        PIDConfig(Kp=Kp, Ki=Ki, Kd=Kd,
                  output_min=output_min, output_max=output_max),
        SimulationConfig(setpoint=setpoint, sim_time=sim_time, dt=dt))
    return r['t'], r['sp'], r['pv'], r['co'], r['error']


# ============================================================
# 比例带 ↔ Kp 换算
# ============================================================

def delta_to_Kp(delta):
    """比例带 → Kp"""
    if abs(delta) < 1e-9:
        return 100.0
    return 100.0 / delta


def Kp_to_delta(Kp):
    """Kp → 比例带"""
    if abs(Kp) < 1e-9:
        return 999.0
    return 100.0 / Kp


# ============================================================
# 1. 临界比例带法
# ============================================================

# 教材表 4-2 整定公式 (delta, TI, TD)
CRITICAL_TABLE = {
    'P':   {'delta_factor': 2.0,  'TI_factor': None, 'TD_factor': None},
    'PI':  {'delta_factor': 2.2,  'TI_factor': 0.85, 'TD_factor': None},
    'PID': {'delta_factor': 1.7,  'TI_factor': 0.5,  'TD_factor': 0.125},
}


def critical_ratio_calculate(delta_cr, T_cr, control_law='PID'):
    """
    根据临界比例带 δ_cr 和临界振荡周期 T_cr 计算整定参数。

    返回:
        dict: {'Kp', 'Ki', 'Kd', 'delta', 'TI', 'TD', 'method_name'}
        或 None (control_law 无效)
    """
    law = control_law.upper()
    if law not in CRITICAL_TABLE:
        return None

    row = CRITICAL_TABLE[law]
    delta = row['delta_factor'] * delta_cr
    Kp = delta_to_Kp(delta)

    Ki = 0.0
    Kd = 0.0
    TI = None
    TD = None

    if row['TI_factor'] is not None:
        TI = row['TI_factor'] * T_cr
        Ki = Kp / TI
    if row['TD_factor'] is not None:
        TD = row['TD_factor'] * T_cr
        Kd = Kp * TD

    return {
        'Kp': round(Kp, 4),
        'Ki': round(Ki, 4),
        'Kd': round(Kd, 4),
        'delta': round(delta, 2),
        'TI': round(TI, 2) if TI else None,
        'TD': round(TD, 2) if TD else None,
        'method_name': f'临界比例带法 ({law})',
    }


def auto_find_critical(K, T, tau, setpoint=1.0, sim_time=80.0, dt=0.01):
    """
    自动搜索临界比例带 δ_cr 和临界振荡周期 T_cr。
    使用纯比例控制, 二分搜索等幅振荡临界点。

    返回:
        dict: {'delta_cr', 'T_cr', 'Kp_cr', 'success', 'message'}
    """
    if tau <= 0:
        return {'success': False, 'message': '一阶无迟延系统不会产生等幅振荡，请设置 τ > 0'}

    # 二分搜索范围
    Kp_low, Kp_high = 0.1, 50.0
    # 先找到能产生振荡的高 Kp
    found_osc = False
    for _ in range(20):
        t, sp, pv, co, err = _sim(K, T, tau, Kp_high, 0, 0, setpoint, sim_time, dt)
        m = compute_metrics(t, sp, pv, err)
        if m['overshoot'] > 5:
            found_osc = True
            break
        Kp_high *= 2
        if Kp_high > 500:
            break

    if not found_osc:
        # 尝试更高的 Kp
        for _ in range(10):
            Kp_high *= 1.5
            t, sp, pv, co, err = _sim(K, T, tau, Kp_high, 0, 0, setpoint, sim_time, dt)
            if max(pv) > setpoint * 2:
                found_osc = True
                break
            if Kp_high > 2000:
                break

    if not found_osc:
        return {'success': False, 'message': f'未找到能引起振荡的 Kp (已尝试到 Kp={Kp_high:.0f})'}

    # 二分搜索临界 Kp
    best_Kp = Kp_high
    best_delta_cr = None
    best_T_cr = None

    for _ in range(25):
        Kp_mid = (Kp_low + Kp_high) / 2
        t, sp, pv, co, err = _sim(K, T, tau, Kp_mid, 0, 0, setpoint, sim_time, dt)
        m = compute_metrics(t, sp, pv, err)

        # 判断是否有持续振荡（超调 > 5% 且有多个峰值）
        peaks = _count_significant_peaks(t, pv, m['final_value'])
        if m['overshoot'] < 2 or peaks < 2:
            Kp_low = Kp_mid  # 增益不够, 加大
        else:
            Kp_high = Kp_mid  # 已振荡, 减小增益
            best_Kp = Kp_mid
            # 计算振荡周期
            T_cr = _measure_period(t, pv, m['final_value'])
            if T_cr:
                best_T_cr = T_cr
                best_delta_cr = Kp_to_delta(Kp_mid)

    if best_delta_cr is None or best_T_cr is None:
        return {'success': False, 'message': '未能精确定位临界振荡点'}

    return {
        'success': True,
        'delta_cr': round(best_delta_cr, 2),
        'T_cr': round(best_T_cr, 3),
        'Kp_cr': round(best_Kp, 4),
        'message': f'临界比例带={best_delta_cr:.1f}%, 振荡周期={best_T_cr:.2f}s',
    }


def manual_critical_measure(K, T, tau, Kp_cr, setpoint=1.0, sim_time=80.0, dt=0.01):
    """
    手动临界比例带法: 用户已将系统调至等幅振荡, 提供当前 Kp_cr,
    程序自动测量振荡周期 T_cr 并返回 δ_cr 和 T_cr。

    返回同 auto_find_critical 格式的 dict。
    """
    if tau <= 0:
        return {'success': False, 'message': '一阶无迟延系统不会产生等幅振荡，请设置 τ > 0'}

    t, sp, pv, co, err = _sim(K, T, tau, Kp_cr, 0.0, 0.0, setpoint, sim_time, dt)
    m = compute_metrics(t, sp, pv, err)

    T_cr = _measure_period(t, pv, m['final_value'])
    if T_cr is None:
        # 尝试用峰值个数判断是否振荡
        n_peaks = _count_significant_peaks(t, pv, m['final_value'])
        if n_peaks < 3:
            return {'success': False,
                    'message': f'未检测到等幅振荡 (仅{n_peaks}个显著峰)，请增大Kp直到出现等幅振荡'}
        return {'success': False, 'message': '无法测量振荡周期，请确认系统处于等幅振荡状态'}

    delta_cr = Kp_to_delta(Kp_cr)

    return {
        'success': True,
        'delta_cr': round(delta_cr, 2),
        'T_cr': round(T_cr, 3),
        'Kp_cr': round(Kp_cr, 4),
        'message': f'临界比例带={delta_cr:.1f}%, 振荡周期={T_cr:.2f}s (手动记录)',
    }


def _count_significant_peaks(t, pv, pv_final):
    """统计超过稳态值的显著峰值个数"""
    from simulation import find_peaks
    peaks_idx = find_peaks(pv)
    count = 0
    for idx in peaks_idx:
        if pv[idx] > pv_final * 1.02:
            count += 1
    return count


def _measure_period(t, pv, pv_final):
    """测量振荡周期 (连续两个峰值的时间差)"""
    from simulation import find_peaks
    peaks_idx = find_peaks(pv)
    sig_peaks = [idx for idx in peaks_idx if pv[idx] > pv_final * 1.02]
    if len(sig_peaks) >= 2:
        return t[sig_peaks[1]] - t[sig_peaks[0]]
    return None


# ============================================================
# 2. 衰减曲线法
# ============================================================

DECAY_TABLE = {
    # phi=0.75
    0.75: {
        'P':   {'delta_factor': 1.0,  'TI_factor': None, 'TD_factor': None},
        'PI':  {'delta_factor': 1.2,  'TI_factor': 0.5,  'TD_factor': None},
        'PID': {'delta_factor': 0.8,  'TI_factor': 0.3,  'TD_factor': 0.1},
    },
    # phi=0.9
    0.9: {
        'P':   {'delta_factor': 1.0,  'TI_factor': None, 'TD_factor': None},
        'PI':  {'delta_factor': 1.2,  'TI_factor': 2.0,  'TD_factor': None},
        'PID': {'delta_factor': 0.8,  'TI_factor': 1.2,  'TD_factor': 0.4},
    },
}


def decay_curve_calculate(delta_s, T_s_or_tr, phi=0.75, control_law='PID'):
    """
    根据衰减曲线法计算整定参数。

    参数:
        delta_s: 达到目标衰减比时的比例带
        T_s_or_tr: φ=0.75时=振荡周期Ts, φ=0.9时=上升时间tr
        phi: 目标衰减率 0.75 或 0.9
        control_law: 'P' / 'PI' / 'PID'

    返回 dict 或 None
    """
    law = control_law.upper()
    if phi not in DECAY_TABLE or law not in DECAY_TABLE[phi]:
        return None

    row = DECAY_TABLE[phi][law]
    delta = row['delta_factor'] * delta_s
    Kp = delta_to_Kp(delta)

    Ki = 0.0
    Kd = 0.0
    TI = None
    TD = None

    if row['TI_factor'] is not None:
        TI = row['TI_factor'] * T_s_or_tr
        Ki = Kp / TI if TI > 0 else 0
    if row['TD_factor'] is not None:
        TD = row['TD_factor'] * T_s_or_tr
        Kd = Kp * TD if TD > 0 else 0

    phi_label = f'φ={phi}'
    Ts_label = 'Ts' if phi == 0.75 else 'tr'

    return {
        'Kp': round(Kp, 4),
        'Ki': round(Ki, 4),
        'Kd': round(Kd, 4),
        'delta': round(delta, 2),
        'TI': round(TI, 2) if TI else None,
        'TD': round(TD, 2) if TD else None,
        'method_name': f'衰减曲线法 {phi_label} ({law})',
    }


def find_decay_curve_params(K, T, tau, phi=0.75, setpoint=1.0,
                              sim_time=60.0, dt=0.01):
    """
    自动调整纯比例控制, 寻找达到目标衰减率 φ 时的 δ_s 和 T_s (或 tr)。

    返回:
        dict: {'success', 'delta_s', 'T_s_or_tr', 'Kp_s', 'message'}
    """
    if tau <= 0:
        return {'success': False, 'message': '一阶无迟延系统无振荡，请设置 τ > 0'}

    Kp_low, Kp_high = 0.5, 30.0
    best_Kp = None
    best_delta_s = None
    best_T_s = None

    for _ in range(30):
        Kp_mid = (Kp_low + Kp_high) / 2
        t, sp, pv, co, err = _sim(K, T, tau, Kp_mid, 0, 0, setpoint, sim_time, dt)
        m = compute_metrics(t, sp, pv, err)

        # 判断是否发散 (超调过大 或 衰减率为负)
        diverging = (max(pv) > setpoint * 5 or m['overshoot'] > 300)

        dr = m.get('decay_rate')
        if dr is None:
            if diverging:
                Kp_high = Kp_mid   # 发散了, 减小增益
            else:
                Kp_low = Kp_mid    # 无振荡, 加大增益
            continue

        if diverging or dr < -0.5:
            Kp_high = Kp_mid       # 发散或负衰减 → 减增益
        elif dr < phi - 0.03:
            Kp_high = Kp_mid       # 衰减太慢(振荡剧烈) → 减增益
        elif dr > phi + 0.03:
            Kp_low = Kp_mid        # 衰减太快(接近无振荡) → 加增益
        else:
            # 找到目标
            best_Kp = Kp_mid
            best_delta_s = Kp_to_delta(Kp_mid)
            if phi == 0.75:
                best_T_s = _measure_period(t, pv, m['final_value'])
            else:
                best_T_s = m.get('rise_time')
            break

        # 记录最佳近似
        if dr is not None and dr > 0 and abs(dr - phi) < 0.1:
            best_Kp = Kp_mid
            best_delta_s = Kp_to_delta(Kp_mid)
            if phi == 0.75:
                best_T_s = _measure_period(t, pv, m['final_value'])
            else:
                best_T_s = m.get('rise_time')

    if best_delta_s is None:
        return {'success': False, 'message': f'未能找到 φ={phi} 的衰减曲线'}

    return {
        'success': True,
        'delta_s': round(best_delta_s, 2),
        'T_s_or_tr': round(best_T_s, 3) if best_T_s else None,
        'Kp_s': round(best_Kp, 4) if best_Kp else None,
        'message': f'φ={phi}: δ_s={best_delta_s:.1f}%',
    }


# ============================================================
# 3. 响应曲线法 / 动态特性参数法
# ============================================================

def response_curve_calculate(K, T, tau, control_law='PID'):
    """
    响应曲线法 (动态特性参数法)。
    适用于有自平衡能力的对象 G(s) = K*e^(-tau*s)/(Ts+1)。

    教材表 4-4 / 4-5:
      ε = K / T (对象特征参数)
      tau/T < 0.2 → 表 4-4 近似公式
      0.2 ≤ tau/T ≤ 1.5 → 表 4-5

    返回 dict 或 None
    """
    law = control_law.upper()
    if T <= 0 or tau <= 0:
        return None

    epsilon = K / T
    ratio = tau / T

    if ratio < 0.2:
        # 近似公式 (表 4-4)
        coeffs = {
            'P':   (1.0,  None,  None),
            'PI':  (1.2,  3.3,   None),
            'PID': (0.8,  2.0,   0.5),
        }
        if law not in coeffs:
            return None
        c_delta, c_TI, c_TD = coeffs[law]
        delta = c_delta * epsilon * tau * 100.0  # 比例制 → 百分制
        TI = c_TI * tau if c_TI else None
        TD = c_TD * tau if c_TD else None

    elif ratio <= 1.5:
        # 精确公式 (表 4-5)
        exact = _response_curve_exact(ratio, epsilon, T)
        if law not in exact:
            return None
        delta, TI, TD = exact[law]
        delta *= 100.0  # 比例制 → 百分制
    else:
        return None

    # 保护: delta 不能为负
    if delta <= 0:
        return None

    Kp = delta_to_Kp(delta)
    Ki = Kp / TI if TI and TI > 0 else 0.0
    Kd = Kp * TD if TD and TD > 0 else 0.0

    return {
        'Kp': round(Kp, 4),
        'Ki': round(Ki, 4),
        'Kd': round(Kd, 4),
        'delta': round(delta, 2),
        'TI': round(TI, 2) if TI else None,
        'TD': round(TD, 2) if TD else None,
        'epsilon': round(epsilon, 4),
        'tau_T_ratio': round(ratio, 4),
        'method_name': f'响应曲线法 ({law}, τ/T={ratio:.2f})',
    }


def _response_curve_exact(ratio, epsilon, T):
    """表 4-5: 0.2 < tau/T <= 1.5 的精确公式"""
    tau = ratio * T  # 从 ratio = tau/T 反算
    return {
        'P': (
            2.6 * epsilon * T * (ratio - 0.8) / (ratio + 0.7),
            None,
            None,
        ),
        'PI': (
            2.6 * epsilon * T * (ratio - 0.08) / (ratio + 0.6),
            0.8 * T,
            None,
        ),
        'PID': (
            2.6 * epsilon * T * (ratio - 0.15) / (ratio + 0.88),
            tau,
            0.25 * tau,
        ),
    }


# ============================================================
# 4. 批量对比
# ============================================================

def batch_compare(K, T, tau, methods, setpoint=1.0, sim_time=30.0, dt=0.01):
    """
    批量运行多组PID参数并对比。

    methods: [{'name': ..., 'Kp': ..., 'Ki': ..., 'Kd': ...}, ...]

    返回 list of dict, 每个包含 name + metrics
    """
    results = []
    for m in methods:
        t, sp, pv, co, err = _sim(K, T, tau,
                                   m['Kp'], m['Ki'], m['Kd'],
                                   setpoint, sim_time, dt)
        metrics = compute_metrics(t, sp, pv, err)
        results.append({
            'name': m['name'],
            'Kp': m['Kp'],
            'Ki': m['Ki'],
            'Kd': m['Kd'],
            't': t, 'sp': sp, 'pv': pv, 'co': co, 'error': err,
            'metrics': metrics,
        })
    return results

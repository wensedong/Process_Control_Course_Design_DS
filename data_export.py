#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据导出模块
- 保存图像 (.png / .svg)
- 导出仿真数据 (.csv)
"""

import csv
import os
from datetime import datetime


def _get_default_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_results')
    os.makedirs(d, exist_ok=True)
    return d


def _timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def save_figure(fig, path, fmt='png'):
    """保存 matplotlib figure"""
    fig.savefig(path, dpi=150, bbox_inches='tight', format=fmt)


def save_csv(path, K, T, tau, Kp, Ki, Kd, sp_val, metrics,
             t, sp, pv, co, error, method_name='手动整定', stable=True):
    """导出仿真数据到CSV"""
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['# PID仿真数据'])
        w.writerow(['# 时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        w.writerow(['# 整定方法', method_name])
        w.writerow(['# 模型', f'K={K}, T={T}, tau={tau}, G(s)=K*e^(-tau*s)/(Ts+1)'])
        w.writerow(['# PID', f'Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}'])
        w.writerow(['# 设定值', str(sp_val)])
        w.writerow(['# 稳定', '是' if stable else '否 (发散)'])
        w.writerow(['#'])
        w.writerow(['# 性能指标'])
        for key, label in [
            ('rise_time', '上升时间(s)'), ('overshoot', '超调量(%)'),
            ('settling_time', '调节时间(s)'), ('steady_state_error', '稳态误差(%)'),
            ('final_value', '稳态值'), ('iae', 'IAE'), ('ise', 'ISE'),
            ('decay_ratio', '衰减比'), ('decay_rate', '衰减率')]:
            val = metrics.get(key)
            w.writerow([f'# {label}', f'{val:.6f}' if val is not None else 'N/A'])
        w.writerow(['#'])
        w.writerow(['Time(s)', 'Setpoint', 'ProcessVariable', 'ControlOutput', 'Error'])
        for i in range(len(t)):
            w.writerow([f'{t[i]:.6f}', f'{sp[i]:.6f}',
                        f'{pv[i]:.6f}', f'{co[i]:.6f}', f'{error[i]:.6f}'])

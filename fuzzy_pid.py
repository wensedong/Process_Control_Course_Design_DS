#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模糊 PID 控制器模块

原理:
  - 输入: 误差 e = SP - PV, 误差变化率 ec = e(k) - e(k-1)
  - 模糊化: e 和 ec 映射到 7 个模糊集合 {NB,NM,NS,ZO,PS,PM,PB}
  - 规则推理: 49 条规则 → ΔKp, ΔKi, ΔKd
  - 解模糊: 重心法 (Centroid) 得到精确修正量
  - 在线调整: Kp' = Kp_base + scale × ΔKp (同 Ki, Kd)

参考文献:
  - 模糊PID控制器的典型设计 (Fuzzy PID Controller Design)
  - 隶属度函数: 三角型 (Triangular MF)
  - 规则表: 工程常用的 7×7 规则库

用法:
  from fuzzy_pid import FuzzyPIDController
  ctrl = FuzzyPIDController(Kp_base=2.0, Ki_base=0.5, Kd_base=0.1, dt=0.01)
  u = ctrl.compute(setpoint, pv)  # 返回值同标准PIDController
"""

import numpy as np


# ============================================================
# 模糊集合定义
# ============================================================

# 语言变量索引
NB, NM, NS, ZO, PS, PM, PB = 0, 1, 2, 3, 4, 5, 6
FUZZY_LABELS = ['NB', 'NM', 'NS', 'ZO', 'PS', 'PM', 'PB']

# 三角隶属度函数参数: (左底, 顶点, 右底) — 定义在 [-1, 1] 上
TRI_MF = {
    NB: (-1.0, -1.0, -0.67),
    NM: (-1.0, -0.67, -0.33),
    NS: (-0.67, -0.33, 0.0),
    ZO: (-0.33, 0.0, 0.33),
    PS: (0.0, 0.33, 0.67),
    PM: (0.33, 0.67, 1.0),
    PB: (0.67, 1.0, 1.0),
}

# 输出单点值 (用于重心法解模糊)
OUTPUT_SINGLETON = {
    NB: -1.0,
    NM: -0.67,
    NS: -0.33,
    ZO: 0.0,
    PS: 0.33,
    PM: 0.67,
    PB: 1.0,
}


# ============================================================
# 模糊规则表 (7×7 → ΔKp, ΔKi, ΔKd)
# ============================================================

# ΔKp 规则: 大误差 → 大比例作用快速消除误差;
#          接近稳态 → 减小比例避免超调
RULE_DELTA_KP = [
    #  ec: NB     NM     NS     ZO     PS     PM     PB     ← ec
    [    PB,    PB,    PM,    PM,    PS,    ZO,    ZO  ],  # e=NB
    [    PB,    PB,    PM,    PS,    PS,    ZO,    NS  ],  # e=NM
    [    PM,    PM,    PM,    PS,    ZO,    NS,    NS  ],  # e=NS
    [    PM,    PM,    PS,    ZO,    NS,    NM,    NM  ],  # e=ZO
    [    PS,    PS,    ZO,    NS,    NS,    NM,    NM  ],  # e=PS
    [    PS,    ZO,    NS,    NM,    NM,    NM,    NB  ],  # e=PM
    [    ZO,    ZO,    NM,    NM,    NM,    NB,    NB  ],  # e=PB
]

# ΔKi 规则: 误差大时减少积分防止饱和;
#          误差小时增加积分以消除稳态误差
RULE_DELTA_KI = [
    #  ec: NB     NM     NS     ZO     PS     PM     PB
    [    NB,    NB,    NM,    NM,    NS,    ZO,    ZO  ],  # e=NB
    [    NB,    NB,    NM,    NS,    NS,    ZO,    ZO  ],  # e=NM
    [    NB,    NM,    NS,    NS,    ZO,    PS,    PS  ],  # e=NS
    [    NM,    NM,    NS,    ZO,    PS,    PM,    PM  ],  # e=ZO
    [    NM,    NS,    ZO,    PS,    PS,    PM,    PB  ],  # e=PS
    [    ZO,    ZO,    PS,    PS,    PM,    PB,    PB  ],  # e=PM
    [    ZO,    ZO,    PS,    PM,    PM,    PB,    PB  ],  # e=PB
]

# ΔKd 规则: 误差变化快时增大微分以抑制振荡;
#          稳态附近减小微分避免噪声放大
RULE_DELTA_KD = [
    #  ec: NB     NM     NS     ZO     PS     PM     PB
    [    PS,    NS,    NB,    NB,    NB,    NM,    PS  ],  # e=NB
    [    PS,    NS,    NB,    NM,    NM,    NS,    ZO  ],  # e=NM
    [    ZO,    NS,    NM,    NM,    NS,    NS,    ZO  ],  # e=NS
    [    ZO,    NS,    NS,    NS,    NS,    NS,    ZO  ],  # e=ZO
    [    ZO,    ZO,    ZO,    ZO,    ZO,    ZO,    ZO  ],  # e=PS
    [    PB,    NS,    PS,    PS,    PS,    PS,    PB  ],  # e=PM
    [    PB,    PM,    PM,    PM,    PS,    PS,    PB  ],  # e=PB
]


# ============================================================
# 隶属度计算 (三角型)
# ============================================================

def _trimf(x, a, b, c):
    """三角隶属度: 返回 x 对三角 (a, b, c) 的隶属度 [0, 1]"""
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    else:  # b < x < c
        return (c - x) / (c - b) if c > b else 1.0


def _fuzzify(value):
    """
    将归一化值 (∈ [-1, 1]) 模糊化 → 返回各集合的隶属度数组 (7,)
    """
    mu = np.zeros(7)
    for i in range(7):
        a, b, c = TRI_MF[i]
        mu[i] = _trimf(value, a, b, c)
    return mu


# ============================================================
# 模糊推理 + 解模糊
# ============================================================

def _infer_and_defuzzify(mu_e, mu_ec, rule_table):
    """
    对给定规则表进行模糊推理 (Mamdani min) 并用重心法解模糊。

    返回: 精确输出值 ∈ [-1, 1]
    """
    numerator = 0.0
    denominator = 0.0

    for i in range(7):      # e 的模糊集
        for j in range(7):  # ec 的模糊集
            # Mamdani: 取较小隶属度作为规则触发强度
            strength = min(mu_e[i], mu_ec[j])
            if strength > 1e-9:
                rule_out = rule_table[i][j]         # 规则结论 (NB~PB)
                singleton = OUTPUT_SINGLETON[rule_out]  # 输出单点值
                numerator += strength * singleton
                denominator += strength

    if denominator < 1e-9:
        return 0.0
    return numerator / denominator


# ============================================================
# 模糊 PID 控制器
# ============================================================

class FuzzyPIDController:
    """
    模糊自适应 PID 控制器。

    参数:
        Kp_base, Ki_base, Kd_base: 基础 PID 参数 (可从教材方法获得)
        scale_Kp, scale_Ki, scale_Kd: 模糊输出缩放因子
        e_max: 误差论域上限 (超出则饱和)
        ec_max: 误差变化率论域上限
        dt: 采样时间
        output_min, output_max: 控制量限幅
    """

    def __init__(self, Kp_base=1.0, Ki_base=0.0, Kd_base=0.0,
                 scale_Kp=0.5, scale_Ki=0.1, scale_Kd=0.05,
                 e_max=1.0, ec_max=0.5,
                 dt=0.01, output_min=-10.0, output_max=10.0):
        # 基础参数
        self.Kp_base = Kp_base
        self.Ki_base = Ki_base
        self.Kd_base = Kd_base

        # 当前参数 (初始 = 基础值)
        self.Kp = Kp_base
        self.Ki = Ki_base
        self.Kd = Kd_base

        # 模糊缩放
        self.scale_Kp = scale_Kp
        self.scale_Ki = scale_Ki
        self.scale_Kd = scale_Kd

        # 论域
        self.e_max = max(e_max, 0.01)
        self.ec_max = max(ec_max, 0.001)

        # PID 内部状态
        self.dt = dt
        self.output_min = output_min
        self.output_max = output_max
        self.reset()

    def reset(self):
        """重置内部状态"""
        self.integral = 0.0
        self.prev_pv = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def compute(self, setpoint, pv):
        """
        每步调用: 模糊推理 → 更新 Kp/Ki/Kd → 计算控制量。

        返回: 控制量 u (float)
        """
        # ① 计算误差和误差变化率
        error = setpoint - pv
        if self.first_run:
            ec = 0.0
        else:
            ec = error - self.prev_error

        # ② 归一化到 [-1, 1]
        e_norm = max(-1.0, min(1.0, error / self.e_max))
        ec_norm = max(-1.0, min(1.0, ec / self.ec_max))

        # ③ 模糊化
        mu_e = _fuzzify(e_norm)
        mu_ec = _fuzzify(ec_norm)

        # ④ 推理 + 解模糊 → ΔKp, ΔKi, ΔKd ∈ [-1, 1]
        dKp_fuzzy = _infer_and_defuzzify(mu_e, mu_ec, RULE_DELTA_KP)
        dKi_fuzzy = _infer_and_defuzzify(mu_e, mu_ec, RULE_DELTA_KI)
        dKd_fuzzy = _infer_and_defuzzify(mu_e, mu_ec, RULE_DELTA_KD)

        # ⑤ 更新 PID 参数 (基础值 + 模糊修正, 并限幅为正)
        self.Kp = max(0.01, self.Kp_base + self.scale_Kp * dKp_fuzzy)
        self.Ki = max(0.0,  self.Ki_base + self.scale_Ki * dKi_fuzzy)
        self.Kd = max(0.0,  self.Kd_base + self.scale_Kd * dKd_fuzzy)

        # ⑥ 标准 PID 计算 (对 PV 微分, 带抗积分饱和)
        P = self.Kp * error

        candidate_integral = self.integral + error * self.dt
        raw = P + self.Ki * candidate_integral

        if self.first_run:
            D = 0.0
            self.first_run = False
        else:
            D = self.Kd * (-(pv - self.prev_pv) / self.dt)

        raw += D
        output = max(self.output_min, min(self.output_max, raw))

        # 抗饱和
        if output == raw or (
            output == self.output_max and error < 0
        ) or (
            output == self.output_min and error > 0
        ):
            self.integral = candidate_integral

        self.prev_pv = pv
        self.prev_error = error

        return output

    # ---- 调试接口 ----

    def debug_step(self, setpoint, pv):
        """
        单步调试版: 返回 (output, debug_info)
        debug_info 包含 e, ec, mu_e, mu_ec, dKp, dKi, dKd
        """
        error = setpoint - pv
        ec = error - self.prev_error if not self.first_run else 0.0
        e_norm = max(-1.0, min(1.0, error / self.e_max))
        ec_norm = max(-1.0, min(1.0, ec / self.ec_max))
        mu_e = _fuzzify(e_norm)
        mu_ec = _fuzzify(ec_norm)
        dKp = _infer_and_defuzzify(mu_e, mu_ec, RULE_DELTA_KP)
        dKi = _infer_and_defuzzify(mu_e, mu_ec, RULE_DELTA_KI)
        dKd = _infer_and_defuzzify(mu_e, mu_ec, RULE_DELTA_KD)

        output = self.compute(setpoint, pv)

        debug = {
            'error': error,
            'ec': ec,
            'e_norm': e_norm,
            'ec_norm': ec_norm,
            'mu_e': mu_e,
            'mu_ec': mu_ec,
            'dKp_fuzzy': dKp,
            'dKi_fuzzy': dKi,
            'dKd_fuzzy': dKd,
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd,
        }
        return output, debug


# ============================================================
# 辅助: 打印规则表 (学习用)
# ============================================================

def print_rule_table(rule_table, name):
    """打印模糊规则表供学习和调试"""
    print(f"\n{'='*60}")
    print(f"  模糊规则表: {name}")
    print(f"{'='*60}")
    header = "e\\ec  " + "  ".join(f"{FUZZY_LABELS[j]:>4}" for j in range(7))
    print(header)
    print("-" * len(header))
    for i in range(7):
        row = f"{FUZZY_LABELS[i]:>4}  " + "  ".join(
            f"{FUZZY_LABELS[rule_table[i][j]]:>4}" for j in range(7))
        print(row)
    print()


def print_all_rules():
    """打印全部三张规则表"""
    print_rule_table(RULE_DELTA_KP, "ΔKp (比例增益修正)")
    print_rule_table(RULE_DELTA_KI, "ΔKi (积分增益修正)")
    print_rule_table(RULE_DELTA_KD, "ΔKd (微分增益修正)")
    print("规则解读:")
    print("  e   = SP - PV      (当前误差)")
    print("  ec  = e(k) - e(k-1) (误差变化率)")
    print("  NB..PB = 负大, 负中, 负小, 零, 正小, 正中, 正大")
    print("  ΔKp: 大误差 → 加大比例快速消除; 稳态 → 减小避免超调")
    print("  ΔKi: 大误差 → 减小积分防饱和; 小误差 → 增大消除静差")
    print("  ΔKd: |ec|大 → 加大微分抑制振荡; 稳态 → 减小防噪声")


if __name__ == '__main__':
    print_all_rules()

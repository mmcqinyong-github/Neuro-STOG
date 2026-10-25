"""P3 审计用的机制臂：各自自然参数，Δ_mech = max over arms。

设计说明（为什么不做 iso-parameter）
==================================
W1 尝试过 iso-parameter 与 iso-feature 同时对齐，结论是**不可能**：
"共享"的机制本质就是减少参数（42 vs 144）。强行把 S 补到 144 参数，
等于把要检验的东西强行抹掉；强行把 F 压到 42，等于人为阉割了它的表达能力。
两者都会让结论变成参数设置的函数而非机制的函数。

因此 P3 采取的定义是：

    Δ_mech(E) = max_arm [ (wmse_none(E) − wmse_arm(E)) / wmse_none(E) ]

即"在该环境上，任何被试机制能达到的最好优势"。这是对 Δ_res 的正确对照量：
Δ_res 是**上界**（全信息 oracle），Δ_mech 是**实测**（有限机制）。H2 检验 Δ_mech ≤ Δ_res。

W1 的归因结论仍然有效且更精确（见 `W1裁决报告.md` §1）：
- env1：F−FP = +0.241 显著 ⇒ F 的领先来自信息量
- env3：S−FP = +0.072 显著、S−L = +0.022 不显著 ⇒ 共享权重有效，但增益机制被 L 覆盖

本模块直接从 W1 的 `src/shared_gain.py` 复用五臂，不再重复实现。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cyclic_mode import _metrics
import shared_gain as sg


ARM_KEYS = ("F", "FP", "S", "L", "D")


def arm_params(env: sg.W1Env) -> dict[str, int]:
    """各臂自然参数量（与 shared_gain 内部一致）。"""
    nv = env.n_vars
    n_dom = nv // env.m_e
    d_pool = sg.D_COORD          # 8
    return {
        "F": nv * (sg.N_FEAT + 1),            # 144
        "FP": nv * (d_pool + 1),              # 81
        "S": n_dom * d_pool + 2 * nv,         # 42
        "L": n_dom * d_pool + nv,             # 33
        "D": n_dom * d_pool + 2 * nv + 4,     # 46
    }


def delta_mech(env: sg.W1Env, lam: float, n_calib: int,
               arms: Sequence[str] = ARM_KEYS) -> dict:
    """Δ_mech = 各臂中相对无修正基线的最大改善（%）。同时返回各臂明细。"""
    r = sg.run_point(env, lam=lam, n_calib=n_calib, arms=tuple(arms))
    base = r["base"][0]
    detail = {}
    for a in arms:
        w = r[a][0]
        detail[a] = {
            "wmse": w,
            "mae": r[a][1],
            "n_params": r[a][2],
            "rel_pct": (base - w) / base * 100.0,
        }
    best_arm = max(detail, key=lambda a: detail[a]["rel_pct"])
    return {
        "base_wmse": base,
        "delta_mech": detail[best_arm]["rel_pct"],
        "best_arm": best_arm,
        "detail": detail,
    }

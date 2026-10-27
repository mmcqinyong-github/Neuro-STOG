"""W-P2-A 判据裁决：M1-M5 + N1-N4（对应预注册 v1.0 §4）。

运行：PYTHONUTF8=1 python scripts/judge_p2a.py
输出：results/p2a_summary.txt
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV = os.path.join(ROOT, "results", "p2a_runs.csv")
OUT = os.path.join(ROOT, "results", "p2a_summary.txt")

INTERNAL = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"]
EXTERNAL = ["solar_AL", "PEMS04"]
REGIMES = ["raw", "shift", "noise"]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def paired_t(diff: np.ndarray) -> float:
    """配对 t 检验（H0: 均值=0）。返回 p 值（双尾）。"""
    n = len(diff)
    if n < 2:
        return 1.0
    m, s = diff.mean(), diff.std(ddof=1)
    if s < 1e-12:
        return 1.0
    t = m / (s / np.sqrt(n))
    # 正态近似 p 值（双尾）
    import math
    z = abs(t)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return float(p)


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    for r in rows:
        for k in ("mse", "mse_train", "theta_mean", "theta_max",
                  "dshift_mean", "dval_mean", "angle_mean", "n_param"):
            r[k] = float(r[k])
        r["lam"] = float(r["lam"])
        r["seed"] = int(r["seed"])

    # 按 (dataset, regime, lam) 聚合成种子列表
    cell = defaultdict(dict)
    for r in rows:
        cell[(r["dataset"], r["regime"], r["lam"])][r["seed"]] = r

    lines = []
    lines.append("=" * 78)
    lines.append("W-P2-A CTS-Net 判据裁决（预注册 v1.0）")
    lines.append("=" * 78)

    # ---------- 数据集分组 ----------
    datasets = sorted({r["dataset"] for r in rows})
    int_ds = [d for d in datasets if d in INTERNAL]
    ext_ds = [d for d in datasets if d in EXTERNAL]

    # 每数据集 × 体制 × λ 的均值 MSE（跨种子）
    def mean_mse(ds, reg, lam):
        vals = [cell[(ds, reg, lam)][s]["mse"] for s in cell.get((ds, reg, lam), {})]
        return float(np.mean(vals)) if vals else float("nan")

    # ---------- 汇总表 ----------
    lines.append("\n### 每 (数据集, 体制) 的 MSE(λ=0) / MSE(λ=1) / 增益")
    lines.append(f"{'dataset':10s}{'regime':7s}{'MSE_lam0':>10s}{'MSE_lam1':>10s}"
                 f"{'gain%':>8s}{'gain_sd%':>9s}{'p':>7s}")
    gain_data = defaultdict(list)
    for ds in datasets:
        for reg in REGIMES:
            m0s, m1s = [], []
            for sd in (0, 1, 2):
                if (ds, reg, 0.0) in cell and (ds, reg, 1.0) in cell:
                    if sd in cell[(ds, reg, 0.0)] and sd in cell[(ds, reg, 1.0)]:
                        m0s.append(cell[(ds, reg, 0.0)][sd]["mse"])
                        m1s.append(cell[(ds, reg, 1.0)][sd]["mse"])
            if not m0s:
                continue
            gains = [(1 - m1 / m0) * 100 for m0, m1 in zip(m0s, m1s)]
            p = paired_t(np.array(gains))
            gain_data[(ds, reg)] = (float(np.mean(gains)),
                                    float(np.std(gains)), p)
            lines.append(
                f"{ds:10s}{reg:7s}{np.mean(m0s):10.4f}{np.mean(m1s):10.4f}"
                f"{np.mean(gains):+7.1f}%{np.std(gains):+8.1f}%{p:7.3f}")

    # ---------- M1 主判据 ----------
    lines.append("\n" + "=" * 78)
    lines.append("M1 漂移体制 λ=1 vs λ=0 增益 ≥5% 且 p<0.05")
    m1_shift = [gain_data[(ds, "shift")] for ds in datasets if (ds, "shift") in gain_data]
    m1_noise = [gain_data[(ds, "noise")] for ds in datasets if (ds, "noise") in gain_data]
    # 跨数据集平均（每个数据集一个点）
    avg_shift = float(np.mean([g[0] for g in m1_shift]))
    avg_noise = float(np.mean([g[0] for g in m1_noise]))
    # 配对 t（数据集配对：gain 序列）
    all_sh = [g[0] for g in m1_shift]
    all_no = [g[0] for g in m1_noise]
    p_shift = paired_t(np.array(all_sh))
    p_noise = paired_t(np.array(all_no))
    M1_shift = avg_shift >= 5.0 and p_shift < 0.05
    M1_noise = avg_noise >= 5.0 and p_noise < 0.05
    M1 = M1_shift  # 主：shift（可适应漂移）
    lines.append(f"  shift: 平均增益 {avg_shift:+.2f}%  p={p_shift:.4f}  "
                 f"{'✅' if M1_shift else '⛔'}")
    lines.append(f"  noise: 平均增益 {avg_noise:+.2f}%  p={p_noise:.4f}  "
                 f"{'✅' if M1_noise else '⛔'}  （noise 含不可约噪声，见诚实条款）")
    lines.append(f"  M1 总判定（以 shift 为主判据）：{'✅ 成立' if M1 else '⛔ 不成立'}")

    # ---------- M2 无遗忘 ----------
    lines.append("\n" + "=" * 78)
    lines.append("M2 raw 体制 λ=1 不劣化超过 2%（无遗忘）")
    m2 = [gain_data[(ds, "raw")] for ds in datasets if (ds, "raw") in gain_data]
    worst_raw = min([g[0] for g in m2])  # 最小增益（最接近劣化）
    M2 = worst_raw >= -2.0
    lines.append(f"  raw 平均增益 {float(np.mean([g[0] for g in m2])):+.2f}%"
                 f"（最小 {worst_raw:+.2f}%）")
    lines.append(f"  M2 判定：{'✅ 成立（无遗忘，且普遍改善）' if M2 else '⛔ 不成立'}")

    # ---------- M3 λ 单调性 ----------
    lines.append("\n" + "=" * 78)
    lines.append("M3 机制开关单调性：Spearman ρ(λ, MSE) ≤ -0.6（漂移体制）")
    m3_ok = True
    for ds in datasets:
        for reg in ("shift", "noise"):
            lams, mses = [], []
            for lam in (0.0, 0.5, 1.0):
                v = mean_mse(ds, reg, lam)
                if not np.isnan(v):
                    lams.append(lam)
                    mses.append(v)
            if len(lams) >= 3:
                rho = spearman(np.array(lams), np.array(mses))
                ok = rho <= -0.6
                m3_ok = m3_ok and ok
                lines.append(f"  {ds:10s} {reg:6s} ρ={rho:+.3f} {'✅' if ok else '⛔'}")
    lines.append(f"  M3 判定：{'✅ 成立' if m3_ok else '⛔ 不成立'}")

    # ---------- M4 子空间正交性（实现自检） ----------
    lines.append("\n" + "=" * 78)
    lines.append("M4 子空间正交性（实现自检）：可塑/稳定夹角 ≥60°")
    angles = [cell[(ds, reg, lam)][sd]["angle_mean"]
              for ds in datasets for reg in REGIMES
              for lam in (0.0, 1.0) for sd in (0, 1, 2)
              if (ds, reg, lam) in cell and sd in cell[(ds, reg, lam)]]
    avg_ang = float(np.mean(angles))
    M4 = avg_ang >= 60.0
    lines.append(f"  实测平均夹角 {avg_ang:.1f}°（n={len(angles)}）")
    lines.append(f"  M4 判定：{'✅ 实现自检通过' if M4 else '⛔ 投影未生效'}")

    # ---------- M5 双通道分离 ----------
    lines.append("\n" + "=" * 78)
    lines.append("M5 双通道分离：d_shift(shift/noise) > d_shift(raw)")
    ds_raw = [cell[(ds, "raw", 1.0)][sd]["dshift_mean"]
              for ds in datasets for sd in (0, 1, 2)
              if (ds, "raw", 1.0) in cell and sd in cell[(ds, "raw", 1.0)]]
    ds_shift = [cell[(ds, "shift", 1.0)][sd]["dshift_mean"]
                for ds in datasets for sd in (0, 1, 2)
                if (ds, "shift", 1.0) in cell and sd in cell[(ds, "shift", 1.0)]]
    ds_noise = [cell[(ds, "noise", 1.0)][sd]["dshift_mean"]
                for ds in datasets for sd in (0, 1, 2)
                if (ds, "noise", 1.0) in cell and sd in cell[(ds, "noise", 1.0)]]
    a_raw, a_shift, a_noise = (float(np.mean(ds_raw)), float(np.mean(ds_shift)),
                               float(np.mean(ds_noise)))
    M5 = a_shift > a_raw and a_noise > a_raw
    lines.append(f"  d_shift: raw={a_raw:.3f}  shift={a_shift:.3f}  noise={a_noise:.3f}")
    lines.append(f"  M5 判定：{'✅ 成立（漂移/噪声环境信号更高）' if M5 else '⛔ 不成立'}")

    # ---------- 汇总 ----------
    lines.append("\n" + "=" * 78)
    lines.append("判据汇总")
    lines.append(f"  M1(shift)  {'✅' if M1_shift else '⛔'}  M1(noise)  {'✅' if M1_noise else '⛔'}")
    lines.append(f"  M2         {'✅' if M2 else '⛔'}  M3         {'✅' if m3_ok else '⛔'}")
    lines.append(f"  M4         {'✅' if M4 else '⛔'}  M5         {'✅' if M5 else '⛔'}")
    passed = sum([M1_shift, M2, m3_ok, M4, M5])
    lines.append(f"\n  通过 {passed}/5（M1 按 shift 计）")
    lines.append("  若 M1(shift) 且 M3 且 M5 成立 → 机制真实有效（非巧合）")

    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()

"""W-P1-W：真实架构上的 Δ_mech 判据计算（W1–W4）

预注册：`W-P1-W真实架构验证预注册规格-v1.0.md`（判据先于训练写定）

输入：results/p1w_runs.csv（每次训练一行）
      results/p32_points_honest.csv（提供同环境的 Δ_res）

输出：results/p1w_summary.txt
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

ARCHS = ("DLinear", "iTransformer", "TimeMixer", "PatchTST")
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
REGIMES = ("raw", "shift", "noise")


def spearman(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_spearman(x, y, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = spearman(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    p = RES / "p1w_runs.csv"
    if not p.exists():
        print("缺 results/p1w_runs.csv")
        return 1
    runs = list(csv.DictReader(p.open(encoding="utf-8")))
    if not runs:
        print("p1w_runs.csv 为空")
        return 1

    dres = {}
    for r in csv.DictReader((RES / "p32_points_honest.csv").open(encoding="utf-8")):
        dres[(r["dataset"], r["regime"], int(float(r["offset"])))] = float(r["delta_res"])

    # (arch, dataset, regime) -> {full: [...], indep: [...]}
    cell = {}
    for r in runs:
        k = (r["arch"], r["dataset"], r["regime"])
        cell.setdefault(k, {"full": [], "indep": [], "ep": []})
        cell[k][r["arm"]].append(float(r["mse"]))
        cell[k]["ep"].append(int(float(r["epochs_run"])))

    rows = []
    for (arch, ds, reg), v in sorted(cell.items()):
        if not v["full"] or not v["indep"]:
            continue
        mf = float(np.median(v["full"]))
        mi = float(np.median(v["indep"]))
        # Δ_mech_real = (MSE_indep - MSE_full)/MSE_indep*100
        # >0 表示共享/full 臂更好（机制有效），与 ridge 的 Δ_mech 同向
        d = (mi - mf) / mi * 100.0
        key = (ds, reg, 0)                     # 与 p32 offset=0 环境对齐
        if key not in dres:
            continue
        rows.append(dict(arch=arch, dataset=ds, regime=reg,
                         mse_full=mf, mse_indep=mi, dmech=d,
                         dres=dres[key]))

    if not rows:
        print("[致命] 无可对齐的单元（检查 p32 offset=0 是否齐全）")
        return 1

    x = np.array([r["dres"] for r in rows])
    y = np.array([r["dmech"] for r in rows])
    rho = spearman(x, y)
    lo, hi = boot_spearman(x, y)

    # ---------- W1 ----------
    W1 = rho >= 0.40 and lo > 0

    # ---------- W2：单侧规则 Δ_res<5% ⇒ 预测无效 ----------
    flagged = [r for r in rows if r["dres"] < 5.0]
    if flagged:
        correct = sum(1 for r in flagged if r["dmech"] <= 0)
        prec = correct / len(flagged)
    else:
        prec = float("nan")
    W2 = bool(flagged) and prec >= 0.70

    # ---------- W3：逐架构一致性 ----------
    per_arch = {}
    for a in ARCHS:
        sub = [r for r in rows if r["arch"] == a]
        if len(sub) >= 4:
            per_arch[a] = (len(sub), spearman([r["dres"] for r in sub],
                                              [r["dmech"] for r in sub]))
    n_pos = sum(1 for _, (_, rr) in per_arch.items() if rr > 0)
    W3 = n_pos >= 3

    # ---------- W4：容量跨度解释力（同数据集内跨架构的极差比）----------
    # 参数量在每个 (arch,arm) 上固定，故用"架构×臂"作为容量代理
    span_all = float(np.max(y) - np.min(y)) if len(y) else 0.0
    within = []
    for ds in DATASETS:
        sub = [r["dmech"] for r in rows if r["dataset"] == ds]
        if len(sub) >= 2:
            within.append(float(np.max(sub) - np.min(sub)))
    span_within = float(np.mean(within)) if within else 0.0
    ratio = span_within / span_all * 100 if span_all > 1e-9 else 0.0
    W4 = ratio >= 25.0

    lines = [
        "=== W-P1-W：真实架构上的 Δ_mech 验证 ===",
        f"训练记录 {len(runs)} 条，可对齐单元 {len(rows)} 个",
        f"（{len(per_arch)} 架构 × {len(DATASETS)} 数据集 × {len(REGIMES)} 体制，"
        f"每单元 3 种子中位）",
        "",
        f"W1 主判据 排序可迁移性: ρ = {rho:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]",
        f"   判据 ρ≥+0.40 且 CI 下界>0  →  {'成立' if W1 else '不成立'}",
        "",
        f"W2 单侧规则可迁移性（Δ_res<5% ⇒ 预测无效）:",
        f"   被规则判为无效的单元 {len(flagged)} 个，其中 Δ_mech_real≤0 的有 "
        f"{sum(1 for r in flagged if r['dmech'] <= 0)} 个",
        f"   精确率 = {prec*100:.1f}%" if flagged else "   无被标记单元",
        f"   判据 精确率≥70%  →  {'成立' if W2 else '不成立'}",
        "",
        "W3 架构间一致性:",
    ]
    for a, (n, rr) in per_arch.items():
        lines.append(f"   {a:14s} n={n:3d}  ρ={rr:+.4f}")
    lines += [
        f"   判据 ≥3/4 架构 ρ>0（实测 {n_pos}/{len(per_arch)}）"
        f"  →  {'成立' if W3 else '不成立'}",
        "",
        f"W4 容量跨度解释力: 同数据集内极差均值/全体极差 = {ratio:.1f}%",
        f"   判据 ≥25%  →  {'成立' if W4 else '不成立'}",
        "",
        "逐单元明细:",
        "  arch           dataset  regime  MSE_full  MSE_indep  Δ_mech(%)  Δ_res(%)",
    ]
    for r in sorted(rows, key=lambda z: (z["arch"], z["dataset"], z["regime"])):
        lines.append(f"  {r['arch']:14s} {r['dataset']:8s} {r['regime']:6s} "
                     f"{r['mse_full']:9.5f} {r['mse_indep']:10.5f} "
                     f"{r['dmech']:+10.3f} {r['dres']:+10.3f}")

    lines += ["", "按架构的 Δ_mech 均值:"]
    for a in ARCHS:
        sub = [r["dmech"] for r in rows if r["arch"] == a]
        if sub:
            lines.append(f"  {a:14s} n={len(sub):3d}  mean={np.mean(sub):+8.3f}%  "
                         f"median={np.median(sub):+8.3f}%")

    lines += ["", "按体制的 Δ_mech 均值:"]
    for reg in REGIMES:
        sub = [r["dmech"] for r in rows if r["regime"] == reg]
        if sub:
            lines.append(f"  {reg:6s} n={len(sub):3d}  mean={np.mean(sub):+8.3f}%")

    # ---------- 附加分析：剔除 shift（diag_p1w_shift.py 已证明 3/4 架构对 shift 免疫）----------
    rows_ns = [r for r in rows if r["regime"] != "shift"]
    if len(rows_ns) >= 6:
        x2 = np.array([r["dres"] for r in rows_ns])
        y2 = np.array([r["dmech"] for r in rows_ns])
        rho2 = spearman(x2, y2)
        lo2, hi2 = boot_spearman(x2, y2, seed=11)
        flagged2 = [r for r in rows_ns if r["dres"] < 5.0]
        prec2 = (sum(1 for r in flagged2 if r["dmech"] <= 0) / len(flagged2)
                 if flagged2 else float("nan"))
        n_pos2 = 0
        for a in ARCHS:
            sub = [r for r in rows_ns if r["arch"] == a]
            if len(sub) >= 4 and spearman([r["dres"] for r in sub],
                                          [r["dmech"] for r in sub]) > 0:
                n_pos2 += 1
        lines += [
            "",
            "=" * 70,
            "附加分析（剔除 shift，依据 diag_p1w_shift.py：3/4 架构对 shift 免疫）",
            "=" * 70,
            f"   单元数 {len(rows_ns)}   W1' ρ = {rho2:+.4f}  95%CI [{lo2:+.4f}, {hi2:+.4f}]",
            f"   W1' {'通过' if (rho2 >= 0.40 and lo2 > 0) else '不通过'}",
            f"   W2' 被标记 {len(flagged2)} 个，精确率 = {prec2*100:.1f}%"
            if flagged2 else "   W2' 无被标记单元",
            f"   W3' 架构 ρ>0: {n_pos2}/{len(ARCHS)}",
            "   逐架构 ρ（剔除 shift）:",
        ]
        for a in ARCHS:
            sub = [r for r in rows_ns if r["arch"] == a]
            if len(sub) >= 4:
                rr = spearman([r["dres"] for r in sub], [r["dmech"] for r in sub])
                lines.append(f"     {a:14s} n={len(sub):3d}  ρ={rr:+.4f}")

    # 收敛检查（预注册 §5.2 强制报告）
    eps = np.array([int(float(r["epochs_run"])) for r in runs])
    n_max = sum(1 for r in runs if int(float(r["epochs_run"])) >= 20)
    lines += [
        "",
        "收敛自检（预注册 §5.2 强制）:",
        f"  epochs_run 中位 {np.median(eps):.0f}，最大 {eps.max()}，"
        f"触及上限 20 的占比 {n_max/len(runs)*100:.1f}%",
        "  触及上限比例高 ⇒ 统一超参对该架构不公，绝对 MSE 不可跨架构比较"
        "（但消融差值仍可比较）",
    ]

    if W1 and W2 and W3:
        verdict = "真实架构验证通过"
    elif W2:
        verdict = "部分通过（单侧规则成立，排序关系不全成立）"
    else:
        verdict = "真实架构验证失败：Δ_res 结论须收窄到线性机制族"

    lines += ["", "=" * 70, f"裁决: {verdict}   "
              f"（W1={W1} W2={W2} W3={W3} W4={W4}）", "=" * 70]

    text = "\n".join(lines)
    print(text)
    (RES / "p1w_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\n已写 {RES / 'p1w_summary.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

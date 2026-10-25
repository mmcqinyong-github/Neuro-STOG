"""W1 kill-test 的 20 种子研究：工作域测度 + 精度 + 机制分辨力审计。

预注册来源：`一区论文创作idea-v1.md` §5 的 W1 行，与 §2 P1 的否决条件 N1/N3/N2。

主指标（工作域测度，Working-Domain Measure, WDM）
==============================================
在 (λ, n_calib) 网格上，某臂「工作」的定义 = 该点测试 wmse 严格优于不修正基线。
WDM(arm) = 工作点比例 ∈ [0, 1]。这是预注册里"训练成功且不退化"的直接操作化。

次指标：winsorized MSE（q99 截尾）、MAE、以及相对基线的改善百分比。

审计量：Δ_res（机制分辨力）
==========================
用全信息滞后特征拟合的 oracle 相对基线的最大可达优势。Δ_res ≈ 0 的环境无法分辨任何机制，
在该环境上得到的负结果与正结果同样不承载信息（预注册 P3 的主张）。

裁决规则（写在跑之前，不允许事后调整）
====================================
N1：若 WDM(S) 未显著大于 WDM(F)（配对 bootstrap CI 跨零，且种子胜率 < 15/20）→ "共享扩展工作域"不成立。
N3：若 WDM(S) 未显著大于 WDM(L) 或未显著大于 WDM(FP) → 分组归一化/池化输入已足够，
    生物共享结构无独立贡献，P1 归零，转 P3。
N2：若最优域尺寸 m* 的单调性方向与预注册（随信噪比下降而增大、随异质性上升而减小）相反
    → 只保留工程技巧，删除"律"的主张。

输出：results/w1_{env}_seeds.csv、results/w1_{env}_gridpoints.csv、results/w1_summary.txt
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyclic_mode import _metrics                                    # noqa: E402
import shared_gain as sg                                            # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ARMS = ("F", "FP", "S", "L", "D")
ENVS = ("env1", "env3")
GRID_LAM = (1e-3, 1e-2, 1e-1, 1.0)
GRID_NCAL = (50, 100, 200, 400)
OMEGA_MAIN = 2.0        # 由 Δ_res 审计确定：ω=2 时环境具备可分辨结构
OMEGA_CTRL = 0.0        # 对照：无共享成分，任何共享机制都不该有优势


def bootstrap_ci(values, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    boots = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def delta_res(env: sg.W1Env, n_calib: int = 400, lags: int = 6) -> float:
    """机制分辨力：全信息滞后 oracle 相对无修正基线的最大可达优势（%）。"""
    backbone, calib, test = sg.prepare(env, n_calib)

    def resid(x, upto):
        idx = list(range(env.lookback, env.lookback + upto))
        base = np.stack([sg.predict_backbone_ci(backbone, x[t - env.lookback : t]) for t in idx])
        return x[idx] - base

    def lagfeat(x, upto):
        return np.stack([
            np.concatenate([x[t - l] for l in range(1, lags + 1)])
            for t in range(env.lookback, env.lookback + upto)
        ])

    r_te, r_ca = resid(test, env.n_test), resid(calib, n_calib)
    Xte, Xca = lagfeat(test, env.n_test), lagfeat(calib, n_calib)
    none, _ = _metrics(r_te)
    best = np.inf
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        A = np.concatenate([Xca, np.ones((len(Xca), 1))], axis=1)
        P = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ r_ca)
        pred = np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1) @ P
        w, _ = _metrics(r_te - pred)
        best = min(best, w)
    return float((none - best) / none * 100.0)


def seed_grid(env_name: str, omega: float, seed: int):
    """一个种子在 (λ, n_calib) 网格上的全部点。"""
    rows = []
    for lam in GRID_LAM:
        for nc in GRID_NCAL:
            env = sg.W1Env(seed=seed, env=env_name, shift_weight=omega)
            r = sg.run_point(env, lam=lam, n_calib=nc)
            base = r["base"][0]
            row = {"seed": seed, "env": env_name, "omega": omega, "lam": lam,
                   "n_calib": nc, "base_wmse": base}
            for a in ARMS:
                w = r[a][0]
                row[f"{a}_wmse"] = w
                row[f"{a}_mae"] = r[a][1]
                row[f"{a}_nparam"] = r[a][2]
                row[f"{a}_works"] = int(w < base)
                row[f"{a}_rel_pct"] = (base - w) / base * 100.0
            row["g_S_dev"] = float(np.abs(r["g_S"] - 1).max())
            row["g_D_dev"] = float(np.abs(r["g_D"] - 1).max())
            rows.append(row)
    return rows


def study(env_name: str, omega: float, n_seeds: int = 20, seed0: int = 101):
    tag = f"{env_name}_w{omega:g}"
    points = []
    for i in range(n_seeds):
        points.extend(seed_grid(env_name, omega, seed0 + i))

    with (RESULTS / f"w1_{tag}_gridpoints.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(points[0].keys()))
        w.writeheader()
        w.writerows(points)

    # 逐种子聚合出 WDM
    seeds = sorted({p["seed"] for p in points})
    per_seed = []
    for s in seeds:
        pts = [p for p in points if p["seed"] == s]
        row = {"seed": s, "n_points": len(pts)}
        for a in ARMS:
            row[f"WDM_{a}"] = float(np.mean([p[f"{a}_works"] for p in pts]))
            row[f"medrel_{a}"] = float(np.median([p[f"{a}_rel_pct"] for p in pts]))
            row[f"meanrel_{a}"] = float(np.mean([p[f"{a}_rel_pct"] for p in pts]))
        row["base_wmse_median"] = float(np.median([p["base_wmse"] for p in pts]))
        row["dres"] = delta_res(sg.W1Env(seed=s, env=env_name, shift_weight=omega))
        per_seed.append(row)

    with (RESULTS / f"w1_{tag}_seeds.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        w.writeheader()
        w.writerows(per_seed)

    lines = [f"=== W1 kill-test | {env_name} | omega={omega:g} | {n_seeds} seeds x {len(GRID_LAM)*len(GRID_NCAL)} grid points ==="]
    lines.append(f"delta_res (oracle 可达优势): mean={np.mean([r['dres'] for r in per_seed]):+.2f}%"
                 f"  CI95={bootstrap_ci([r['dres'] for r in per_seed])}")
    lines.append("")
    lines.append(f"{'arm':>4} {'WDM mean':>9} {'CI95':>20} {'medrel%':>9} {'meanrel%':>9} {'nparam':>7}")
    for a in ARMS:
        vals = np.array([r[f"WDM_{a}"] for r in per_seed], dtype=float)
        lo, hi = bootstrap_ci(vals)
        medrel = np.mean([r[f"medrel_{a}"] for r in per_seed])
        meanrel = np.mean([r[f"meanrel_{a}"] for r in per_seed])
        npar = int(np.mean([p[f"{a}_nparam"] for p in points]))
        lines.append(f"{a:>4} {vals.mean():9.3f} [{lo:+7.3f},{hi:+7.3f}] {medrel:9.2f} {meanrel:9.2f} {npar:7d}")

    lines.append("")
    for pair in (("S", "F"), ("S", "L"), ("S", "FP"), ("S", "D")):
        a, b = pair
        d = np.array([r[f"WDM_{a}"] - r[f"WDM_{b}"] for r in per_seed])
        lo, hi = bootstrap_ci(d)
        wins = int((d > 0).sum())
        lines.append(f"WDM({a}) - WDM({b}): mean={d.mean():+.3f} CI95=[{lo:+.3f},{hi:+.3f}]"
                     f" seeds_won={wins}/{n_seeds}")

    txt = "\n".join(lines)
    print(txt)
    (RESULTS / f"w1_{tag}_summary.txt").write_text(txt + "\n", encoding="utf-8")
    return per_seed


if __name__ == "__main__":
    all_summary = {}
    for env_name in ENVS:
        for omega in (OMEGA_MAIN, OMEGA_CTRL):
            ps = study(env_name, omega)
            all_summary[f"{env_name}_w{omega:g}"] = {
                a: float(np.mean([r[f"WDM_{a}"] for r in ps])) for a in ARMS
            }
            print()
    (RESULTS / "w1_overall.json").write_text(
        json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("汇总写入 results/w1_overall.json")

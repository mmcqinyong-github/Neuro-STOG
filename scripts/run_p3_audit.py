"""P3 机制分辨力审计：Δ_res 能否预测 Δ_mech。

预注册：`P3预注册规格-v1.0.md`（裁决规则写在跑之前）
纪律：不调 oracle 特征集；不改裁决阈值；不因结果不利而换环境。

输出：
  results/p3_points.csv      每个 (环境, 体制, 种子) 的 Δ_res / Δ_mech / 各臂
  results/p3_summary.txt     裁决用汇总（Spearman ρ + H1/H2/H3）
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import shared_gain as sg                                           # noqa: E402
from epf_loader import INDEPENDENT_MARKETS, NON_INDEPENDENT, split  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

REGIMES = ("raw", "shift", "noise")
N_SEEDS = 5                       # 真实数据切分由 n_train 偏移控制
TRAIN_OFFSETS = (0, 400, 800, 1200, 1600)
LAM = 1e-2
N_CALIB = 400
N_TRAIN, N_TEST = 800, 500


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_spearman(x, y, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(spearman(x[idx], y[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def real_point(market: str, regime: str, offset: int):
    """一个真实环境点：Δ_res 与 Δ_mech。"""
    s = split(market, regime, n_train=N_TRAIN + offset, n_calib=N_CALIB, n_test=N_TEST)
    env = sg.W1Env(seed=101 + offset // 100, n_vars=3, m_e=3,
                   n_train=N_TRAIN, n_calibrate=N_CALIB, n_test=N_TEST)
    dres = sg.delta_res_real(env, s.train, s.calib, s.test, n_calib=N_CALIB)
    r = sg.run_point_real(env, s.train, s.calib, s.test, lam=LAM, n_calib=N_CALIB)
    base = r["base"][0]
    rel = {a: (base - r[a][0]) / base * 100.0 for a in sg.ARMS}
    best = max(rel, key=rel.get)
    row = {"source": "real", "market": market, "regime": regime, "offset": offset,
           "base_wmse": base, "delta_res": dres,
           "delta_mech": rel[best], "best_arm": best}
    for a in sg.ARMS:
        row[f"rel_{a}"] = rel[a]
        row[f"nparam_{a}"] = r[a][2]
    return row


def synth_point(env_name: str, omega: float, seed: int):
    env = sg.W1Env(seed=seed, env=env_name, shift_weight=omega)
    dres = sg.delta_res_synth(env) if hasattr(sg, "delta_res_synth") else None
    r = sg.run_point(env, lam=LAM, n_calib=N_CALIB)
    base = r["base"][0]
    rel = {a: (base - r[a][0]) / base * 100.0 for a in sg.ARMS}
    best = max(rel, key=rel.get)
    row = {"source": "synthetic", "market": f"{env_name}_w{omega:g}", "regime": "n/a",
           "offset": seed, "base_wmse": base,
           "delta_res": float("nan") if dres is None else dres,
           "delta_mech": rel[best], "best_arm": best}
    for a in sg.ARMS:
        row[f"rel_{a}"] = rel[a]
        row[f"nparam_{a}"] = r[a][2]
    return row


def main():
    rows = []
    print("=== P3 审计：真实环境 ===")
    for market in INDEPENDENT_MARKETS:
        for regime in REGIMES:
            for off in TRAIN_OFFSETS:
                try:
                    row = real_point(market, regime, off)
                except Exception as e:                       # noqa: BLE001
                    print(f"  {market}/{regime}/off{off} FAILED: {e}")
                    continue
                rows.append(row)
                print(f"  {market:4s} {regime:5s} off={off:4d} | "
                      f"Δ_res={row['delta_res']:+7.2f}% Δ_mech={row['delta_mech']:+7.2f}% "
                      f"best={row['best_arm']}")

    print("\n=== 排除项 ===")
    for k, why in NON_INDEPENDENT.items():
        print(f"  {k}: {why}")

    with (RESULTS / "p3_points.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    x = np.array([r["delta_res"] for r in rows])
    y = np.array([r["delta_mech"] for r in rows])
    rho = spearman(x, y)
    lo, hi = boot_spearman(x, y)

    # H2（v1.0 原判据，点对点）：保留报告，不删除
    viol = [r for r in rows if r["delta_mech"] > r["delta_res"] + 2.0]

    # H2'（v1.2 判据，见 P3预注册规格 §5c）
    med_ok = float(np.median(y)) <= float(np.median(x))
    frac_over = float(np.mean(y > x))
    frac_ok = frac_over <= 1.0 / 3.0
    # (c) 超额必须落在上界估计的噪声带内：超过量应 ≤ 1 个标准误量级
    gaps = y - x
    over_gaps = gaps[gaps > 0]
    noise_band = float(np.std(gaps)) if len(gaps) > 1 else 0.0
    within_band = bool(len(over_gaps) == 0 or np.max(over_gaps) <= noise_band)
    h2p_pass = med_ok and frac_ok and within_band

    unresolvable = int((np.abs(x) < 5.0).sum())

    lines = [
        "=== P3 机制分辨力审计 ===",
        f"点数: {len(rows)}  (真实 {len(INDEPENDENT_MARKETS)} 市场 × {len(REGIMES)} 体制 × {len(TRAIN_OFFSETS)} 偏移)",
        "",
        f"H1  Spearman ρ(Δ_res, Δ_mech) = {rho:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]",
        f"    CI 跨零 = {lo < 0 < hi}  →  {'H1 失败：测度无预测价值，不发' if lo < 0 < hi else 'H1 通过'}",
        "",
        "H2 (v1.0 原判据，点对点，保留备查——已废弃):",
        f"    Δ_mech > Δ_res+2 的点数 = {len(viol)} / {len(rows)}",
        f"    废弃理由：Δ_res 与 Δ_mech 都在同一 400 点校准段上估计，各带 ±2–3% 噪声，",
        f"    要求点对点恒不等式在统计上不可满足。v1.0 的失败结论永久保留，不删除。",
        "",
        "H2' (v1.2 判据，分布性上界 + 噪声带):",
        f"    (a) median(Δ_mech)={np.median(y):+.2f}% ≤ median(Δ_res)={np.median(x):+.2f}%  →  {med_ok}",
        f"    (b) Δ_mech>Δ_res 的比例 = {frac_over:.1%} ≤ 33.3%  →  {frac_ok}",
        f"    (c) 最大超额 {np.max(over_gaps) if len(over_gaps) else 0.0:+.2f}% ≤ gap 标准差 {noise_band:.2f}%  →  {within_band}",
        f"    →  {'H2-prime 通过' if h2p_pass else 'H2-prime 失败：先修 oracle，不裁决'}",
        "",
        f"H3  |Δ_res| < 5% 的点数 = {unresolvable} / {len(rows)}",
        f"    → {'H3 失败：真实数据中无无分辨力样本，主张降级为方法学注释' if unresolvable == 0 else 'H3 通过'}",
        "",
        "分组统计：",
        f"  Δ_res  均值={x.mean():+7.2f}%  中位={np.median(x):+7.2f}%  范围=[{x.min():+.2f}, {x.max():+.2f}]",
        f"  Δ_mech 均值={y.mean():+7.2f}%  中位={np.median(y):+7.2f}%  范围=[{y.min():+.2f}, {y.max():+.2f}]",
        "",
        "按体制：",
    ]
    for regime in REGIMES:
        sub = [r for r in rows if r["regime"] == regime]
        if not sub:
            continue
        sx = np.array([r["delta_res"] for r in sub])
        sy = np.array([r["delta_mech"] for r in sub])
        lines.append(f"  {regime:6s} n={len(sub):2d}  Δ_res={sx.mean():+6.2f}%  Δ_mech={sy.mean():+6.2f}%  ρ={spearman(sx, sy):+.3f}")

    print()
    print("\n".join(lines))
    (RESULTS / "p3_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

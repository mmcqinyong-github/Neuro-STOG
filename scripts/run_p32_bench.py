"""W-P3-2：Δ_res 跨 7 个公开 TSF 基准的分辨力排行 + 与原 EPF 结果的兼容性。

预注册：`W-P3-2预注册规格-v1.0.md`（判据先于数据写定）
纪律：不改通道采样规则、不改判据阈值、不因结果不利而换数据集。

输出：
  results/p32_points.csv    105 点 = 7 数据集 × 3 体制 × 5 偏移
  results/p32_summary.txt   H-B1 ~ H-B4 裁决
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import shared_gain as sg                                    # noqa: E402
from tsf_loader import BENCHMARKS, EXCLUDED, split          # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

REGIMES = ("raw", "shift", "noise")
TRAIN_OFFSETS = (0, 400, 800, 1200, 1600)
LAM = 1e-2
N_CALIB = 400
N_TRAIN, N_TEST = 800, 500
LOOKBACK = 24


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
        v = spearman(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cluster_boot_spearman(x, y, groups, n_boot=10_000, seed=0):
    """按数据集分块（预注册 §3 稳健性）。"""
    uniq = sorted(set(groups))
    blocks = {g: np.where(np.array(groups) == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    k = len(uniq)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        idx = np.concatenate([blocks[uniq[p]] for p in pick])
        v = spearman(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bench_point(name: str, regime: str, offset: int) -> dict:
    """一个基准环境点：Δ_res 与 Δ_mech（五臂中最好一臂）。"""
    s = split(name, regime, lookback=LOOKBACK, n_train=N_TRAIN,
              n_calib=N_CALIB, n_test=N_TEST, offset=offset)
    env = sg.W1Env(seed=101 + offset // 100, n_vars=s.n_vars, m_e=3,
                   n_train=N_TRAIN, n_calibrate=N_CALIB, n_test=N_TEST,
                   lookback=LOOKBACK)
    dres = sg.delta_res_real(env, s.train, s.calib, s.test, n_calib=N_CALIB)
    r = sg.run_point_real(env, s.train, s.calib, s.test, lam=LAM, n_calib=N_CALIB)
    base = r["base"][0]
    rel = {a: (base - r[a][0]) / base * 100.0 for a in sg.ARMS}
    best = max(rel, key=rel.get)
    row = {
        "dataset": name, "regime": regime, "offset": offset,
        "n_vars": s.n_vars, "n_domains": s.n_vars // 3,
        "base_wmse": base, "delta_res": dres,
        "delta_mech": rel[best], "best_arm": best,
    }
    for a in sg.ARMS:
        row[f"rel_{a}"] = rel[a]
        row[f"nparam_{a}"] = r[a][2]
    return row


def load_epf_points() -> list[dict]:
    """复用 P3 已落盘的 60 点，做 H-B4 合并检验（不重算）。"""
    p = RESULTS / "p3_points.csv"
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return [
            {"dataset": f"EPF-{r['market']}", "regime": r["regime"],
             "offset": int(r["offset"]), "delta_res": float(r["delta_res"]),
             "delta_mech": float(r["delta_mech"])}
            for r in csv.DictReader(f)
        ]


def main():
    t0 = time.time()
    rows: list[dict] = []
    print("=== W-P3-2：Δ_res 跨基准 ===", flush=True)
    for name in BENCHMARKS:
        for regime in REGIMES:
            for off in TRAIN_OFFSETS:
                try:
                    row = bench_point(name, regime, off)
                except Exception as e:                        # noqa: BLE001
                    print(f"  {name}/{regime}/off{off} FAILED: {type(e).__name__}: {e}",
                          flush=True)
                    continue
                rows.append(row)
                print(f"  {name:12s} {regime:5s} off={off:4d} nv={row['n_vars']} | "
                      f"Δ_res={row['delta_res']:+7.2f}% "
                      f"Δ_mech={row['delta_mech']:+7.2f}% best={row['best_arm']}",
                      flush=True)

    if not rows:
        print("[致命] 无有效点", flush=True)
        return 1

    with (RESULTS / "p32_points.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    x = np.array([r["delta_res"] for r in rows])
    y = np.array([r["delta_mech"] for r in rows])
    groups = [r["dataset"] for r in rows]
    rho = spearman(x, y)
    lo, hi = boot_spearman(x, y)
    clo, chi = cluster_boot_spearman(x, y, groups)

    # ---------- H-B1 ----------
    if rho >= 0.5 and lo > 0:
        hb1 = "通过"
    elif rho > 0 and lo <= 0:
        hb1 = "部分通过（点估计为正但 CI 含零）"
    else:
        hb1 = "否决"

    # ---------- H-B2 ----------
    unres = int((np.abs(x) < 5.0).sum())
    unres_frac = unres / len(rows)
    hb2 = "通过" if unres_frac >= 0.25 else ("否决" if unres_frac < 0.15 else "部分通过")

    # ---------- H-B3 ----------
    per_ds = {}
    for name in BENCHMARKS:
        sub = [r for r in rows if r["dataset"] == name]
        if sub:
            per_ds[name] = np.median([r["delta_res"] for r in sub])
    spread = max(per_ds.values()) - min(per_ds.values()) if per_ds else 0.0
    hb3 = "通过" if spread >= 5.0 else ("否决" if spread < 2.0 else "部分通过")

    # ---------- H-B4 ----------
    epf = load_epf_points()
    if epf:
        mx = np.concatenate([x, np.array([r["delta_res"] for r in epf])])
        my = np.concatenate([y, np.array([r["delta_mech"] for r in epf])])
        rho_merge = spearman(mx, my)
        mlo, mhi = boot_spearman(mx, my, seed=7)
        hb4 = ("通过" if rho_merge >= 0.5
               else ("部分通过" if rho_merge >= 0.3 else "否决：EPF 结果为数据集特异"))
    else:
        rho_merge, mlo, mhi, hb4 = float("nan"), float("nan"), float("nan"), "无法检验（缺 p3_points.csv）"

    lines = [
        "=== W-P3-2：Δ_res 跨公开 TSF 基准 ===",
        f"点数: {len(rows)}  ({len(per_ds)} 数据集 × {len(REGIMES)} 体制 × {len(TRAIN_OFFSETS)} 偏移)",
        f"耗时: {time.time() - t0:.1f}s",
        "",
        "H-B1 跨基准正相关（主判据）:",
        f"    Spearman ρ(Δ_res, Δ_mech) = {rho:+.3f}  逐点 95%CI [{lo:+.3f}, {hi:+.3f}]",
        f"    cluster bootstrap（按 {len(per_ds)} 数据集分块）95%CI [{clo:+.3f}, {chi:+.3f}]",
        f"    判据: ρ≥+0.5 且 CI 下界>0  →  {hb1}",
        "",
        "H-B2 无分辨力环境的普遍性:",
        f"    |Δ_res| < 5% 的点数 = {unres} / {len(rows)} = {unres_frac:.1%}",
        f"    判据: ≥25% 通过 / <15% 否决  →  {hb2}",
        "",
        "H-B3 数据集间分辨力差异:",
        f"    各数据集中位 Δ_res 极差 = {spread:.2f} 个百分点",
        f"    判据: ≥5pp 通过 / <2pp 否决  →  {hb3}",
        "",
        "H-B4 与 P3 原 EPF 结果的兼容性:",
        f"    合并 {len(rows)} 新点 + {len(epf)} EPF 点 = {len(rows) + len(epf)} 点",
        f"    合并 ρ = {rho_merge:+.3f}  95%CI [{mlo:+.3f}, {mhi:+.3f}]",
        f"    判据: ≥+0.5 通过 / <+0.3 否决  →  {hb4}",
        "",
        "按数据集（H-B3 依据）:",
        "  数据集          n   median(Δ_res)  mean(Δ_res)  median(Δ_mech)  ρ(组内)",
    ]
    for name, med in sorted(per_ds.items(), key=lambda kv: -kv[1]):
        sub = [r for r in rows if r["dataset"] == name]
        sx = np.array([r["delta_res"] for r in sub])
        sy = np.array([r["delta_mech"] for r in sub])
        r_in = spearman(sx, sy) if len(sub) > 2 else float("nan")
        lines.append(
            f"  {name:12s} {len(sub):3d}  {med:+10.2f}%  {sx.mean():+9.2f}%  "
            f"{np.median(sy):+13.2f}%  {r_in:+.3f}"
        )

    lines += ["", "按体制:"]
    for regime in REGIMES:
        sub = [r for r in rows if r["regime"] == regime]
        if not sub:
            continue
        sx = np.array([r["delta_res"] for r in sub])
        sy = np.array([r["delta_mech"] for r in sub])
        lines.append(
            f"  {regime:6s} n={len(sub):3d}  Δ_res={sx.mean():+7.2f}%  "
            f"Δ_mech={sy.mean():+7.2f}%  ρ={spearman(sx, sy):+.3f}"
        )

    lines += ["", "best_arm 计数:"]
    for a in sg.ARMS:
        c = sum(1 for r in rows if r["best_arm"] == a)
        lines.append(f"  {a:3s} {c:3d}")

    lines += ["", "排除项（预注册 §1）:"]
    for k, why in EXCLUDED.items():
        lines.append(f"  {k}: {why}")

    text = "\n".join(lines)
    print("\n" + text, flush=True)
    (RESULTS / "p32_summary.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

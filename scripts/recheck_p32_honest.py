"""W-P1-T：修复 Δ_res 的测试集泄漏后，重算 105 点并重做 H-B1 裁决。

背景
====
`src/shared_gain.py::delta_res_real` 原实现在**测试段**上挑 ridge 的 lam
（`w = _metrics(r_te - pred)`，取最小）——测试集泄漏。已在 W-P1-S 中修复
（`leak=False` 默认走 calib 对半切分、测试段只评不选）。

本脚本
------
1. 用 leak=True 复现旧数值，确认复现一致（数值自检）
2. 用 leak=False（honest）重算 105 点
3. 对两种口径分别重做 H-B1（Spearman(Δ_res, Δ_mech) ≥ +0.5）
4. 记录每个数据集的 Δ_res 符号/幅度变化

预注册判据（沿用 W-P3-2 H-B1 原阈值，不放宽）
-----------------------------------------------
  H-B1: ρ ≥ +0.5 且 cluster bootstrap 95% CI 下界 > 0 → 通过
  H-B1'（新增，诚实性要求）: 若 honest 口径下 H-B1 不通过，
        则 W-P3-2 的结论必须降级为"泄漏驱动的假阳性"

输出：results/p32_points_honest.csv, results/p32_recheck.txt
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
from tsf_loader import split                                # noqa: E402

RESULTS = ROOT / "results"
REGIMES = ("raw", "shift", "noise")
TRAIN_OFFSETS = (0, 400, 800, 1200, 1600)
LAM = 1e-2
N_CALIB = 400
N_TRAIN, N_TEST = 800, 500
LOOKBACK = 24


def _rank(x):
    """平均秩。"""
    x = np.asarray(x, float)
    order = x.argsort()
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def cluster_boot(x, y, groups, n_boot=2000, seed=7):
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
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def point(name, regime, offset, leak):
    s = split(name, regime, lookback=LOOKBACK, n_train=N_TRAIN,
              n_calib=N_CALIB, n_test=N_TEST, offset=offset)
    env = sg.W1Env(seed=101 + offset // 100, n_vars=s.n_vars, m_e=3,
                   n_train=N_TRAIN, n_calibrate=N_CALIB, n_test=N_TEST,
                   lookback=LOOKBACK)
    dres = sg.delta_res_real(env, s.train, s.calib, s.test, n_calib=N_CALIB,
                             leak=leak)
    r = sg.run_point_real(env, s.train, s.calib, s.test, lam=LAM, n_calib=N_CALIB)
    base = r["base"][0]
    rel = {a: (base - r[a][0]) / base * 100.0 for a in sg.ARMS}
    best = max(rel, key=rel.get)
    row = {"dataset": name, "regime": regime, "offset": offset,
           "base_wmse": base, "delta_res": dres,
           "delta_mech": rel[best], "best_arm": best}
    for a in sg.ARMS:
        row[f"rel_{a}"] = rel[a]
        row[f"nparam_{a}"] = r[a][2]
    return row


def main():
    with (RESULTS / "p32_points.csv").open(newline="", encoding="utf-8") as f:
        old = list(csv.DictReader(f))
    datasets = sorted({r["dataset"] for r in old})
    print("=" * 92)
    print("Δ_res 泄漏修复后的重算（105 点 × 2 口径）")
    print("=" * 92)
    print(f"数据集 {len(datasets)}: {datasets}")

    out = {"leak": [], "honest": []}
    t0 = time.time()
    for leak in (True, False):
        tag = "leak" if leak else "honest"
        print(f"\n--- 口径 {tag} ---", flush=True)
        for ds in datasets:
            for regime in REGIMES:
                for off in TRAIN_OFFSETS:
                    try:
                        out[tag].append(point(ds, regime, off, leak))
                    except Exception as e:  # noqa: BLE001
                        print(f"  [fail] {ds}/{regime}/{off}: {e}")
            sub = [r for r in out[tag] if r["dataset"] == ds]
            print(f"  {ds:14s} 中位 Δ_res = "
                  f"{np.median([r['delta_res'] for r in sub]):+8.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 数值自检：leak 口径应与旧文件一致 ----
    print("\n--- 自检：leak 口径 vs 旧 p32_points.csv ---")
    oldmap = {(r["dataset"], r["regime"], int(float(r["offset"]))):
              float(r["delta_res"]) for r in old}
    diffs = []
    for r in out["leak"]:
        k = (r["dataset"], r["regime"], r["offset"])
        if k in oldmap:
            diffs.append(abs(r["delta_res"] - oldmap[k]))
    print(f"  可比点 {len(diffs)}，最大 |差| = {max(diffs):.6f} pp  "
          f"中位 = {np.median(diffs):.6f} pp")
    print(f"  → {'复现一致（<0.01pp）' if max(diffs) < 0.01 else '存在差异，需查'}")

    # ---- H-B1 两口径 ----
    print("\n--- H-B1 重做（ρ(Δ_res, Δ_mech) ≥ +0.5 且 CI 下界 > 0）---")
    res = {}
    for tag in ("leak", "honest"):
        rows = out[tag]
        x = np.array([r["delta_res"] for r in rows])
        y = np.array([r["delta_mech"] for r in rows])
        g = [r["dataset"] for r in rows]
        rho = spearman(x, y)
        lo, hi = cluster_boot(x, y, g)
        ok = (rho >= 0.5) and (lo > 0)
        res[tag] = (rho, lo, hi, ok)
        print(f"  {tag:7s}: ρ = {rho:+.4f}  cluster 95%CI [{lo:+.4f}, {hi:+.4f}]  "
              f"→ {'通过' if ok else '不通过'}")

    # ---- 逐数据集对照 ----
    print(f"\n{'dataset':14s} {'Δ_res leak':>12s} {'Δ_res honest':>14s} {'差':>10s} {'符号':>6s}")
    sign_flip = 0
    for ds in datasets:
        a = np.median([r["delta_res"] for r in out["leak"] if r["dataset"] == ds])
        b = np.median([r["delta_res"] for r in out["honest"] if r["dataset"] == ds])
        fl = (np.sign(a) != np.sign(b))
        sign_flip += fl
        print(f"{ds:14s} {a:+12.2f} {b:+14.2f} {b-a:+10.2f} {'翻转' if fl else '同号':>6s}")

    # ---- 裁决 ----
    print("\n" + "=" * 92)
    print(f"H-B1 (leak 口径，即 W-P3-2 原结论) : "
          f"ρ={res['leak'][0]:+.4f} → {'通过' if res['leak'][3] else '不通过'}")
    print(f"H-B1 (honest 口径，无泄漏)         : "
          f"ρ={res['honest'][0]:+.4f} → {'通过' if res['honest'][3] else '不通过'}")
    print(f"数据集 Δ_res 符号翻转             : {sign_flip}/{len(datasets)}")
    verdict = ("W-P3-2 结论在无泄漏口径下存活"
               if res["honest"][3]
               else "W-P3-2 结论依赖测试集泄漏，必须降级为假阳性")
    print(f"\n裁决：{verdict}")

    with (RESULTS / "p32_points_honest.csv").open("w", newline="", encoding="utf-8") as f:
        keys = list(out["honest"][0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out["honest"])

    with (RESULTS / "p32_recheck.txt").open("w", encoding="utf-8") as f:
        f.write(f"rho_leak={res['leak'][0]:.6f} ci=[{res['leak'][1]:.4f},{res['leak'][2]:.4f}] pass={res['leak'][3]}\n")
        f.write(f"rho_honest={res['honest'][0]:.6f} ci=[{res['honest'][1]:.4f},{res['honest'][2]:.4f}] pass={res['honest'][3]}\n")
        f.write(f"sign_flip_datasets={sign_flip}/{len(datasets)}\n")
        f.write(f"verdict={verdict}\n")
    print(f"\n已写 {RESULTS / 'p32_points_honest.csv'}")
    print(f"已写 {RESULTS / 'p32_recheck.txt'}")


if __name__ == "__main__":
    main()

"""W-P1-U：修复"oracle 非上界"缺陷后的 Δ_res（决定性检验）。

根因（W-P1-S 的续查）
====================
`raft_dres.py::_oracle_features` 的 oracle 特征 = 
    6 阶滞后(C×6) + 15 维多尺度(C×15) + 3 维池化坐标(C×3)
而基线 `fit_backbone_ci` 的输入是**全部 720 阶滞后**（每通道 720+1 参数）。

⇒ oracle 看到的信息是基线的**真子集**，"oracle 是上界"这一前提**不成立**。
   Δ_res = (base − oracle)/base 因而可任意为负（exchange_rate −302%）。

修复
----
令 oracle 的特征**包含基线特征**：直接把 `_oracle_features` 与
「全部 720 阶滞后（逐通道）」拼接，构成真正的超集。
此时理论上 oracle 至少能达到基线水平（lam→∞ 时退化为基线），
故 **Δ_res ≥ 0 必然成立** —— 这个"符号必然性"本身就是修复成功的验证。

判据（先写下）
-------------
  U1. 修复后所有点的 Δ_res ≥ −1e-6（数值容差）→ 上界性质恢复
  U2. 修复后 ρ(Δ_res, Δ_mech) 是否仍 ≥ +0.5 且 CI 下界 > 0
      → 若仍成立，则"分辨力预测机制增益"这一命题**独立于构造缺陷**而成立
      → 若不成立，则 W-P3-2 与 W-P3-3 一并降级

输出：results/p33_dres_nested.csv, results/wp1u_verdict.txt
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raft_dres import (  # noqa: E402
    _oracle_features, fit_backbone_ci, raft_split,
)

DATA_BASE = r"C:\paper-QuantMo\Experiments\dataset"
ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")

DATASETS = [
    ("ETTh1", "ETTh1", os.path.join(DATA_BASE, "ETT-small", "ETTh1.csv")),
    ("ETTh2", "ETTh2", os.path.join(DATA_BASE, "ETT-small", "ETTh2.csv")),
    ("ETTm1", "ETTm1", os.path.join(DATA_BASE, "ETT-small", "ETTm1.csv")),
    ("ETTm2", "ETTm2", os.path.join(DATA_BASE, "ETT-small", "ETTm2.csv")),
    ("weather", "custom", os.path.join(DATA_BASE, "weather", "weather.csv")),
    ("exchange_rate", "custom",
     os.path.join(DATA_BASE, "exchange_rate", "exchange_rate.csv")),
]

SEQ_LEN, PRED_LEN, LAGS = 720, 96, 6
LAM_GRID = (1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8)
MAX_PTS = 200


def step_idx(seg):
    s, e = SEQ_LEN, seg.shape[0] - PRED_LEN
    return np.arange(s, e) if e > s else np.array([], dtype=int)


def nested_features(window: np.ndarray, lags: int) -> np.ndarray:
    """**真超集**：基线所需的全部 720 阶滞后 + 原 oracle 特征。

    基线每通道用自己 720 阶窗口拟合 ⇒ 把 window 拉平（L×C）就包含了基线的全部输入。
    """
    L, C = window.shape
    return np.concatenate([
        window.T.ravel(),                    # C×L = 全部滞后（含基线所需）
        _oracle_features(window, lags),      # 原 oracle 特征（多尺度 + 池化坐标）
    ])


def run(name, kind, path):
    df = pd.read_csv(path)
    x = df.iloc[:, 1:].to_numpy(dtype=float)
    tr, va, te = raft_split(x, kind, SEQ_LEN, PRED_LEN)
    tr_i, va_i, te_i = step_idx(tr), step_idx(va), step_idx(te)
    if len(va_i) == 0 or len(te_i) == 0:
        raise ValueError("段太短")

    bb = fit_backbone_ci(tr, SEQ_LEN)

    def base_direct(seg, idx):
        wins = np.stack([seg[t - SEQ_LEN:t] for t in idx])
        p = np.einsum("jl,nlj->nj", bb["W"], wins) + bb["b"]
        return np.repeat(p[:, None, :], PRED_LEN, axis=1)

    sel_t = np.linspace(0, len(te_i) - 1, min(MAX_PTS, len(te_i))).astype(int)
    sel_v = np.linspace(0, len(va_i) - 1, min(MAX_PTS, len(va_i))).astype(int)

    def build(seg, idx, sub):
        X, Y = [], []
        for j in sub:
            t = idx[j]
            X.append(nested_features(seg[t - SEQ_LEN:t], LAGS))
            Y.append(seg[t:t + PRED_LEN])
        return np.stack(X), np.stack(Y)

    Xv, Yv = build(va, va_i, sel_v)
    Xt, Yt = build(te, te_i, sel_t)
    C = x.shape[1]

    # calib 对半：前半拟合、后半选 lam（无泄漏）
    h = len(Xv) // 2
    Af = np.concatenate([Xv[:h], np.ones((h, 1))], axis=1)
    As = np.concatenate([Xv[h:], np.ones((len(Xv) - h, 1))], axis=1)
    At = np.concatenate([Xt, np.ones((len(Xt), 1))], axis=1)
    yf, ys = Yv[:h], Yv[h:]

    U, s, Vt = np.linalg.svd(Af, full_matrices=False)
    AsV = As @ Vt.T
    AtV = At @ Vt.T

    pred = np.zeros_like(Yt)
    for step in range(PRED_LEN):
        for j in range(C):
            Uy = U.T @ yf[:, step, j]
            best = None
            for lam in LAM_GRID:
                d = s / (s * s + lam)
                e = float(np.mean((AsV @ (d * Uy) - ys[:, step, j]) ** 2))
                if best is None or e < best[0]:
                    best = (e, lam)
            d = s / (s * s + best[1])
            pred[:, step, j] = AtV @ (d * Uy)

    base_sel = base_direct(te, te_i)[sel_t]
    b_wmse = float(np.mean((base_sel - Yt) ** 2))
    o_wmse = float(np.mean((pred - Yt) ** 2))
    return dict(dataset=name, n_vars=C, feat_dim=int(Af.shape[1]),
                dim_over_n=round(Af.shape[1] / h, 3),
                base_wmse=b_wmse, oracle_wmse=o_wmse,
                delta_res=(b_wmse - o_wmse) / b_wmse * 100.0,
                n_oracle_train=h)


def _rank(x):
    x = np.asarray(x, float)
    o = x.argsort()
    r = np.empty(len(x), float)
    r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def main():
    print("=" * 88)
    print("修复 oracle 非上界缺陷：嵌套特征 + 无泄漏选参")
    print("=" * 88)
    rows = []
    t0 = time.time()
    for name, kind, path in DATASETS:
        try:
            r = run(name, kind, path)
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {name}: {e}")
            continue
        rows.append(r)
        print(f"  {name:14s} C={r['n_vars']:2d} d={r['feat_dim']:5d} "
              f"d/n={r['dim_over_n']:7.3f}  Δ_res={r['delta_res']:+9.3f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)

    neg = [r for r in rows if r["delta_res"] < -1e-6]
    print(f"\nU1 上界性质（Δ_res ≥ 0）: {'恢复' if not neg else '仍失败'}  "
          f"(负值点 {len(neg)}/{len(rows)})")
    if neg:
        for r in neg:
            print(f"    {r['dataset']}: {r['delta_res']:+.3f}%")

    # 与 Δ_mech 对照
    dmech = {}
    p = os.path.join(RES, "p33_dmech.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = r.get("dmech_B") or r.get("dmech_A") or ""
                if v not in ("", None):
                    dmech[r["dataset"]] = float(v)

    paired = [(r["delta_res"], dmech[r["dataset"]]) for r in rows
              if r["dataset"] in dmech]
    print(f"\n可配对 {len(paired)} 个数据集")
    verdict_rho = float("nan")
    if len(paired) >= 3:
        from scipy.stats import spearmanr  # noqa: PLC0415
        dr = [a for a, _ in paired]
        dm = [b for _, b in paired]
        rho, pv = spearmanr(dr, dm)
        verdict_rho = rho
        print(f"U2 ρ(Δ_res_nested, Δ_mech) = {rho:+.4f}  (p={pv:.4f}, n={len(paired)})")
        for r in rows:
            if r["dataset"] in dmech:
                print(f"   {r['dataset']:14s} Δ_res={r['delta_res']:+9.3f}%  "
                      f"Δ_mech={dmech[r['dataset']]:+8.3f}%")

    csvp = os.path.join(RES, "p33_dres_nested.csv")
    if rows:
        with open(csvp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n已写 {csvp}")

    with open(os.path.join(RES, "wp1u_verdict.txt"), "w", encoding="utf-8") as f:
        f.write(f"U1_upper_bound_restored={not neg} negative={len(neg)}/{len(rows)}\n")
        f.write(f"U2_rho={verdict_rho}\n")
        for r in rows:
            f.write(f"{r['dataset']},{r['delta_res']:.4f},{dmech.get(r['dataset'],'')}\n")


if __name__ == "__main__":
    main()

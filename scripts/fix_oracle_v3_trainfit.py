"""W-P1-V：Δ_res 的最终修正尝试 —— oracle 与基线同源（同在 train 拟合）。

失败链条（本工作区第四次证伪的完整谱系）
======================================
  v1  `raft_dres` 原版：oracle 特征只有最后 6 阶滞后（C×24）
      → oracle 信息是基线的**真子集** ⇒ 上界不成立 ⇒ Δ_res 可任意为负

  v2  `fix_oracle_nested.py`：特征改真超集（全 720 阶 + 原特征）
      → d/n 涨到 52–156，ETTh2/ETTm2 仍为 −160%/−279% ⇒ 上界仍不成立

  根因（v2 暴露）：**oracle 只用 calib 的 200–400 个点拟合，基线用 train 的数千个点拟合。**
  即使特征是真超集，"样本量不平等"也足以让 oracle 输给基线。
  ⇒ 上界性质的前提不只是"信息超集"，还必须"样本量不劣"。

本版修正（v3）
--------------
oracle 与基线都在 **train** 上拟合：
  * 基线：`fit_backbone_ci(train)`，逐通道 720 阶 ridge（原样不动）
  * oracle：在 train 上取 (窗口, 目标) 对，特征 = 全 720 阶滞后 + 多尺度 + 池化坐标
            （真超集），ridge 的 lam 用 train 内部前后切分选择（无测试泄漏）
  * 评估：test 段（与基线同一批点）

此时 d/n 应为可控量级（ETT: d≈5209 / n≈6000 ⇒ 0.87），
且理论上 lam→∞ 时 oracle 退化为基线的特例 ⇒ **Δ_res ≥ 0 应当恢复**。

判据（先写下）
-------------
  V1'. 所有点 Δ_res ≥ −1e-6（上界性质恢复）
  V2'. ρ(Δ_res, Δ_mech) 是否仍 ≥ +0.5
  V3'. d/n 是否 ≤ 2（否则仍是病态）

只跑 ETT 四个（train 长度约 12 个月，够用），weather/exchange_rate 训练段
与特征维的比值过高，若 ETT 上 V1' 都无法恢复，则不必再跑。
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raft_dres import _oracle_features, fit_backbone_ci, raft_split  # noqa: E402

DATA_BASE = r"C:\paper-QuantMo\Experiments\dataset"
ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")

DATASETS = [
    ("ETTh1", "ETTh1", os.path.join(DATA_BASE, "ETT-small", "ETTh1.csv")),
    ("ETTh2", "ETTh2", os.path.join(DATA_BASE, "ETT-small", "ETTh2.csv")),
    ("ETTm1", "ETTm1", os.path.join(DATA_BASE, "ETT-small", "ETTm1.csv")),
    ("ETTm2", "ETTm2", os.path.join(DATA_BASE, "ETT-small", "ETTm2.csv")),
]
SEQ_LEN, PRED_LEN, LAGS = 720, 96, 6
LAM_GRID = (1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e10)
# oracle 在 train 上最多取的点数（控制 SVD 成本；仍是基线的同量级样本）
MAX_TRAIN_FIT = 3000
MAX_EVAL = 150


def step_idx(seg):
    s, e = SEQ_LEN, seg.shape[0] - PRED_LEN
    return np.arange(s, e) if e > s else np.array([], dtype=int)


def nested_features(window):
    """真超集：全部 L 阶滞后（逐通道拉平）+ 原 oracle 特征。"""
    L, C = window.shape
    return np.concatenate([window.T.ravel(), _oracle_features(window, LAGS)])


def run(name, kind, path):
    x = pd.read_csv(path).iloc[:, 1:].to_numpy(dtype=float)
    tr, va, te = raft_split(x, kind, SEQ_LEN, PRED_LEN)
    tr_i, te_i = step_idx(tr), step_idx(te)
    C = x.shape[1]

    bb = fit_backbone_ci(tr, SEQ_LEN)

    def base_pred(seg, idx):
        wins = np.stack([seg[t - SEQ_LEN:t] for t in idx])
        p = np.einsum("jl,nlj->nj", bb["W"], wins) + bb["b"]
        return np.repeat(p[:, None, :], PRED_LEN, axis=1)

    # ---- oracle 的拟合集：train ----
    # 注意：linspace 产生的是**子集下标**，必须再映射回真实时间下标 tr_i[...]
    sel_f = np.arange(len(tr_i))
    if len(sel_f) > MAX_TRAIN_FIT:
        sel_f = np.linspace(0, len(tr_i) - 1, MAX_TRAIN_FIT).astype(int)
    fit_t = tr_i[sel_f]
    Xf, Yf = [], []
    for t in fit_t:
        Xf.append(nested_features(tr[t - SEQ_LEN:t]))
        Yf.append(tr[t:t + PRED_LEN])
    Xf, Yf = np.stack(Xf), np.stack(Yf)

    sel_e = np.arange(len(te_i))
    if len(sel_e) > MAX_EVAL:
        sel_e = np.linspace(0, len(te_i) - 1, MAX_EVAL).astype(int)
    ev_t = te_i[sel_e]
    Xe, Ye = [], []
    for t in ev_t:
        Xe.append(nested_features(te[t - SEQ_LEN:t]))
        Ye.append(te[t:t + PRED_LEN])
    Xe, Ye = np.stack(Xe), np.stack(Ye)

    # ---- train 内部前后切分选 lam（无测试泄漏）----
    h = len(Xf) // 2
    Af = np.concatenate([Xf[:h], np.ones((h, 1))], axis=1)
    As = np.concatenate([Xf[h:], np.ones((len(Xf) - h, 1))], axis=1)
    Ae = np.concatenate([Xe, np.ones((len(Xe), 1))], axis=1)

    # 一次特征分解 A'A = V diag(ev) V'，复用给所有 lam
    G = Af.T @ Af
    ev, V = np.linalg.eigh(G)
    ev = np.maximum(ev, 0.0)
    # Af 形状 (h, d)：(样本 t, 特征 d)。Af.T @ Yf 会按 matmul 广播错解，用 einsum 明确语义
    AfTy = np.einsum("td,tpj->dpj", Af, Yf[:h])       # (d, P, C)
    AsV = As @ V
    AeV = Ae @ V
    Vs = V.T

    pred = np.zeros_like(Ye)
    for step in range(PRED_LEN):
        for j in range(C):
            b = Vs @ AfTy[:, step, j]          # (d,)
            best = None
            for lam in LAM_GRID:
                coef = b / (ev + lam)
                e = float(np.mean((AsV @ coef - Yf[h:, step, j]) ** 2))
                if best is None or e < best[0]:
                    best = (e, coef)
            pred[:, step, j] = AeV @ best[1]

    b_wmse = float(np.mean((base_pred(te, ev_t) - Ye) ** 2))
    o_wmse = float(np.mean((pred - Ye) ** 2))
    return dict(dataset=name, n_vars=C, feat_dim=int(Af.shape[1]),
                n_fit=h, dim_over_n=round(Af.shape[1] / h, 3),
                base_wmse=b_wmse, oracle_wmse=o_wmse,
                delta_res=(b_wmse - o_wmse) / b_wmse * 100.0)


def main():
    print("=" * 88)
    print("v3：oracle 与基线同在 train 拟合（样本量对齐 + 特征真超集 + 无泄漏选参）")
    print("=" * 88)
    rows, t0 = [], time.time()
    for name, kind, path in DATASETS:
        try:
            r = run(name, kind, path)
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {name}: {type(e).__name__}: {e}")
            continue
        rows.append(r)
        print(f"  {name:14s} d={r['feat_dim']:5d} n={r['n_fit']:5d} "
              f"d/n={r['dim_over_n']:6.3f}  Δ_res={r['delta_res']:+9.3f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        print("无结果")
        return

    neg = [r for r in rows if r["delta_res"] < -1e-6]
    print(f"\nV1' 上界性质（Δ_res ≥ 0）: {'恢复' if not neg else '仍失败'}  "
          f"（负值 {len(neg)}/{len(rows)}）")
    for r in neg:
        print(f"    {r['dataset']}: {r['delta_res']:+.3f}%")
    print(f"V3' d/n ≤ 2: "
          f"{'全部满足' if all(r['dim_over_n'] <= 2 for r in rows) else '有超限'}"
          f"  (max {max(r['dim_over_n'] for r in rows):.3f})")

    dmech = {}
    p = os.path.join(RES, "p33_dmech.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = r.get("dmech_B") or r.get("dmech_A") or ""
                if v not in ("", None):
                    dmech[r["dataset"]] = float(v)

    print(f"\n{'dataset':14s} {'Δ_res_v3':>11s} {'Δ_mech':>10s}")
    for r in rows:
        print(f"{r['dataset']:14s} {r['delta_res']:+11.3f} "
              f"{dmech.get(r['dataset'], float('nan')):+10.3f}")

    pair = [(r["delta_res"], dmech[r["dataset"]]) for r in rows if r["dataset"] in dmech]
    if len(pair) >= 3:
        from scipy.stats import spearmanr  # noqa: PLC0415
        rho, pv = spearmanr([a for a, _ in pair], [b for _, b in pair])
        print(f"\nV2' ρ(Δ_res_v3, Δ_mech) = {rho:+.4f} (p={pv:.4f}, n={len(pair)})")
        print(f"    判据 ρ ≥ +0.5: {'通过' if rho >= 0.5 else '不通过'}")

    with open(os.path.join(RES, "p33_dres_v3.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写 {os.path.join(RES, 'p33_dres_v3.csv')}")


if __name__ == "__main__":
    main()

"""Δ_res 构造有效性诊断（W-P1-S 的前置诚信核查）。

触发原因
--------
在准备回答"层 2 的 Δ_res 符号发现能否支撑一区创新"时，复查 `src/raft_dres.py`
发现两处**结构性缺陷**，它们可能使 Δ_res 的符号主要由过拟合伪影决定，
而非由"环境分辨力"决定：

  D1. **测试集泄漏**：oracle 的 ridge 惩罚 lam 是在 test 段上逐 (步, 通道)
      挑最小误差得到的（`err = mean((p - Yt)**2)` 用 Yt = test 真值）。
      这是用测试集选超参，Δ_res 系统性偏乐观。

  D2. **oracle 维度 >> 样本数**：oracle 特征维数 = C×24（lags 6 + 多尺度 15
      + 池化 3），而 oracle 的拟合样本被 `max_oracle_pts=400` 截断。
      weather (C=21) ⇒ 504 维 / 400 样本；exchange_rate (C=8) ⇒ 192 维 / 400 样本。
      维度超过样本数时 ridge 只是插值，其"能力"不可解释为信息上界。

  D3. **基线不是 oracle 的特例**：基线是"所有未来步共用同一预测"的直接外推，
      oracle 是"每步独立拟合"。在近随机游走序列上，直接外推近似最优，
      而高维 oracle 必然更差 ⇒ Δ_res 可以为负且幅度巨大
      （exchange_rate 实测 −302%）。

后果
----
若 Δ_res 的符号主要由 D1/D2/D3 决定，则 W-P3-3 的 H-L2（ρ=+0.9429）存在
**共同原因混淆**：Δ_res 与 Δ_mech 都被"环境对小样本高维模型的友好程度"
这一共同因子驱动，二者的相关不构成"分辨力预测机制增益"的证据。

本脚本做四件事
--------------
  A. 复算 Δ_res 的两种口径，比较泄漏影响：
       leak   = 现行实现（test 选 lam）
       honest = val 选 lam（test 完全不参与选参）
  B. 记录每个数据集的 oracle 维度 / 样本数比（d/n）
  C. 检验 Δ_res 的对数幅度是否与 d/n 共变（Spearman）
  D. 用"基线自身"充当零信息参照：把 oracle 换成 ridge 基线在 val 上选 lam
     的版本，看 Δ_res 是否仍大 —— 若 oracle 只是"更灵活的基线"而非信息上界，
     则 Δ_res 测的是容量不是信息。

判据（事后诊断，非预注册判据；仅用于判定既有结论是否失效）
------------------------------------------------------------
  V1. 若 |Δ_res_honest − Δ_res_leak| > 5pp 的数据集 ≥ 2 个 ⇒ 泄漏实质影响结论
  V2. 若某数据集 d/n > 0.5 ⇒ 该数据集的 Δ_res 不可解释为信息上界
  V3. 若 Spearman(|Δ_res|, d/n) ≥ +0.5 ⇒ 幅度主要由容量而非信息驱动
  V4. 若 Δ_res 的符号在 leak/honest 两种口径下不一致 ⇒ 符号结论不稳健

输出：results/dres_validity_diag.csv, results/dres_validity_diag.txt
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raft_dres import (  # noqa: E402
    _oracle_features, fit_backbone_ci, raft_split,
)

DATA_BASE = r"C:\paper-QuantMo\Experiments\dataset"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

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
LAM_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5)
MAX_ORACLE_PTS = 400
RIDGE_LAM = 10.0


def load_csv(path: str) -> np.ndarray:
    """与 run_p33_dres.py 完全一致：丢首列（时间戳）。"""
    df = pd.read_csv(path)
    return df.iloc[:, 1:].to_numpy(dtype=float)


def step_idx(seg: np.ndarray):
    start, end = SEQ_LEN, seg.shape[0] - PRED_LEN
    if end <= start:
        return np.array([], dtype=int)
    return np.arange(start, end)


def compute(dataset: str, kind: str, n_vars: int, train, val, test,
            leak: bool) -> dict:
    """leak=True 复刻现行实现（test 选 lam）；leak=False 用 val 选 lam。"""
    C = train.shape[1]
    tr = step_idx(train)
    va = step_idx(val)
    te = step_idx(test)
    if len(va) == 0 or len(te) == 0:
        raise ValueError(f"{dataset}: 段太短 va={len(va)} te={len(te)}")

    bb = fit_backbone_ci(train, SEQ_LEN, lam=RIDGE_LAM)

    def base_direct(seg, idx):
        wins = np.stack([seg[t - SEQ_LEN:t] for t in idx])
        p = np.einsum("jl,nlj->nj", bb["W"], wins) + bb["b"]
        return np.repeat(p[:, None, :], PRED_LEN, axis=1)

    sel_te = np.linspace(0, len(te) - 1, min(MAX_ORACLE_PTS, len(te))).astype(int)
    sel_va = np.linspace(0, len(va) - 1, min(MAX_ORACLE_PTS, len(va))).astype(int)

    def build(seg, idx, sub):
        Xs, Ys = [], []
        for j in sub:
            t = idx[j]
            Xs.append(_oracle_features(seg[t - SEQ_LEN:t], LAGS))
            Ys.append(seg[t:t + PRED_LEN])
        return np.stack(Xs), np.stack(Ys)

    Xv, Yv = build(val, va, sel_va)
    Xt, Yt = build(test, te, sel_te)

    Av = np.concatenate([Xv, np.ones((len(Xv), 1))], axis=1)
    At = np.concatenate([Xt, np.ones((len(Xt), 1))], axis=1)
    U, s, Vt = np.linalg.svd(Av, full_matrices=False)
    AtV = At @ Vt.T

    # honest 与 leak 的差别只在"用哪段数据选 lam"
    Xsel = Av if leak else Av           # leak 时下面用 Yt 评估；honest 用 Yv
    Ysel = Yt if leak else Yv
    Asel_V = AtV if leak else Av @ Vt.T

    base_sel = base_direct(test, te)[sel_te]
    base_wmse = float(np.mean((base_sel - Yt) ** 2))

    pred = np.zeros_like(Yt)
    chosen = np.zeros((PRED_LEN, C), dtype=int)
    for h in range(PRED_LEN):
        for j in range(C):
            yv = Yv[:, h, j]
            Uy = U.T @ yv
            best = None
            for li, lam in enumerate(LAM_GRID):
                d = s / (s * s + lam)
                # 取参臂：leak 用 test 特征、honest 用 val 特征
                coef = d * Uy
                if leak:
                    p_sel = AtV @ coef                      # (n_te,)
                else:
                    p_sel = (Av @ Vt.T) @ coef              # (n_va,)
                err = float(np.mean((p_sel - Ysel[:, h, j]) ** 2))
                if best is None or err < best[0]:
                    best = (err, li)
            lam_best = LAM_GRID[best[1]]
            chosen[h, j] = best[1]
            d = s / (s * s + lam_best)
            pred[:, h, j] = AtV @ (d * Uy)                  # 始终在 test 上评

    oracle_wmse = float(np.mean((pred - Yt) ** 2))
    d_res = (base_wmse - oracle_wmse) / base_wmse * 100.0

    return {
        "dataset": dataset,
        "n_vars": C,
        "feat_dim": int(Av.shape[1]),
        "n_oracle_train": int(len(Xv)),
        "n_oracle_test": int(len(Xt)),
        "dim_over_n": float(Av.shape[1] / max(len(Xv), 1)),
        "base_wmse": base_wmse,
        "oracle_wmse": oracle_wmse,
        "delta_res": float(d_res),
        "root_mse_oracle": float(np.sqrt(oracle_wmse)),
        "root_mse_base": float(np.sqrt(base_wmse)),
        "lam_mode_mean": float(np.mean(chosen)),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    print("=" * 92)
    print("Δ_res 构造有效性诊断：test 泄漏 × 维度/样本比")
    print("=" * 92)
    for name, kind, path in DATASETS:
        try:
            data = load_csv(path)
        except FileNotFoundError:
            print(f"  [skip] {name}: 数据缺失")
            continue
        tr, va, te = raft_split(data, kind, SEQ_LEN, PRED_LEN)
        for leak in (True, False):
            try:
                r = compute(name, kind, data.shape[1], tr, va, te, leak=leak)
            except Exception as e:  # noqa: BLE001
                print(f"  [fail] {name} leak={leak}: {e}")
                continue
            r["lam_mode"] = "test(leak)" if leak else "val(honest)"
            rows.append(r)
            print(f"  {name:14s} {r['lam_mode']:12s} "
                  f"C={r['n_vars']:2d} d={r['feat_dim']:4d} n={r['n_oracle_train']:4d} "
                  f"d/n={r['dim_over_n']:5.2f}  Δ_res={r['delta_res']:+9.2f}%")

    # ---- V1–V4 ----
    print()
    print("-" * 92)
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["lam_mode"]] = r

    diffs, sign_flip = [], []
    print(f"{'dataset':14s} {'Δ_res(leak)':>13s} {'Δ_res(honest)':>15s} {'差(pp)':>9s} {'符号变?':>8s} {'d/n':>7s}")
    for ds, d in by_ds.items():
        if len(d) < 2:
            continue
        a = d["test(leak)"]["delta_res"]
        b = d["val(honest)"]["delta_res"]
        flip = (np.sign(a) != np.sign(b))
        diffs.append(abs(a - b))
        sign_flip.append(flip)
        print(f"{ds:14s} {a:+13.2f} {b:+15.2f} {abs(a-b):9.2f} {'YES' if flip else 'no':>8s} "
              f"{d['val(honest)']['dim_over_n']:7.2f}")

    V1 = sum(1 for x in diffs if x > 5.0) >= 2
    V2 = [ds for ds, d in by_ds.items() if d.get("val(honest)", {}).get("dim_over_n", 0) > 0.5]
    ds_list = [r["dataset"] for r in rows if r["lam_mode"] == "val(honest)"]
    dn = [by_ds[ds]["val(honest)"]["dim_over_n"] for ds in ds_list]
    mag = [abs(by_ds[ds]["val(honest)"]["delta_res"]) for ds in ds_list]
    rho_dn = np.nan
    if len(ds_list) >= 3:
        from scipy.stats import spearmanr  # noqa: PLC0415
        rho_dn, _ = spearmanr(dn, mag)
    V3 = (not np.isnan(rho_dn)) and rho_dn >= 0.5
    V4 = any(sign_flip)

    print()
    print(f"V1 泄漏实质影响结论 (≥2 数据集 |差|>5pp) : {'触发' if V1 else '未触发'}  "
          f"(|差|>5pp 的数据集 {sum(1 for x in diffs if x > 5)} 个)")
    print(f"V2 维度超样本 (d/n>0.5)               : {'触发' if V2 else '未触发'}  "
          f"({', '.join(V2) if V2 else '无'})")
    print(f"V3 幅度由容量驱动 (Spearman(|Δ_res|,d/n)≥0.5): "
          f"{'触发' if V3 else '未触发'}  (ρ={rho_dn:+.4f}, n={len(ds_list)})")
    print(f"V4 符号不稳健 (leak/honest 符号不一致) : {'触发' if V4 else '未触发'}  "
          f"({sum(sign_flip)}/{len(sign_flip)} 个数据集翻转)")
    print()
    verdict = "Δ_res 符号不可作为分辨力度量" if (V1 or V2 or V3 or V4) else "Δ_res 符号基本稳健"
    print(f"总判读：{verdict}")

    csv = os.path.join(OUT_DIR, "dres_validity_diag.csv")
    keys = list(rows[0].keys())
    with open(csv, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    txt = os.path.join(OUT_DIR, "dres_validity_diag.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"V1={V1} V2={V2} V3={V3} V4={V4} rho_dn={rho_dn}\nverdict={verdict}\n")
    print(f"\n已写 {csv}\n已写 {txt}")


if __name__ == "__main__":
    main()

"""W-P1-U：机制增益的可筛选性（留一数据集前瞻检验）。

预注册：`W-P1-U可筛选性预注册规格-v1.0.md`（判据已先写定）

核心问题
--------
给定一个从未参与分析的（数据集，体制，偏移）点，能否只用**跑机制实验之前
就能算出的量**预测机制增益是否可信？

切分：**留一数据集交叉验证（LOO-DS）**，7 折，每折留 1 个数据集的 15 点。
      这是严格的数据集级外推，不做点级随机切分（那样会因同数据集点高度
      相关而严重高估外推能力）。

判据（预注册 §5，不得事后调整）
------------------------------
  U-a  LOO Spearman ρ(预测, 实际)  ≥ +0.40 通过 / < +0.15 否决
  U-b  LOO balanced accuracy       ≥ 0.65 通过 / < 0.55 否决
  U-c  LOO AUC                     ≥ 0.65 通过 / < 0.55 否决
  U-d  样本内（随机 5 折）ρ 应显著高于 LOO-DS

输出：results/loo_screen.csv, results/loo_screen.txt
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
ARMS = ["F", "FP", "S", "L", "D"]


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


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(y, s):
    """Mann-Whitney AUC。"""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rank(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bal_acc(y, pred):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    vals = []
    for c in (0, 1):
        m = y == c
        if m.sum() == 0:
            continue
        vals.append((pred[m] == c).mean())
    return float(np.mean(vals)) if vals else float("nan")


def fit_linear(X, y):
    """带截距最小二乘。"""
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def pred_linear(coef, X):
    A = np.column_stack([np.ones(len(X)), X])
    return A @ coef


def load():
    with open(os.path.join(RES, "p32_points_honest.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # honest CSV 未带 n_vars/n_domains，从原始文件按 (dataset, regime, offset) 合并
    extra = {}
    with open(os.path.join(RES, "p32_points.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["dataset"], r["regime"], int(float(r["offset"])))
            extra[k] = (float(r["n_vars"]), float(r["n_domains"]))
    miss = 0
    for r in rows:
        r["delta_res"] = float(r["delta_res"])
        r["delta_mech"] = float(r["delta_mech"])
        r["abs_dres"] = abs(r["delta_res"])
        r["offset"] = int(float(r["offset"]))
        k = (r["dataset"], r["regime"], r["offset"])
        if k in extra:
            r["n_vars"], r["n_domains"] = extra[k]
        else:
            r["n_vars"], r["n_domains"] = float("nan"), float("nan")
            miss += 1
        r["base_wmse"] = float(r["base_wmse"])
        r["rel"] = {a: float(r["rel_" + a]) for a in ARMS}
        r["npar"] = {a: int(float(r["nparam_" + a])) for a in ARMS}
    if miss:
        print(f"  [warn] {miss} 点缺 n_vars，M2 将跳过这些点")
    return rows


def cap_slope(r):
    return spearman([r["npar"][a] for a in ARMS], [r["rel"][a] for a in ARMS])


def loo_dataset(rows, feats, target="delta_mech"):
    """留一数据集交叉验证。返回逐点 (实际, 预测)。"""
    ds = sorted({r["dataset"] for r in rows})
    out = []
    for held in ds:
        tr = [r for r in rows if r["dataset"] != held]
        te = [r for r in rows if r["dataset"] == held]
        Xtr = np.array([[r[f] for f in feats] for r in tr], float)
        ytr = np.array([r[target] for r in tr], float)
        Xte = np.array([[r[f] for f in feats] for r in te], float)
        yte = np.array([r[target] for r in te], float)
        coef = fit_linear(Xtr, ytr)
        p = pred_linear(coef, Xte)
        for r, yt, yp in zip(te, yte, p):
            out.append(dict(dataset=held, regime=r["regime"], offset=r["offset"],
                            actual=yt, pred=float(yp),
                            actual_sign=int(yt > 0), pred_sign=int(yp > 0)))
    return out


def kfold_within(rows, feats, k=5, seed=11, target="delta_mech"):
    """样本内随机 k 折（正对照）。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    folds = np.array_split(idx, k)
    out = []
    for f in folds:
        te = [rows[i] for i in f]
        tr = [rows[i] for i in idx if i not in set(f.tolist())]
        Xtr = np.array([[r[x] for x in feats] for r in tr], float)
        ytr = np.array([r[target] for r in tr], float)
        Xte = np.array([[r[x] for x in feats] for r in te], float)
        coef = fit_linear(Xtr, ytr)
        for r, yp in zip(te, pred_linear(coef, Xte)):
            out.append(dict(actual=r[target], pred=float(yp),
                            actual_sign=int(r[target] > 0), pred_sign=int(yp > 0)))
    return out


def report(tag, res):
    y = np.array([r["actual"] for r in res])
    p = np.array([r["pred"] for r in res])
    ys = np.array([r["actual_sign"] for r in res])
    ps = np.array([r["pred_sign"] for r in res])
    rho = spearman(p, y)
    a = auc(ys, p)
    ba = bal_acc(ys, ps)
    print(f"  {tag:32s} ρ={rho:+.4f}  AUC={a:.4f}  balAcc={ba:.4f}  n={len(res)}")
    return dict(tag=tag, rho=rho, auc=a, bal_acc=ba, n=len(res))


def main():
    rows = load()
    for r in rows:
        r["cap_slope"] = cap_slope(r) or 0.0

    print("=" * 92)
    print("W-P1-U 机制增益的可筛选性：留一数据集前瞻检验")
    print("=" * 92)
    print(f"点 {len(rows)}，数据集 {len({r['dataset'] for r in rows})}")
    print(f"Δ_mech > 0 的点: {sum(1 for r in rows if r['delta_mech'] > 0)}/{len(rows)}")

    MODELS = [
        ("M1: Δ_res", ["delta_res"], True),
        ("M2: Δ_res + n_vars", ["delta_res", "n_vars"], True),
        ("M3: |Δ_res|", ["abs_dres"], True),
        ("C1: cap_slope（含机制信息，仅诊断）", ["cap_slope"], False),
        ("C2: base_wmse", ["base_wmse"], True),
    ]

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("\n--- LOO-DS（严格数据集级外推）---")
    loo_results = {}
    for name, feats, primary in MODELS:
        res = loo_dataset(rows, feats)
        st = report(name, res)
        st["primary"] = primary
        st["feats"] = feats
        loo_results[name] = (st, res)

    out("\n--- 正对照：样本内随机 5 折（同特征）---")
    infold = {}
    for name, feats, primary in MODELS:
        if not primary:
            continue
        res = kfold_within(rows, feats)
        infold[name] = report(name, res)

    out("\n--- 零模型（训练折均值）---")
    ds = sorted({r["dataset"] for r in rows})
    zres = []
    for held in ds:
        tr = [r for r in rows if r["dataset"] != held]
        te = [r for r in rows if r["dataset"] == held]
        m = np.mean([r["delta_mech"] for r in tr])
        for r in te:
            zres.append(dict(actual=r["delta_mech"], pred=m,
                             actual_sign=int(r["delta_mech"] > 0),
                             pred_sign=int(m > 0)))
    zst = report("Z: 训练折均值（零模型）", zres)

    # ---- 裁决（主模型 M1）----
    out("\n" + "=" * 92)
    m1 = loo_results["M1: Δ_res"][0]
    Ua = m1["rho"] >= 0.40
    Ub = m1["bal_acc"] >= 0.65
    Uc = m1["auc"] >= 0.65
    out(f"U-a LOO ρ ≥ +0.40        : {'通过' if Ua else '未通过'}  (ρ={m1['rho']:+.4f})")
    out(f"U-b LOO balAcc ≥ 0.65    : {'通过' if Ub else '未通过'}  (={m1['bal_acc']:.4f})")
    out(f"U-c LOO AUC ≥ 0.65       : {'通过' if Uc else '未通过'}  (={m1['auc']:.4f})")

    # 中间区判定
    def zone(rho, acc):
        if rho >= 0.40 or acc >= 0.65:
            return "通过"
        if rho < 0.15 and acc < 0.55:
            return "否决"
        return "弱信号"

    z = zone(m1["rho"], m1["bal_acc"])
    out(f"\n中间区判定规则 → {z}")

    # U-d
    if "M1: Δ_res" in infold:
        gap = infold["M1: Δ_res"]["rho"] - m1["rho"]
        out(f"U-d 样本内 ρ={infold['M1: Δ_res']['rho']:+.4f} vs LOO ρ={m1['rho']:+.4f}  "
            f"差={gap:+.4f}  "
            f"→ {'存在外推损失（正常）' if gap > 0.05 else '无外推损失（可疑/无过拟合）'}")
    out(f"零模型 ρ={zst['rho']:+.4f}  balAcc={zst['bal_acc']:.4f}")

    verdict = ("存在可外推的可筛选性" if z == "通过"
               else ("撤回机制可比性判据路线（第五次失败）" if z == "否决"
                     else "弱信号，不足以支撑实用性主张"))
    out(f"\n总裁决：{verdict}")

    # ---- 写文件 ----
    res = loo_results["M1: Δ_res"][1]
    with open(os.path.join(RES, "loo_screen.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)

    with open(os.path.join(RES, "loo_screen.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写 {os.path.join(RES, 'loo_screen.csv')}")
    print(f"已写 {os.path.join(RES, 'loo_screen.txt')}")


if __name__ == "__main__":
    main()

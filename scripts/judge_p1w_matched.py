"""W-P1-W 附加分析：按"参数量是否匹配"分层重算 W1/W2/W3

动机
----
4 个架构的 full/indep 两臂参数量比值差异极大：

    DLinear       18,624 / 111,744  → indep 是 full 的 6.0 倍
    iTransformer  79,520 /  79,520  → 恰好 1.00 倍（纯机制对照）
    TimeMixer    149,870 /  89,753  → full 是 indep 的 1.67 倍
    PatchTST   2,722,880 / 141,920  → full 是 indep 的 19.2 倍（极端不匹配）

**这是严重混淆**：若 indep 臂赢，可能只是"参数更多"的正则化/拟合优势，
不是"不共享"的机制优势。反过来 PatchTST 的 full 臂 19 倍参数，
其胜出更可能是容量而非机制。

预注册 W1–W4 没有区分这一层（§1 只写了"两臂定义"，没控制参数量）。
按纪律：**原判据结果照报，额外报本分层分析**，以本分析的解释为准。

预注册判据
----------
  P1. 参数量匹配子集的可识别性
      判据：存在 ≥1 个架构 |log2(indep/full)| ≤ 0.1（即参数比在 0.93–1.07）
  P2. 匹配子集上 W1 是否仍成立
      判据：匹配子集 ρ ≥ +0.40 且 bootstrap CI 下界 > 0
  P3. 匹配子集上 W2 是否仍成立
      判据：Δ_res<5% 的单元中 Δ_mech_real≤0 的精确率 ≥ 0.70
  P4. 不匹配子集与匹配子集的 Δ_mech 均值差异
      判据：|差| ≥ 5pp 则说明"参数量混淆"确实影响结论

输出：results/p1w_param_matched.txt
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
    if not vals:
        return float("nan"), float("nan")
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    runs = list(csv.DictReader((RES / "p1w_runs.csv").open(encoding="utf-8")))
    dres = {}
    for r in csv.DictReader((RES / "p32_points_honest.csv").open(encoding="utf-8")):
        dres[(r["dataset"], r["regime"], int(float(r["offset"])))] = float(r["delta_res"])

    cell, nparam = {}, {}
    for r in runs:
        k = (r["arch"], r["dataset"], r["regime"])
        cell.setdefault(k, {"full": [], "indep": []})
        cell[k][r["arm"]].append(float(r["mse"]))
        nparam[(r["arch"], r["arm"])] = int(float(r["n_param"]))

    rows = []
    for (arch, ds, reg), v in sorted(cell.items()):
        if not v["full"] or not v["indep"]:
            continue
        mf, mi = float(np.median(v["full"])), float(np.median(v["indep"]))
        key = (ds, reg, 0)
        if key not in dres:
            continue
        rows.append(dict(arch=arch, dataset=ds, regime=reg,
                         mse_full=mf, mse_indep=mi,
                         dmech=(mi - mf) / mi * 100.0, dres=dres[key]))

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 78)
    out("W-P1-W 附加分析：按参数量匹配分层")
    out("=" * 78)
    out(f"{'arch':14s}{'P_full':>12s}{'P_indep':>12s}{'比值':>9s}{'log2':>8s}{'判定':>8s}")
    ratio = {}
    for a in ARCHS:
        pf, pi = nparam.get((a, "full")), nparam.get((a, "indep"))
        if not pf or not pi:
            continue
        rr = pi / pf
        lg = float(np.log2(rr))
        matched = abs(lg) <= 0.1
        ratio[a] = (pf, pi, rr, lg, matched)
        out(f"{a:14s}{pf:12d}{pi:12d}{rr:9.3f}{lg:+8.3f}"
            f"{'匹配' if matched else '不匹配':>8s}")

    matched_archs = [a for a, v in ratio.items() if v[4]]
    unmatched_archs = [a for a, v in ratio.items() if not v[4]]
    P1 = len(matched_archs) >= 1
    out(f"\nP1 存在参数匹配架构（|log2 比值|≤0.1）: "
        f"{'成立' if P1 else '不成立'}  匹配={matched_archs or '无'}")

    def evaluate(sub_rows, label):
        if len(sub_rows) < 6:
            out(f"\n[{label}] 单元数 {len(sub_rows)} < 6，不足以检验")
            return None
        x = np.array([r["dres"] for r in sub_rows])
        y = np.array([r["dmech"] for r in sub_rows])
        rho = spearman(x, y)
        lo, hi = boot_spearman(x, y, seed=3)
        flag = [r for r in sub_rows if r["dres"] < 5.0]
        prec = (sum(1 for r in flag if r["dmech"] <= 0) / len(flag)) if flag else float("nan")
        out(f"\n[{label}] n={len(sub_rows)}")
        out(f"   ρ = {rho:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
        out(f"   Δ_mech 均值 = {np.mean(y):+.3f}%  中位 = {np.median(y):+.3f}%")
        if flag:
            out(f"   W2 被标记 {len(flag)}，精确率 = {prec*100:.1f}%")
        return dict(n=len(sub_rows), rho=rho, lo=lo, hi=hi,
                    mean=float(np.mean(y)), prec=prec, nflag=len(flag))

    r_match = evaluate([r for r in rows if r["arch"] in matched_archs], "P2/P3 参数匹配子集")
    r_unmatch = evaluate([r for r in rows if r["arch"] in unmatched_archs],
                         "对照 参数不匹配子集")

    P2 = bool(r_match) and r_match["rho"] >= 0.40 and r_match["lo"] > 0
    P3 = bool(r_match) and r_match["nflag"] > 0 and r_match["prec"] >= 0.70
    P4 = (bool(r_match) and bool(r_unmatch)
          and abs(r_match["mean"] - r_unmatch["mean"]) >= 5.0)

    out("\n" + "=" * 78)
    out("判据裁决")
    out("=" * 78)
    out(f"  P1 存在参数匹配架构       : {'成立' if P1 else '不成立'}")
    out(f"  P2 匹配子集 W1 仍成立     : {'成立' if P2 else '不成立'}"
        + (f"   ρ={r_match['rho']:+.4f}" if r_match else ""))
    out(f"  P3 匹配子集 W2 仍成立     : {'成立' if P3 else '不成立'}"
        + (f"   精确率={r_match['prec']*100:.1f}%" if r_match and r_match["nflag"] else ""))
    out(f"  P4 参数量混淆影响结论     : {'成立（混淆显著）' if P4 else '不成立（混淆不显著）'}"
        + (f"   |Δ|={abs(r_match['mean']-r_unmatch['mean']):.2f}pp"
           if r_match and r_unmatch else ""))

    out("\n结论解释（据此写论文）:")
    if P2 and not P4:
        out("  → 参数匹配后 Δ_res–Δ_mech 关系仍成立，且混淆不显著。")
        out("    可以把 Δ_res 结论推广到真实架构，不受参数量混淆干扰。")
    elif P2 and P4:
        out("  → 匹配子集上关系成立，但参数量混淆显著影响绝对幅度。")
        out("    只能写『机制方向成立，但绝对增益含容量成分』。")
    elif P1 and not P2:
        out("  → ⛔ 参数匹配后关系消失。这意味着原 W1 的 ρ 主要由参数量差异驱动，")
        out("    不是『共享 vs 独立』的机制效应。必须如实降级。")
    else:
        out("  → ⛔ 无参数匹配架构可用，无法排除容量混淆。")
        out("    必须声明『Δ_mech_real 的机理无法与参数量分离』。")

    (RES / "p1w_param_matched.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n已写 {RES / 'p1w_param_matched.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

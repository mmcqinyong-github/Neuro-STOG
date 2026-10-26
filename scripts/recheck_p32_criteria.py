"""W-P1-T：用无泄漏 Δ_res 重算 W-P3-2 的 H-B2/H-B3 与容量归因 T4。

背景
----
`scripts/recheck_p32_honest.py` 结果：
  - H-B1 存活：ρ 从 +0.8469（leak）→ +0.8420（honest），CI 下界仍 >0
  - 但 Δ_res 的**绝对值**被系统性虚高 5–6pp：
      ETTm1   +6.89 → +0.82 (−6.07pp)
      electricity +8.02 → +1.89 (−6.13pp)
      traffic +5.61 → +0.66 (−4.95pp)

后果：H-B2（"无分辨力环境 |Δ_res|<5% 的占比 ≥25%"）在 leak 口径下是 47.6%，
但泄漏把数值整体抬高，**honest 口径下该比例必然大幅上升**。
这不是小修正，而是把核心命题的强度改掉。必须重算。

判据（沿用 W-P3-2 原阈值，不放宽）
----------------------------------
  H-B2 : |Δ_res| < 5% 的点数占比 ≥25% → 通过；<15% → 否决
  H-B3 : 各数据集中位 Δ_res 的极差 ≥5pp → 通过；<2pp → 否决
  T4'   : 控制容量后 Δ_res→Δ_mech 是否消失（用 honest Δ_res 重做）

输出：results/p32_hb_recheck.txt
"""

from __future__ import annotations

import csv
import os
import statistics as st

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")


def _rank(x):
    x = np.asarray(x, float)
    order = x.argsort()
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def partial_spearman(a, b, c):
    ra, rb, rc = _rank(a), _rank(b), _rank(c)
    A = np.column_stack([np.ones(len(c)), rc])
    ea = ra - A @ np.linalg.lstsq(A, ra, rcond=None)[0]
    eb = rb - A @ np.linalg.lstsq(A, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])


def main():
    with open(os.path.join(RES, "p32_points.csv"), encoding="utf-8") as f:
        leak = list(csv.DictReader(f))
    with open(os.path.join(RES, "p32_points_honest.csv"), encoding="utf-8") as f:
        hon = list(csv.DictReader(f))

    ARMS = ["F", "FP", "S", "L", "D"]
    for rows in (leak, hon):
        for r in rows:
            r["delta_res"] = float(r["delta_res"])
            r["delta_mech"] = float(r["delta_mech"])
            r["rel"] = {a: float(r["rel_" + a]) for a in ARMS}
            r["npar"] = {a: int(float(r["nparam_" + a])) for a in ARMS}

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 88)
    out("用无泄漏 Δ_res 重算 W-P3-2 判据")
    out("=" * 88)
    out(f"leak 点 {len(leak)}，honest 点 {len(hon)}")

    THR = 5.0
    out("\n--- H-B2：无分辨力环境（|Δ_res| < 5%）的普遍性 ---")
    for tag, rows in (("leak(原结论)", leak), ("honest(修复后)", hon)):
        n = sum(1 for r in rows if abs(r["delta_res"]) < THR)
        pct = n / len(rows) * 100
        verdict = "通过" if pct >= 25 else ("否决" if pct < 15 else "灰区")
        out(f"  {tag:16s}: {n}/{len(rows)} = {pct:5.1f}%  → {verdict}")

    out("\n--- H-B3：各数据集中位 Δ_res 的极差 ---")
    for tag, rows in (("leak(原结论)", leak), ("honest(修复后)", hon)):
        byds = {}
        for r in rows:
            byds.setdefault(r["dataset"], []).append(r["delta_res"])
        meds = {k: st.median(v) for k, v in byds.items()}
        span = max(meds.values()) - min(meds.values())
        verdict = "通过" if span >= 5 else ("否决" if span < 2 else "灰区")
        out(f"  {tag:16s}: 极差 = {span:5.2f} pp  → {verdict}")
        out("      逐数据集中位: " + "  ".join(
            f"{k}={v:+.2f}" for k, v in sorted(meds.items(), key=lambda x: -x[1])))

    out("\n--- 逐数据集 |Δ_res| 中位数与『无分辨力』判定（honest）---")
    out(f"{'dataset':14s} {'med Δ_res':>11s} {'med|Δ_res|':>12s} {'<5%?':>7s} {'med Δ_mech':>12s}")
    for ds in sorted({r["dataset"] for r in hon}):
        sub = [r for r in hon if r["dataset"] == ds]
        md = st.median([r["delta_res"] for r in sub])
        ma = st.median([abs(r["delta_res"]) for r in sub])
        mm = st.median([r["delta_mech"] for r in sub])
        out(f"{ds:14s} {md:+11.2f} {ma:12.2f} {'是' if ma < THR else '否':>7s} {mm:+12.2f}")

    out("\n--- T4'：控制容量后 Δ_res→Δ_mech 是否消失（honest Δ_res）---")
    for tag, rows in (("leak", leak), ("honest", hon)):
        dr = np.array([r["delta_res"] for r in rows])
        dm = np.array([r["delta_mech"] for r in rows])
        cap = np.array([spearmanr([r["npar"][a] for a in ARMS],
                                  [r["rel"][a] for a in ARMS]).statistic
                        for r in rows])
        r0 = spearmanr(dr, dm).statistic
        r1 = partial_spearman(dr, dm, cap)
        out(f"  {tag:7s}: 原始 ρ={r0:+.4f}  控制容量后偏ρ={r1:+.4f}  "
            f"保留 {abs(r1)/max(abs(r0),1e-9)*100:5.1f}%")

    out("\n--- 关键诊断：honest Δ_res 的分布 vs Δ_mech 的分布 ---")
    for tag, rows in (("leak", leak), ("honest", hon)):
        dr = np.array([r["delta_res"] for r in rows])
        dm = np.array([r["delta_mech"] for r in rows])
        out(f"  {tag:7s}: Δ_res  中位 {np.median(dr):+7.2f}  IQR "
            f"[{np.percentile(dr,25):+7.2f},{np.percentile(dr,75):+7.2f}]")
        out(f"           Δ_mech 中位 {np.median(dm):+7.2f}  IQR "
            f"[{np.percentile(dm,25):+7.2f},{np.percentile(dm,75):+7.2f}]")
        out(f"           |Δ_mech| 中位 {np.median(np.abs(dm)):+7.2f}")

    # ---- 汇总 ----
    n_hon = sum(1 for r in hon if abs(r["delta_res"]) < THR) / len(hon) * 100
    out("\n" + "=" * 88)
    out(f"结论 1：H-B1 存活（ρ +0.8469→+0.8420），Δ_res 的信号不是泄漏伪影")
    out(f"结论 2：但 |Δ_res| 被泄漏系统性抬高 5–6pp，"
        f"「无分辨力」占比从 47.6% 升至 {n_hon:.1f}%")
    out(f"结论 3：P3-3 层 2 的崩溃（3/6 符号翻转）源于 d/n 失控"
        f"（weather 1.26），p32 的 d/n≈0.22 未失控 —— 两者不矛盾")

    with open(os.path.join(RES, "p32_hb_recheck.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写 {os.path.join(RES, 'p32_hb_recheck.txt')}")


if __name__ == "__main__":
    main()

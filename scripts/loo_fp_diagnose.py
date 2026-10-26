"""W-P1-U 附加诊断：假阳性（FP）集中于哪类环境？

动机
----
LOO-DS 的符号判定 FP=17 / 真实负 39 ⇒ 假阳性率 **43.6%**。
这个数字决定了筛选器只能用于"排除"而不能用于"确认"。
必须查明 FP 的结构，否则无法给出使用边界。

预注册（先写）
-------------
  F1. FP 是否集中于 Δ_res 中等区？
      假设：Δ_res 很小（\u003c2%）时 oracle 优势接近噪声，Δ_mech 符号近随机 ⇒ 高 FP
      判据：|Δ_res| \u003c 2% 的点里 FP 占比 是否 ≥ 全体的 1.5 倍

  F2. FP 是否集中在特定体制（shift/noise/raw）？
      判据：某体制的 FP 率 ≥ 2× 其他体制的中位

  F3. 若用"只对 |Δ_res| ≥ 阈值的点做判定"能否降低 FP？
      这是**可操作的补救**：报出阈值-精度曲线，让使用者自选工作点。

输出：控制台 + results/loo_fp_diag.txt
"""

from __future__ import annotations

import csv
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")


def main():
    rows = list(csv.DictReader(open(os.path.join(RES, "loo_screen.csv"), encoding="utf-8")))
    hon = {(r["dataset"], r["regime"], int(float(r["offset"]))): float(r["delta_res"])
           for r in csv.DictReader(open(os.path.join(RES, "p32_points_honest.csv"),
                                        encoding="utf-8"))}
    for r in rows:
        r["actual"] = float(r["actual"])
        r["pred"] = float(r["pred"])
        r["offset"] = int(float(r["offset"]))
        r["ares"] = abs(hon[(r["dataset"], r["regime"], r["offset"])])
        r["dres"] = hon[(r["dataset"], r["regime"], r["offset"])]
        r["fp"] = (r["actual"] <= 0) and (r["pred"] > 0)
        r["tp"] = (r["actual"] > 0) and (r["pred"] > 0)
        r["fn"] = (r["actual"] > 0) and (r["pred"] <= 0)
        r["tn"] = (r["actual"] <= 0) and (r["pred"] <= 0)

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    n_neg = sum(1 for r in rows if r["actual"] <= 0)
    n_pos = len(rows) - n_neg
    n_fp = sum(r["fp"] for r in rows)
    out("=" * 84)
    out("FP 结构诊断")
    out("=" * 84)
    out(f"总 {len(rows)} 点：真实正 {n_pos}，真实负 {n_neg}，FP {n_fp} "
        f"（假阳性率 {n_fp/max(n_neg,1)*100:.1f}%）")

    # ---- F1：按 |Δ_res| 分箱 ----
    out("\n--- F1：FP 是否集中于 |Δ_res| 小区间 ---")
    out(f"{'|Δ_res| 区间':>16s}{'n':>5s}{'n_neg':>7s}{'FP':>5s}{'FP率':>8s}{'占全部FP':>10s}")
    bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
    hit = 0
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r["ares"] < hi]
        if not sub:
            continue
        ng = [r for r in sub if r["actual"] <= 0]
        f = sum(r["fp"] for r in sub)
        lab = f"[{lo}, {hi if hi < 1e8 else '∞'})"
        out(f"{lab:>16s}{len(sub):5d}{len(ng):7d}{f:5d}"
            f"{f/max(len(ng),1)*100:7.1f}%{f/max(n_fp,1)*100:9.1f}%")
        if lo == 0:
            hit = f

    small = [r for r in rows if r["ares"] < 2]
    small_neg = [r for r in small if r["actual"] <= 0]
    small_fp = sum(r["fp"] for r in small)
    base_rate = n_fp / max(n_neg, 1)
    small_rate = small_fp / max(len(small_neg), 1)
    out(f"\n  |Δ_res|<2% 区间: n={len(small)}, 真实负 {len(small_neg)}, FP {small_fp}, "
        f"FP率 {small_rate*100:.1f}%")
    out(f"  全体 FP 率 {base_rate*100:.1f}%  → 比值 {small_rate/max(base_rate,1e-9):.2f}×")
    F1 = small_rate >= 1.5 * base_rate
    out(f"  F1（≥1.5×）: {'成立' if F1 else '不成立'}")

    # ---- F2：按体制 ----
    out("\n--- F2：FP 是否集中于特定体制 ---")
    out(f"{'regime':>10s}{'n':>5s}{'n_neg':>7s}{'FP':>5s}{'FP率':>8s}")
    regrate = {}
    for reg in sorted({r["regime"] for r in rows}):
        sub = [r for r in rows if r["regime"] == reg]
        ng = [r for r in sub if r["actual"] <= 0]
        f = sum(r["fp"] for r in sub)
        regrate[reg] = f / max(len(ng), 1)
        out(f"{reg:>10s}{len(sub):5d}{len(ng):7d}{f:5d}{regrate[reg]*100:7.1f}%")
    mx = max(regrate.values())
    md = np.median(list(regrate.values()))
    F2 = mx >= 2 * max(md, 1e-9)
    out(f"  F2（最大 ≥2× 中位）: {'成立' if F2 else '不成立'} "
        f"(max {mx*100:.1f}% vs 中位 {md*100:.1f}%)")

    # ---- F3：阈值-精度曲线（可操作补救）----
    out("\n--- F3：只对 |Δ_res| ≥ 阈值的点做判定，FP 如何变化 ---")
    out(f"{'阈值':>6s}{'保留n':>7s}{'n_neg':>7s}{'FP':>5s}{'FP率':>8s}"
        f"{'TP':>5s}{'FN':>5s}{'召回':>8s}{'精确率':>8s}")
    best = None
    for thr in (0, 1, 2, 3, 5, 8, 10, 15):
        sub = [r for r in rows if r["ares"] >= thr]
        ng = [r for r in sub if r["actual"] <= 0]
        pg = [r for r in sub if r["actual"] > 0]
        f = sum(r["fp"] for r in sub)
        tp = sum(r["tp"] for r in sub)
        fn = sum(r["fn"] for r in sub)
        npred = f + tp
        rec = tp / max(tp + fn, 1)
        prec = tp / max(npred, 1)
        out(f"{thr:6d}{len(sub):7d}{len(ng):7d}{f:5d}"
            f"{f/max(len(ng),1)*100:7.1f}%{tp:5d}{fn:5d}{rec*100:7.1f}%{prec*100:7.1f}%")
        # 记录"召回 >=0.85 前提下 FP 率最低"的工作点
        if rec >= 0.85 and (best is None or f / max(len(ng), 1) < best[1]):
            best = (thr, f / max(len(ng), 1), rec, prec)

    if best:
        out(f"\n  推荐工作点: |Δ_res| ≥ {best[0]}%  "
            f"→ FP率 {best[1]*100:.1f}%, 召回 {best[2]*100:.1f}%, 精确率 {best[3]*100:.1f}%")

    # ---- 汇总与边界 ----
    out("\n" + "=" * 84)
    out("使用边界（据此写论文）")
    out("=" * 84)
    out(f"F1={F1}（FP 集中于小区间）  F2={F2}（FP 集中于某体制）")
    if best:
        out(f"\n可操作结论：")
        out(f"  1) Δ_res 低的环境（|Δ_res| < {best[0]}%）不要下「机制有效」的结论 —— "
            f"此区间 FP 率 {base_rate*100:.0f}%")
        out(f"  2) 只在 |Δ_res| ≥ {best[0]}% 的点上做机制比较，"
            f"此时召回 {best[2]*100:.1f}% / 精确率 {best[3]*100:.1f}%")
        out(f"  3) 这正好对应 p32 的 H-B2：54.3% 的点落在 |Δ_res|<5% 的无分辨力区")

    with open(os.path.join(RES, "loo_fp_diag.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写 {os.path.join(RES, 'loo_fp_diag.txt')}")


if __name__ == "__main__":
    main()

"""机制排名反转的归因：容量适配 vs 机制优越（W-P1-T）。

问题
----
W-P3-2 的 105 个点显示"排名在同一数据集内随评测偏移翻转"。翻转的原因是什么？

两种候选解释：
  H_cap  **容量适配**：各档位参数量不同（F=96, FP=54, S=28, L=22, D=32），
         "赢家"只是碰巧容量适配那一刀切分。「机制」本身没有贡献。
  H_mech **机制优越**：在容量被控制后，机制仍有稳定的优劣排序。

判据（先写下）
-------------
  T1. **容量斜率**：对每个点，计算 Spearman(nparam, rel) over 5 个档位。
      若负斜率与正斜率**都**显著存在（各 ≥30%），则"容量偏好"本身随环境翻转
      ⇒ 支持 H_cap。

  T2. **近匹配子集**：只保留 nparam ∈ [22, 32] 的三个档位（L=22, S=28, D=32），
       它们的容量差 ≤10 参数（相对 96 的 F 是小容量族）。
       计算该子集内的 rel 极差 median，与**全 5 档**的 rel 极差 median 比较：
       若子集极差显著更小（<50%），则 Δ_mech 主要由容量跨度造成 ⇒ 支持 H_cap。

  T3. **容量方向的 dataset×offset 翻转率**：
       在同一数据集的 5 个 offset 之间，容量斜率的符号翻转比例。
       若翻转率 ≥40%，则"容量该往哪边走"在同一数据集内都不稳定 ⇒ 机制排名
       在同一数据集内不可能稳定 ⇒ 支持 H_cap。

  T4. **Δ_res 无关性检查**：Δ_mech 与 Δ_res 的偏相关（控制 nparam_best 后）。
       若控制容量后 Δ_res → Δ_mech 的相关消失，则 W-P3-2 的 ρ=+0.847 也是混淆。

输出：results/capacity_attribution.csv, results/capacity_attribution.txt
"""

from __future__ import annotations

import csv
import os
import statistics as st
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
PTS = os.path.join(ROOT, "results", "p32_points.csv")
OUT = os.path.join(ROOT, "results")

ARMS = ["F", "FP", "S", "L", "D"]


def rankdata(x):
    """平均秩。"""
    x = np.asarray(x, float)
    order = x.argsort()
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1)
    # 处理并列
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def main():
    with open(PTS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("delta_res", "delta_mech", "nparam_F", "nparam_FP",
                  "nparam_S", "nparam_L", "nparam_D", "base_wmse"):
            r[k] = float(r[k])
        r["offset"] = int(float(r["offset"]))
        r["rel"] = {a: float(r["rel_" + a]) for a in ARMS}
        r["npar"] = {a: int(float(r["nparam_" + a])) for a in ARMS}

    print("=" * 92)
    print("机制排名反转的归因：容量适配 vs 机制优越")
    print("=" * 92)
    print(f"点数 {len(rows)}  数据集 {len({r['dataset'] for r in rows})}")

    npar_vals = sorted({r["npar"][a] for r in rows for a in ARMS})
    print(f"\n各档位参数量（全点一致）: {dict((a, rows[0]['npar'][a]) for a in ARMS)}")

    # ---- T1 容量斜率 ----
    slopes = []
    for r in rows:
        npv = [r["npar"][a] for a in ARMS]
        rv = [r["rel"][a] for a in ARMS]
        rho, _ = spearmanr(npv, rv)
        slopes.append(rho)
        r["cap_slope"] = rho
    slopes = np.array(slopes)
    neg = int(np.sum(slopes < -0.5))
    pos = int(np.sum(slopes > 0.5))
    print(f"\n--- T1 容量斜率 Spearman(nparam, rel) over 5 档 ---")
    print(f"  强负斜率(<-0.5): {neg}/{len(rows)} = {neg/len(rows)*100:.1f}%")
    print(f"  强正斜率(>+0.5): {pos}/{len(rows)} = {pos/len(rows)*100:.1f}%")
    print(f"  中位斜率 {st.median(slopes):+.4f}")
    T1 = (neg >= 0.3 * len(rows)) and (pos >= 0.3 * len(rows))
    print(f"  → 双向均显著（各≥30%）: {'成立' if T1 else '不成立'}")

    # ---- T2 近匹配子集 ----
    near = ["L", "S", "D"]   # 22, 28, 32
    span_all, span_near = [], []
    for r in rows:
        va = [r["rel"][a] for a in ARMS]
        vn = [r["rel"][a] for a in near]
        span_all.append(max(va) - min(va))
        span_near.append(max(vn) - min(vn))
    m_all, m_near = st.median(span_all), st.median(span_near)
    print(f"\n--- T2 近匹配子集（{near}, nparam={[rows[0]['npar'][a] for a in near]}）---")
    print(f"  全 5 档 rel 极差 中位数 = {m_all:.3f} pp")
    print(f"  近匹配 3 档 rel 极差 中位数 = {m_near:.3f} pp")
    print(f"  比值 = {m_near/max(m_all,1e-9)*100:.1f}%")
    T2 = m_near < 0.5 * m_all
    print(f"  → 子集极差 < 50% 全档极差: {'成立' if T2 else '不成立'}")

    # ---- T3 同一数据集内容量方向的 offset 翻转率 ----
    print(f"\n--- T3 同一数据集内 容量斜率 的 offset 间符号翻转 ---")
    byds = defaultdict(list)
    for r in rows:
        byds[r["dataset"]].append(r)
    flips, total = 0, 0
    for ds, rs in sorted(byds.items()):
        rs = sorted(rs, key=lambda x: (x["regime"], x["offset"]))
        byreg = defaultdict(list)
        for r in rs:
            byreg[r["regime"]].append(r["cap_slope"])
        f_here, t_here = 0, 0
        for reg, sl in byreg.items():
            for i in range(len(sl)):
                for j in range(i + 1, len(sl)):
                    t_here += 1
                    if np.sign(sl[i]) != np.sign(sl[j]):
                        f_here += 1
        flips += f_here
        total += t_here
        print(f"  {ds:14s} 翻转 {f_here}/{t_here}  "
              f"斜率范围 [{min(min(v) for v in byreg.values()):+.2f}, "
              f"{max(max(v) for v in byreg.values()):+.2f}]")
    rate = flips / max(total, 1)
    print(f"  总翻转率 = {flips}/{total} = {rate*100:.1f}%")
    T3 = rate >= 0.40
    print(f"  → ≥40%: {'成立' if T3 else '不成立'}")

    # ---- T4 控制容量后 Δ_res 与 Δ_mech 的偏相关 ----
    print(f"\n--- T4 控制容量后 Δ_res→Δ_mech 的偏相关 ---")
    dr = np.array([r["delta_res"] for r in rows])
    dm = np.array([r["delta_mech"] for r in rows])
    # 容量代理：最优档位的参数量
    nbest = np.array([r["npar"][max(ARMS, key=lambda a: r["rel"][a])] for r in rows])
    # 也用一个连续容量代理：各档 rel 对 nparam 回归的斜率
    cap = np.array([r["cap_slope"] for r in rows])

    def partial(a, b, c):
        ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
        # 偏 Spearman：对秩做线性回归后取残差相关
        A = np.column_stack([np.ones(len(c)), rc])
        ea = ra - A @ np.linalg.lstsq(A, ra, rcond=None)[0]
        eb = rb - A @ np.linalg.lstsq(A, rb, rcond=None)[0]
        return float(np.corrcoef(ea, eb)[0, 1])

    r0 = spearmanr(dr, dm).statistic
    r1 = partial(dr, dm, cap)
    print(f"  原始 Spearman(Δ_res, Δ_mech)        = {r0:+.4f}")
    print(f"  控制容量斜率后的偏 Spearman         = {r1:+.4f}")
    print(f"  相关衰减 = {abs(r0)-abs(r1):+.4f} ({abs(r1)/max(abs(r0),1e-9)*100:.1f}% 保留)")
    T4 = abs(r1) < 0.5 * abs(r0)
    print(f"  → 控制后保留 <50%: {'成立' if T4 else '不成立'}")

    # ---- 汇总 ----
    print("\n" + "=" * 92)
    print(f"T1 容量偏好双向同时存在 : {'成立' if T1 else '不成立'}")
    print(f"T2 容量跨度解释大部分 Δ_mech : {'成立' if T2 else '不成立'}")
    print(f"T3 同数据集内容量方向不稳定 : {'成立' if T3 else '不成立'}")
    print(f"T4 控制容量后 Δ_res 信号消失 : {'成立' if T4 else '不成立'}")
    hits = sum([T1, T2, T3, T4])
    verdict = ("H_cap 成立：机制排名主要由容量适配决定" if hits >= 3
               else ("H_cap 部分成立" if hits == 2 else "H_cap 不成立：机制效应独立于容量"))
    print(f"\n{hits}/4 → {verdict}")

    csvp = os.path.join(OUT, "capacity_attribution.csv")
    with open(csvp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "regime", "offset", "cap_slope",
                    "rel_span_all", "rel_span_near", "delta_res", "delta_mech"])
        for r, s_all, s_near in zip(rows, span_all, span_near):
            w.writerow([r["dataset"], r["regime"], r["offset"],
                        round(r["cap_slope"], 4), round(s_all, 4),
                        round(s_near, 4), r["delta_res"], r["delta_mech"]])
    txtp = os.path.join(OUT, "capacity_attribution.txt")
    with open(txtp, "w", encoding="utf-8") as f:
        f.write(f"T1={T1} T2={T2} T3={T3} T4={T4}\n")
        f.write(f"slope_neg={neg} slope_pos={pos} n={len(rows)}\n")
        f.write(f"span_all={m_all:.4f} span_near={m_near:.4f} ratio={m_near/m_all:.4f}\n")
        f.write(f"flip_rate={rate:.4f}\n")
        f.write(f"rho_dres_dmech={r0:.4f} partial={r1:.4f}\n")
        f.write(f"verdict={verdict}\n")
    print(f"\n已写 {csvp}\n已写 {txtp}")


if __name__ == "__main__":
    main()

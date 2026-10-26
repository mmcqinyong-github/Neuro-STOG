"""机制比较的可判定性：统计功效要求 vs TSF 文献的种子惯例。

动机（W-P1-T）
--------------
两次自我证伪（LIH 被文献降级、Δ_res 被自检推翻）暴露的共同问题不是"假说不够多"，
而是**我们没有先问"这个比较本身是否可判定"**。

一个具体的、可立即用现有数据回答的问题：
> 在时间序列预测里，宣称"机制 A 优于机制 B"需要多少个随机种子？

现有数据：`results/p33_raw/` 中有 40 次官方 RAFT 复现（6 数据集 × 3 档 × 3 种子，
档 A 按 v1.3 §C.2 缩减），每次给出 test MSE。同一 (数据集, 档) 的多次种子
给出**同一配置下的实现波动** —— 这正好是"种子噪声"的直接估计，
无需任何额外训练。

方法
----
对每个 (数据集, 档) 有 n_seed 个 MSE 值（n_seed = 2 或 3）：

1. **种子噪声 σ_seed**：用合并的组内标准差（pooled within-group SD）估计。
   由于每组只有 2–3 个观测，直接算组内 SD 噪声很大，故用 pooled 形式：
       σ_seed² = Σ_g Σ_i (x_gi − x̄_g)² / Σ_g (n_g − 1)
   并报告其自由度和 95% CI（卡方分布）。

2. **机制效应 δ**：Δ_mech 的绝对值（%），转成 MSE 绝对差：
       δ_abs = |MSE_A − MSE_B|（同一数据集、同一随机种子配对）

3. **所需种子数**（双侧 α=0.05、power=0.80、配对设计）：
       n_req = (z_{1−α/2} + z_{power})² · σ_seed² / δ_abs²   × 2（配对效率折半）
   更标准的形式：配对 t 检验 n = (z_{0.975}+z_{0.80})² σ_d²/δ²，
   其中 σ_d 为**差值**的标准差。若两臂独立且同方差，σ_d² = 2σ_seed²。

4. **与文献惯例对照**：TSF 论文几乎一律用 1 个种子（部分 3–5 个）。

判据（先写下，再计算）
--------------------
  P1. 若"多数 (数据集, 机制) 组合的 n_req > 5"，则说明 5 种子惯例不足
      （>5 判为"超惯例"，统计上意味着文献中大量排名翻转是预期的）

  P2. 若 n_req 的中位数 > 20，则说明"在 TSF 上宣称机制优越"在常规成本下
      不可判定 ⇒ 支持"机制比较 ill-posed"命题

输出：results/mc_power.csv、results/mc_power.txt
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "results", "p33_raw")
OUT = os.path.join(ROOT, "results")

Z_975, Z_80 = 1.959964, 0.8416212
ALPHA, POWER = 0.05, 0.80


def load() -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def main():
    recs = load()
    print("=" * 88)
    print("机制比较可判定性：所需种子数 vs 文献惯例（基于 40 次官方 RAFT 复现）")
    print("=" * 88)
    print(f"原始记录 {len(recs)} 条")

    # ---- 组： (dataset, arm) → [mse] ----
    grp: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in recs:
        grp[(r["dataset"], r["arm"])].append(float(r["mse"]))

    # ---- 种子噪声 pooled within-group ----
    ss, df = 0.0, 0
    detail = []
    for (ds, arm), v in sorted(grp.items()):
        if len(v) < 2:
            continue
        m = st.mean(v)
        ss_g = sum((x - m) ** 2 for x in v)
        df_g = len(v) - 1
        ss += ss_g
        df += df_g
        detail.append((ds, arm, len(v), m, st.stdev(v), st.stdev(v) / m * 100))
    if df == 0:
        print("组内自由度不足，无法估计种子噪声")
        return
    sigma2 = ss / df
    sigma = sigma2 ** 0.5

    print(f"\n组内（同配置不同种子）统计：")
    print(f"{'dataset':14s} {'arm':>4s} {'n':>3s} {'MSE均值':>11s} {'SD':>10s} {'CV%':>7s}")
    for ds, arm, n, m, sd, cv in detail:
        print(f"{ds:14s} {arm:>4s} {n:3d} {m:11.5f} {sd:10.5f} {cv:7.3f}")

    # chi2 CI for sigma2
    from scipy.stats import chi2  # noqa: PLC0415
    lo = df * sigma2 / chi2.ppf(0.975, df)
    hi = df * sigma2 / chi2.ppf(0.025, df)
    print(f"\n合并种子噪声 σ_seed = {sigma:.5f}  (df={df})")
    print(f"σ_seed² 的 95% CI = [{lo**0.5:.5f}, {hi**0.5:.5f}]")
    print(f"中位 CV = {st.median([c for *_, c in detail]):.3f}%  "
          f"(相对 MSE 量级)")

    # ---- 机制效应：同数据集、同种子、两臂配对 ----
    def get(ds, arm, seed):
        for r in recs:
            if r["dataset"] == ds and r["arm"] == arm and r["seed"] == seed:
                return float(r["mse"])
        return None

    pairs = []
    datasets = sorted({r["dataset"] for r in recs})
    for ds in datasets:
        for a1, a2 in (("on", "B"), ("on", "A"), ("A", "B")):
            for s in (0, 1, 2):
                x1, x2 = get(ds, a1, s), get(ds, a2, s)
                if x1 is not None and x2 is not None:
                    pairs.append(dict(dataset=ds, a1=a1, a2=a2, seed=s,
                                      mse1=x1, mse2=x2, d=x1 - x2,
                                      rel=(x2 - x1) / x2 * 100))
    print(f"\n可配对比较 {len(pairs)} 对（同数据集、同种子、两档）")

    print(f"\n{'dataset':14s} {'臂对':>8s} {'ΔMSE':>11s} {'相对%':>9s} "
          f"{'|δ|/σ':>8s} {'n_req(pair)':>12s}")
    rows = []
    for ds in datasets:
        for a1, a2 in (("on", "B"), ("on", "A"), ("A", "B")):
            sub = [p for p in pairs if p["dataset"] == ds and p["a1"] == a1 and p["a2"] == a2]
            if not sub:
                continue
            dmean = st.mean(p["d"] for p in sub)
            # 差值 SD：优先用配对差值的实际 SD；样本 <2 时退化为独立假设 2σ²
            if len(sub) >= 2:
                sd_d = st.stdev(p["d"] for p in sub)
                sd_d = max(sd_d, (2 * sigma2) ** 0.5 * 1e-3)
            else:
                sd_d = (2 * sigma2) ** 0.5
            rel = st.mean(p["rel"] for p in sub)
            if abs(dmean) < 1e-12:
                n_req = float("inf")
            else:
                n_req = (Z_975 + Z_80) ** 2 * sd_d ** 2 / dmean ** 2
            rows.append(dict(dataset=ds, arm1=a1, arm2=a2, n_pair=len(sub),
                             d_mse=dmean, rel_pct=rel, sd_diff=sd_d,
                             effect_over_sigma=abs(dmean) / sigma,
                             n_req=n_req))
            print(f"{ds:14s} {a1+'-'+a2:>8s} {dmean:+11.5f} {rel:+9.3f} "
                  f"{abs(dmean)/sigma:8.2f} {n_req:12.1f}")

    # ---- P1 / P2 ----
    finite = [r["n_req"] for r in rows if np.isfinite(r["n_req"])]
    med = st.median(finite) if finite else float("inf")
    over5 = sum(1 for x in finite if x > 5)
    P1 = over5 > len(finite) / 2
    P2 = med > 20
    print()
    print(f"P1 多数组合 n_req > 5（5 种子惯例不足）: {'成立' if P1 else '不成立'}  "
          f"({over5}/{len(finite)} 个组合 > 5)")
    print(f"P2 n_req 中位数 > 20（常规成本下不可判定）: {'成立' if P2 else '不成立'}  "
          f"(中位数 {med:.1f})")

    # ---- 与"论文报告差异"对照 ----
    print()
    print("关键对照：n_req 与『论文声称的效应量』")
    for r in rows:
        if r["dataset"] == "weather" and r["arm1"] == "on":
            print(f"  weather on-B: 实测相对 {r['rel_pct']:+.3f}%, "
                  f"效应/噪声 = {r['effect_over_sigma']:.2f}σ, n_req={r['n_req']:.1f}")
            print(f"    → 论文声称 +9.34%，实测 {r['rel_pct']:+.3f}%，"
                  f"即声称效应是实测的 {9.34/max(abs(r['rel_pct']),1e-9):.0f} 倍")

    csv = os.path.join(OUT, "mc_power.csv")
    with open(csv, "w", encoding="utf-8") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    txt = os.path.join(OUT, "mc_power.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"sigma_seed={sigma:.6f} df={df} median_CV={st.median([c for *_, c in detail]):.4f}%\n")
        f.write(f"n_req_median={med:.2f} over5={over5}/{len(finite)} P1={P1} P2={P2}\n")
    print(f"\n已写 {csv}\n已写 {txt}")


if __name__ == "__main__":
    main()

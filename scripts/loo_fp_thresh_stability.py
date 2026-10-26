"""W-P1-U 附加诊断 2：|Δ_res| ≥ τ 规则的**折外**稳定性检验

动机
----
loo_fp_diagnose.py 报出"|Δ_res| ≥ 15% ⇒ FP 率 0%, 召回 100%, 精确率 100%"。
但该阈值是**在同一批 105 点上扫出来的**——典型的样本内膜拜。
必须做严格的 LOO-DS 折外检验，否则这条规则不能写进论文。

预注册（先写判据，再计算）
---------------------------
  对 7 个数据集逐折留出（留出折 = 1 个数据集全部 15 点，训练折 = 其余 6 个数据集）：

  S1. 折外可迁移性
      在训练折上选 τ*(约束：召回 ≥ 0.85；目标：最小 FP 率)，
      把 τ* 原样用到留出折，记留出折 FP 率 fp_out。
      判据：median(fp_out) ≤ 21.8%（= 全体 43.6% 的一半）

  S2. 阈值稳定性
      判据：7 折 τ* 的极差 ≤ 5 个百分点

  S3. 可用覆盖率
      判据：留出折中被保留（|Δ_res| ≥ τ*）的点占比 median ≥ 50%

  若 S1 不成立 ⇒ "≥15%" 规则**不可外推**，只能报"样本内观察"，
  论文中不得写成使用建议。这是硬约束。

输出：控制台 + results/loo_fp_thresh_stability.txt
"""

from __future__ import annotations

import csv
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")

THRS = (0, 0.5, 1, 1.5, 2, 3, 5, 8, 10, 15, 20, 25, 30)


def load():
    rows = list(csv.DictReader(open(os.path.join(RES, "loo_screen.csv"), encoding="utf-8")))
    hon = {(r["dataset"], r["regime"], int(float(r["offset"]))): float(r["delta_res"])
           for r in csv.DictReader(open(os.path.join(RES, "p32_points_honest.csv"),
                                        encoding="utf-8"))}
    for r in rows:
        r["actual"] = float(r["actual"])
        r["pred"] = float(r["pred"])
        r["offset"] = int(float(r["offset"]))
        r["ares"] = abs(hon[(r["dataset"], r["regime"], r["offset"])])
        r["fp"] = (r["actual"] <= 0) and (r["pred"] > 0)
        r["tp"] = (r["actual"] > 0) and (r["pred"] > 0)
        r["fn"] = (r["actual"] > 0) and (r["pred"] <= 0)
        r["neg"] = r["actual"] <= 0
    return rows


def stats(sub):
    ng = sum(r["neg"] for r in sub)
    f = sum(r["fp"] for r in sub)
    tp = sum(r["tp"] for r in sub)
    fn = sum(r["fn"] for r in sub)
    return ng, f, tp, fn


def fp_rate(sub):
    ng, f, _, _ = stats(sub)
    return f / max(ng, 1)


def recall(sub):
    ng, f, tp, fn = stats(sub)
    return tp / max(tp + fn, 1)


def pick_thr(sub, min_recall=0.85):
    """在给定点集上选 τ*：召回≥min_recall 前提下 FP 率最低，同分取较小 τ。"""
    best = None
    for t in THRS:
        s = [r for r in sub if r["ares"] >= t]
        if not s:
            continue
        rec = recall(s)
        if rec < min_recall:
            continue
        fpr = fp_rate(s)
        if best is None or fpr < best[1] - 1e-12:
            best = (t, fpr, rec)
    return best


def main():
    rows = load()
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    datasets = sorted({r["dataset"] for r in rows})
    base_fpr = fp_rate(rows)
    out("=" * 84)
    out("|Δ_res| ≥ τ 规则的折外稳定性检验（LOO-DS）")
    out("=" * 84)
    out(f"全体 {len(rows)} 点，全体 FP 率 {base_fpr*100:.1f}%（S1 门槛 = 半值 {base_fpr*50:.1f}%）")

    out(f"\n{'留出数据集':>12s}{'训练折τ*':>10s}{'训练FP率':>10s}{'训练召回':>10s}"
        f"{'留出n':>7s}{'保留n':>7s}{'覆盖率':>8s}{'留出FP率':>10s}{'留出召回':>10s}")
    out("-" * 84)

    res = []
    for ds in datasets:
        tr = [r for r in rows if r["dataset"] != ds]
        te = [r for r in rows if r["dataset"] == ds]
        b = pick_thr(tr)
        if b is None:
            out(f"{ds:>12s}  训练折无满足召回≥0.85 的阈值")
            continue
        t, fpr_tr, rec_tr = b
        kept = [r for r in te if r["ares"] >= t]
        cov = len(kept) / len(te)
        fpr_te = fp_rate(kept) if kept else float("nan")
        rec_te = recall(kept) if kept else float("nan")
        res.append((ds, t, fpr_tr, rec_tr, len(te), len(kept), cov, fpr_te, rec_te))
        out(f"{ds:>12s}{t:10.1f}{fpr_tr*100:9.1f}%{rec_tr*100:9.1f}%"
            f"{len(te):7d}{len(kept):7d}{cov*100:7.1f}%"
            f"{(fpr_te*100 if kept else float('nan')):9.1f}%"
            f"{(rec_te*100 if kept else float('nan')):9.1f}%")

    taus = [x[1] for x in res]
    fprs = [x[7] for x in res if x[5] > 0]
    covs = [x[6] for x in res]

    out("-" * 84)
    out(f"τ* 分布: min {min(taus):.1f}  中位 {np.median(taus):.1f}  max {max(taus):.1f}  "
        f"极差 {max(taus)-min(taus):.1f}pp")
    out(f"留出折 FP 率: 中位 {np.median(fprs)*100:.1f}%  范围 "
        f"[{min(fprs)*100:.1f}%, {max(fprs)*100:.1f}%]")
    out(f"留出折覆盖率: 中位 {np.median(covs)*100:.1f}%  最低 {min(covs)*100:.1f}%")

    S1 = np.median(fprs) <= base_fpr * 0.5
    S2 = (max(taus) - min(taus)) <= 5.0
    S3 = np.median(covs) >= 0.50

    out("\n" + "=" * 84)
    out("判据裁决")
    out("=" * 84)
    out(f"  S1 折外可迁移（median 折外 FP 率 ≤ {base_fpr*50:.1f}%）: "
        f"{'成立' if S1 else '不成立'}  实测 {np.median(fprs)*100:.1f}%")
    out(f"  S2 阈值稳定（50% 极差 ≤ 5pp）: "
        f"{'成立' if S2 else '不成立'}  实测 {max(taus)-min(taus):.1f}pp")
    out(f"  S3 覆盖率（中位 ≥50%）: {'成立' if S3 else '不成立'}  "
        f"实测 {np.median(covs)*100:.1f}%")

    n_ok = sum([S1, S2, S3])
    out(f"\n结论：{n_ok}/3 项成立。")
    if S1 and S3:
        out("  → 该规则可作为**使用建议**写入论文，但阈值需按 S2 报出不确定性。")
    elif S1:
        out("  → 折外 FP 率确实下降，但覆盖率不足 ⇒ 只能作为『高置信子集』报告，"
            "不能当作全环境筛选器。")
    else:
        out("  → ⛔ 规则**不可外推**。样本内 0% FP 是过拟合产物，"
            "论文中禁止写成使用建议，只能作为『样本内观察，未通过折外检验』记录。")

    with open(os.path.join(RES, "loo_fp_thresh_stability.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写 {os.path.join(RES, 'loo_fp_thresh_stability.txt')}")


if __name__ == "__main__":
    main()

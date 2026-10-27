"""W-P2-B 图表（Nature 风格，代码生成，300 dpi）。

四面板：
  A  读出门 2×2 列主效应（分数据集，预算精确匹配）
  B  各体制下所有变体相对冻结基线的增益
  C  预算 vs 增益散点 —— 暴露"平均预算"这一混淆（诚实展示）
  D  三条通道的时间轨迹示例（说明两条通道确实不同）
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS = os.path.join(ROOT, "results", "p2b_runs.csv")
FIGDIR = os.path.join(ROOT, "figs")

INTERNAL = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"]
EXTERNAL = ["solar", "PEMS04"]
ALLDS = INTERNAL + EXTERNAL
REGIMES = ["raw", "shift", "noise", "rewire"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "lines.linewidth": 1.2,
    "axes.linewidth": 0.8, "pdf.fonttype": 42,
})


def load():
    with open(RUNS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    extra = os.path.join(ROOT, "results", "p2b_extra.csv")
    if os.path.exists(extra):
        with open(extra, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    for r in rows:
        for k in ("mse", "budget"):
            r[k] = float(r[k])
        r["seed"] = int(r["seed"])
    return rows


def cell(rows, ds, reg, var):
    v = [r["mse"] for r in rows
         if r["dataset"] == ds and r["regime"] == reg and r["variant"] == var]
    return np.mean(v) if v else float("nan")


def gain(rows, ds, reg, var):
    b = cell(rows, ds, reg, "none")
    a = cell(rows, ds, reg, var)
    return (1 - a / b) * 100 if (np.isfinite(a) and np.isfinite(b)) else np.nan


def main() -> None:
    rows = load()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = axes[0][0]

    # ---------- A：2×2 列主效应（每个数据集一根，两种写源）----------
    x = np.arange(len(ALLDS))
    w = 0.38
    for k, (lbl, v_nov, v_cre, col) in enumerate([
            ("write = novelty (ACh)", "ach", "credit", "#4878A8"),
            ("write = credit (DA)", "cw_ar", "cw_cr", "#B4532A")]):
        d = [np.mean([gain(rows, ds, r, v_cre) - gain(rows, ds, r, v_nov)
                      for r in REGIMES]) for ds in ALLDS]
        ax.bar(x + (k - 0.5) * w, d, width=w, label=lbl, color=col,
               alpha=0.9, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(ALLDS, rotation=30, ha="right")
    ax.set_ylabel("credit-read minus novelty-read\n(gain pp, all regimes)")
    ax.set_title("A  Read-out gate should follow DA credit", fontsize=9.5)
    ax.legend(frameon=False, loc="upper right")
    ax.axvline(4.5, color="grey", ls=":", lw=0.8)
    ax.text(2.0, ax.get_ylim()[0] + 0.25, "internal", ha="center", fontsize=7,
            color="grey")
    ax.text(5.5, ax.get_ylim()[0] + 0.25, "external (untouched)",
            ha="center", fontsize=7, color="grey")

    # ---------- B：各体制增益 ----------
    ax = axes[0][1]
    adv = ["ach", "credit", "cw_ar", "cw_cr", "da", "const", "moe2"]
    cols = ["#4878A8", "#1B4F72", "#B4532A", "#7B241C", "#6C7A89", "#8E44AD",
            "#2E7D32"]
    x = np.arange(len(REGIMES))
    w = 0.14
    for k, v in enumerate(adv):
        g = [np.nanmean([gain(rows, ds, r, v) for ds in ALLDS])
             for r in REGIMES]
        ax.bar(x + (k - 2.5) * w, g, width=w, label=v, color=cols[k],
               edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIMES)
    ax.set_ylabel("gain vs frozen backbone (%)")
    ax.set_title("B  All variants, all regimes", fontsize=9.5)
    ax.legend(frameon=False, ncol=3, fontsize=6.5)

    # ---------- C：预算 vs 增益（混淆展示）----------
    ax = axes[1][0]
    for k, v in enumerate(adv):
        bs = [r["budget"] for r in rows if r["variant"] == v]
        gs = [gain(rows, r["dataset"], r["regime"], v) for r in rows
              if r["variant"] == v]
        keep = np.isfinite(gs)
        ax.scatter(np.array(bs)[keep], np.array(gs)[keep], s=14,
                   color=cols[k], label=v, alpha=0.75, linewidths=0)
    # 线性趋势
    bb = np.array([r["budget"] for r in rows if r["variant"] != "none"])
    gg = np.array([gain(rows, r["dataset"], r["regime"], r["variant"])
                   for r in rows if r["variant"] != "none"])
    kk = np.isfinite(gg)
    co = np.polyfit(np.array(bb)[kk], np.array(gg)[kk], 1)
    xs = np.linspace(np.nanmin(bb), np.nanmax(bb), 20)
    ax.plot(xs, np.polyval(co, xs), "k--", lw=1,
            label="linear trend (confound)")
    ax.set_xlabel("total plasticity budget  (sum of 1+k*drive)")
    ax.set_ylabel("gain vs frozen backbone (%)")
    ax.set_title("C  Budget confound: interpolate honestly", fontsize=9.5)
    ax.legend(frameon=False, ncol=3, fontsize=6.5)

    # ---------- D：通道轨迹 ----------
    ax = axes[1][1]
    traj_p = os.path.join(ROOT, "results", "p2b_traj.csv")
    if os.path.exists(traj_p):
        tr = list(csv.DictReader(open(traj_p, newline="", encoding="utf-8")))
        t = np.array([int(r["t"]) for r in tr])
        ax.plot(t, [float(r["d_shift"]) for r in tr], color="#4878A8",
                label="structural novelty (ACh-type)")
        ax.plot(t, [float(r["d_val"]) for r in tr], color="#B4532A",
                label="value surprise (DA-type)")
        ax.plot(t, [float(r["credit"]) for r in tr], color="#1B4F72",
                label="read-out credit (DA-type)")
        ax.plot(t, [float(r["theta"]) for r in tr], color="black", ls="--",
                lw=1, label="read gate theta")
        ax.set_xlabel("evaluation step")
        ax.set_ylabel("normalised signal")
        ax.set_title("D  Two signals behave differently in time\n(ETTh1, shift)",
                     fontsize=9.5)
        ax.legend(frameon=False, fontsize=6.5, ncol=2)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "panel D: run results/p2b_traj.csv first",
                ha="center", va="center", fontsize=8)

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(FIGDIR, f"fig_p2b_dual.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("saved", p)


if __name__ == "__main__":
    main()

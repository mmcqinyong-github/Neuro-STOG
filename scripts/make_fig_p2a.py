"""W-P2-A 图表：CTS-Net 四面板（Nature 风格，300dpi）。

运行：PYTHONUTF8=1 python scripts/make_fig_p2a.py
输出：figs/fig_p2a_cts.png / .pdf
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV = os.path.join(ROOT, "results", "p2a_runs.csv")
INTERNAL = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"]
EXTERNAL = ["solar_AL", "PEMS04"]
REGIMES = ["raw", "shift", "noise"]
REG_COLOR = {"raw": "#4C72B0", "shift": "#DD8452", "noise": "#55A868"}
REG_LABEL = {"raw": "raw (stable)", "shift": "shift (drift)", "noise": "noise"}


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    for r in rows:
        r["mse"] = float(r["mse"])
        r["lam"] = float(r["lam"])
        r["seed"] = int(r["seed"])
        r["dshift_mean"] = float(r["dshift_mean"])
    cell = defaultdict(dict)
    for r in rows:
        cell[(r["dataset"], r["regime"], r["lam"])][r["seed"]] = r

    def mm(ds, reg, lam):
        v = [cell[(ds, reg, lam)][s]["mse"] for s in cell.get((ds, reg, lam), {})]
        return float(np.mean(v)) if v else float("nan")

    datasets = sorted({r["dataset"] for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("CTS-Net: complementary teaching-signal subspace network "
                 "improves forecasting under drift", fontsize=13, y=0.985)

    # ---- A: λ=0 vs λ=1 散点（漂移体制）----
    ax = axes[0, 0]
    for reg in ("shift", "noise"):
        xs, ys = [], []
        for ds in datasets:
            m0, m1 = mm(ds, reg, 0.0), mm(ds, reg, 1.0)
            if not (np.isnan(m0) or np.isnan(m1)):
                xs.append(m0)
                ys.append(m1)
        ax.scatter(xs, ys, s=60, alpha=0.85, color=REG_COLOR[reg],
                   edgecolor="white", linewidth=0.6, label=REG_LABEL[reg], zorder=3)
    lim = [0, max([mm(ds, reg, 0.0) for ds in datasets for reg in REGIMES]) * 1.05]
    ax.plot(lim, lim, "k--", lw=1.0, zorder=2)
    ax.fill_between(lim, [0, 0], lim, color="red", alpha=0.05, zorder=1)
    ax.set_xlabel("MSE  λ=0 (backbone, no mechanism)", fontsize=10)
    ax.set_ylabel("MSE  λ=1 (CTS-Net)", fontsize=10)
    ax.set_title("A  Drift regimes: mechanism switch reduces error", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.grid(alpha=0.25, lw=0.5)

    # ---- B: λ 单调性 ----
    ax = axes[0, 1]
    for reg in ("shift", "noise"):
        vals = []
        for ds in datasets:
            v = [mm(ds, reg, l) for l in (0.0, 0.5, 1.0)]
            if not any(np.isnan(v)):
                vals.append(v)
        arr = np.array(vals)
        ax.plot([0, 0.5, 1], arr.mean(0), "o-", color=REG_COLOR[reg],
                lw=2, ms=6, label=REG_LABEL[reg])
        ax.fill_between([0, 0.5, 1], arr.mean(0) - arr.std(0),
                        arr.mean(0) + arr.std(0), color=REG_COLOR[reg],
                        alpha=0.15, lw=0)
    ax.set_xlabel("mechanism switch λ", fontsize=10)
    ax.set_ylabel("MSE (mean±sd, cross-dataset)", fontsize=10)
    ax.set_title("B  Mechanism-switch monotonicity (λ↑ ⇒ error↓)", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xticks([0, 0.5, 1.0])

    # ---- C: 增益柱状图（内部 vs 外部）----
    ax = axes[1, 0]
    groups = [("内部", INTERNAL), ("外部", EXTERNAL)]
    xpos = np.arange(len(groups))
    width = 0.25
    for k, reg in enumerate(REGIMES):
        means, sds = [], []
        for _, dslist in groups:
            g = [(1 - mm(ds, reg, 1.0) / mm(ds, reg, 0.0)) * 100
                 for ds in dslist if not (np.isnan(mm(ds, reg, 0.0))
                                          or np.isnan(mm(ds, reg, 1.0)))]
            means.append(float(np.mean(g)) if g else float("nan"))
            sds.append(float(np.std(g)) if g else float("nan"))
        ax.bar(xpos + (k - 1) * width, means, width, yerr=sds,
               color=REG_COLOR[reg], label=REG_LABEL[reg], alpha=0.9,
               error_kw=dict(lw=0.8, ecolor="k", capsize=2))
    ax.axhline(5, color="red", ls="--", lw=1, label="M1 threshold 5%")
    ax.set_xticks(xpos); ax.set_xticklabels(["internal 5 sets", "external 2 sets"])
    ax.set_ylabel("gain (%)  λ=1 vs λ=0", fontsize=10)
    ax.set_title("C  Gain consistent: internal & external", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5, axis="y")

    # ---- D: d_shift 通道分离 ----
    ax = axes[1, 1]
    for reg in REGIMES:
        vals = [cell[(ds, reg, 1.0)][s]["dshift_mean"]
                for ds in datasets for s in (0, 1, 2)
                if (ds, reg, 1.0) in cell and s in cell[(ds, reg, 1.0)]]
        bp = ax.boxplot(vals, positions=[REGIMES.index(reg)], widths=0.45,
                        patch_artist=True, showfliers=False)
        for el in bp["boxes"]:
            el.set_facecolor(REG_COLOR[reg]); el.set_alpha(0.7)
        for el in bp["medians"]:
            el.set_color("k")
    ax.set_xticks(range(3)); ax.set_xticklabels(["raw", "shift", "noise"])
    ax.set_ylabel("drift signal d_shift (λ=1)", fontsize=10)
    ax.set_title("D  Dual-channel separation: higher in drift", fontsize=11)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.set_ylim(0, 1.05)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.join(ROOT, "figs"), exist_ok=True)
    png = os.path.join(ROOT, "figs", "fig_p2a_cts.png")
    pdf = os.path.join(ROOT, "figs", "fig_p2a_cts.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[written] {png} / {pdf}")


if __name__ == "__main__":
    main()

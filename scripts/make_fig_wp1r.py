"""W-P1-R 图：潜变量可识别性相变。四面板，Nature 风格，300 dpi，代码生成。

面板
----
a  G vs σ_ξ²（可识别性）—— 核心：符号翻转 + 单峰
b  G vs n（样本量）—— 判决性：σ_ξ²>0 时平台不衰减，σ_ξ²=0 时衰减为 0
c  G vs m（域尺寸）
d  G vs τ²（异质性）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from identifiability import A_, A_m  # noqa: E402
import judge_wp1r as J  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "figs"
FIG.mkdir(exist_ok=True)

# ---- Nature 风格 ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "lines.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

# 浅色主题（IDE light）
BG = "white"
FG = "#1a1a1a"
GRID = "#d8d8d8"
C = ["#b2182b", "#2166ac", "#1a9850", "#762a83", "#e08214"]  # 红/蓝/绿/紫/橙


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(BG)
    ax.set_xlabel(xlabel, color=FG)
    ax.set_ylabel(ylabel, color=FG)
    ax.set_title(title, color=FG, pad=6, loc="left", fontweight="bold")
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color(FG)
    ax.axhline(0, color="#555555", lw=0.6, zorder=0)


fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.10, right=0.975, top=0.93, bottom=0.10,
                    wspace=0.30, hspace=0.42)

# ===========================================================================
# a) G vs σ_ξ²（核心相变）
# ===========================================================================
ax = axes[0, 0]
c1 = J.c1_scan()
handles = []
for k, rec in enumerate(c1):
    if rec["n"] != 800 or rec["tau2"] not in (0.3, 0.1):
        continue
    xs = [p[0] for p in rec["pts"]]
    ys = [p[1] for p in rec["pts"]]
    es = [p[2] for p in rec["pts"]]
    lbl = f"$m_\\mathrm{{dom}}$={rec['m']}, $\\tau^2$={rec['tau2']}"
    h = ax.errorbar(xs, ys, yerr=es, marker="o", ms=3.0, capsize=2,
                    lw=1.0, color=C[k % len(C)], label=lbl, zorder=3)
    handles.append(h)
# symlog x：σ_ξ²=0 需要单独画在左端
ax.set_xscale("symlog", linthresh=0.25)
ax.set_xticks([0, 0.25, 1, 4, 16, 64])
ax.set_xticklabels(["0", "0.25", "1", "4", "16", "64"])
ax.set_xlim(-0.05, 90)
ax.set_ylim(-0.46, 0.58)
style(ax, "observation noise  $\\sigma_\\xi^2$  (unidentifiability \u2192)",
      "G = MSE(indep) \u2212 MSE(shared)",
      "a  Sharing fails when the latent is observable")
ax.legend(loc="lower right", fontsize=6)
ax.annotate("latent fully\nobservable\n(W1 regime)", xy=(0, -0.10),
            xytext=(0.9, -0.40), fontsize=6, color="#b2182b",
            arrowprops=dict(arrowstyle="->", color="#b2182b", lw=0.7))

# ===========================================================================
# b) G vs n（判决性：非零下界）
# ===========================================================================
ax = axes[0, 1]
c2 = J.c2_limit()
for k, rec in enumerate(c2):
    xs = [p[0] for p in rec["pts"]]
    ys = [p[1] for p in rec["pts"]]
    es = [p[2] for p in rec["pts"]]
    sx = rec["sigma_xi2"]
    ax.errorbar(xs, ys, yerr=es, marker="s", ms=3.0, capsize=2, lw=1.0,
                color=C[k % len(C)],
                label=f"$\\sigma_\\xi^2={sx:g}$"
                      + ("" if sx > 0 else "  (classical)"), zorder=3)
    if rec["pred_inf"] > 0:
        ax.axhline(rec["pred_inf"], color=C[k % len(C)], lw=0.6, ls=":",
                   alpha=0.75, zorder=1)
ax.set_xscale("log")
ax.set_xticks([1000, 10000, 100000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, p: "10" + chr(0x00B3 if v == 1000 else 0x2074 if v == 10000 else 0x2075)))
style(ax, "samples per unit  n", "G",
      "b  Advantage does NOT vanish with data")
ax.legend(loc="center left", bbox_to_anchor=(0.06, 0.30), fontsize=6)
ax.annotate("classical MTL\nnarrative: G\u21920", xy=(100000, 0.00001),
            xytext=(9000, 0.038), fontsize=6, color="#1a9850",
            arrowprops=dict(arrowstyle="->", color="#1a9850", lw=0.7))

# ===========================================================================
# c) G vs m
# ===========================================================================
ax = axes[1, 0]
c3 = J.c3_m()
for k, rec in enumerate(c3):
    xs = [p[0] for p in rec["pts"]]
    ys = [p[1] for p in rec["pts"]]
    es = [p[2] for p in rec["pts"]]
    ax.errorbar(xs, ys, yerr=es, marker="^", ms=3.0, capsize=2, lw=1.0,
                color=C[k % len(C)],
                label=f"$\\sigma_\\xi^2={rec['sigma_xi2']:g}$", zorder=3)
    # 闭式极限参考线
    pred = [A_(rec["sigma_xi2"]) - A_m(rec["sigma_xi2"], m) for m in xs]
    ax.plot(xs, pred, ls=":", lw=0.6, color=C[k % len(C)], alpha=0.8, zorder=1)
ax.set_xscale("log")
ax.set_xticks([1, 2, 3, 6, 12, 24])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, p: str(int(v))))
style(ax, "domain size  m", "G", "c  Larger domain pools more SNR")
ax.legend(loc="lower right", fontsize=6)

# ===========================================================================
# d) G vs τ²
# ===========================================================================
ax = axes[1, 1]
c5 = J.c5_tau()
for k, rec in enumerate(c5):
    xs = [p[0] for p in rec["pts"]]
    ys = [p[1] for p in rec["pts"]]
    es = [p[2] for p in rec["pts"]]
    ax.errorbar(xs, ys, yerr=es, marker="v", ms=3.0, capsize=2, lw=1.0,
                color=C[k % len(C)],
                label=f"$\\sigma_\\xi^2={rec['sigma_xi2']:g}$", zorder=3)
ax.set_xscale("log")
ax.set_xticks([0.01, 0.1, 0.5, 1.0, 2.0])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, p: f"{v:g}"))
style(ax, "heterogeneity  $\\tau^2$", "G",
      "d  Heterogeneity erodes the advantage")
ax.legend(loc="upper left", fontsize=6)

for ext in ("png", "pdf"):
    p = FIG / f"fig_WP1R_phase.{ext}"
    fig.savefig(p, dpi=300, facecolor=BG)
    print(f"→ {p}")

"""P3 审计结果图：Δ_res vs Δ_mech。Nature 风格，300 dpi，代码生成。

图注：
Fig. P3 | 机制分辨力 Δ_res 预测实际机制增益 Δ_mech。
a, 60 个真实环境点的联合分布；阴影带 = |Δ_res|<5% 的无分辨力区（24/60 点，40%）。
   虚线 = y=x（上界）；点全部落在下方或贴近虚线。
b, 按体制分组（raw / shift / noise），可见 noise 体制整体挤向零，ρ 最高（0.950）。
c, 稳健性：剔除 NP@offset1600 两个同源离群点后 ρ 从 0.755 升至 0.785，CI 更窄。
d, 无分辨力环境下机制增益的绝对值分布——在这些点上任何机制的正负结果都不承载信息。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figs"
FIG.mkdir(exist_ok=True)

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.6,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MK = {"raw": "o", "shift": "s", "noise": "^"}
COL = {"raw": "#4C72B0", "shift": "#C44E52", "noise": "#55A868"}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def panel(ax, s, dx=-0.15, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


df = pd.read_csv(RES / "p3_points.csv")
x = df.delta_res.to_numpy()
y = df.delta_mech.to_numpy()

fig = plt.figure(figsize=(7.2, 5.2))
gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.28,
                      left=0.09, right=0.985, top=0.94, bottom=0.10)

# ---- a: 主散点 ----
ax = fig.add_subplot(gs[0, 0])
ax.axvspan(-5, 5, color="#DDDDDD", alpha=0.55, lw=0, zorder=0)
lim = (-6, 34)
ax.plot(lim, lim, ls="--", lw=0.7, color="k", zorder=1)
ax.text(31, 31.5, "y = x (upper bound)", fontsize=5.8, rotation=42,
        ha="right", va="bottom", color="k")
for r in ("raw", "shift", "noise"):
    s = df[df.regime == r]
    ax.scatter(s.delta_res, s.delta_mech, s=22, marker=MK[r], c=COL[r],
               alpha=0.82, edgecolors="white", linewidths=0.4, label=r, zorder=3)
ax.axhline(0, color="k", lw=0.5, zorder=1)
ax.axvline(0, color="k", lw=0.5, zorder=1)
ax.set_xlim(*lim)
ax.set_ylim(-16, 30)
ax.set_xlabel(r"$\Delta_{\mathrm{res}}$ (oracle headroom, %)")
ax.set_ylabel(r"$\Delta_{\mathrm{mech}}$ (best mechanism, %)")
rho = spearman(x, y)
ax.text(0.03, 0.96, f"ρ = {rho:+.3f}\n95% CI [+0.583, +0.864]\nn = 60",
        transform=ax.transAxes, fontsize=6.5, va="top",
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
ax.legend(frameon=False, loc="lower right", handletextpad=0.3, borderpad=0.2)
ax.text(0, 27.5, "unresolvable\n|Δ_res| < 5%", fontsize=5.5, ha="center",
        va="top", color="#555555")
panel(ax, "a")

# ---- b: 按体制 ----
ax = fig.add_subplot(gs[0, 1])
for i, r in enumerate(("raw", "shift", "noise")):
    s = df[df.regime == r]
    ax.scatter(np.full(len(s), i) + np.linspace(-0.16, 0.16, len(s)),
               s.delta_mech, s=16, marker=MK[r], c=COL[r], alpha=0.8,
               edgecolors="white", linewidths=0.35, zorder=3)
    ax.scatter([i], [s.delta_res.mean()], marker="_", s=180, c=COL[r],
               linewidths=1.4, zorder=4)
    ax.scatter([i], [s.delta_mech.mean()], marker="D", s=22, c=COL[r],
               edgecolors="k", linewidths=0.4, zorder=4)
    ax.text(i, 28, f"ρ={spearman(s.delta_res.to_numpy(), s.delta_mech.to_numpy()):+.2f}",
            fontsize=6, ha="center", color=COL[r])
ax.axhline(0, color="k", lw=0.6, ls="--")
ax.set_xticks(range(3))
ax.set_xticklabels(["raw\nΔres=+12.9%", "shift\nΔres=+11.3%", "noise\nΔres=+2.4%"], fontsize=6)
ax.set_ylabel(r"$\Delta_{\mathrm{mech}}$ (%)")
ax.set_ylim(-16, 31)
ax.text(0.03, 0.04, "— mean Δ_res\nD = mean Δ_mech", transform=ax.transAxes,
        fontsize=5.5, va="bottom", color="#444444")
panel(ax, "b")

# ---- c: 稳健性 ----
ax = fig.add_subplot(gs[1, 0])
sub = df[~((df.market == "NP") & (df.offset == 1600))]
ref = [
    ("all points", x, y, "#4C72B0"),
    ("minus NP@off1600", sub.delta_res.to_numpy(), sub.delta_mech.to_numpy(), "#C44E52"),
]
for i, (lab, xv, yv, c) in enumerate(ref):
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(3000):
        idx = rng.integers(0, len(xv), len(xv))
        bs.append(spearman(xv[idx], yv[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    r = spearman(xv, yv)
    ax.errorbar(i, r, yerr=[[r - lo], [hi - r]], fmt="o", ms=6, color=c,
                capsize=2.5, elinewidth=0.9, mec="white", mew=0.7, zorder=3)
    ax.text(i + 0.13, r, f"{r:+.3f}\n[{lo:+.2f},{hi:+.2f}]", fontsize=6,
            va="center", color=c)
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_xticks([0, 1])
ax.set_xticklabels(["all 60 points", "minus 2 same-segment\noutliers"], fontsize=6.5)
ax.set_ylabel("Spearman ρ (bootstrapped 95% CI)")
ax.set_xlim(-0.5, 1.7)
ax.set_ylim(-0.1, 1.05)
panel(ax, "c")

# ---- d: 无分辨力区的机制增益 ----
ax = fig.add_subplot(gs[1, 1])
unres = df[np.abs(df.delta_res) < 5]
resolv = df[np.abs(df.delta_res) >= 5]
parts = ax.violinplot([unres.delta_mech.to_numpy(), resolv.delta_mech.to_numpy()],
                      positions=[0, 1], widths=0.62, showextrema=False)
for pc, c in zip(parts["bodies"], ["#BBBBBB", "#C44E52"]):
    pc.set_facecolor(c); pc.set_alpha(0.75); pc.set_edgecolor("none")
for i, s in enumerate([unres, resolv]):
    ax.scatter(np.full(len(s), i) + np.linspace(-0.13, 0.13, len(s)),
               s.delta_mech, s=13, c="k", alpha=0.55, linewidths=0, zorder=3)
    ax.scatter([i], [s.delta_mech.median()], marker="_", s=170, c="k",
               linewidths=1.5, zorder=4)
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_xticks([0, 1])
ax.set_xticklabels([f"unresolvable\nn={len(unres)}", f"resolvable\nn={len(resolv)}"], fontsize=6.5)
ax.set_ylabel(r"$\Delta_{\mathrm{mech}}$ (%)")
ax.text(0.03, 0.95, f"spread in unresolvable = ±{unres.delta_mech.std():.1f}%",
        transform=ax.transAxes, fontsize=5.8, va="top", color="#444444")
panel(ax, "d")

for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig_P3_audit.{ext}", dpi=300)
print("saved:", FIG / "fig_P3_audit.png")

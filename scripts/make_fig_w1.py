"""W1 kill-test 结果图：Nature 风格，300 dpi，全部由数据生成，可复现。

图注（与正文一致）：
Fig. W1 | 共享胶质增益在同等条件下未显示独立贡献。
a, 工作域测度 WDM（16 点网格上优于不修正基线的比例）四环境 × 五臂，误差棒为种子 bootstrap 95% CI。
b, 逐点相对基线改善的分布（中位数 ± IQR），负值 = 劣于不修正。
c, 机制分辨力 Δ_res：全信息 oracle 相对基线的可达优势。ω=0 时 CI 跨零 ⇒ 环境无分辨力。
d, 参数规模 vs 中位改善：S 参数最少但并不更优，否定"降自由度换工作域"的直觉。
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

ARMS = ["F", "FP", "S", "L", "D"]
COL = {"F": "#4C72B0", "FP": "#8C8C8C", "S": "#C44E52", "L": "#55A868", "D": "#8172B2"}
LBL = {"F": "Free per-synapse (144)", "FP": "Per-syn. pooled (81)",
       "S": "Shared domain (42)", "L": "Grouped baseline (33)", "D": "Shared + diffusion (46)"}
CONDS = [("env1_w2", "env1, \u03c9=2"), ("env3_w2", "env3, \u03c9=2"),
         ("env1_w0", "env1, \u03c9=0"), ("env3_w0", "env3, \u03c9=0")]


def boot_ci(v, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(n)])
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def panel_label(ax, s, dx=-0.16, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


fig = plt.figure(figsize=(7.2, 5.4))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                      left=0.085, right=0.985, top=0.94, bottom=0.10)

# ---------------- a: WDM ----------------
ax = fig.add_subplot(gs[0, 0])
w = 0.15
xs = np.arange(len(CONDS))
for k, arm in enumerate(ARMS):
    means, los, his = [], [], []
    for tag, _ in CONDS:
        df = pd.read_csv(RES / f"w1_{tag}_seeds.csv")
        v = df[f"WDM_{arm}"].to_numpy()
        lo, hi = boot_ci(v)
        means.append(v.mean()); los.append(v.mean() - lo); his.append(hi - v.mean())
    ax.bar(xs + (k - 2) * w, means, w, color=COL[arm], label=LBL[arm],
           yerr=[los, his], capsize=1.6, error_kw={"lw": 0.6}, edgecolor="none")
ax.set_xticks(xs)
ax.set_xticklabels([c[1] for c in CONDS], rotation=18, ha="right")
ax.set_ylabel("Working-domain measure")
ax.set_ylim(0, 0.72)
ax.axhline(0, color="k", lw=0.5)
ax.legend(frameon=False, ncol=1, loc="upper right", handlelength=1.2,
          borderpad=0.2, labelspacing=0.28)
panel_label(ax, "a")

# ---------------- b: 逐点相对改善 ----------------
ax = fig.add_subplot(gs[0, 1])
data, pos, colors = [], [], []
p = 0
for c_i, (tag, _) in enumerate(CONDS):
    df = pd.read_csv(RES / f"w1_{tag}_gridpoints.csv")
    for arm in ARMS:
        v = df[f"{arm}_rel_pct"].to_numpy()
        data.append(v); pos.append(p); colors.append(COL[arm]); p += 1
    p += 1
bp = ax.boxplot(data, positions=pos, widths=0.66, patch_artist=True,
                showfliers=False, medianprops={"lw": 1.0, "color": "k"},
                whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("none")
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_yscale("symlog", linthresh=10)
ax.set_ylabel("Relative improvement vs. no correction (%, symlog)")
ticks = [np.mean(pos[i * 6:i * 6 + 5]) for i in range(4)]
ax.set_xticks(ticks)
ax.set_xticklabels([c[1] for c in CONDS], rotation=18, ha="right")
ax.set_ylim(-3000, 400)
panel_label(ax, "b")

# ---------------- c: Δ_res ----------------
ax = fig.add_subplot(gs[1, 0])
means, los, his = [], [], []
for tag, _ in CONDS:
    df = pd.read_csv(RES / f"w1_{tag}_seeds.csv")
    v = df["dres"].to_numpy()
    lo, hi = boot_ci(v)
    means.append(v.mean()); los.append(v.mean() - lo); his.append(hi - v.mean())
bars = ax.bar(np.arange(4), means, 0.55,
              color=["#C44E52", "#C44E52", "#BBBBBB", "#BBBBBB"],
              yerr=[los, his], capsize=2.2, error_kw={"lw": 0.7}, edgecolor="none")
ax.axhline(0, color="k", lw=0.7)
ax.axhline(5, color="#4C72B0", lw=0.7, ls=":")
ax.text(3.42, 5.6, "5% threshold", fontsize=6, color="#4C72B0", ha="right")
ax.set_xticks(np.arange(4))
ax.set_xticklabels([c[1] for c in CONDS], rotation=18, ha="right")
ax.set_ylabel(r"$\Delta_{\mathrm{res}}$ oracle headroom (%)")
ax.set_ylim(-2, 23)
panel_label(ax, "c")

# ---------------- d: 参数 vs 中位改善 ----------------
ax = fig.add_subplot(gs[1, 1])
for arm in ARMS:
    npar = pd.read_csv(RES / "w1_env1_w2_gridpoints.csv")[f"{arm}_nparam"].iloc[0]
    med = []
    for tag, _ in CONDS:
        df = pd.read_csv(RES / f"w1_{tag}_seeds.csv")
        med.append(df[f"medrel_{arm}"].to_numpy())
    y = np.mean(med)
    yerr = np.std(med) / np.sqrt(len(med))
    ax.errorbar(npar, y, yerr=yerr, fmt="o", ms=4.5, color=COL[arm],
                mec="white", mew=0.6, capsize=1.8, elinewidth=0.7,
                label=arm, zorder=3)
    ax.annotate(arm, (npar, y), textcoords="offset points", xytext=(6, 4),
                fontsize=6.5, color=COL[arm], fontweight="bold")
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_xlabel("Number of parameters")
ax.set_ylabel("Median relative improvement (%)")
ax.set_xlim(20, 165)
ax.set_ylim(-40, 6)
panel_label(ax, "d")

for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig_W1_killtest.{ext}", dpi=300)
print("saved:", FIG / "fig_W1_killtest.png", FIG / "fig_W1_killtest.pdf")

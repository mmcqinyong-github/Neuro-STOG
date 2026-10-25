"""W-P3-2 结果图：跨 7 个公开 TSF 基准的 Δ_res 分辨力排行 + 文献反查。

Nature 风格，300 dpi，代码生成（禁 AI 生成）。
四面板：
  a  105 点散点：Δ_res vs Δ_mech，灰带 |Δ_res|<5%，虚线 y=x
  b  7 个数据集的 Δ_res 排行（中位 + 逐 offset 点），横虚线 = 5% 阈
  c  无分辨力区 vs 有分辨力区的 Δ_mech 分布（violin + 个体点）
  d  文献反查：15 篇论文 × 7 基准的命中矩阵
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

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

THR = 5.0
MK = {"raw": "o", "shift": "s", "noise": "^"}
COL = {"raw": "#4C72B0", "shift": "#C44E52", "noise": "#55A868"}
DS_ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]

# 文献池（与 run_p32_lit_audit.py 保持一致）
LIT = [
    ("MWENet 2025", ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather"]),
    ("UniWave TII 2026", DS_ORDER),
    ("STAIR 2026", DS_ORDER),
    ("RAFT 2025", DS_ORDER),
    ("TFEM 2025", DS_ORDER),
    ("DEEP TPC 2026", DS_ORDER),
    ("PhaseFormer ICLR26", DS_ORDER),
    ("MixLinear ICLR26", ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic"]),
    ("TimeMixer++ 2025", DS_ORDER),
    ("BEAT 2025", DS_ORDER),
    ("FiCoTS 2025", DS_ORDER),
    ("TALON/CALF 2025", DS_ORDER),
    ("Waveformer TNNLS25", ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather"]),
    ("iBACon TKDE25", ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic"]),
    ("STNet 2025", ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather"]),
]


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def panel(ax, s, dx=-0.16, dy=1.07):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


def main():
    df = pd.read_csv(RES / "p32_points.csv")
    x = df["delta_res"].to_numpy()
    y = df["delta_mech"].to_numpy()

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                          left=0.09, right=0.985, top=0.93, bottom=0.075)

    # ---------------- a: 散点 ----------------
    ax = fig.add_subplot(gs[0, 0])
    ax.axvspan(-THR, THR, color="0.90", zorder=0, lw=0)
    lim = [min(x.min(), y.min()) - 6, max(x.max(), y.max()) + 6]
    ax.plot(lim, lim, ls="--", lw=0.7, color="0.45", zorder=1)
    for reg in ("raw", "shift", "noise"):
        m = df["regime"] == reg
        ax.scatter(x[m], y[m], s=9, marker=MK[reg], c=COL[reg],
                   alpha=0.75, linewidths=0, label=reg, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.axhline(0, color="0.35", lw=0.5, zorder=2)
    ax.set_xlabel(r"$\Delta_{\mathrm{res}}$ (%)  oracle headroom")
    ax.set_ylabel(r"$\Delta_{\mathrm{mech}}$ (%)  best arm")
    rho = spearman(x, y)
    ax.text(0.035, 0.955, f"$\\rho$ = {rho:+.3f}   n = {len(x)}",
            transform=ax.transAxes, fontsize=7, va="top")
    ax.text(0.035, 0.875, "grey: unresolvable\n(|$\\Delta_{res}$| < 5%)",
            transform=ax.transAxes, fontsize=6, va="top", color="0.35")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3,
              borderpad=0.2, markerscale=1.3)
    ax.spines[["top", "right"]].set_visible(False)
    panel(ax, "a")

    # ---------------- b: 数据集分辨力排行 ----------------
    ax = fig.add_subplot(gs[0, 1])
    meds = []
    for ds in DS_ORDER:
        v = df.loc[(df.dataset == ds) & (df.regime == "raw"), "delta_res"].to_numpy()
        meds.append((ds, np.median(v), v))
    meds.sort(key=lambda t: t[1])
    ypos = np.arange(len(meds))
    for i, (ds, m, v) in enumerate(meds):
        jitter = np.linspace(-0.13, 0.13, len(v))
        ax.scatter(v, i + jitter, s=8, c="0.55", linewidths=0, zorder=2)
        c = "#C44E52" if abs(m) < THR else "#4C72B0"
        ax.scatter([m], [i], s=26, marker="D", c=c, zorder=3, linewidths=0)
    ax.axvline(THR, ls="--", lw=0.7, color="#C44E52")
    ax.axvline(-THR, ls="--", lw=0.7, color="#C44E52")
    ax.axvline(0, lw=0.5, color="0.35")
    ax.set_yticks(ypos)
    ax.set_yticklabels([t[0] for t in meds], fontsize=6.5)
    ax.set_xlabel(r"$\Delta_{\mathrm{res}}$ (%, raw regime)")
    ax.set_xlim(-8, 30)
    n_un = sum(1 for t in meds if abs(t[1]) < THR)
    ax.text(0.98, 0.06, f"{n_un}/{len(meds)} datasets\nunresolvable",
            transform=ax.transAxes, fontsize=6.2, ha="right", va="bottom",
            color="#C44E52")
    ax.spines[["top", "right"]].set_visible(False)
    panel(ax, "b")

    # ---------------- c: 两区 Δ_mech 分布 ----------------
    ax = fig.add_subplot(gs[1, 0])
    unres = np.abs(x) < THR
    groups = [y[unres], y[~unres]]
    labels = [f"unresolvable\nn={unres.sum()}", f"resolvable\nn={(~unres).sum()}"]
    parts = ax.violinplot(groups, positions=[0, 1], widths=0.62,
                          showextrema=False, showmedians=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor("#C44E52" if i == 0 else "#4C72B0")
        b.set_alpha(0.28)
        b.set_edgecolor("none")
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        ax.scatter(i + rng.normal(0, 0.055, len(g)), g, s=7,
                   c="#C44E52" if i == 0 else "#4C72B0", alpha=0.6,
                   linewidths=0, zorder=3)
        ax.plot([i - 0.24, i + 0.24], [np.median(g)] * 2, lw=1.4,
                color="#222222", zorder=4)
        ax.text(i, np.median(g) + 3.5, f"{np.median(g):+.1f}%",
                ha="center", fontsize=6.2, zorder=5)
    ax.axhline(0, lw=0.5, color="0.35")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel(r"$\Delta_{\mathrm{mech}}$ (%)")
    ax.set_ylim(-40, 45)
    neg_u = (y[unres] < 0).mean()
    neg_r = (y[~unres] < 0).mean()
    ax.text(0.03, 0.04, f"neg. rate: {neg_u:.0%} vs {neg_r:.0%}",
            transform=ax.transAxes, fontsize=6.2, color="0.3")
    ax.spines[["top", "right"]].set_visible(False)
    panel(ax, "c")

    # ---------------- d: 文献命中矩阵 ----------------
    ax = fig.add_subplot(gs[1, 1])
    dr = {ds: np.median(df.loc[(df.dataset == ds) & (df.regime == "raw"),
                               "delta_res"].to_numpy()) for ds in DS_ORDER}
    M = np.full((len(LIT), len(DS_ORDER)), np.nan)
    for i, (_, dss) in enumerate(LIT):
        for j, ds in enumerate(DS_ORDER):
            if ds in dss:
                M[i, j] = 1.0 if abs(dr[ds]) < THR else 0.0
    cmap = ListedColormap(["#4C72B0", "#C44E52"])
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, aspect="auto",
              vmin=0, vmax=1)
    for i in range(len(LIT)):
        for j in range(len(DS_ORDER)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, "\u25cf", ha="center", va="center",
                        fontsize=4.2, color="white")
    ax.set_xticks(range(len(DS_ORDER)))
    ax.set_xticklabels(DS_ORDER, rotation=55, ha="right", fontsize=5.8)
    ax.set_yticks(range(len(LIT)))
    ax.set_yticklabels([n for n, _ in LIT], fontsize=5.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    n_tot = int(np.sum(~np.isnan(M)))
    n_un2 = int(np.nansum(M))
    ax.set_title(f"red = unresolvable: {n_un2}/{n_tot} = {n_un2 / n_tot:.0%}",
                 fontsize=6.5, pad=3)
    panel(ax, "d")

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_P32_bench.{ext}")
    print(f"saved {FIG / 'fig_P32_bench.png'}")


if __name__ == "__main__":
    main()

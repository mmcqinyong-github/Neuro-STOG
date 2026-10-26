"""W-P1-V 结果图：外部数据集验证 + FP 结构诊断。

Nature 风格，300 dpi，代码生成（禁 AI 生成）。
四面板：
  a  外部 3 数据集 45 点：Δ_res vs Δ_mech（与 p32 的 105 点并列，灰带 |Δ_res|<5%）
  b  10 个数据集的 median Δ_res（7 内部 + 3 外部，外部用不同标记）
  c  FP 结构：|Δ_res| 分箱的 FP 数量与占全部 FP 比例（双轴柱 + 折线）
  d  阈值-精度曲线：样本内（实线）vs 折外中位（点），标出 S3 覆盖率塌陷
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

THR = 5.0
MK = {"raw": "o", "shift": "s", "noise": "^"}
COL = {"raw": "#4C72B0", "shift": "#C44E52", "noise": "#55A868"}
EXT = ["solar", "PEMS04", "PEMS08"]
INNER7 = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]


def spearman(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    p32 = pd.read_csv(RES / "p32_points_honest.csv")
    for c in ("offset",):
        p32[c] = p32[c].astype(int)
    ext = pd.read_csv(RES / "p1v_points.csv")
    ext["offset"] = ext["offset"].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    ax = axes[0, 0]

    # ---------------- a ----------------
    ax.axhline(0, color="0.55", lw=0.5, zorder=1)
    ax.axvline(0, color="0.55", lw=0.5, zorder=1)
    ax.axvspan(-THR, THR, color="0.90", zorder=0)
    for reg in ("raw", "shift", "noise"):
        s = ext[ext.regime == reg]
        ax.scatter(s.delta_res, s.delta_mech, s=16, marker=MK[reg],
                   facecolor=COL[reg], edgecolor="white", linewidth=0.35,
                   alpha=0.95, label=f"external / {reg}", zorder=3)
    s = p32
    ax.scatter(s.delta_res, s.delta_mech, s=11, marker="o",
               facecolor="none", edgecolor="0.60", linewidth=0.4,
               alpha=0.7, label="internal 7 (n=105)", zorder=2)
    rho_e = spearman(ext.delta_res, ext.delta_mech)
    ax.set_xlabel(r"$\Delta_{\mathrm{res}}$ (%)")
    ax.set_ylabel(r"$\Delta_{\mathrm{mech}}$ (%)")
    ax.set_xscale("symlog", linthresh=10)
    ax.text(0.03, 0.96, f"$\\rho_{{ext}}={rho_e:+.3f}$\n$n_{{{len(ext)}}}$",
            transform=ax.transAxes, va="top", fontsize=6.5)
    ax.text(-THR + 0.6, ax.get_ylim()[1] * 0.72, "no resolution",
            fontsize=5.8, color="0.45", rotation=90, va="top")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3,
              labelspacing=0.25, ncol=1)
    ax.set_title("a  External-data forward test", loc="left", fontweight="bold")

    # ---------------- b ----------------
    ax = axes[0, 1]
    med = {}
    for name in INNER7:
        sub = p32[p32.dataset == name]
        if len(sub):
            med[name] = np.median(sub.delta_res)
    for name in EXT:
        sub = ext[ext.dataset == name]
        if len(sub):
            med[name] = np.median(sub.delta_res)
    order = [k for k in INNER7 + EXT if k in med]
    vals = [med[k] for k in order]
    colors = ["#B0B0B0" if k in INNER7 else "#C44E52" for k in order]
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, vals, height=0.62, color=colors, edgecolor="white", lw=0.4)
    ax.axvline(0, color="0.35", lw=0.5)
    ax.axvline(THR, color="#C44E52", lw=0.6, ls="--")
    ax.axvline(-THR, color="#C44E52", lw=0.6, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(order, fontsize=6)
    ax.set_xlabel(r"median $\Delta_{\mathrm{res}}$ (%)")
    ax.text(THR + 0.5, len(order) - 0.6, "5% threshold", fontsize=5.8,
            color="#C44E52", rotation=90, va="top")
    ax.set_title("b  Resolution across 10 datasets", loc="left", fontweight="bold")

    # ---------------- c：FP 结构 ----------------
    ax = axes[1, 0]
    hon = pd.read_csv(RES / "p32_points_honest.csv")
    hon["offset"] = hon["offset"].astype(int)
    loo = pd.read_csv(RES / "loo_screen.csv")
    loo["offset"] = loo["offset"].astype(int)
    key = hon.set_index(["dataset", "regime", "offset"])["delta_res"]
    loo["ares"] = [abs(key.loc[(r.dataset, r.regime, r.offset)])
                   for r in loo.itertuples()]
    loo["fp"] = (loo.actual <= 0) & (loo.pred > 0)
    loo["neg"] = loo.actual <= 0

    bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
    labs = ["[0,2)", "[2,5)", "[5,10)", "[10,20)", "≥20"]
    fpc, tot = [], []
    for lo_, hi_ in bins:
        s = loo[(loo.ares >= lo_) & (loo.ares < hi_)]
        fpc.append(int(s.fp.sum()))
        tot.append(len(s))
    fpc = np.array(fpc)
    xs = np.arange(len(bins))
    ax.bar(xs, fpc, width=0.55, color="#C44E52", edgecolor="white", lw=0.4,
           label="false positives")
    ax2 = ax.twinx()
    ax2.plot(xs, fpc / max(fpc.sum(), 1) * 100, marker="o", ms=3.2,
             color="#4C72B0", lw=1.0, label="share of all FP")
    ax2.set_ylabel("share of all FP (%)", color="#4C72B0", fontsize=6.5)
    ax2.tick_params(axis="y", colors="#4C72B0", labelsize=6)
    ax2.set_ylim(0, 105)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set_xlabel(r"$|\Delta_{\mathrm{res}}|$ bin (%)")
    ax.set_ylabel("false positives (count)")
    ax.text(0, fpc[0] * 0.52, f"{fpc[0]/max(fpc.sum(),1)*100:.1f}%",
            ha="center", fontsize=6.2, color="white", fontweight="bold")
    ax.set_title("c  Where the false positives live", loc="left", fontweight="bold")

    # ---------------- d：阈值-精度 ----------------
    ax = axes[1, 1]
    taus = np.array([0, 1, 2, 3, 5, 8, 10, 15])
    fpr, rec, cov = [], [], []
    for t in taus:
        s = loo[loo.ares >= t]
        ng = s.neg.sum()
        fpr.append(s.fp.sum() / max(ng, 1) * 100)
        tp = ((s.actual > 0) & (s.pred > 0)).sum()
        fn = ((s.actual > 0) & (s.pred <= 0)).sum()
        rec.append(tp / max(tp + fn, 1) * 100)
        cov.append(len(s) / len(loo) * 100)
    ax.plot(taus, fpr, marker="o", ms=3.0, lw=1.0, color="#C44E52",
            label="FP rate (in-sample)")
    ax.plot(taus, rec, marker="s", ms=3.0, lw=1.0, color="#4C72B0",
            label="recall")
    ax.plot(taus, cov, marker="^", ms=3.0, lw=1.0, color="#55A868",
            label="coverage")
    ax.axhline(21.8, color="0.45", lw=0.6, ls=":")
    ax.text(14.8, 24, "S1 gate 21.8%", fontsize=5.6, color="0.45", ha="right")
    ax.axvline(15, color="0.55", lw=0.6, ls="--")
    ax.text(14.4, 60, r"$\tau^{*}=15\%$", fontsize=6, rotation=90,
            va="center", ha="right", color="0.35")
    ax.set_xlabel(r"threshold $\tau$ (%): keep $|\Delta_{\mathrm{res}}|\geq\tau$")
    ax.set_ylabel("percent (%)")
    ax.set_ylim(-4, 108)
    ax.legend(loc="center right", frameon=False, handletextpad=0.3,
              labelspacing=0.25)
    ax.set_title("d  Threshold trade-off (coverage collapses)", loc="left",
                 fontweight="bold")

    for a in axes.ravel():
        a.grid(axis="y", color="0.93", lw=0.4, zorder=0)
        a.set_axisbelow(True)

    fig.tight_layout(pad=0.7, w_pad=1.0, h_pad=1.0)
    for ext_ in ("png", "pdf"):
        fig.savefig(FIG / f"fig_p1v_external.{ext_}", dpi=300)
    print(f"已写 {FIG / 'fig_p1v_external.png'} / .pdf")


if __name__ == "__main__":
    main()

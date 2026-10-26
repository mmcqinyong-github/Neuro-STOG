"""W-P1-W 结果图：真实架构验证失败 + 根因分解。

Nature 风格，300 dpi，代码生成（禁 AI 生成）。
四面板：
  a  48 单元：Δ_res vs Δ_mech_real，按架构着色，标 ρ=+0.1775
  b  逐架构 ρ 与 Δ_mech 均值（双轴），显示四架构符号不一致
  c  参数量不匹配 → Δ_mech 的关系（log2 参数比 vs Δ_mech 均值）
  d  W2 去偏：各阈值的精确率 vs 常数基线（阴影=随机 95% 带）
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
    "legend.fontsize": 6.2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ARCHS = ["DLinear", "iTransformer", "TimeMixer", "PatchTST"]
COL = {"DLinear": "#4C72B0", "iTransformer": "#DD8452",
       "TimeMixer": "#55A868", "PatchTST": "#C44E52"}
MK = {"raw": "o", "shift": "s", "noise": "^"}
REGIMES = ("raw", "shift", "noise")
THR = 5.0


def spearman(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def load():
    runs = pd.read_csv(RES / "p1w_runs.csv")
    dres = pd.read_csv(RES / "p32_points_honest.csv")
    dmap = {(r.dataset, r.regime): float(r.delta_res)
            for r in dres[dres.offset == 0].itertuples()}
    cell = {}
    for r in runs.itertuples():
        k = (r.arch, r.dataset, r.regime)
        cell.setdefault(k, {"full": [], "indep": []})[r.arm].append(float(r.mse))
    rows = []
    for (a, ds, reg), v in cell.items():
        if not v["full"] or not v["indep"]:
            continue
        mf, mi = np.median(v["full"]), np.median(v["indep"])
        rows.append(dict(arch=a, dataset=ds, regime=reg,
                         dm=(mi - mf) / mi * 100.0, dr=dmap[(ds, reg)]))
    return pd.DataFrame(rows), runs


def main():
    df, runs = load()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))

    # ---------------- a ----------------
    ax = axes[0, 0]
    ax.axhline(0, color="0.55", lw=0.5)
    ax.axvline(0, color="0.55", lw=0.5)
    ax.axvspan(-THR, THR, color="0.90", zorder=0)
    for a in ARCHS:
        for reg in REGIMES:
            s = df[(df.arch == a) & (df.regime == reg)]
            if s.empty:
                continue
            ax.scatter(s.dr, s.dm, s=18, marker=MK[reg], facecolor=COL[a],
                       edgecolor="white", linewidth=0.35, alpha=0.92, zorder=3)
    rho = spearman(df.dr, df.dm)
    ax.set_xlabel(r"$\Delta_{\mathrm{res}}$ (%)")
    ax.set_ylabel(r"$\Delta_{\mathrm{mech}}^{\mathrm{real}}$ (%)")
    ax.set_xscale("symlog", linthresh=10)
    ax.text(0.03, 0.96, f"$\\rho={rho:+.3f}$\n$n={len(df)}$", transform=ax.transAxes,
            va="top", fontsize=6.5)
    ax.text(-THR + 0.6, 30, "no resolution", fontsize=5.6, color="0.45",
            rotation=90, va="top")
    # 自绘图例
    import matplotlib.lines as mlines
    hs = [mlines.Line2D([], [], marker="o", ls="", ms=3.5,
                        markerfacecolor=COL[a], markeredgecolor="white",
                        label=a) for a in ARCHS]
    hs += [mlines.Line2D([], [], marker=MK[r], ls="", ms=3.5,
                         markerfacecolor="0.55", markeredgecolor="white",
                         label=r) for r in REGIMES]
    ax.legend(handles=hs, loc="lower left", frameon=False, ncol=2,
              handletextpad=0.25, labelspacing=0.2, columnspacing=0.8)
    ax.set_title("a  Real architectures: relation does not transfer",
                 loc="left", fontweight="bold")

    # ---------------- b ----------------
    ax = axes[0, 1]
    rhos, means = [], []
    for a in ARCHS:
        s = df[df.arch == a]
        rhos.append(spearman(s.dr, s.dm))
        means.append(float(np.mean(s.dm)))
    xs = np.arange(len(ARCHS))
    ax.bar(xs, rhos, width=0.5, color="#B0B0B0", edgecolor="white", lw=0.4,
           label=r"$\rho$ (within arch.)")
    ax.axhline(0.40, color="#4C72B0", lw=0.7, ls="--")
    ax.text(3.42, 0.43, "W1 gate 0.40", fontsize=5.6, color="#4C72B0", ha="right")
    ax.axhline(0, color="0.4", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(ARCHS, rotation=18, ha="right", fontsize=6)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(-0.05, 0.75)
    ax2 = ax.twinx()
    ax2.plot(xs, means, marker="D", ms=4, color="#C44E52", lw=1.1,
             label="mean " + r"$\Delta_{\mathrm{mech}}$")
    ax2.axhline(0, color="#C44E52", lw=0.4, ls=":")
    ax2.set_ylabel(r"mean $\Delta_{\mathrm{mech}}^{\mathrm{real}}$ (%)",
                   color="#C44E52", fontsize=6.5)
    ax2.tick_params(axis="y", colors="#C44E52", labelsize=6)
    for i, m in enumerate(means):
        ax2.annotate(f"{m:+.1f}", (i, m), textcoords="offset points",
                     xytext=(0, 7 if m < 0 else -11), ha="center",
                     fontsize=5.6, color="#C44E52")
    ax.set_title("b  Architecture-specific sign reversal", loc="left",
                 fontweight="bold")

    # ---------------- c ----------------
    ax = axes[1, 0]
    np_by = {}
    for r in runs.itertuples():
        np_by[(r.arch, r.arm)] = int(r.n_param)
    lgs, mns = [], []
    for a in ARCHS:
        pf, pi = np_by[(a, "full")], np_by[(a, "indep")]
        lgs.append(float(np.log2(pi / pf)))
        mns.append(float(np.mean(df[df.arch == a].dm)))
    ax.axhline(0, color="0.5", lw=0.5)
    ax.axvline(0, color="#4C72B0", lw=0.7, ls="--")
    ax.text(0.06, 6, "parameter-matched", fontsize=5.8, color="#4C72B0",
            rotation=90, va="bottom")
    for a in ARCHS:
        i = ARCHS.index(a)
        ax.scatter(lgs[i], mns[i], s=42, facecolor=COL[a], edgecolor="white",
                   linewidth=0.5, zorder=3)
        ax.annotate(a, (lgs[i], mns[i]), textcoords="offset points",
                    xytext=(0, 9) if mns[i] > 0 else (0, -12),
                    ha="center", fontsize=6, color=COL[a])
    if len(lgs) >= 2:
        z = np.polyfit(lgs, mns, 1)
        xx = np.linspace(min(lgs) - 0.3, max(lgs) + 0.3, 20)
        ax.plot(xx, np.polyval(z, xx), color="0.55", lw=0.7, ls=":")
        rr = np.corrcoef(lgs, mns)[0, 1]
        ax.text(0.03, 0.06, f"r={rr:+.3f}", transform=ax.transAxes, fontsize=6.2)
    ax.set_xlabel(r"$\log_2(P_{\mathrm{indep}}/P_{\mathrm{full}})$")
    ax.set_ylabel(r"mean $\Delta_{\mathrm{mech}}^{\mathrm{real}}$ (%)")
    ax.set_title("c  Parameter mismatch drives the sign", loc="left",
                 fontweight="bold")

    # ---------------- d ----------------
    ax = axes[1, 1]
    base = float((df.dm <= 0).mean())
    thrs = [1, 2, 3, 5, 8, 10]
    prec, lo95, hi95, ns = [], [], [], []
    rng = np.random.default_rng(0)
    for t in thrs:
        f = df[df.dr < t]
        if f.empty:
            prec.append(np.nan); lo95.append(np.nan); hi95.append(np.nan)
            ns.append(0); continue
        p = float((f.dm <= 0).mean())
        prec.append(p); ns.append(len(f))
        vals = np.array([float((df.dm.sample(len(f), replace=False,
                                             random_state=int(rng.integers(1e9))) <= 0).mean())
                         for _ in range(4000)])
        lo95.append(float(np.percentile(vals, 2.5)))
        hi95.append(float(np.percentile(vals, 97.5)))
    # 顺序必须与 thrs 对齐（prec 已按 thrs 顺序构建）
    ax.axhspan(min(lo95) * 100, max(hi95) * 100, color="0.92", zorder=0,
               label="random-subset 95% band")
    ax.axhline(base * 100, color="0.45", lw=0.8, ls="--",
               label=f"constant predictor ({base*100:.1f}%)")
    ax.plot(thrs, np.array(prec) * 100, marker="o", ms=3.2, lw=1.1,
            color="#C44E52", label="W2 precision")
    for t, p, n in zip(thrs, prec, ns):
        ax.annotate(f"n={n}", (t, p * 100), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=5.4, color="0.4")
    ax.set_xlabel(r"threshold $\tau$ (%): flag if $|\Delta_{\mathrm{res}}|<\tau$")
    ax.set_ylabel("precision (%)")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.35,
              labelspacing=0.22)
    ax.set_title("d  W2 is indistinguishable from guessing", loc="left",
                 fontweight="bold")

    for a in axes.ravel():
        a.grid(axis="y", color="0.93", lw=0.4, zorder=0)
        a.set_axisbelow(True)

    fig.tight_layout(pad=0.7, w_pad=1.0, h_pad=1.1)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_p1w_realarch.{ext}", dpi=300)
    print(f"已写 {FIG / 'fig_p1w_realarch.png'} / .pdf")


if __name__ == "__main__":
    main()

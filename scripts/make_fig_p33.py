"""W-P3-3 层 2 图：Nature 风格四面板。

a  Δ_res 与 Δ_mech 的散点（含 |Δ_res|<5% 无分辨力灰带）
b  逐数据集 Δ_mech 的种子分布（箱线 + 散点，叠论文 Table 8 的对照）
c  H-L1 判据：高低分辨力环境下的机制增益稳健性
d  复现质量：官方代码实测 vs 论文报告值
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Paper-Neuro-STOG")
RES = ROOT / "results"
FIG = ROOT / "figs"

plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.6,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PAPER_TABLE8 = {
    "ETTh1": dict(raft=0.367, without_retrieval=0.379),
    "ETTh2": dict(raft=0.276, without_retrieval=0.282),
    "ETTm1": dict(raft=0.302, without_retrieval=0.306),
    "ETTm2": dict(raft=0.164, without_retrieval=0.167),
    "weather": dict(raft=0.165, without_retrieval=0.182),
    "exchange_rate": dict(raft=0.091, without_retrieval=0.089),
}
INK = "#1a1a1a"
BLUE = "#3B6FB6"
RED = "#C0392B"
GREY = "#9AA0A6"
GREEN = "#2E8B57"


def load_dmech():
    p = RES / "p33_dmech.csv"
    if not p.exists():
        return []
    return list(csv.DictReader(p.open(encoding="utf-8")))


def load_raw():
    out = []
    for f in sorted((RES / "p33_raw").glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


def main() -> None:
    rows = load_dmech()
    raw = load_raw()
    if not rows:
        print("无可用数据")
        return

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                          left=0.09, right=0.985, top=0.93, bottom=0.09)

    # ---------------- a: Δ_res vs Δ_mech ----------------
    ax = fig.add_subplot(gs[0, 0])
    sel = [r for r in rows if r.get("dmech_B") not in ("", None)
           and r.get("delta_res") not in ("", None)]
    xs = [float(r["delta_res"]) for r in sel]
    ys = [float(r["dmech_B"]) for r in sel]
    lbl = [r["dataset"] for r in sel]
    # exchange_rate 的 Δ_res = −302% 属极端值，会压扁坐标轴：
    # 主轴用对称对数刻度，保留全部点，不删样本。
    ax.axhspan(-1, 1, color=GREY, alpha=0.14, lw=0,
               label=r"$|\Delta_{\rm mech}| < 1\%$")
    ax.axhline(0, color=INK, lw=0.6, zorder=1)
    ax.axvline(0, color=INK, lw=0.6, zorder=1)
    sc = ax.scatter(xs, ys, s=30, c=BLUE, edgecolors="white", linewidths=0.6, zorder=3)
    # 手工错开标注，避免 exchange_rate / ETTh2 / ETTm2 / weather 标签重叠
    offsets = {"exchange_rate": (8, -11), "ETTh2": (8, 3), "ETTm2": (4, 7),
               "weather": (4, -12), "ETTh1": (-20, 6), "ETTm1": (7, 1)}
    for x, y, t in zip(xs, ys, lbl):
        dx, dy = offsets.get(t, (4, 3))
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=5.5, color=INK)
    ax.set_xscale("symlog", linthresh=20)
    ax.set_xlabel(r"$\Delta_{\rm res}$ in RAFT environment (%, symlog)")
    ax.set_ylabel(r"$\Delta_{\rm mech}$ from official code (%)")
    ax.set_title("a  Signed resolution predicts mechanism gain", loc="left", fontsize=7.5)
    ax.legend(loc="lower left", frameon=False)

    # ---------------- b: per-dataset Δ_mech seed distribution ----------------
    ax = fig.add_subplot(gs[0, 1])
    dss = [r["dataset"] for r in rows if r.get("dmech_B") not in ("", None)]
    data, paper_gain = [], []
    for ds in dss:
        vals = [r.get("dmech_B_seedvals", "") for r in rows if r["dataset"] == ds]
        v = [float(x) for x in vals[0].split(";") if x] if vals else []
        data.append(v if v else [np.nan])
        if ds in PAPER_TABLE8:
            p = PAPER_TABLE8[ds]
            paper_gain.append((p["without_retrieval"] - p["raft"]) / p["without_retrieval"] * 100)
        else:
            paper_gain.append(np.nan)
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.0),
                    boxprops=dict(facecolor="#DCE6F2", edgecolor=BLUE, lw=0.7),
                    whiskerprops=dict(color=BLUE, lw=0.7),
                    capprops=dict(color=BLUE, lw=0.7))
    for i, v in enumerate(data):
        ax.scatter(np.full(len(v), i + 1) + np.random.default_rng(i).normal(0, 0.05, len(v)),
                   v, s=9, c=BLUE, alpha=0.65, zorder=3, edgecolors="none")
    ax.scatter(np.arange(1, len(dss) + 1), paper_gain, marker="D", s=16,
               facecolors="none", edgecolors=RED, lw=0.9, zorder=4,
               label="paper Table 8 (reported)")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(1, len(dss) + 1))
    ax.set_xticklabels(dss, rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta_{\rm mech}$ (retrieval on vs off) (%)")
    ax.set_title("b  Official code vs reported gain", loc="left", fontsize=7.5)
    ax.legend(loc="upper left", frameon=False)
    # weather 的论文值 +9.3% 把其余数据集挤到底部 ⇒ 断轴显示
    ymin, ymax = -3.4, 10.2
    split = 4.2
    ax.axhspan(split, ymax, color="white", zorder=0)
    ax.axhline(split, color=INK, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.set_ylim(ymin, ymax)
    # 论文值：折断点以上的用压缩映射画，避免与实测箱线混淆
    for i, v in enumerate(paper_gain):
        if np.isnan(v):
            continue
        if v > split:
            yv = split + (v - split) * 0.42
            ax.scatter([i + 1], [yv], marker="D", s=16, facecolors="none",
                       edgecolors=RED, lw=0.9, zorder=4)
            ax.annotate(f"{v:.1f}%", (i + 1, yv), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=5.5, color=RED)
    # 断轴标记（斜杠）
    for xpos in (0.0, 1.0):
        ax.plot([xpos, xpos], [split - 0.010, split + 0.010], color=INK, lw=0.6,
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)
    ax.text(1.008, split, "break", transform=ax.get_yaxis_transform(),
            va="center", fontsize=5, color=INK, rotation=90)
    # ---------------- c: H-L1 robustness ----------------
    ax = fig.add_subplot(gs[1, 0])
    hi = [r for r in rows if r.get("res_zone") == "高" and r.get("dmech_B") not in ("", None)]
    lo = [r for r in rows if r.get("res_zone") == "低" and r.get("dmech_B") not in ("", None)]
    groups = [("High $|\\Delta_{res}|$ half\n(upper by median)", hi, BLUE),
              ("Low $|\\Delta_{res}|$ half\n(lower by median)", lo, RED)]
    for gi, (name, g, c) in enumerate(groups):
        for i, r in enumerate(g):
            sv = [float(x) for x in r.get("dmech_B_seedvals", "").split(";") if x]
            if not sv:
                continue
            x = gi * 1.0 + (i - (len(g) - 1) / 2) * 0.12
            ax.scatter([x] * len(sv), sv, s=14, c=c, alpha=0.75,
                       edgecolors="none", zorder=3)
            ax.plot([x, x], [min(sv), max(sv)], color=c, lw=0.6, alpha=0.5, zorder=2)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(r"$\Delta_{\rm mech}$ per seed (%)")
    ax.set_title("c  H-L1': gain robustness by resolution", loc="left", fontsize=7.5)

    # ---------------- d: reproduction fidelity ----------------
    ax = fig.add_subplot(gs[1, 1])
    ds_ok = [ds for ds in PAPER_TABLE8 if any(r["dataset"] == ds for r in rows)]
    if ds_ok:
        meas, rep, labels = [], [], []
        for ds in ds_ok:
            mm = [x["mse"] for x in raw
                  if x["dataset"] == ds and x["arm"] == "on"]
            if not mm:
                continue
            meas.append(float(np.median(mm)))
            rep.append(PAPER_TABLE8[ds]["raft"])
            labels.append(ds)
        idx = np.arange(len(labels))
        w = 0.36
        ax.bar(idx - w / 2, rep, w, color=GREY, edgecolor=INK, lw=0.4,
               label="paper reported")
        ax.bar(idx + w / 2, meas, w, color=BLUE, edgecolor=INK, lw=0.4,
               label="measured (official code)")
        for i, (a_, b_) in enumerate(zip(rep, meas)):
            ax.text(i, max(a_, b_) * 1.02, f"{(b_ - a_) / a_ * 100:+.1f}%",
                    ha="center", fontsize=5.5, color=INK)
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("test MSE (96-step, multivariate)")
        ax.set_title("d  H-L4: reproduction fidelity", loc="left", fontsize=7.5)
        ax.legend(loc="upper left", frameon=False)

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_P33_layer2.png", facecolor="white")
    fig.savefig(FIG / "fig_P33_layer2.pdf", facecolor="white")
    print(f"写入 {FIG / 'fig_P33_layer2.png'}")


if __name__ == "__main__":
    main()

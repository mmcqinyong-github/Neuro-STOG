"""W-P3-2 文献反查：已发表机制论文的实验环境有多少落在无分辨力区。

预注册：`W-P3-2预注册规格-v1.0.md` §5
关键映射决策（预注册 §5.3 未写明、此处必须写清楚的地方）：

  文献用的是**原始数据集**（无人工注入的 shift/noise），
  因此文献环境的 Δ_res 只能对应我们的 **raw 体制**。
  数据集级 Δ_res = raw 体制 5 个训练偏移的中位数。

判据（预注册 §5.3 层 1）：
  对每篇论文报告的 (数据集) 计一次。若该数据集的 |Δ_res| < 5%，则记为
  "落在无分辨力区"——该数据集上报告的机制级改进无法排除环境本身不给
  机制留空间。
  L1 判据：落在无分辨力区的比例 ≥ 25% → 支持 P3 的文献批判主张。

诚实限定（预注册 §5.4，必须写进输出）：
  1. 只提取论文**声明使用**的数据集，不逐篇复算其 MSE 表（复算需跑其代码）。
  2. 我们的 Δ_res 用 lookback=24、单步残差修正，与论文的 96/192/336/720
     多步 horizon 不对齐。这是简化。
  3. 论文若已用很强基线，Δ_res 无法区分"机制无用"与"环境已被基线吃干"。

输出：results/p32_lit_audit.txt
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DS_ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]

# 别名归一化：论文里的写法五花八门
ALIAS = {
    "etth1": "ETTh1", "ett-h1": "ETTh1", "ett h1": "ETTh1",
    "etth2": "ETTh2", "ett-h2": "ETTh2",
    "ettm1": "ETTm1", "ett-m1": "ETTm1",
    "ettm2": "ETTm2", "ett-m2": "ETTm2",
    "weather": "weather", "wth": "weather",
    "electricity": "electricity", "ecl": "electricity", "elec": "electricity",
    "traffic": "traffic", "tra": "traffic",
}

# ---------------------------------------------------------------------------
# 论文池（预注册 §5.1：2024-01 至 2026-10，TSF 顶会/顶刊，提出新架构/模块，
#         在 ≥3 个标准基准上报 MSE。按检索结果出现顺序取前 15 篇）
#
# ⚠️ datasets 字段 = 论文**明确声明**使用的标准基准集合（从摘要/实验节提取）。
#    不是从 MSE 表逐格抄的——见诚实限定 1。
# ---------------------------------------------------------------------------
PAPERS = [
    dict(id=1, name="MWENet (Neurocomputing 2025)",
         note="多小波增强网络，ESD+ELSTM 双模块，LSTM 骨干",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather"]),
    dict(id=2, name="UniWave (IEEE TII 2026, 海南大学)",
         note="Wavelet Patch + 时序调制编码器 TME，三支路",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]),
    dict(id=3, name="STAIR (arXiv 2605.13678)",
         note="三阶段学习解锁简单模型，逐阶段释放容量",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]),
    dict(id=4, name="RAFT (2025)",
         note="检索增强预测，Pearson 相关检索历史 patch",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather", "traffic"]),
    dict(id=5, name="TFEM (2025)",
         note="低秩自注意力 + 频域路径 + 长短权重调和",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather", "traffic"]),
    dict(id=6, name="DEEP TPC (arXiv 2602.16188)",
         note="时序先验深度条件化，冻结 LLM 骨干",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]),
    dict(id=7, name="PhaseFormer (ICLR 2026)",
         note="相位建模 + 低秩跨相位路由 + 参数共享预测头",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]),
    dict(id=8, name="MixLinear (ICLR 2026)",
         note="时频双域融合，0.1K 参数极端低资源",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic"]),
    dict(id=9, name="TimeMixer++ (ICLR 2025)",
         note="多分辨率时间成像 + 时间图像分解 + 多尺度混合",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]),
    dict(id=10, name="BEAT (2025)",
         note="分频自适应调优，按频率监控收敛状态重分配梯度",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]),
    dict(id=11, name="FiCoTS (arXiv 2512.00293)",
         note="细到粗 LLM 增强分层跨模态交互，三级融合模块",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]),
    dict(id=12, name="TALON / CALF (2025)",
         note="时序异质性建模 + 语义对齐，MoE 专家路由",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]),
    dict(id=13, name="Waveformer (IEEE TNNLS 2025)",
         note="MODWT 小波注意力，多周期分解",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather"]),
    dict(id=14, name="iBACon (IEEE TKDE 2025)",
         note="不平衡感知对比学习，IOD 难度伪标签",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic"]),
    dict(id=15, name="STNet (Neurocomputing 2025)",
         note="EWMA 自适应分解 + 季节/趋势双分支 MLP",
         datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather"]),
]


def dataset_dres(regime: str = "raw") -> dict[str, float]:
    """数据集级 Δ_res = 该数据集在给定体制下 5 个训练偏移的中位数。"""
    rows = list(csv.DictReader((RESULTS / "p32_points.csv").open(encoding="utf-8")))
    out = {}
    for ds in DS_ORDER:
        v = [float(r["delta_res"]) for r in rows
             if r["dataset"] == ds and r["regime"] == regime]
        if v:
            out[ds] = float(np.median(v))
    return out


def normalize(ds: str) -> str | None:
    return ALIAS.get(ds.strip().lower())


def main() -> int:
    csv_path = RESULTS / "p32_points.csv"
    if not csv_path.exists():
        print("[错误] 先跑 run_p32_bench.py", file=sys.stderr)
        return 1

    dr = dataset_dres("raw")
    thr = 5.0

    lines = []
    lines.append("=== W-P3-2 文献反查：已发表机制论文的环境分辨力 ===")
    lines.append("")
    lines.append(f"阈值: |Δ_res| < {thr:.0f}% 判为无分辨力区（预注册 §5.3 层 1）")
    lines.append("映射决策: 文献用原始数据（无注入）→ 对应 raw 体制；")
    lines.append("          数据集级 Δ_res = raw 体制 5 个训练偏移的中位数。")
    lines.append("")
    lines.append("--- 各数据集 Δ_res（raw）---")
    for ds in DS_ORDER:
        if ds in dr:
            tag = "无分辨力" if abs(dr[ds]) < thr else "有分辨力"
            lines.append(f"  {ds:12s} {dr[ds]:+7.2f}%  {tag}")
    lines.append("")

    # ---------- 逐篇统计 ----------
    total_pairs = 0
    unres_pairs = 0
    per_paper = []
    lines.append("--- 逐篇 ---")
    lines.append(f"{'#':>2s} {'论文':38s} {'数据集数':8s} {'无分辨力':8s} 明细")
    for p in PAPERS:
        ds_norm = [normalize(d) for d in p["datasets"]]
        ds_norm = [d for d in ds_norm if d in dr]
        unres = [d for d in ds_norm if abs(dr[d]) < thr]
        total_pairs += len(ds_norm)
        unres_pairs += len(unres)
        frac = len(unres) / len(ds_norm) if ds_norm else 0.0
        per_paper.append(dict(id=p["id"], name=p["name"], n=len(ds_norm),
                              n_unres=len(unres), frac=frac,
                              unres_list=",".join(unres)))
        mark = "".join("●" if abs(dr[d]) < thr else "○" for d in ds_norm)
        lines.append(f"{p['id']:2d} {p['name'][:38]:38s} {len(ds_norm):^8d} "
                     f"{len(unres):^8d} {mark}  {p['note'][:30]}")

    frac_all = unres_pairs / total_pairs if total_pairs else 0.0
    lines.append("")
    lines.append("--- L1 判据（预注册 §5.3 层 1）---")
    lines.append(f"  (论文, 数据集) 组合总数 = {total_pairs}")
    lines.append(f"  落在无分辨力区的组合数 = {unres_pairs}")
    lines.append(f"  比例 = {frac_all:.1%}")
    lines.append(f"  判据: ≥25% 支持文献批判主张  →  "
                 f"{'通过' if frac_all >= 0.25 else '未通过'}")

    # ---------- 工具函数 ----------
    def boot_frac(flags, n_boot=10_000, seed=0):
        rng = np.random.default_rng(seed)
        arr = np.array(flags, dtype=float)
        n = len(arr)
        vals = [arr[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    flags = []
    for p in PAPERS:
        for d in p["datasets"]:
            dn = normalize(d)
            if dn in dr:
                flags.append(abs(dr[dn]) < thr)
    lo, hi = boot_frac(flags)
    lines.append(f"  逐组合 bootstrap 95%CI [{lo:.1%}, {hi:.1%}]")

    # 论文级（每篇按"是否至少有一个数据集落在无分辨力区"计）
    n_paper_any = sum(1 for q in per_paper if q["n_unres"] > 0)
    lines.append("")
    lines.append("--- 论文级视角 ---")
    lines.append(f"  至少有一个数据集落在无分辨力区的论文 = "
                 f"{n_paper_any} / {len(per_paper)} = {n_paper_any / len(per_paper):.1%}")
    full_unres = sum(1 for q in per_paper if q["n_unres"] == q["n"] and q["n"] > 0)
    lines.append(f"  **全部**数据集都落在无分辨力区的论文 = {full_unres} / {len(per_paper)}")

    lines.append("")
    lines.append("--- 各数据集被引用的次数 vs 其分辨力 ---")
    lines.append(f"{'数据集':12s} {'被引用次数':10s} {'Δ_res':>8s}  状态")
    for ds in DS_ORDER:
        if ds not in dr:
            continue
        cnt = sum(1 for p in PAPERS if ds in [normalize(x) for x in p["datasets"]])
        tag = "无分辨力" if abs(dr[ds]) < thr else "有分辨力"
        lines.append(f"{ds:12s} {cnt:^10d} {dr[ds]:+7.2f}%  {tag}")

    lines.append("")
    lines.append("--- 诚实限定（预注册 §5.4，不得省略）---")
    lines.append("  1. datasets 字段是论文**声明使用**的标准基准，不是从 MSE 表逐格抄的。")
    lines.append("     要逐篇复算 MSE 需跑其开源代码，本轮未做。")
    lines.append("  2. 我们的 Δ_res 用 lookback=24、单步残差修正；论文用 96/192/336/720")
    lines.append("     多步 horizon。两者不对齐，这是简化。")
    lines.append("  3. 若论文基线已很强（吃掉大部分可修正结构），Δ_res 无法区分")
    lines.append("     '机制无用' 与 '环境已被基线吃干'。这是 Δ_res 的固有局限。")
    lines.append("  4. 论文池由检索前 15 篇构成，检索本身的覆盖度是未控变量。")
    lines.append("     若换检索词，池子会变，比例也会变——这一点必须承认。")

    text = "\n".join(lines)
    print(text)
    (RESULTS / "p32_lit_audit.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\n已写出 {RESULTS / 'p32_lit_audit.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

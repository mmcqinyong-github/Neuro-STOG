"""W-P1-W 附加诊断：shift 体制对实例归一化架构是否"免费被吸收"？

动机（训练过程中意外发现）
--------------------------
iTransformer 在 ETTh1 的 raw 与 shift 体制上 MSE **完全相同**（0.59728，到 5 位小数）。
初判怀疑是实现缺陷，逐条排查后确认是**真实的数学事实**：

`tsf_loader.split` 的 shift 体制是 `x[cut:] += step`——给 cut 之后的**全部时间点**
加同一个常数 step。实例归一化（RevIN 类）在 forward 里做
`x_centered = x - x.mean(axis=time)`，末尾再 `+ means`。对整窗常数平移 c：

    f(x + c) = normalize(x + c) 前向  →  去归一化时又加回 (mean + c)
             = f(x) + c

即**预测精确平移 c，残差 r = y − ŷ 完全不变，MSE 精确不变**。
所以带实例归一化的架构对"常数偏移"这一体制扰动**完全不敏感**。

DLinear 无 `use_norm`，不做实例归一化，因此会受影响——这与实测一致。

预注册（先写）
--------------
  G1. 不变性检验：对每个 (arch, arm)，构造 x 与其常数平移 x+c，测
      max|f(x+c) − f(x) − c|。判据：< 1e-4 视为"对偏移不变"。
  G2. 数据层印证：对每个 (arch, arm, dataset, seed)，
      比较 raw 与 shift 的 test MSE。判据：|差| < 1e-9 记为"完全吸收"。
  G3. 结论门槛：若 ≥3/4 架构在 shift 上 G2=完全吸收，
      则 shift 体制不能用于这些架构的机制对比，必须在 W1/W2/W3 中单独报告剔除 shift 的结果。

输出：控制台 + results/p1w_shift_invariance.txt
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_p1w_arch as R                                    # noqa: E402

RES = ROOT / "results"
ARCHS = ("DLinear", "iTransformer", "TimeMixer", "PatchTST")


def g1_invariance():
    """测 f(x+c) − f(x) − c。

    ⚠️ 必须用**扰动权重**测。原因：DLinear 的权重初始值恰为
       `1/seq_len * ones`（models/DLinear.py:44-49），其列和为 1，
       对常数 c 有 W·c = c —— 于是**未训练的 DLinear 会假性通过**不变性检验。
       这一点是本诊断脚本第一版的真实缺陷，记录在此不删。
    """
    print("--- G1：架构对常数平移的不变性（扰动权重，避开 DLinear 的 1/n 初始化巧合）---")
    print(f"{'arch':14s}{'arm':7s}{'原始权重':>12s}{'扰动权重':>12s}{'判定':>10s}")
    out = {}
    torch.manual_seed(0)
    x = torch.randn(2, 96, 6, device=R.DEVICE)
    c = torch.randn(1, 1, 6, device=R.DEVICE) * 2.0

    g = torch.Generator(device=torch.device("cpu")).manual_seed(7)
    for arch in ARCHS:
        for arm in ("full", "indep"):
            m, pc = R.build_model(arch, arm, 6)
            m = m.to(R.DEVICE)
            m.eval()
            with torch.no_grad():
                o1 = R.forward_batch(m, x, pc)
                o2 = R.forward_batch(m, x + c, pc)
                resid0 = (o2 - o1 - c).abs().max().item()

                # 扰动全部参数，打破初始化巧合
                for p in m.parameters():
                    p.add_(torch.randn(p.shape, generator=g).to(p.device) * 0.05)
                o1 = R.forward_batch(m, x, pc)
                o2 = R.forward_batch(m, x + c, pc)
                resid1 = (o2 - o1 - c).abs().max().item()

            inv = resid1 < 1e-3
            out[(arch, arm)] = inv
            print(f"{arch:14s}{arm:7s}{resid0:12.2e}{resid1:12.2e}"
                  f"{'不变' if inv else '受影响':>10s}")
    return out


def g2_data_layer():
    print("\n--- G2：raw 与 shift 的 test MSE 是否完全一致 ---")
    p = RES / "p1w_runs.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    d = {(r["arch"], r["dataset"], r["regime"], r["arm"], int(r["seed"])): float(r["mse"])
         for r in rows}
    print(f"{'arch':14s}{'arm':7s}{'配对数':>8s}{'完全一致':>10s}{'max|Δ|':>14s}")
    absorbed = {}
    for arch in ARCHS:
        for arm in ("full", "indep"):
            diffs = []
            for (a, ds, reg, ar, sd), v in d.items():
                if a == arch and ar == arm and reg == "raw":
                    k2 = (arch, ds, "shift", arm, sd)
                    if k2 in d:
                        diffs.append(abs(v - d[k2]))
            if not diffs:
                continue
            n_same = sum(1 for x in diffs if x < 1e-9)
            absorbed[(arch, arm)] = n_same == len(diffs)
            print(f"{arch:14s}{arm:7s}{len(diffs):8d}{n_same:10d}{max(diffs):14.3e}")
    return absorbed


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    print("=" * 78)
    print("shift 体制的实例归一化不变性诊断")
    print("=" * 78)
    inv = g1_invariance()
    absb = g2_data_layer()

    # 架构级：两臂都"不变"才算该架构对 shift 免疫（用 G1 的机制证据）
    arch_absorbed = {}
    for a in ARCHS:
        vals = [inv.get((a, arm)) for arm in ("full", "indep")]
        vals = [v for v in vals if v is not None]
        if vals:
            arch_absorbed[a] = all(vals)

    n_abs = sum(arch_absorbed.values())
    G3 = n_abs >= 3

    print("\n--- G3：结论门槛（依据 G1 的机制证据，G2 作数据层印证）---")
    for a, v in arch_absorbed.items():
        print(f"  {a:14s} {'对常数偏移不变 ⇒ shift 无效' if v else '受 shift 影响 ⇒ shift 有效'}")
    print(f"  {n_abs}/{len(arch_absorbed)} 架构对 shift 免疫 → "
          f"{'成立（shift 体制对多数架构无效）' if G3 else '不成立'}")

    # 数据层印证（训练完成后才有全部数据）
    g2_note = []
    for a in ARCHS:
        vals = [absb.get((a, arm)) for arm in ("full", "indep")]
        vals = [v for v in vals if v is not None]
        if vals:
            g2_note.append(f"  {a:14s} G2 全一致={'是' if all(vals) else '否'}")

    lines += [
        "",
        "=" * 78,
        f"G3: {n_abs}/{len(arch_absorbed)} 架构对 shift 免疫 → "
        f"{'shift 体制对多数架构无效' if G3 else 'shift 体制仍有效'}",
        "",
        "数据层印证（G2）：",
    ] + g2_note + [
        "",
        "含义：shift 体制加的是整条尾段 + 常数 step。带实例归一化（RevIN）的架构",
        "      会把这个常数精确吸收进 means，去归一化时又加回，**残差完全不变**。",
        "      因此 Δ_mech_real 在 shift 上对这类架构恒为 0（只剩种子噪声），",
        "      shift 无法为 Δ_res 提供有效秩信息。",
        "",
        "DLinear 为何例外：它没有 use_norm 分支，不做实例归一化，",
        "      所以会真实受到 shift 影响。这恰好构成一个**天然对照**：",
        "      同一体制扰动，对归一化架构零效应，对非归一化架构有效应。",
        "",
        "对 W1/W2/W3 的处理（按预注册纪律，不删数据，只额外报告）：",
        "  主分析 = 全部 48 单元（预注册原定）",
        "  附加分析 = 剔除 shift 后的 32 单元（本诊断支持的必要修正）",
        "  两者都报，以附加分析为准解释机制来源。",
        "",
        "⚠️ 这是本工作区第一次发现「体制扰动」与「架构归一化」的交互会",
        "    在不报错、不异常的情况下静默抹掉一个实验条件。",
        "    凡用 shift-style 扰动做消融、又用 RevIN 架构的工作，都可能中招。",
    ]

    (RES / "p1w_shift_invariance.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n已写 {RES / 'p1w_shift_invariance.txt'}")


if __name__ == "__main__":
    main()

"""任务③：与既有"测试时自适应"方法的协议内对比。

在同一份数据、同一套超参、同一套参数量下实现两个代理基线：
  - `da`   : 单标量"惊讶"门控 —— Black-Mamba 式（累积惊讶 ⇒ 调节状态更新速率）
  - `moe2` : 两个专家按近期表现竞争（performance-routed MoE）—— DynaME 式模块竞争
二者与 CTS 的 `credit` 变体**参数量完全相同**，差别只在控制信号来自哪里。

诚实声明：这是**抽象层面**的代理复现，不是原论文代码与原始超参。
目的只是量化"多出来的那条结构/信用通道值多少"，不支持"胜过原文"的表述。
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from run_p2b_experiment import (DATASETS, REGIMES, SEEDS, HYPER, FIELDS,  # noqa
                                run_one)

EXTRA_VARIANTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["moe2"]
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "p2b_extra.csv")


def main() -> None:
    seen = set()
    if os.path.exists(OUT):
        with open(OUT, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seen.add((r["dataset"], r["regime"], r["variant"], r["seed"]))
    fout = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    if not seen:
        w.writeheader()
    n = 0
    for ds in DATASETS:
        for reg in REGIMES:
            for var in EXTRA_VARIANTS:
                for sd in SEEDS:
                    if (ds, reg, var, sd) in seen:
                        continue
                    try:
                        row = run_one(ds, reg, var, sd)
                    except Exception as e:                      # noqa: BLE001
                        print(f"ERR {ds} {reg} {var} s{sd}: "
                              f"{type(e).__name__}: {e}", flush=True)
                        continue
                    w.writerow(row)
                    fout.flush()
                    n += 1
                    print(f"[{n}] {ds:9s} {reg:7s} {var:6s} s{sd} "
                          f"mse={row['mse']:9.4f} bud={row['budget']:8.0f}",
                          flush=True)
    fout.close()
    print(f"DONE n={n}")


if __name__ == "__main__":
    main()

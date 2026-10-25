"""P3 稳健性补做：按市场分块的 cluster bootstrap。

动机：60 个点 = 4 市场 × 3 体制 × 5 偏移，5 个偏移是同一序列的滑动窗口，
存在自相关。逐点 i.i.d. bootstrap 会把置信区间报窄。本脚本按**市场**分块
（4 块）做重抽，每块内部结构完整保留，只在块层面做有放回抽样。

输出：results/p3_cluster_boot.txt
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "p3_points.csv"
OUT_PATH = ROOT / "results" / "p3_cluster_boot.txt"

N_BOOT = 10_000
SEED = 20261025


def load_points(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                dict(
                    market=r["market"],
                    regime=r["regime"],
                    offset=int(r["offset"]),
                    delta_res=float(r["delta_res"]),
                    delta_mech=float(r["delta_mech"]),
                )
            )
    return rows


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """秩相关；用 argsort 求秩，不做 ties 平均（与 run_p3_audit.py 保持一致）。"""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / denom) if denom > 0 else float("nan")


def point_bootstrap(rows, n_boot=N_BOOT, seed=SEED):
    """原报口径：逐点 i.i.d. 有放回重抽（保留以便对比）。"""
    rng = np.random.default_rng(seed)
    x = np.array([r["delta_res"] for r in rows])
    y = np.array([r["delta_mech"] for r in rows])
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = spearman(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    return vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def cluster_bootstrap(rows, n_boot=N_BOOT, seed=SEED):
    """按市场分块：块层面有放回重抽 4 块，块内所有点整体进入样本。"""
    markets = sorted({r["market"] for r in rows})
    blocks = {m: [r for r in rows if r["market"] == m] for m in markets}
    rng = np.random.default_rng(seed)
    k = len(markets)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        sample = []
        for p in pick:
            sample.extend(blocks[markets[p]])
        x = np.array([r["delta_res"] for r in sample])
        y = np.array([r["delta_mech"] for r in sample])
        v = spearman(x, y)
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    return vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main() -> int:
    if not CSV_PATH.exists():
        print(f"[错误] 找不到 {CSV_PATH}", file=sys.stderr)
        return 1

    rows = load_points(CSV_PATH)
    markets = sorted({r["market"] for r in rows})
    print(f"点数 {len(rows)}  市场块 {markets}  每块点数 "
          f"{[sum(1 for r in rows if r['market'] == m) for m in markets]}")

    x = np.array([r["delta_res"] for r in rows])
    y = np.array([r["delta_mech"] for r in rows])
    rho_full = spearman(x, y)

    pm, plo, phi = point_bootstrap(rows)
    cm, clo, chi = cluster_bootstrap(rows)

    lines = []
    lines.append("=== P3 稳健性：cluster bootstrap（按市场分块）===")
    lines.append(f"点数: {len(rows)}  市场块: {len(markets)} ({', '.join(markets)})")
    lines.append(f"每块点数: {[sum(1 for r in rows if r['market'] == m) for m in markets]}")
    lines.append("")
    lines.append(f"点估计   ρ = {rho_full:+.4f}")
    lines.append("")
    lines.append(f"逐点 bootstrap（原报口径, n_boot={N_BOOT}）:")
    lines.append(f"    均值 = {pm:+.4f}   95%CI [{plo:+.4f}, {phi:+.4f}]   "
                 f"宽度 = {phi - plo:.4f}")
    lines.append(f"cluster bootstrap（按 {len(markets)} 市场分块, n_boot={N_BOOT}）:")
    lines.append(f"    均值 = {cm:+.4f}   95%CI [{clo:+.4f}, {chi:+.4f}]   "
                 f"宽度 = {chi - clo:.4f}")
    lines.append("")
    lines.append(f"宽度变化 = {(chi - clo) - (phi - plo):+.4f}")
    lines.append(f"下界变化 = {clo - plo:+.4f}   "
                 f"下界是否仍远离零 = {bool(clo > 0)}")
    lines.append("")
    lines.append("判读：若下界仍显著大于 0，则 H1 不受滑动窗口自相关影响。")

    text = "\n".join(lines)
    print("\n" + text)
    OUT_PATH.write_text(text + "\n", encoding="utf-8")
    print(f"\n已写出 {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

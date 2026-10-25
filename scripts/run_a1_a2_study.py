"""对照 A1/A2 的 20 种子研究：循环 vs 反应式门控；结构化 vs 无结构转移。

输出 results/a1_env{1,3}_seeds.csv 与 results/a2_env{1,3}_seeds.csv，打印汇总。
不在这里写任何性能假设——结果由数据裁决。
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cyclic_mode import EnvConfig, run_a1, run_a2, run_a4

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def bootstrap_ci(values, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    boots = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def study_a1(env, n_seeds=20, seed0=101):
    rows = []
    for i in range(n_seeds):
        cfg = EnvConfig(seed=seed0 + i, env=env)
        r = run_a1(cfg, backbone=None)
        rows.append({
            "seed": cfg.seed,
            "base_wmse": r.base_wmse, "cyclic_wmse": r.cyclic_wmse, "reactive_wmse": r.reactive_wmse,
            "base_mae": r.base_mae, "cyclic_mae": r.cyclic_mae, "reactive_mae": r.reactive_mae,
            "rel_cyclic_vs_reactive_pct": (r.reactive_wmse - r.cyclic_wmse) / r.reactive_wmse * 100,
            "rel_cyclic_vs_base_pct": (r.base_wmse - r.cyclic_wmse) / r.base_wmse * 100,
            "cyclic_beats_reactive": int(r.cyclic_wmse < r.reactive_wmse),
            "cyclic_beats_base": int(r.cyclic_wmse < r.base_wmse),
            "n_predictions": r.n_predictions,
        })
    out = RESULTS / f"a1_{env}_seeds.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    col = lambda n: np.array([r[n] for r in rows], dtype=float)
    rel = col("rel_cyclic_vs_reactive_pct")
    lo, hi = bootstrap_ci(rel)
    print(f"[A1/{env}] cyclic>reactive: {int(col('cyclic_beats_reactive').sum())}/{n_seeds} seeds | "
          f"rel vs reactive: mean={rel.mean():+.2f}% CI95=[{lo:+.2f},{hi:+.2f}]")
    print(f"[A1/{env}] cyclic>base: {int(col('cyclic_beats_base').sum())}/{n_seeds} seeds | "
          f"mean cyclic_wmse={col('cyclic_wmse').mean():.4f} base={col('base_wmse').mean():.4f} "
          f"reactive={col('reactive_wmse').mean():.4f}")
    print(f"[A1/{env}] MAE: cyclic={col('cyclic_mae').mean():.4f} reactive={col('reactive_mae').mean():.4f} base={col('base_mae').mean():.4f}")
    return rows


def study_a2(env, n_seeds=20, seed0=101):
    rows = []
    for i in range(n_seeds):
        cfg = EnvConfig(seed=seed0 + i, env=env)
        r = run_a2(cfg, backbone=None)
        rows.append({
            "seed": cfg.seed,
            "structured_wmse": r.structured_wmse, "unstructured_wmse": r.unstructured_wmse,
            "structured_mae": r.structured_mae, "unstructured_mae": r.unstructured_mae,
            "rel_structured_pct": (r.unstructured_wmse - r.structured_wmse) / r.unstructured_wmse * 100,
            "structured_beats": int(r.structured_wmse < r.unstructured_wmse),
        })
    out = RESULTS / f"a2_{env}_seeds.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    col = lambda n: np.array([r[n] for r in rows], dtype=float)
    rel = col("rel_structured_pct")
    lo, hi = bootstrap_ci(rel)
    print(f"[A2/{env}] structured>unstructured: {int(col('structured_beats').sum())}/{n_seeds} seeds | "
          f"rel: mean={rel.mean():+.2f}% CI95=[{lo:+.2f},{hi:+.2f}]")
    return rows


def study_a4(env, n_seeds=20, seed0=101):
    rows = []
    for i in range(n_seeds):
        cfg = EnvConfig(seed=seed0 + i, env=env)
        r = run_a4(cfg, backbone=None)
        rows.append({
            "seed": cfg.seed,
            "isolated_wmse": r.isolated_wmse, "global_wmse": r.global_wmse,
            "isolated_mae": r.isolated_mae, "global_mae": r.global_mae,
            "isolated_pollution": r.isolated_pollution,
            "global_pollution": r.global_pollution,
            "rel_isolated_pct": (r.global_wmse - r.isolated_wmse) / r.global_wmse * 100,
            "isolated_beats": int(r.isolated_wmse < r.global_wmse),
        })
    out = RESULTS / f"a4_{env}_seeds.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    col = lambda n: np.array([r[n] for r in rows], dtype=float)
    rel = col("rel_isolated_pct")
    lo, hi = bootstrap_ci(rel)
    print(f"[A4/{env}] isolated>global: {int(col('isolated_beats').sum())}/{n_seeds} seeds | "
          f"rel: mean={rel.mean():+.2f}% CI95=[{lo:+.2f},{hi:+.2f}] | "
          f"pollution iso={col('isolated_pollution').mean():.4f} glob={col('global_pollution').mean():.4f}")
    return rows


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    for env in ("env1", "env3"):
        study_a1(env)
        study_a2(env)
        study_a4(env)
    print("\n结果已写入 results/ 下 6 个 CSV。")

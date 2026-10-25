"""对照 C1 的 20 种子研究：学习更新 vs E1 调参 EMA vs E2 元学习率。

环境 env1/env3/env5/env6；输出 results/c1_{env}_seeds.csv 并打印汇总。
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from residual_state import ResidualConfig, run_c1

RESULTS = Path(__file__).resolve().parents[1] / "results"


def bootstrap_ci(values, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    boots = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def study(env, n_seeds=20, seed0=101):
    rows = []
    for i in range(n_seeds):
        cfg = ResidualConfig(seed=seed0 + i, env=env)
        r = run_c1(cfg, backbone=None)
        rows.append({
            "seed": cfg.seed,
            "base_wmse": r.base_wmse, "learned_wmse": r.learned_wmse,
            "e1_wmse": r.e1_wmse, "e2_wmse": r.e2_wmse,
            "base_mae": r.base_mae, "learned_mae": r.learned_mae,
            "e1_mae": r.e1_mae, "e2_mae": r.e2_mae,
            "rel_vs_e1_pct": (r.e1_wmse - r.learned_wmse) / r.e1_wmse * 100,
            "rel_vs_e2_pct": (r.e2_wmse - r.learned_wmse) / r.e2_wmse * 100,
            "learned_beats_e1": int(r.learned_wmse < r.e1_wmse),
            "learned_beats_e2": int(r.learned_wmse < r.e2_wmse),
            "learned_beats_base": int(r.learned_wmse < r.base_wmse),
            "n_predictions": r.n_predictions,
        })
        print(f"  seed={cfg.seed} base={r.base_wmse:.4f} learned={r.learned_wmse:.4f} "
              f"e1={r.e1_wmse:.4f} e2={r.e2_wmse:.4f}")
    out = RESULTS / f"c1_{env}_seeds.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    col = lambda n: np.array([r[n] for r in rows], dtype=float)
    for name, beats in (("rel_vs_e1_pct", "learned_beats_e1"), ("rel_vs_e2_pct", "learned_beats_e2")):
        rel = col(name)
        lo, hi = bootstrap_ci(rel)
        print(f"[C1/{env}] {name}: {int(col(beats).sum())}/{n_seeds} seeds | "
              f"mean={rel.mean():+.2f}% CI95=[{lo:+.2f},{hi:+.2f}]")
    print(f"[C1/{env}] MAE mean: learned={col('learned_mae').mean():.4f} "
          f"e1={col('e1_mae').mean():.4f} e2={col('e2_mae').mean():.4f} base={col('base_mae').mean():.4f}")
    return rows


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    for env in ("env1", "env5", "env6", "env3"):
        print(f"=== {env} ===")
        study(env)
    print("\n结果已写入 results/c1_*_seeds.csv")

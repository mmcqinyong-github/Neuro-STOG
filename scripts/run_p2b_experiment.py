"""W-P2-B 双通道解离主实验（预注册 §5）。

规模：7 数据集（内部 5 + 外部 2）× 4 体制 × 7 变体 × 3 种子 = 588 单元。
超参沿用 W-P2-A 在 ETTh1 开发集选定的平衡点；外部数据集不参与任何调参。
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cts_v2 import CTSV2, _rewire_matrix            # noqa: E402

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "solar", "PEMS04"]
REGIMES = ["raw", "shift", "noise", "rewire"]
VARIANTS = ["none", "ach", "credit", "cw_ar", "cw_cr", "da", "const"]
SEEDS = [0, 1, 2]

HYPER = dict(n_epochs=300, lr=1e-3, batch=64, eta_base=0.01, kappa=8.0,
             alpha_shift=0.05, alpha_val=0.10, theta_max=1.0)

FIELDS = ["dataset", "regime", "variant", "seed", "mse", "mse_train",
          "eta_mean", "theta_mean", "theta_abs", "dshift", "dval", "credit",
          "budget", "n_param", "seconds"]

LOOKBACK, N_TRAIN, N_CALIB, N_TEST = 96, 800, 400, 500


def _load(ds: str, regime: str):
    if ds in ("solar", "PEMS04"):
        from tsf_external import split_external as sp
    else:
        from tsf_loader import split as sp
    # rewire 在 raw 序列上做，漂移体现在「目标映射」里，输入分布不变
    return sp(ds, "raw" if regime == "rewire" else regime,
              lookback=LOOKBACK, n_train=N_TRAIN,
              n_calib=N_CALIB, n_test=N_TEST, offset=0)


def run_one(ds: str, regime: str, variant: str, seed: int) -> dict:
    t0 = time.time()
    s = _load(ds, regime)
    L = LOOKBACK
    Xtr, Ytr = CTSV2.make_windows(s.train, L, 1)
    S_roll = np.concatenate([s.calib, s.test], axis=0)
    tmap = _rewire_matrix(s.n_vars) if regime == "rewire" else None
    net = CTSV2(L, s.n_vars, seed=seed, variant=variant, **HYPER)
    fit_info = net.fit(Xtr, Ytr)
    res = net.rolling(S_roll, len(s.calib), N_TEST, target_map=tmap)
    n_param = (sum(p.numel() for p in net.enc.parameters())
               + sum(p.numel() for p in net.dec.parameters())
               + sum(p.numel() for p in net.experts.parameters()))
    return dict(dataset=ds, regime=regime, variant=variant, seed=seed,
                mse=round(res["mse"], 6),
                mse_train=round(fit_info["mse_train"], 6),
                eta_mean=round(float(np.mean(res["eta"])), 6),
                theta_mean=round(float(np.mean(res["theta"])), 6),
                theta_abs=round(float(np.mean(np.abs(res["theta"]))), 6),
                dshift=round(float(np.mean(res["d_shift"])), 6),
                dval=round(float(np.mean(res["d_val"])), 6),
                credit=round(float(np.mean(res["credit"])), 6),
                budget=round(res["budget"], 3), n_param=n_param,
                seconds=round(time.time() - t0, 2))


def main() -> None:
    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "p2b_runs.csv")
    seen = set()
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seen.add((r["dataset"], r["regime"], r["variant"], r["seed"]))
    fout = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=FIELDS)
    if not seen:
        writer.writeheader()
    done = 0
    for ds in DATASETS:
        for reg in REGIMES:
            for var in VARIANTS:
                for sd in SEEDS:
                    key = (ds, reg, var, sd)
                    if key in seen:
                        continue
                    try:
                        row = run_one(ds, reg, var, sd)
                    except Exception as e:                     # noqa: BLE001
                        print(f"ERR {ds} {reg} {var} s{sd}: "
                              f"{type(e).__name__}: {e}", flush=True)
                        continue
                    writer.writerow(row)
                    fout.flush()
                    done += 1
                    print(f"[{done}] {ds:9s} {reg:7s} {var:7s} s{sd} "
                          f"mse={row['mse']:9.4f} bud={row['budget']:8.0f} "
                          f"{row['seconds']:5.1f}s", flush=True)
    fout.close()
    print(f"DONE rows_appended={done}")


if __name__ == "__main__":
    main()

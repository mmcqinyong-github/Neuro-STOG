"""W-P2-A 主实验：CTS-Net（互补教学信号-低维子空间网络）。

运行：PYTHONUTF8=1 C:/code/T/Scripts/python.exe scripts/run_p2a_experiment.py
输出：results/p2a_runs.csv（每次训练一行）

判据对应（W-P2-A预注册规格 v1.0）：
  M1 漂移体制 λ=1 vs λ=0 MSE 降低 ≥5% 且配对 t 检验 p<0.05
  M2 raw 体制 λ=1 不劣化超过 2%
  M3 λ 单调性 Spearman ρ(λ, MSE) ≤ -0.6
  M4 子空间正交性 ≥60°（实现自检）
  M5 双通道分离：d_shift(shift/noise) > d_shift(raw)

超参在 ETTh1（开发集）上选定，不触碰外部验证集（solar_AL/PEMS04）。
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cts_net as CN  # noqa: E402

# ---- 数据集（内部/外部分开，外部用 split_external）----
INTERNAL = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"]
EXTERNAL = ["solar", "PEMS04"]
REGIMES = ["raw", "shift", "noise"]
LAMS = [0.0, 0.5, 1.0]
SEEDS = [0, 1, 2]

# ---- 固定超参（ETTh1 开发集选定；外部不参与选择）----
HYPER = dict(n_epochs=300, eta_base=0.01, kappa_shift=8.0, alpha_shift=0.05)


def run_one(ds: str, regime: str, lam: float, seed: int) -> dict:
    if ds in EXTERNAL:
        from tsf_external import split_external as sp
    else:
        from tsf_loader import split as sp
    s = sp(ds, regime, lookback=96, n_train=800, n_calib=400, n_test=500, offset=0)
    L = 96
    Xtr, Ytr = CN.CTSNet.make_windows(s.train, L, 1)
    S_roll = np.concatenate([s.calib, s.test], axis=0)
    net = CN.CTSNet(L, s.n_vars, lam=lam, seed=seed, **HYPER)
    fit_info = net.fit(Xtr, Ytr)
    res = net.rolling(S_roll, len(s.calib), 500)
    return dict(dataset=ds, regime=regime, lam=lam, seed=seed,
                mse=res["mse"], mse_train=fit_info["mse_train"],
                theta_mean=float(np.mean(res["theta"])),
                theta_max=float(np.max(res["theta"])),
                dshift_mean=float(np.mean(res["d_shift"])),
                dval_mean=float(np.mean(res["d_val"])),
                angle_mean=float(np.mean(res["angle"])),
                n_param=sum(p.numel() for p in net.enc.parameters())
                        + sum(p.numel() for p in net.dec.parameters())
                        + sum(p.numel() for p in net.adapter.parameters()))


def main():
    import numpy as np  # noqa: F401
    out = "results/p2a_runs.csv"
    datasets = INTERNAL + EXTERNAL
    total = len(datasets) * len(REGIMES) * len(LAMS) * len(SEEDS)
    # 断点续跑：已存在的 (dataset,regime,lam,seed) 跳过
    done = set()
    if os.path.exists(out):
        with open(out, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["regime"], r["lam"], r["seed"]))
    header = ["dataset", "regime", "lam", "seed", "mse", "mse_train",
              "theta_mean", "theta_max", "dshift_mean", "dval_mean",
              "angle_mean", "n_param"]
    newfile = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=header)
    if newfile:
        w.writeheader()
    t0 = time.time()
    cnt = 0
    for ds in datasets:
        for reg in REGIMES:
            for lam in LAMS:
                for sd in SEEDS:
                    key = (ds, reg, str(lam), str(sd))
                    if key in done:
                        continue
                    try:
                        r = run_one(ds, reg, lam, sd)
                        w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                                    for k, v in r.items()})
                        f.flush()
                        cnt += 1
                        el = time.time() - t0
                        print(f"[{cnt}/{total-len(done)}] {ds} {reg} lam={lam} "
                              f"seed={sd} mse={r['mse']:.4f} ({el:.0f}s)", flush=True)
                    except Exception as e:
                        print(f"ERR {ds} {reg} lam={lam} seed={sd}: "
                              f"{type(e).__name__}: {e}", flush=True)
    f.close()
    print(f"DONE {cnt} units in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

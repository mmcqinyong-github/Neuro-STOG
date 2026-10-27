"""W-P2-A 组件消融：三轴（θ 动态性 / 漂移门控 / 子空间投影）。

对应预注册 N4 否决检查与 §6 消融表。
运行：PYTHONUTF8=1 C:/code/T/Scripts/python.exe scripts/run_p2a_ablation.py
输出：results/p2a_ablation.csv
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cts_net as CN  # noqa: E402

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "solar", "PEMS04"]
REGIME = "shift"          # 主可适应漂移
VARIANTS = ["full", "no_theta", "no_gate", "no_proj"]
SEEDS = [0, 1, 2]
HYPER = dict(n_epochs=300, eta_base=0.01, kappa_shift=8.0, alpha_shift=0.05)


def run_one(ds, variant, seed):
    if ds in ("solar", "PEMS04"):
        from tsf_external import split_external as sp
    else:
        from tsf_loader import split as sp
    s = sp(ds, REGIME, lookback=96, n_train=800, n_calib=400, n_test=500, offset=0)
    L = 96
    Xtr, Ytr = CN.CTSNet.make_windows(s.train, L, 1)
    S_roll = np.concatenate([s.calib, s.test], axis=0)
    net = CN.CTSNet(L, s.n_vars, lam=1.0, seed=seed, variant=variant, **HYPER)
    fi = net.fit(Xtr, Ytr)
    res = net.rolling(S_roll, len(s.calib), 500)
    return dict(dataset=ds, variant=variant, seed=seed, mse=res["mse"],
                mse_train=fi["mse_train"], theta_max=float(np.max(res["theta"])),
                n_param=sum(p.numel() for p in net.enc.parameters())
                        + sum(p.numel() for p in net.dec.parameters())
                        + sum(p.numel() for p in net.adapter.parameters()))


def main():
    out = "results/p2a_ablation.csv"
    header = ["dataset", "variant", "seed", "mse", "mse_train", "theta_max", "n_param"]
    done = set()
    if os.path.exists(out):
        with open(out, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["variant"], r["seed"]))
    newfile = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=header)
    if newfile:
        w.writeheader()
    total = len(DATASETS) * len(VARIANTS) * len(SEEDS)
    cnt = 0
    for ds in DATASETS:
        for va in VARIANTS:
            for sd in SEEDS:
                if (ds, va, str(sd)) in done:
                    continue
                try:
                    r = run_one(ds, va, sd)
                    w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                                for k, v in r.items()})
                    f.flush()
                    cnt += 1
                    print(f"[{cnt}] {ds} {va} seed={sd} mse={r['mse']:.4f}", flush=True)
                except Exception as e:
                    print(f"ERR {ds} {va} seed={sd}: {type(e).__name__}: {e}", flush=True)
    f.close()
    print(f"DONE {cnt}")


if __name__ == "__main__":
    main()

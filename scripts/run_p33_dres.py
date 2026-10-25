"""W-P3-3 层 2：在 RAFT 环境上批量计算 Δ_res。

与层 1 (W-P3-2) 的 Δ_res 分开算，因为环境不同：
  * 层 1：tsf_loader 构造的环境（lookback=24、单步、9 通道子集、800/400/500）
  * 层 2：RAFT 官方环境（官方切分、lookback=720、pred_len=96、全通道）

输出 results/p33_dres.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Paper-Neuro-STOG")
sys.path.insert(0, str(ROOT / "src"))

from raft_dres import delta_res_raft, raft_split  # noqa: E402

DATA_BASE = Path(r"C:\paper-QuantMo\Experiments\dataset")

SPECS = {
    "ETTh1": dict(path=DATA_BASE / "ETT-small" / "ETTh1.csv", kind="ETTh1",
                  seq_len=720, pred_len=96),
    "ETTh2": dict(path=DATA_BASE / "ETT-small" / "ETTh2.csv", kind="ETTh2",
                  seq_len=720, pred_len=96),
    "ETTm1": dict(path=DATA_BASE / "ETT-small" / "ETTm1.csv", kind="ETTm1",
                  seq_len=720, pred_len=96),
    "ETTm2": dict(path=DATA_BASE / "ETT-small" / "ETTm2.csv", kind="ETTm2",
                  seq_len=720, pred_len=96),
    "weather": dict(path=DATA_BASE / "weather" / "weather.csv", kind="custom",
                    seq_len=720, pred_len=96),
    "exchange_rate": dict(path=DATA_BASE / "exchange_rate" / "exchange_rate.csv",
                          kind="custom", seq_len=720, pred_len=96),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(SPECS))
    ap.add_argument("--lags", type=int, default=6)
    ap.add_argument("--max_oracle_pts", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "results" / "p33_dres.csv"))
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        sp = SPECS[ds]
        print(f"[{ds}] 读取数据 ...", flush=True)
        df = pd.read_csv(sp["path"])
        x = df.iloc[:, 1:].to_numpy(dtype=float)
        tr, va, te = raft_split(x, sp["kind"], sp["seq_len"], sp["pred_len"])
        print(f"[{ds}] 切分 {tr.shape} {va.shape} {te.shape}，计算 Δ_res ...",
              flush=True)
        t0 = time.time()
        r = delta_res_raft(tr, va, te, seq_len=sp["seq_len"],
                           pred_len=sp["pred_len"], lags=args.lags,
                           max_oracle_pts=args.max_oracle_pts)
        row = dict(dataset=ds, kind=sp["kind"], seq_len=sp["seq_len"],
                   pred_len=sp["pred_len"], n_vars=r["C"],
                   delta_res=round(r["delta_res"], 4),
                   base_wmse=round(r["base_wmse"], 6),
                   base_wmse_sel=round(r["base_wmse_sel"], 6),
                   oracle_wmse=round(r["oracle_wmse"], 6),
                   n_oracle_pts=r["n_oracle_pts"],
                   n_test_pts=r["n_test_pts"],
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
        print(f"{ds:14s} Δ_res={r['delta_res']:+8.3f}%  "
              f"base={r['base_wmse_sel']:.4f} oracle={r['oracle_wmse']:.4f}  "
              f"({row['seconds']}s)", flush=True)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n写入 {outp}")


if __name__ == "__main__":
    main()

"""W-P1-V：外部数据集加载器（solar_AL / PEMS04 / PEMS08）

严格复刻 `tsf_loader` 的三处缺陷修正，只替换"读原始数组"这一步。
通道采样、z-score 基准、阶跃锚定、阶跃幅度全部继承，**不做任何适配**。

预注册：`W-P1-V外部验证预注册规格-v1.0.md`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tsf_loader import _pick_channels, NOISE_SIGMA, SHIFT_SIGMA

DATA_DIR = Path("C:/paper-QuantMo/Experiments/dataset")

EXTERNAL: dict[str, tuple[str, str]] = {
    "solar":   ("txt", "solar/solar_AL.txt"),
    "PEMS04":  ("npz", "PEMS/PEMS04.npz"),
    "PEMS08":  ("npz", "PEMS/PEMS08.npz"),
}


def load_raw_external(name: str) -> np.ndarray:
    """返回 (n_rows, n_cols) 浮点数组。"""
    kind, rel = EXTERNAL[name]
    path = DATA_DIR / rel
    if kind == "txt":
        arr = pd.read_csv(path, header=None).to_numpy(dtype=float)
    elif kind == "npz":
        d = np.load(path)
        raw = d["data"]
        arr = raw[:, :, 0].astype(float)     # flow 通道
    else:
        raise ValueError(kind)
    if np.isnan(arr).any():
        where = np.isnan(arr)
        med = np.nanmedian(arr, axis=0)
        arr[where] = np.take(med, np.where(where)[1])
    return arr


@dataclass
class BenchSplit:
    train: np.ndarray
    calib: np.ndarray
    test: np.ndarray
    n_vars: int


def split_external(name: str, regime: str = "raw", lookback: int = 24,
                   n_train: int = 800, n_calib: int = 400, n_test: int = 500,
                   offset: int = 0) -> BenchSplit:
    """与 tsf_loader.split 逐行同构，只有 arr 的来源不同。"""
    a = n_train + lookback
    b = a + n_calib + lookback
    c = b + n_test + lookback
    span = c

    arr = load_raw_external(name)
    if arr.shape[0] < offset + span:
        raise ValueError(f"{name} 只有 {arr.shape[0]} 行，需 {offset + span}")
    idx = _pick_channels(arr.shape[1])
    x = arr[offset: offset + span][:, idx].copy()

    head = x[:a]
    mu = head.mean(axis=0)
    sd = head.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    x = (x - mu) / sd

    if regime == "shift":
        cut = a + n_calib // 3
        step = head.std(axis=0) * SHIFT_SIGMA
        step = step / sd
        x[cut:] += step
    elif regime == "noise":
        rng = np.random.default_rng(20_241_025)
        x = x + NOISE_SIGMA * rng.normal(size=x.shape)
    elif regime != "raw":
        raise ValueError(regime)

    return BenchSplit(train=x[:a], calib=x[a - lookback: b],
                      test=x[b - lookback: c], n_vars=x.shape[1])


if __name__ == "__main__":
    for n in EXTERNAL:
        arr = load_raw_external(n)
        idx = _pick_channels(arr.shape[1])
        print(f"{n:8s} raw {arr.shape}  zero_frac={(arr == 0).mean()*100:5.1f}%  "
              f"n_vars={len(idx)}  idx={idx[:9].tolist()}")

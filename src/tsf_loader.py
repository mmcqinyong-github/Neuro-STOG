"""公开 TSF 基准加载器：把 ETTh/ETTm/weather/electricity/traffic 转成 W1Env 兼容序列。

通道选择规则（W-P3-2 预注册 §1，一次写定）：
    n_vars = min(9, 3 * floor(C / 3))
    idx    = unique(round(linspace(0, C-1, n_vars)))
即等间隔采样，**不按方差/相关性/前 n 个挑**。规则固定后不因结果调整。

为什么不用前 n 个：electricity/traffic 的通道按地理/道路相邻编号，
前 9 个是高度相关的邻近节点，不具代表性，会把 Δ_res 系统性压低。

------------------------------------------------------------------
实现缺陷记录（2026-10-25 首轮跑完后发现并修正，共 3 处）
------------------------------------------------------------------
**缺陷 1：阶跃位置随 offset 漂移**
原实现 `cut = int(0.7 * n_obs)`，n_obs 随 offset 增长，导致阶跃在序列中的
相对位置从 calib 尾部（1240）一路滑到 train 段内（760）。后果：
  traffic shift off=0 的 calib 几乎全是切换前数据，机制在旧体制拟合、
  在新体制评估 → Δ_mech = −79.30% 的灾难值。这不是机制的效应。
修正：`cut` 由 `split` 按「校准段 1/3 处」这一**固定相对位置**传入。

**缺陷 2：标准化基准随 offset 变化**
原实现 `head = x[:0.4*n_obs]`，n_obs=1772（off=0）时 head=708 行，
n_obs=3372（off=1600）时 head=1348 行——**已经越过 train 段（824 行）
进入 calib 段**，等于用校准段统计量做标准化，构成信息泄漏，
且各 offset 的 z-score 基准不同、不可比。
修正：标准化统计量固定取「该 offset 下的 train 段」。

**缺陷 3：阶跃幅度用局部 std，窗口依赖**
`step = x[:c].std * 1.5` 的 std 来自切换前窗口，不同 offset 的窗口波动率不同
（实测 ETTh1 阶跃幅度 1.99σ vs 2.54σ），把数据异质性混入环境差异。
修正：改用**该 offset 的 train 段 std**（固定段、固定长度），使阶跃幅度在
各 offset 间可比。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path("C:/paper-QuantMo/Experiments/dataset")

BENCHMARKS: dict[str, str] = {
    "ETTh1": "ETT-small/ETTh1.csv",
    "ETTh2": "ETT-small/ETTh2.csv",
    "ETTm1": "ETT-small/ETTm1.csv",
    "ETTm2": "ETT-small/ETTm2.csv",
    "weather": "weather/weather.csv",
    "electricity": "electricity/electricity.csv",
    "traffic": "traffic/traffic.csv",
}

# 预注册 §1：不纳入主分析
EXCLUDED = {
    "exchange_rate": "随机游走型，Δ_res 预期极端，单独作附注",
    "solar": "非 CSV 格式（.txt），读取路径不同",
    "illness": "单变量，与多通道残差修正设定不符",
    "PEMS": "非标准表结构",
    "m4": "竞赛格式，非等间隔",
}

SHIFT_SIGMA = 1.5      # 阶跃幅度（以 train 段 std 为单位）
NOISE_SIGMA = 1.5      # 噪声幅度


def _pick_channels(n_cols: int) -> np.ndarray:
    """预注册 §1 的等间隔通道采样规则。"""
    n_vars = min(9, 3 * (n_cols // 3))
    if n_vars < 3:
        raise ValueError(f"通道数 {n_cols} 太少，无法构成域")
    return np.unique(np.round(np.linspace(0, n_cols - 1, n_vars)).astype(int))


def load_raw(name: str, max_rows: int | None = None) -> np.ndarray:
    """读原始 CSV，返回 (n_rows, n_cols) 的浮点数组。"""
    if name not in BENCHMARKS:
        raise KeyError(f"未知数据集 {name}；可选 {list(BENCHMARKS)}")
    df = pd.read_csv(DATASET_DIR / BENCHMARKS[name], nrows=max_rows)
    cols = [c for c in df.columns if c.lower() != "date"]
    arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    keep = ~np.isnan(arr).all(axis=1)
    arr = arr[keep]
    if np.isnan(arr).any():
        med = np.nanmedian(arr, axis=0)
        where = np.isnan(arr)
        arr[where] = np.take(med, np.where(where)[1])
    return arr


@dataclass
class BenchSplit:
    train: np.ndarray
    calib: np.ndarray
    test: np.ndarray
    n_vars: int


def split(name: str, regime: str = "raw", lookback: int = 24,
          n_train: int = 800, n_calib: int = 400, n_test: int = 500,
          offset: int = 0) -> BenchSplit:
    """三段切分，含 offset 滑动训练窗（预注册 §2 的 5 个训练偏移）。

    处理顺序（全部三个缺陷修正后）：
      1. 取 `offset` 起的连续窗口，长度 = n_train + n_calib + n_test + 3*lookback
      2. 用**该窗口的前 n_train+lookback 行（即 train 段）**统计量做 z-score
         → 无泄漏，且各 offset 的基准同定义
      3. 在 z-score 后的序列上注入 shift / noise，阶跃幅度用 train 段 std（≈1）
         → 阶跃位置锚定 calib 段 1/3 处，不随 offset 漂移

    阶跃的段结构：
      train 全程切换前 → 骨干无法预先吸收
      calib 前 1/3 切换前、后 2/3 切换后 → 机制可见两种体制
      test  全程切换后 → 检验外推
    """
    a = n_train + lookback
    b = a + n_calib + lookback
    c = b + n_test + lookback
    span = c                      # 相对窗口内的总长度

    arr = load_raw(name)
    if arr.shape[0] < offset + span:
        raise ValueError(f"{name} 只有 {arr.shape[0]} 行，需 {offset + span}")
    idx = _pick_channels(arr.shape[1])
    x = arr[offset : offset + span][:, idx].copy()

    # --- 缺陷 2 修正：标准化基准 = 该 offset 的 train 段 ---
    head = x[:a]
    mu = head.mean(axis=0)
    sd = head.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    x = (x - mu) / sd

    # --- 缺陷 1/3 修正：阶跃锚定 calib 1/3 处，幅度用 train 段 std ---
    if regime == "shift":
        cut = a + n_calib // 3
        step = head.std(axis=0) * SHIFT_SIGMA          # head 已是原始尺度的副本
        step = step / sd                                # 换到 z-score 尺度
        x[cut:] += step
    elif regime == "noise":
        rng = np.random.default_rng(20_241_025)
        x = x + NOISE_SIGMA * rng.normal(size=x.shape)  # z 尺度下 std≈1
    elif regime != "raw":
        raise ValueError(regime)

    return BenchSplit(
        train=x[:a],
        calib=x[a - lookback : b],
        test=x[b - lookback : c],
        n_vars=x.shape[1],
    )


def channel_report() -> list[dict]:
    out = []
    for name in BENCHMARKS:
        arr = load_raw(name)
        idx = _pick_channels(arr.shape[1])
        out.append(dict(name=name, n_cols_raw=arr.shape[1],
                        n_vars=len(idx), idx=idx.tolist()))
    return out


if __name__ == "__main__":
    for r in channel_report():
        print(f"{r['name']:12s} 原始 {r['n_cols_raw']:4d} 列  →  n_vars={r['n_vars']}  "
              f"idx={r['idx']}")

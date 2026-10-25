"""EPF 真实数据加载器：把 5 个电力市场转成 W1Env 兼容的多变量序列。

数据源：`C:/paper-QuantMo/Experiments/论文模板1的数据集-EPF/{BE,DE,FR,NP,PJM}.csv`
每文件 52,416 行小时级，列 = [date, 两个外生预测, OT(实际电价)]。

三体制构造（预注册 P3 规格 §3）：
- raw   原始序列，骨干 = 逐通道 ridge
- shift 后 30% 叠加阶跃（模拟体制切换）
- noise σ×2.5（模拟高噪声）

设计约束：W1Env 的 make_series 是合成的。这里用 `SeriesOverride` 把真实序列注入
shared_gain 的特征流水线，复用同一套五臂与指标，保证合成/真实可比。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPF_DIR = Path("C:/paper-QuantMo/Experiments/论文模板1的数据集-EPF")
MARKETS = ("BE", "DE", "FR", "NP", "PJM")

# ⚠️ 数据完整性缺陷（2026-10-25 实测）
# BE 与 FR 的两个外生预测列**逐位完全相同**（np.allclose → True，量级 8.4e3 / 1.2e4），
# 仅电价列不同（mean|diff| = 1.38，价 scale 55.5）。
# 这不是巧合：该数据集把同一份外生预测复制给了两个市场。
# 后果：若把 BE/FR 当两个独立环境，会人为抬高 Spearman ρ 的置信度。
# 处置：保留 FR（其电价 lag1 自相关 0.897 更典型），**BE 标记为非独立**，主分析排除，
# 只在附注里报告"含 BE 时 ρ 如何变化"，作为敏感性检查。
INDEPENDENT_MARKETS = ("DE", "FR", "NP", "PJM")
NON_INDEPENDENT = {"BE": "外生预测列与 FR 逐位相同（重复数据）"}


def load_market(market: str) -> pd.DataFrame:
    p = EPF_DIR / f"{market}.csv"
    df = pd.read_csv(p)
    df.columns = ["date", "exog1", "exog2", "price"]
    for c in ("exog1", "exog2", "price"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def build_series(market: str, regime: str = "raw", n_obs: int = 4000,
                 standardize: bool = True) -> np.ndarray:
    """返回 (n_obs, 3) 的多变量序列：exog1, exog2, price。

    standardize=True（默认）：用**训练段**统计量做 z-score。
    必要性：外生预测量级 8.4e3–1.2e4，电价仅 10²。不标准化时逐通道 ridge 的
    残差被外生变量的绝对量级主导（实测 base wmse 达 8e4–2e6），机制差异被淹没，
    Δ_res 与 Δ_mech 都退化为噪声比。这是 P3 规格 §6 未预见的量纲问题。
    注意：标准化会改变 wmse 的绝对尺度，但不改变**相对**改善百分比——后者才是审计量。
    """
    df = load_market(market)
    x = df[["exog1", "exog2", "price"]].to_numpy(dtype=float)[:n_obs].copy()
    assert x.shape[0] == n_obs, f"{market} 只有 {x.shape[0]} 行"

    if standardize:
        # 只用前 40%（训练段）估计，避免校准/测试信息泄漏
        head = x[: max(int(0.4 * n_obs), 50)]
        mu = head.mean(axis=0)
        sd = head.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        x = (x - mu) / sd

    if regime == "shift":
        cut = int(0.7 * n_obs)
        step = x[:cut].std(axis=0) * 1.5
        x[cut:] += step
    elif regime == "noise":
        rng = np.random.default_rng(20_241_025)
        x = x + 1.5 * x.std(axis=0) * rng.normal(size=x.shape)
    elif regime != "raw":
        raise ValueError(regime)
    return x


@dataclass
class RealSplit:
    """三段切分：train / calib / test，全部来自同一真实序列（时序不重叠）。"""
    train: np.ndarray
    calib: np.ndarray
    test: np.ndarray


def split(market: str, regime: str = "raw", lookback: int = 24,
          n_train: int = 800, n_calib: int = 400, n_test: int = 500) -> RealSplit:
    total = n_train + n_calib + n_test + 3 * lookback
    x = build_series(market, regime, n_obs=total)
    a = n_train + lookback
    b = a + n_calib + lookback
    c = b + n_test + lookback
    return RealSplit(train=x[:a], calib=x[a - lookback : b], test=x[b - lookback : c])

"""Domain loading, day blocking and chronological splitting.

All eleven testbeds are reduced to a common hourly grid with H = 24 steps per
day so that a "day" means the same thing everywhere; sub-hourly testbeds are
aggregated (mean) up to hourly before blocking.  Calendar covariates are derived
positionally, never from future timestamps.

Leakage contract enforced here
------------------------------
* the z-score statistics come from the training split only;
* the window of day t is y[t*H - L : t*H], i.e. ends strictly before day t;
* every day-level statistic used downstream is computed causally.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

H = 24                      # steps per day after aggregation
LOOKBACK_DAYS = 7
L = H * LOOKBACK_DAYS       # 168 input steps

DATA_ROOT = r"C:\paper-STOG\Experiments\dataset"
EPF_DIR = os.path.join(DATA_ROOT, "EPF")
LTF_DIR = os.path.join(DATA_ROOT, "Long-term multivariate dataset")


@dataclass(frozen=True)
class DomainSpec:
    key: str
    path: str
    target: str            # column name, or "__mean__" for the row mean
    date_col: Optional[str]
    agg: int               # native steps averaged into one hourly step
    family: str            # spectral family label used in the universality plot


DOMAINS: Dict[str, DomainSpec] = {
    "EPF-NP": DomainSpec("EPF-NP", os.path.join(EPF_DIR, "NP.csv"), "OT", "date", 1, "power"),
    "EPF-PJM": DomainSpec("EPF-PJM", os.path.join(EPF_DIR, "PJM.csv"), "OT", "date", 1, "power"),
    "EPF-BE": DomainSpec("EPF-BE", os.path.join(EPF_DIR, "BE.csv"), "OT", "date", 1, "power"),
    "EPF-FR": DomainSpec("EPF-FR", os.path.join(EPF_DIR, "FR.csv"), "OT", "date", 1, "power"),
    "EPF-DE": DomainSpec("EPF-DE", os.path.join(EPF_DIR, "DE.csv"), "OT", "date", 1, "power"),
    "ETTh1": DomainSpec("ETTh1", os.path.join(LTF_DIR, "ETT-small", "ETTh1.csv"), "OT", "date", 1, "sensor"),
    "ETTh2": DomainSpec("ETTh2", os.path.join(LTF_DIR, "ETT-small", "ETTh2.csv"), "OT", "date", 1, "sensor"),
    "ETTm1": DomainSpec("ETTm1", os.path.join(LTF_DIR, "ETT-small", "ETTm1.csv"), "OT", "date", 4, "sensor"),
    "ETTm2": DomainSpec("ETTm2", os.path.join(LTF_DIR, "ETT-small", "ETTm2.csv"), "OT", "date", 4, "sensor"),
    "Weather": DomainSpec("Weather", os.path.join(LTF_DIR, "weather", "weather.csv"), "OT", "date", 6, "climate"),
    "ECL": DomainSpec("ECL", os.path.join(LTF_DIR, "electricity", "electricity.csv"), "__mean__", "date", 1, "load"),
}

DOMAIN_ORDER = list(DOMAINS.keys())

# Regime-switching stress benchmarks built by splicing contiguous blocks of
# different real series (see scripts/build_stress.py).  STRESS-0 is the matched
# control: the same splicing applied to a single series.
STRESS_ORDER = ["STAT-0", "STAT-1", "SW-2", "SW-4", "SW-X"]
ALL_ORDER = DOMAIN_ORDER + STRESS_ORDER


def _read_series(spec: DomainSpec) -> Tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
    if spec.path.endswith(".txt"):
        arr = np.loadtxt(spec.path, delimiter=",")
        if arr.ndim == 1:
            arr = arr[:, None]
        vals = arr.mean(axis=1)
        idx = pd.date_range("2006-01-01", periods=len(vals), freq="10min")
    else:
        df = pd.read_csv(spec.path)
        if spec.date_col is not None and spec.date_col in df.columns:
            ts = pd.to_datetime(df[spec.date_col], errors="coerce")
            df = df.loc[~ts.isna()].copy()
            ts = ts.loc[~ts.isna()]
            df.index = pd.DatetimeIndex(ts)
        else:
            df.index = pd.RangeIndex(len(df))
        if spec.target == "__mean__":
            num = df.select_dtypes(include=[np.number])
            vals = num.mean(axis=1).to_numpy(dtype=float)
        else:
            vals = df[spec.target].to_numpy(dtype=float)
        idx = df.index

    vals = np.array(vals, dtype=float, copy=True)
    bad = ~np.isfinite(vals) | (np.abs(vals) >= 9999.0)
    if bad.any():
        good = np.where(~bad)[0]
        if len(good) >= 2:
            vals[bad] = np.interp(np.where(bad)[0], good, vals[good])
        else:
            vals[bad] = 0.0
    return np.asarray(vals, dtype=float), idx


def load_domain(key: str) -> Dict:
    """Return the hourly series plus the day-blocked matrices."""
    spec = DOMAINS[key]
    vals, idx = _read_series(spec)

    # aggregate sub-hourly grids up to hourly
    if spec.agg > 1:
        n = (len(vals) // spec.agg) * spec.agg
        vals = vals[:n].reshape(-1, spec.agg).mean(axis=1)
        if isinstance(idx, pd.DatetimeIndex):
            idx = idx[::spec.agg][: len(vals)]

    # align the grid so that step 0 is hour 0 of some day
    if isinstance(idx, pd.DatetimeIndex) and len(idx) == len(vals):
        off = int(idx[0].hour) % 24
        if off != 0:
            vals = vals[24 - off:]
            idx = idx[24 - off:]

    # sentinel / missing handling: -9999 style markers are treated as gaps and
    # linearly interpolated, never as observations.
    vals = np.asarray(vals, dtype=float)
    bad = ~np.isfinite(vals) | (np.abs(vals) >= 9999.0)
    if bad.any():
        idx_good = np.where(~bad)[0]
        if len(idx_good) >= 2:
            vals[bad] = np.interp(np.where(bad)[0], idx_good, vals[idx_good])
        else:
            vals[bad] = 0.0
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

    n_days_total = len(vals) // H
    vals = vals[: n_days_total * H]
    Y = vals.reshape(n_days_total, H)                     # (D, H)

    # calendar covariates (positional, never future)
    hour = np.arange(H)
    if isinstance(idx, pd.DatetimeIndex) and len(idx) >= H:
        dow = np.array([idx[i * H].dayofweek for i in range(n_days_total)])
    else:
        dow = np.arange(n_days_total) % 7

    # day blocks: a window is available from day LOOKBACK_DAYS onwards
    first = LOOKBACK_DAYS
    X = np.stack([Y[t - LOOKBACK_DAYS:t].ravel() for t in range(first, n_days_total)])
    Yd = Y[first:]
    dow_d = dow[first:]

    return {
        "key": key,
        "family": spec.family,
        "series": vals,
        "X": X.astype(np.float64),           # (D, 168) raw windows
        "Y": Yd.astype(np.float64),          # (D, 24)
        "dow": dow_d.astype(np.int64),       # (D,)
        "hour": hour,
        "n_days": int(Yd.shape[0]),
    }


def split_indices(n_days: int, split=(0.7, 0.1, 0.2)) -> Dict[str, np.ndarray]:
    ntr = int(n_days * split[0])
    nva = int(n_days * split[1])
    return {
        "train": np.arange(0, ntr),
        "val": np.arange(ntr, ntr + nva),
        "test": np.arange(ntr + nva, n_days),
    }


def fit_scaler(Ytr: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(Ytr))
    sd = float(np.std(Ytr))
    if sd < 1e-8:
        sd = 1.0
    return mu, sd


def seasonal_naive_day(X: np.ndarray) -> np.ndarray:
    """Forecast day t with the observed profile of day t-1."""
    return X[:, -H:]


def seasonal_naive_week(X: np.ndarray) -> np.ndarray:
    """Forecast day t with the observed profile of day t-7."""
    return X[:, -7 * H: -6 * H]

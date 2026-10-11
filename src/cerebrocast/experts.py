"""The forecasting expert pool.

Twelve experts spanning four model families (naive/seasonal, linear, kernel,
tree/neural) are fitted once on the training split and then used to produce a
day-ahead forecast for every day of the stream.  Their per-day predictions are
cached, so the streaming layer never touches the raw series again and the whole
maintenance experiment is reproducible from the cache.

Nothing here is retrained during the test phase except by the baselines that are
explicitly defined as retraining baselines (B2, B6).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from .data import H, LOOKBACK_DAYS

EXPERT_NAMES: List[str] = [
    "snaive_day",
    "snaive_week",
    "naive_last",
    "hour_mean",
    "theta_damped",
    "ridge_lag",
    "ridge_lag_strong",
    "knn_lag",
    "dlinear",
    "random_forest",
    "hist_gbm",
    "mlp",
]

STOCHASTIC = {"random_forest", "hist_gbm", "mlp"}


# ---------------------------------------------------------------------------
# feature blocks
# ---------------------------------------------------------------------------
def lag_features(X: np.ndarray) -> np.ndarray:
    D = X.shape[0]
    prof = X.reshape(D, LOOKBACK_DAYS, H)          # days t-7 .. t-1
    means = prof.mean(axis=2)
    stds = prof.std(axis=2)
    last = prof[:, -1, :]
    eps = 1e-6
    extra = np.column_stack([
        means[:, -1] - means[:, 0],
        stds[:, -1],
        stds[:, -1] / (stds[:, 0] + eps),
        last.max(axis=1) - last.min(axis=1),
        np.abs(np.diff(last, axis=1)).mean(axis=1),
        means.mean(axis=1),
        prof[:, -1, :3].mean(axis=1) - prof[:, -2, :3].mean(axis=1),
    ])
    return np.column_stack([X, means, stds, extra]).astype(np.float64)


def dow_features(dow: np.ndarray) -> np.ndarray:
    d = np.asarray(dow) % 7
    return np.column_stack([
        np.sin(2 * np.pi * d / 7.0),
        np.cos(2 * np.pi * d / 7.0),
        (d >= 5).astype(float),
    ])


def hour_features(hour: np.ndarray) -> np.ndarray:
    h = np.asarray(hour) % 24
    return np.column_stack([np.sin(2 * np.pi * h / 24.0), np.cos(2 * np.pi * h / 24.0)])


def row_features(lag: np.ndarray, dw: np.ndarray, hf: np.ndarray) -> np.ndarray:
    """(D, Fl+Fd) x (H, Fh) -> (D*H, Fl+Fd+Fh), step h of day d in row d*H+h."""
    D = lag.shape[0]
    Hh = hf.shape[0]
    base = np.concatenate([lag, dw], axis=1)                     # (D, F)
    base = np.repeat(base, Hh, axis=0)                           # (D*H, F)
    hh = np.tile(hf, (D, 1))                                     # (D*H, Fh)
    return np.concatenate([base, hh], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# expert implementations
# ---------------------------------------------------------------------------
class Expert:
    def __init__(self, name: str, seed: int = 0):
        self.name = name
        self.seed = seed
        self._model = None
        self._stat = None
        self._row_model = False

    # -- fit ---------------------------------------------------------------
    def fit(self, Xtr, Ytr, dow_tr, hour) -> "Expert":
        lag = lag_features(Xtr)
        dw = dow_features(dow_tr)
        Xlin = np.column_stack([lag, dw])
        hf = hour_features(hour)

        n = self.name
        if n == "snaive_day":
            self._stat = None
        elif n == "snaive_week":
            self._stat = None
        elif n == "naive_last":
            self._stat = None
        elif n == "hour_mean":
            self._stat = Ytr.mean(axis=0)
        elif n == "theta_damped":
            self._stat = None
        elif n == "dlinear":
            self._stat = None
        elif n in ("ridge_lag", "ridge_lag_strong"):
            a = 1.0 if n == "ridge_lag" else 30.0
            self._model = Ridge(alpha=a, fit_intercept=True)
            self._model.fit(Xlin, Ytr)
        elif n == "knn_lag":
            self._model = KNeighborsRegressor(n_neighbors=40, weights="distance")
            self._model.fit(Xlin, Ytr)
        elif n == "random_forest":
            Xr = row_features(lag, dw, hf)
            yr = Ytr.reshape(-1)
            idx = np.random.default_rng(self.seed).choice(
                Xr.shape[0], size=min(24000, Xr.shape[0]), replace=False)
            self._model = RandomForestRegressor(
                n_estimators=160, max_depth=14, min_samples_leaf=5,
                n_jobs=-1, random_state=self.seed)
            self._model.fit(Xr[idx], yr[idx])
            self._row_model = True
        elif n == "hist_gbm":
            Xr = row_features(lag, dw, hf)
            yr = Ytr.reshape(-1)
            self._model = HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.06, max_depth=6,
                min_samples_leaf=20, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.12,
                random_state=self.seed)
            self._model.fit(Xr, yr)
            self._row_model = True
        elif n == "mlp":
            Xr = row_features(lag, dw, hf)
            yr = Ytr.reshape(-1)
            mu, sd = Xr.mean(axis=0), Xr.std(axis=0) + 1e-8
            self._stat = (mu, sd)
            self._model = MLPRegressor(
                hidden_layer_sizes=(160, 80), activation="relu",
                alpha=1e-3, learning_rate_init=3e-3, max_iter=180,
                early_stopping=True, n_iter_no_change=12,
                random_state=self.seed)
            self._model.fit((Xr - mu) / sd, yr)
            self._row_model = True
        else:
            raise ValueError(n)
        return self

    # -- predict -----------------------------------------------------------
    def predict(self, X: np.ndarray, dow: np.ndarray, hour) -> np.ndarray:
        D = X.shape[0]
        n = self.name
        if n == "snaive_day":
            return X[:, -H:]
        if n == "snaive_week":
            return X[:, -7 * H: -6 * H]
        if n == "naive_last":
            return np.repeat(X[:, -1:], H, axis=1)
        if n == "hour_mean":
            return np.tile(self._stat, (D, 1))
        if n == "theta_damped":
            prof = X.reshape(D, LOOKBACK_DAYS, H)
            week = prof[:, 0, :]
            means = prof.mean(axis=2)
            trend = means[:, -1] - means[:, 0]
            return week + 0.5 * trend[:, None]
        if n == "dlinear":
            prof = X.reshape(D, LOOKBACK_DAYS, H)
            means = prof.mean(axis=2)                       # (D,7)
            t = np.arange(LOOKBACK_DAYS)
            tm = t - t.mean()
            slope = (means * tm).sum(axis=1) / (tm ** 2).sum()
            level = means[:, -1] + slope
            seasonal = prof[:, -1, :] - means[:, -1:]
            return level[:, None] + seasonal
        if self._row_model:
            lag = lag_features(X)
            dw = dow_features(dow)
            Xr = row_features(lag, dw, hour_features(hour))
            if n == "mlp":
                mu, sd = self._stat
                Xr = (Xr - mu) / sd
            return self._model.predict(Xr).reshape(D, H)
        lag = lag_features(X)
        dw = dow_features(dow)
        return self._model.predict(np.column_stack([lag, dw]))


def heterogeneous_experts(X, Y, dow, idx, seed: int, hour,
                          n_regimes: int = 3) -> Tuple[Dict[str, np.ndarray], int]:
    """Experts whose competence genuinely depends on the regime.

    A deployed pool is never homogeneous: it contains models of different
    vintages (trained on shorter or older windows) and models specialised to the
    sub-populations they were fitted on.  Both families are necessary for the
    maintenance problem to be well posed -- with a homogeneous pool the best
    fixed combination is already near-optimal and there is nothing to maintain.
    """
    from sklearn.cluster import KMeans

    from .fingerprints import Fingerprint

    tr = idx["train"]
    lag = lag_features(X)
    dw = dow_features(dow)
    Xlin = np.column_stack([lag, dw])
    hf = hour_features(hour)
    out: Dict[str, np.ndarray] = {}

    # --- vintages: the same model fitted on a prefix of the training split ---
    for f in (0.25, 0.50, 0.75):
        sub = tr[: max(30, int(len(tr) * f))]
        m = Ridge(alpha=1.0).fit(Xlin[sub], Y[sub])
        out[f"vintage_ridge_{int(f * 100)}"] = m.predict(Xlin)

    # --- regime specialists: fitted on fingerprint clusters of the training days
    fp = Fingerprint(K=32).fit(X[tr])
    PHI = fp.transform(X)
    km = KMeans(n_clusters=n_regimes, n_init=5, random_state=seed).fit(PHI[tr])
    for c in range(n_regimes):
        sub = tr[km.labels_ == c]
        if len(sub) < 40:
            continue
        m = Ridge(alpha=1.0).fit(Xlin[sub], Y[sub])
        out[f"regime_ridge_{c}"] = m.predict(Xlin)
        Xr = row_features(lag[sub], dw[sub], hf)
        yr = Y[sub].reshape(-1)
        g = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.06, max_depth=6, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.12,
            random_state=seed).fit(Xr, yr)
        out[f"regime_hgb_{c}"] = g.predict(row_features(lag, dw, hf)).reshape(-1, H)
    return out, n_regimes


def build_pool(X, Y, dow, idx, seed: int, hour) -> Tuple[np.ndarray, List[str]]:
    """Fit every expert on the training days and predict the whole stream.

    Returns
    -------
    P : (D, M, H) array of expert forecasts (normalised units)
    names : list of expert names
    """
    tr = idx["train"]
    preds, names = [], []
    for name in EXPERT_NAMES:
        s = seed if name in STOCHASTIC else 0
        e = Expert(name, seed=s).fit(X[tr], Y[tr], dow[tr], hour)
        preds.append(e.predict(X, dow, hour))
        names.append(name)
    extra, _ = heterogeneous_experts(X, Y, dow, idx, seed, hour)
    for k in sorted(extra):
        preds.append(extra[k])
        names.append(k)
    P = np.stack(preds, axis=1).astype(np.float64)
    return P, names

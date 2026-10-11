"""Stage 1 -- fit the expert pool and cache every expert's day-ahead forecast.

Protocol (strictly out of sample)
---------------------------------
The series is cut into four chronological blocks:

    |---- fit ----|-- warm --|-- val --|------- test -------|
        55%           10%        10%            25%

The expert pool is refitted twice as data accumulates, so that *every* day the
architecture ever sees is forecast by a model that has never seen it:

    warm days  <- pool fitted on  fit
    val  days  <- pool fitted on  fit + warm
    test days  <- pool fitted on  fit + warm + val

The `fit` block is used only to train the pool; its own forecasts are not stored
(they would be in-sample).  This removes an in-sample artefact that the previous
single-fit version had: the distance-weighted k-NN expert reproduced its own
training targets exactly on 46% of the training days, which made the pool look
far more skilful than it is and corrupted the architecture's warm-up.

The z-score statistics (mu, sd) are computed on the `fit` block alone.

Output: results/pool/<domain>__seed<s>.npz with
    P    (D, M, H) expert forecasts, NaN inside the fit block
    Y    (D, H)    realised day
    X    (D, 168)  input windows
    dow  (D,)      day of week
    idx_fit / idx_warm / idx_val / idx_test
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.config import SEEDS  # noqa: E402
from cerebrocast.data import DOMAIN_ORDER, load_domain  # noqa: E402
from cerebrocast.experts import EXPERT_NAMES, Expert  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "pool")

BLOCK_SPLIT = (0.55, 0.10, 0.10, 0.25)      # fit / warm / val / test
# One pool realisation shared by every architecture seed: the comparison of
# interest is paired (same pool, same days), which maximises power and keeps the
# cost of the growing-window protocol at three refits per benchmark.
POOL_SEEDS = (7,)


def block_indices(n_days: int) -> dict:
    a, b, c, d = BLOCK_SPLIT
    i0 = int(n_days * a)
    i1 = i0 + int(n_days * b)
    i2 = i1 + int(n_days * c)
    return {"fit": np.arange(0, i0), "warm": np.arange(i0, i1),
            "val": np.arange(i1, i2), "test": np.arange(i2, n_days)}


def _hetero_fixed(X, Y, dow, hour, fit_idx, target_idx, seed, n_regimes=3):
    """Vintage and regime-specialist experts with a *fixed* roster.

    The stage-by-stage refit means the fitting set grows; a naive
    implementation would drop a regime specialist whenever one of its clusters
    happened to fall below the minimum size, and the pool would change shape
    between blocks.  Here a starved cluster simply falls back to the whole
    fitting block, so the roster (and therefore the expert count) is identical
    in every block.
    """
    from sklearn.cluster import KMeans
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    from cerebrocast.experts import dow_features, hour_features, lag_features, row_features
    from cerebrocast.fingerprints import Fingerprint

    H = Y.shape[1]
    lag = lag_features(X)
    dw = dow_features(dow)
    hf = hour_features(hour)
    Xlin = np.column_stack([lag, dw])
    out = {}
    for f in (0.25, 0.50, 0.75):
        sub = fit_idx[: max(30, int(len(fit_idx) * f))]
        out[f"vintage_ridge_{int(f * 100)}"] = (
            Ridge(alpha=1.0).fit(Xlin[sub], Y[sub]).predict(Xlin[target_idx]))
    fp = Fingerprint(K=32).fit(X[fit_idx])
    PHI = fp.transform(X)
    km = KMeans(n_clusters=n_regimes, n_init=5, random_state=seed).fit(PHI[fit_idx])
    for c in range(n_regimes):
        sub = fit_idx[km.labels_ == c]
        if len(sub) < 40:
            sub = fit_idx
        out[f"regime_ridge_{c}"] = (
            Ridge(alpha=1.0).fit(Xlin[sub], Y[sub]).predict(Xlin[target_idx]))
        g = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.06, max_depth=6, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.12,
            random_state=seed).fit(row_features(lag[sub], dw[sub], hf),
                                   Y[sub].reshape(-1))
        out[f"regime_hgb_{c}"] = g.predict(
            row_features(lag[target_idx], dw[target_idx], hf)).reshape(-1, H)
    return out


def fit_predict(X, Y, dow, hour, fit_idx, target_idx, seed):
    """Fit every expert on `fit_idx`, forecast `target_idx` only."""
    names = list(EXPERT_NAMES)
    cols = []
    for name in names:
        s = seed if name in {"random_forest", "hist_gbm", "mlp"} else 0
        e = Expert(name, seed=s).fit(X[fit_idx], Y[fit_idx], dow[fit_idx], hour)
        cols.append(e.predict(X[target_idx], dow[target_idx], hour))
    extra = _hetero_fixed(X, Y, dow, hour, fit_idx, target_idx, seed)
    for k in sorted(extra):
        cols.append(extra[k])
        names.append(k)
    return np.stack(cols, axis=1).astype(np.float64), names


def build(Xn, Yn, dow, hour, bi, seed):
    D = len(Yn)
    blocks = {}
    # 1) pool fitted on the fit block -> warm block
    Pw, names = fit_predict(Xn, Yn, dow, hour, bi["fit"], bi["warm"], seed)
    # 2) pool fitted on fit+warm -> val block
    fw = np.concatenate([bi["fit"], bi["warm"]])
    Pv, _ = fit_predict(Xn, Yn, dow, hour, fw, bi["val"], seed)
    # 3) pool fitted on fit+warm+val -> test block
    fwv = np.concatenate([fw, bi["val"]])
    Pt, _ = fit_predict(Xn, Yn, dow, hour, fwv, bi["test"], seed)
    M = Pw.shape[1]
    P = np.full((D, M, Yn.shape[1]), np.nan)
    for idx, blk in ((bi["warm"], Pw), (bi["val"], Pv), (bi["test"], Pt)):
        P[idx] = blk
    blocks["stage_sizes"] = dict(warm=len(bi["warm"]), val=len(bi["val"]),
                                 test=len(bi["test"]))
    return P, names, blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=DOMAIN_ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(POOL_SEEDS))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for dom in a.domains:
        t0 = time.time()
        d = load_domain(dom)
        bi = block_indices(d["n_days"])
        mu = float(np.mean(d["Y"][bi["fit"]]))
        sd = float(np.std(d["Y"][bi["fit"]]))
        sd = sd if sd > 1e-8 else 1.0
        Xn = (d["X"] - mu) / sd
        Yn = (d["Y"] - mu) / sd
        print(f"[{dom}] days={d['n_days']} fit/warm/val/test="
              f"{len(bi['fit'])}/{len(bi['warm'])}/{len(bi['val'])}/{len(bi['test'])}",
              flush=True)
        for s in a.seeds:
            P, names, _ = build(Xn, Yn, d["dow"], d["hour"], bi, s)
            path = os.path.join(a.out, f"{dom}__seed{s}.npz")
            np.savez_compressed(
                path, P=P, names=np.array(names), X=Xn, Y=Yn, dow=d["dow"],
                hour=d["hour"], mu=mu, sd=sd, family=d["family"],
                idx_fit=bi["fit"], idx_warm=bi["warm"],
                idx_val=bi["val"], idx_test=bi["test"],
                # legacy alias: the warm-up block plays the role the old
                # `train` split used to play, but with out-of-sample forecasts
                idx_train=bi["warm"])
            oos = np.concatenate([bi["warm"], bi["val"], bi["test"]])
            mse = np.nanmean((P[oos] - Yn[oos][:, None, :]) ** 2, axis=(0, 2))
            print(f"    seed {s}: out-of-sample expert MSE "
                  f"{np.round(mse, 4)}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

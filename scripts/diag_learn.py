"""Is the per-day choice of expert learnable at all?

For every benchmark this script asks one question, using only a clean
out-of-sample pool and only features that are observable *before* day t:

    can a regularised linear map from observable features predict which expert
    will have the lowest loss on day t?

If it can, a router can capture part of the oracle headroom and the whole
programme is worth pursuing.  If it cannot, then the headroom is real but
causally unreachable, and no amount of architecture will help.

Features (all known at the end of day t-1)
    * the spectral fingerprint phi_t of the input window
    * the log per-expert loss on the previous day
    * the exponential moving average of the log per-expert loss (halflife 7 and 28)
    * a constant

Target
    the day-normalised per-expert loss  y[t, m] = lm[t, m] / median_m lm[t, m]
    (normalising by the day removes the trivial "hard day" component)

Fitting is done on warm+val only; everything reported is test-set.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.data import ALL_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.protocol import build_envelope  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def features(phi, loglm):
    """(D, p + 2M + 1) observable feature matrix, causal by construction."""
    D, M = loglm.shape
    e7 = np.zeros_like(loglm)
    e28 = np.zeros_like(loglm)
    for t in range(1, D):
        e7[t] = 0.906 * e7[t - 1] + 0.094 * loglm[t - 1]
        e28[t] = 0.976 * e28[t - 1] + 0.024 * loglm[t - 1]
    lag = np.vstack([np.zeros((1, M)), loglm[:-1]])
    return np.column_stack([phi, lag, e7, e28, np.ones(D)])


def ridge_fit(X, Y, lam=1.0):
    d = X.shape[1]
    A = X.T @ X + lam * np.trace(X.T @ X) / d * np.eye(d)
    return np.linalg.solve(A, X.T @ Y)


def main(seed=7):
    print(f"{'domain':<10}{'corr':>8}{'ridge':>10}{'const':>10}{'bestfix':>10}"
          f"{'oracle':>10}{'captured%':>11}")
    for dom in ALL_ORDER:
        p = os.path.join(POOL, f"{dom}__seed{seed}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        P, Y, X, dow = z["P"], z["Y"], z["X"], z["dow"]
        warm, va, te = z["idx_warm"], z["idx_val"], z["idx_test"]
        M = P.shape[1]
        fit_tr = np.concatenate([warm, va])
        fp = Fingerprint(K=32).fit(X[fit_tr], dow[fit_tr])
        phi = fp.transform(X, dow)

        lm = np.full((len(Y), M), np.nan)
        oos = np.concatenate([warm, va, te])
        lm[oos] = np.array([[np.mean((P[t][m] - Y[t]) ** 2) for m in range(M)]
                            for t in oos])
        med = np.nanmedian(lm, axis=1)
        Yn = lm / med[:, None]

        loglm = np.log(np.maximum(lm, 1e-12))
        F = features(phi, loglm)

        Wm = ridge_fit(F[fit_tr], Yn[fit_tr])
        pred = F[te] @ Wm
        truth = Yn[te]
        corr = float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1])

        # losses (winsorised, arm independent) of four policies on the test days
        capd = build_envelope(P, Y, oos)
        caps = np.array([capd[int(t)] for t in te])
        raw = lm[te]

        def score(idx):
            return float(np.mean(np.minimum(raw[np.arange(len(te)), idx], caps)))

        ridge_pick = np.argmin(pred, axis=1)
        const_pick = np.full(len(te), int(np.argmin(np.nanmean(Yn[fit_tr], axis=0))))
        best_fix = int(np.argmin(raw.mean(axis=0)))
        fix_pick = np.full(len(te), best_fix)
        oracle = np.argmin(raw, axis=1)

        r, c, bf, o = score(ridge_pick), score(const_pick), score(fix_pick), score(oracle)
        cap = 100 * (bf - r) / max(1e-12, bf - o)
        print(f"{dom:<10}{corr:>8.3f}{r:>10.4f}{c:>10.4f}{bf:>10.4f}{o:>10.4f}"
              f"{cap:>11.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

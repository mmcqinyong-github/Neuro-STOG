"""Causal routers, with the gate driven by *predicted* rather than past loss.

`diag_learn.py` establishes that the per-expert loss on day t is predictable
from features observable at the end of day t-1 (test-set correlation up to 0.85
on the ladder streams, positive everywhere) -- but that a hard `argmin` of the
prediction is a bad decision rule, because the extreme of a noisy prediction is
dominated by noise.  This script replaces the hard pick with a soft, tempered
gate and asks whether *that* beats the classical routers.

Everything is causal: the prediction map is fitted on warm+val, the temperature
is tuned on val, and only test days are scored.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.combo import simplex_ls  # noqa: E402
from cerebrocast.data import ALL_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.protocol import build_envelope  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def softmax(v):
    v = v - v.max()
    e = np.exp(np.clip(v, -60, 0))
    return e / e.sum()


def causal_features(phi, lm, oos, D):
    """Observable feature matrix; the fit block is never used."""
    M = lm.shape[1]
    loglm = np.zeros((D, M))
    idx = {int(t): i for i, t in enumerate(oos)}
    loglm[oos] = np.log(np.maximum(lm[oos], 1e-12))
    lag = np.zeros((D, M))
    e7 = np.zeros((D, M))
    e28 = np.zeros((D, M))
    prev = None
    for i, t in enumerate(oos):
        if i == 0:
            lag[t] = loglm[t]
        else:
            lag[t] = loglm[oos[i - 1]]
            e7[t] = 0.906 * e7[oos[i - 1]] + 0.094 * loglm[oos[i - 1]]
            e28[t] = 0.976 * e28[oos[i - 1]] + 0.024 * loglm[oos[i - 1]]
    return np.column_stack([phi, lag, e7, e28, np.ones(D)])


def ridge_fit(X, Y, lam=1.0):
    d = X.shape[1]
    A = X.T @ X + lam * np.trace(X.T @ X) / d * np.eye(d)
    return np.linalg.solve(A, X.T @ Y)


def run_domain(dom, seed=7, verbose=False):
    z = np.load(os.path.join(POOL, f"{dom}__seed{seed}.npz"), allow_pickle=True)
    P, Y, X, dow = z["P"], z["Y"], z["X"], z["dow"]
    warm, va, te = z["idx_warm"], z["idx_val"], z["idx_test"]
    D, M = len(Y), P.shape[1]
    oos = np.concatenate([warm, va, te])
    fit_tr = np.concatenate([warm, va])

    fp = Fingerprint(K=32).fit(X[fit_tr], dow[fit_tr])
    phi = fp.transform(X, dow)

    lm = np.full((D, M), np.nan)
    lm[oos] = np.array([[np.mean((P[t][m] - Y[t]) ** 2) for m in range(M)] for t in oos])
    med = np.nanmedian(lm, axis=1)
    Yn = lm / med[:, None]
    F = causal_features(phi, lm, oos, D)

    Wm = ridge_fit(F[fit_tr], Yn[fit_tr])
    pred = F @ Wm                       # predicted normalised loss, all days

    capd = build_envelope(P, Y, oos)
    caps = np.array([capd[int(t)] for t in oos])
    pos = {int(t): i for i, t in enumerate(oos)}
    raw = lm[oos]

    def wk(lams, days):
        """Winsorised mean loss of a weight sequence aligned with `days`."""
        p = np.array([pos[int(t)] for t in days])
        lo = np.einsum("tm,tmh->th", lams, P[oos][p]) - Y[oos][p]
        l = np.mean(lo ** 2, axis=1)
        return float(np.mean(np.minimum(l, caps[p])))

    def hedge_seq(scores, eta, days, ref=None):
        w = np.full(M, 1.0 / M)
        out = []
        for t in days:
            out.append(w)
            i = pos[int(t)]
            r = med[int(t)] if ref is None else ref
            inc = np.clip(scores[i] / max(1e-9, r), -5.0, 5.0)
            w = w * np.exp(-eta * inc)
            s = w.sum()
            w = w / s if (np.isfinite(s) and s > 0) else np.full(M, 1.0 / M)
        return np.array(out)

    def pure_gate(scores, eta, days):
        """No memory: weights depend only on today's predicted quality."""
        return np.array([softmax(-eta * np.clip(scores[pos[int(t)]], -8, 8))
                         for t in days])

    def rwls_seq(hl, days):
        rho = 0.5 ** (1.0 / hl)
        G = np.zeros((M, M)); h = np.zeros(M); n = 0.0
        w = np.full(M, 1.0 / M)
        out = []
        for t in days:
            out.append(w)
            i = pos[int(t)]
            wt = 1.0 / max(1e-9, med[int(t)])
            Pp = P[oos][i]
            G = rho * G + wt * (Pp @ Pp.T)
            h = rho * h + wt * (Pp @ Y[oos][i])
            n = rho * n + wt
            if n > 1e-8:
                s = simplex_ls(G / n, h / n, 1e-2, 0.0)
                if s is not None:
                    w = s
        return np.array(out)

    def static_seq(days):
        from cerebrocast.baselines import _simplex_fit
        s = _simplex_fit(P[va], Y[va])
        return np.tile(s, (len(days), 1))

    def tune(fn, grid, days):
        best, bv = None, np.inf
        for g in grid:
            v = wk(fn(g, days), days)
            if v < bv:
                best, bv = g, v
        return best

    Ynz = np.nan_to_num(Yn)
    # day-standardised scores: only the *relative* quality of experts matters
    def std_rows(A):
        mu = A.mean(axis=1, keepdims=True)
        sd = A.std(axis=1, keepdims=True) + 1e-9
        return np.clip((A - mu) / sd, -8, 8)

    Z_real = std_rows(Ynz)
    Z_pred = std_rows(np.nan_to_num(pred))

    out = {}
    mean_fit = np.array([np.mean([np.mean((P[t][m] - Y[t]) ** 2) for t in fit_tr])
                         for m in range(M)])
    bf = int(np.argmin(mean_fit))
    out["best_fixed"] = wk(np.tile(np.eye(M)[bf], (len(oos), 1)), oos)
    out["static"] = wk(static_seq(oos), oos)
    hl = tune(rwls_seq, (2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 40.0), va)
    out["rwls"] = wk(rwls_seq(hl, oos), oos)
    eta_r = tune(lambda e, d: hedge_seq(Z_real, e, d), (0.05, 0.2, 0.5, 1.0, 2.0), va)
    out["hedge_real"] = wk(hedge_seq(Z_real, eta_r, oos), oos)
    eta_g = tune(lambda e, d: pure_gate(Z_pred, e, d), (0.25, 0.5, 1.0, 2.0, 4.0), va)
    out["gate_pred"] = wk(pure_gate(Z_pred, eta_g, oos), oos)
    for a in (0.25, 0.5, 1.0):
        eta_m = tune(lambda e, d: hedge_seq(Z_real + a * Z_pred, e, d),
                     (0.05, 0.2, 0.5, 1.0, 2.0), va)
        out[f"mix{a}"] = wk(hedge_seq(Z_real + a * Z_pred, eta_m, oos), oos)
    # reference = champion of the classical causal family
    classic = min(out["best_fixed"], out["static"], out["rwls"], out["hedge_real"])
    out["_bestBL"] = classic
    out["_oracle"] = wk(np.eye(M)[np.argmin(raw, axis=1)], oos)
    out["_hl"] = hl
    out["_eta_r"] = eta_r
    out["_eta_g"] = eta_g
    if verbose:
        print(f"   halflife={hl} eta_real={eta_r} eta_gate={eta_g}")
    return out


def main(seed=7):
    keys = ["best_fixed", "static", "rwls", "hedge_real",
            "gate_pred", "mix0.25", "mix0.5", "mix1.0"]
    print(f"{'domain':<9}{'bestBL':>9}{'oracle':>9}" + "".join(f"{k[:9]:>10}" for k in keys))
    agg = {}
    for dom in ALL_ORDER:
        if not os.path.exists(os.path.join(POOL, f"{dom}__seed{seed}.npz")):
            continue
        r = run_domain(dom, seed)
        bl = r["_bestBL"]
        print(f"{dom:<9}{bl:>9.4f}{r['_oracle']:>9.4f}" +
              "".join(f"{100*(bl-r[k])/bl:>+10.2f}" for k in keys))
        for k in keys:
            agg.setdefault(k, []).append(100 * (bl - r[k]) / bl)
    print()
    for k in keys:
        v = np.array(agg[k])
        print(f"  {k:<14} median {np.median(v):+7.2f}%   mean {np.mean(v):+7.2f}%   "
              f"wins {int((v > 0).sum())}/{len(v)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

"""Normalised recency-weighted simplex least squares.

The earlier recency-LS accumulated the raw second moments, so on the price
domains a handful of spike days determined the whole fit.  Dividing each day's
contribution by that day's own difficulty makes every day count equally, which
is the objective the combination is actually supposed to minimise.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.baselines import (  # noqa: E402
    B1Frozen, B2MonthlyRetrain, B4Hedge, B5BestExpert, B6ADWIN, B7OnlineMeta,
    _simplex_fit)
from cerebrocast.config import SystemConfig  # noqa: E402
from cerebrocast.data import DOMAIN_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.protocol import build_envelope  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def simplex_from(G, h, ridge=1e-2, shrink_u=0.0):
    M = G.shape[0]
    tr = np.trace(G)
    if tr <= 0:
        return np.full(M, 1.0 / M)
    G = G + ridge * tr / M * np.eye(M)
    try:
        L = np.linalg.cholesky(G + 1e-9 * np.eye(M))
    except np.linalg.LinAlgError:
        return np.full(M, 1.0 / M)
    rhs = np.linalg.solve(L, h)
    pen = 1e3 * (np.abs(rhs).mean() + 1e-6)
    A = np.vstack([L.T, pen * np.ones((1, M))])
    b = np.concatenate([rhs, [pen]])
    w, _ = nnls(A, b)
    s = w.sum()
    lam = w / s if s > 1e-9 else np.full(M, 1.0 / M)
    return (1 - shrink_u) * lam + shrink_u / M if shrink_u > 0 else lam


def run(P, Y, hl, wt, ridge=1e-2, shrink_u=0.0):
    """wt[t] is the day weight; the loss recorded for t uses data before t."""
    M, n = P.shape[1], len(P)
    rho = 0.5 ** (1.0 / hl)
    G = np.zeros((M, M)); h = np.zeros(M); nn = 0.0
    lam = np.full(M, 1.0 / M)
    LAM = np.zeros((n, M))
    for t in range(n):
        LAM[t] = lam
        w = wt[t]
        G = rho * G + w * (P[t] @ P[t].T)
        h = rho * h + w * (P[t] @ Y[t])
        nn = rho * nn + w
        if nn < 1e-8:
            continue
        lam = simplex_from(G / nn, h / nn, ridge, shrink_u)
    return LAM


def main(seed=7):
    HL = (10, 20, 40, 60, 90, 150, 250)
    SU = (0.0, 0.15, 0.35)
    out_rows = []
    for dom in DOMAIN_ORDER:
        z = np.load(os.path.join(POOL, f"{dom}__seed{seed}.npz"), allow_pickle=True)
        P, Y, X, dow = z["P"], z["Y"], z["X"], z["dow"]
        idx = {k: z[f"idx_{k}"] for k in ("train", "val", "test")}
        M = int(P.shape[1])
        tr, va, te = idx["train"], idx["val"], idx["test"]
        cfg = SystemConfig(seed=seed)
        fp = Fingerprint(K=cfg.n_spec_bins)
        fp.fit(X[tr], dow[tr])
        PHI = fp.transform(X, dow)
        caps = build_envelope(P, Y, np.concatenate([tr, va, te]))
        capv = np.array([caps[int(t)] for t in range(len(P))])
        lm = np.array([[np.mean((P[t][m] - Y[t]) ** 2) for m in range(M)]
                       for t in range(len(P))])
        scale = np.median(lm, axis=1) + 1e-9

        def loss_of(LAM):
            yhat = np.einsum("tm,tmh->th", LAM, P)
            return np.mean((yhat - Y) ** 2, axis=1)

        def met(lo):
            return (float(lo[te].mean()), float(np.minimum(lo[te], capv[te]).mean()),
                    float((lo[te] / scale[te]).mean()))

        # ---- baselines ----------------------------------------------------
        Pv, Yv = P[va], Y[va]
        bl = {}
        bl["B1_frozen"] = loss_of(np.tile(_simplex_fit(Pv, Yv), (len(P), 1)))
        b2 = B2MonthlyRetrain(M)
        L2 = np.zeros((len(P), M))
        for t in range(len(P)):
            L2[t] = b2.lam
            b2.step(int(t), PHI[t], P[t], Y[t])
        bl["B2_monthly"] = loss_of(L2)
        best = (np.inf, None)
        for e in (0.1, 0.2, 0.5, 1.0, 2.0):
            h4 = B4Hedge(M, eta=e)
            L4 = np.zeros((len(P), M))
            for t in range(len(P)):
                h4.step(int(t), PHI[t], P[t], Y[t])
                L4[t] = h4.lam
            lo = loss_of(L4)
            v = np.minimum(lo[va], capv[va]).mean()
            if v < best[0]:
                best = (v, lo)
        bl["B4_hedge"] = best[1]
        mse_v = np.array([np.mean((Pv[:, m] - Yv) ** 2) for m in range(M)])
        bl["B5_best"] = loss_of(np.tile(B5BestExpert(int(np.argmin(mse_v)), M).lam,
                                        (len(P), 1)))
        b6 = B6ADWIN(M)
        L6 = np.zeros((len(P), M))
        for t in range(len(P)):
            L6[t] = b6.lam
            b6.step(int(t), PHI[t], P[t], Y[t])
        bl["B6_adwin"] = loss_of(L6)
        best7 = (np.inf, None)
        for e in (0.005, 0.02, 0.05, 0.15, 0.4):
            o = B7OnlineMeta(M, PHI.shape[1], eta=e)
            L7 = np.zeros((len(P), M))
            for t in range(len(P)):
                o.step(int(t), PHI[t], P[t], Y[t])
                L7[t] = o.lam
            lo = loss_of(L7)
            v = np.minimum(lo[va], capv[va]).mean()
            if v < best7[0]:
                best7 = (v, lo)
        bl["B7_meta"] = best7[1]
        bl["_uniform"] = loss_of(np.tile(np.full(M, 1.0 / M), (len(P), 1)))

        # ---- day weights ---------------------------------------------------
        WT = {
            "raw": np.ones(len(P)),
            "norm": 1.0 / scale,
            "norm2": 1.0 / (np.mean(Y ** 2, axis=1) + 1e-9),
        }
        res = {}
        for wn, wt in WT.items():
            cands = {}
            for hl in HL:
                for su in SU:
                    lo = loss_of(run(P, Y, hl, wt, shrink_u=su))
                    v = np.minimum(lo[va], capv[va]).mean()
                    cands[(hl, su)] = (v, lo)
            kb = min(cands, key=lambda k: cands[k][0])
            res[wn] = (cands[kb][1], kb)

        row = dict(dom=dom, bl={k: met(v) for k, v in bl.items()},
                   ls={k: (met(v[0]), v[1]) for k, v in res.items()})
        out_rows.append(row)
        print(f"{dom:<9} bestBL(skill)={min(row['bl'][k][2] for k in row['bl'] if k.startswith('B')):.5f} "
              f"| raw={row['ls']['raw'][0][2]:.5f}{row['ls']['raw'][1]} "
              f"norm={row['ls']['norm'][0][2]:.5f}{row['ls']['norm'][1]} "
              f"norm2={row['ls']['norm2'][0][2]:.5f}{row['ls']['norm2'][1]}", flush=True)

    print("\n=== gains of normalised recency-LS over BEST baseline ===")
    print(f"{'domain':<9} {'bestBL':>8} {'raw%':>7} {'norm%':>7} {'norm2%':>7}   (skill metric)")
    gs = {"raw": [], "norm": [], "norm2": []}
    for r in out_rows:
        bb = min(r["bl"][k][2] for k in r["bl"] if k.startswith("B"))
        vals = {}
        for wn in ("raw", "norm", "norm2"):
            g = 100 * (bb - r["ls"][wn][0][2]) / bb
            vals[wn] = g
            gs[wn].append(g)
        print(f"{r['dom']:<9} {bb:>8.5f} {vals['raw']:>+7.2f} {vals['norm']:>+7.2f} "
              f"{vals['norm2']:>+7.2f}")
    for wn in ("raw", "norm", "norm2"):
        g = gs[wn]
        print(f"  {wn:<6} median={np.median(g):+.2f}% mean={np.mean(g):+.2f}% "
              f"n>=8%: {sum(1 for x in g if x >= 8)}/{len(g)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

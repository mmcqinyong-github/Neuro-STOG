"""Which metric should be primary?

On the price domains the worst 1% of test days carry 25-65% of the total MSE,
so raw MSE is a lottery ticket rather than a measurement.  Compare three
arm-independent metrics on the same runs:

  mse    : mean raw daily squared error
  win    : mean daily loss truncated at the shared rolling envelope
  skill  : mean of  loss_t / median_m l(t,m)   -- day-normalised, MASE-like
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.baselines import (  # noqa: E402
    B1Frozen, B2MonthlyRetrain, B3FFORMA, B4Hedge, B5BestExpert, B6ADWIN,
    B7OnlineMeta, _simplex_fit)
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


def run_ls(P, Y, days, hl, ridge=1e-2, shrink_u=0.0):
    M = P.shape[1]
    rho = 0.5 ** (1.0 / hl)
    G = np.zeros((M, M)); h = np.zeros(M); n = 0.0
    lam = np.full(M, 1.0 / M)
    out = np.zeros((len(P), M))
    for t in range(len(P)):
        out[t] = lam
        G = rho * G + P[t] @ P[t].T
        h = rho * h + P[t] @ Y[t]
        n = rho * n + 1.0
        if n < 12:
            continue
        lam = simplex_from(G / n, h / n, ridge, shrink_u)
    return out


def main(seed=7):
    HL = (20, 40, 60, 90, 150, 250)
    rows = []
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
        scale = np.median(lm, axis=1) + 1e-9        # day-normalisation, arm free

        def met(losses):
            """losses: array over ALL days -> (mse, win, skill) on test"""
            lt = losses[te]
            return (float(lt.mean()),
                    float(np.minimum(lt, capv[te]).mean()),
                    float((lt / scale[te]).mean()))

        LAMS = {}

        def reg(name, lam_of_t):
            yhat = np.einsum("tm,tmh->th", lam_of_t, P)
            LAMS[name] = np.mean((yhat - Y) ** 2, axis=1)

        Pv, Yv = P[va], Y[va]
        lam_static = _simplex_fit(Pv, Yv)
        reg("B1_frozen", np.tile(lam_static, (len(P), 1)))

        b2 = B2MonthlyRetrain(M)
        L2 = np.zeros((len(P), M))
        for t in range(len(P)):
            L2[t] = b2.lam
            b2.step(int(t), PHI[t], P[t], Y[t])
        reg("B2_monthly", L2)

        best = (np.inf, None)
        for e in (0.1, 0.2, 0.5, 1.0, 2.0):
            h4 = B4Hedge(M, eta=e)
            out = np.zeros((len(P), M))
            for t in range(len(P)):
                h4.step(int(t), PHI[t], P[t], Y[t])
                out[t] = h4.lam
            v = np.minimum(np.einsum("tm,tmh->th", out, P).__sub__(Y).__pow__(2).mean(1)[va],
                           capv[va]).mean()
            if v < best[0]:
                best = (v, out)
        reg("B4_hedge", best[1])

        b3 = B3FFORMA(M, PHI.shape[1])
        L3 = np.zeros((len(P), M))
        for t in range(len(P)):
            b3.step(int(t), PHI[t], P[t], Y[t])
            L3[t] = b3.lam
        reg("B3_fforma", L3)
        mse_v = np.array([np.mean((Pv[:, m] - Yv) ** 2) for m in range(M)])
        b5 = B5BestExpert(int(np.argmin(mse_v)), M)
        reg("B5_best", np.tile(b5.lam, (len(P), 1)))
        b6 = B6ADWIN(M)
        L6 = np.zeros((len(P), M))
        for t in range(len(P)):
            L6[t] = b6.lam
            b6.step(int(t), PHI[t], P[t], Y[t])
        L6[va[0]:] = L6[va[0]:]
        reg("B6_adwin", L6)
        best7 = (np.inf, None)
        for e in (0.005, 0.02, 0.05, 0.15, 0.4):
            o = B7OnlineMeta(M, PHI.shape[1], eta=e)
            out = np.zeros((len(P), M))
            for t in range(len(P)):
                o.step(int(t), PHI[t], P[t], Y[t])
                out[t] = o.lam
            v = np.minimum(np.einsum("tm,tmh->th", out, P).__sub__(Y).__pow__(2).mean(1)[va],
                           capv[va]).mean()
            if v < best7[0]:
                best7 = (v, out)
        reg("B7_meta", best7[1])
        reg("_uniform", np.tile(np.full(M, 1.0 / M), (len(P), 1)))

        # recency LS: half-life and uniform shrinkage picked on VAL
        cands = {}
        for hl in HL:
            for su in (0.0, 0.15, 0.35):
                LAM = run_ls(P, Y, range(len(P)), hl, shrink_u=su)
                lo = np.einsum("tm,tmh->th", LAM, P)
                lo = np.mean((lo - Y) ** 2, axis=1)
                v = np.minimum(lo[va], capv[va]).mean()
                cands[(hl, su)] = (v, lo)
        kbest = min(cands, key=lambda k: cands[k][0])
        LAMS["LS_recency"] = cands[kbest][1]
        rows.append((dom, {k: met(v) for k, v in LAMS.items()}, kbest))

    for mi, mname in enumerate(("mse", "win", "skill")):
        print(f"\n=== metric: {mname} ===")
        keys = ["B1_frozen", "B2_monthly", "B3_fforma", "B4_hedge", "B5_best",
                "B6_adwin", "B7_meta", "_uniform", "LS_recency"]
        print(f"{'domain':<9} " + " ".join(f"{k[:8]:>8}" for k in keys) + "  bestBL   LSgain%")
        med = []
        for dom, m, kb in rows:
            vals = [m[k][mi] for k in keys]
            bb = min(m[k][mi] for k in keys if k.startswith("B"))
            g = 100 * (bb - m["LS_recency"][mi]) / bb
            med.append(g)
            print(f"{dom:<9} " + " ".join(f"{v:>8.5f}" for v in vals) +
                  f"  {bb:>8.5f} {g:>+8.2f}   cfg={kb}")
        print(f"  median LS gain: {np.median(med):+.2f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

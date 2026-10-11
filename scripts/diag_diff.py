"""Is the *advantage of being adaptive* predictable?

For each test day t we have, causally, two combinations that were both available
before y_t was revealed:

    lam_long   simplex-LS refit monthly on a long trailing window (B2-like)
    lam_short  simplex-LS on an exponentially weighted short window (hl ~ 5)

Question: is   d_t = loss_short(t) - loss_long(t)   predictable from features
visible at t?  If yes, a gate can harvest it; if no, adaptivity has nothing to
bite on in these data and the honest answer is a negative result.

Also reported:
    oracle switch  - always take the better of the two (non-causal upper bound
                     on any gating mechanism that mixes the two)
    always short / always long
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.combo import simplex_ls  # noqa: E402
from cerebrocast.config import SystemConfig  # noqa: E402
from cerebrocast.data import DOMAIN_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.protocol import build_envelope  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def main(seed=7, doms=DOMAIN_ORDER):
    print(f"{'domain':<9} {'long':>9} {'short':>9} {'oracle':>9} {'harvest':>9} "
          f"{'ridgeSign':>10} {'frac_short_better':>18}")
    HR = []
    for dom in doms:
        z = np.load(os.path.join(POOL, f"{dom}__seed{seed}.npz"), allow_pickle=True)
        P, Y, X, dow = z["P"], z["Y"], z["X"], z["dow"]
        idx = {k: z[f"idx_{k}"] for k in ("train", "val", "test")}
        M = int(P.shape[1])
        tr, va, te = idx["train"], idx["val"], idx["test"]
        cfg0 = SystemConfig(seed=seed)
        fp = Fingerprint(K=cfg0.n_spec_bins)
        fp.fit(X[tr], dow[tr])
        PHI = fp.transform(X, dow)
        capd = build_envelope(P, Y, np.concatenate([tr, va, te]))
        caps = np.array([capd[int(t)] for t in range(len(P))])
        lm = np.array([[np.mean((P[t][m] - Y[t]) ** 2) for m in range(M)]
                       for t in range(len(P))])

        # ---- long window: refit on a trailing window of 180 days, monthly
        def long_lam():
            out = {}
            G = np.zeros((M, M)); h = np.zeros(M); n = 0.0
            lam = np.full(M, 1.0 / M); last = -10 ** 9
            buf = []
            for t in range(len(P)):
                out[int(t)] = lam.copy()
                buf.append(int(t))
                if len(buf) > 180:
                    buf.pop(0)
                if t - last >= 30 and len(buf) >= 30:
                    last = t
                    A = P[buf].reshape(-1, P.shape[2]).T if False else None
                    G = np.zeros((M, M)); h = np.zeros(M)
                    for s in buf:
                        G += P[s] @ P[s].T
                        h += P[s] @ Y[s]
                    n = float(len(buf))
                    s_ = simplex_ls(G / n, h / n, 1e-2, 0.0)
                    if s_ is not None:
                        lam = s_
            return out

        # ---- short window: exponential, half-life hl
        def short_lam(hl):
            out = {}
            rho = 0.5 ** (1.0 / hl)
            G = np.zeros((M, M)); h = np.zeros(M); n = 0.0
            lam = np.full(M, 1.0 / M)
            for t in range(len(P)):
                out[int(t)] = lam.copy()
                w = 1.0 / (np.median(lm[t]) + 1e-9)
                G = rho * G + w * (P[t] @ P[t].T)
                h = rho * h + w * (P[t] @ Y[t])
                n = rho * n + w
                if n > 1e-8:
                    s_ = simplex_ls(G / n, h / n, 1e-2, 0.0)
                    if s_ is not None:
                        lam = s_
            return out

        LL = long_lam()
        # hl chosen on validation
        best = (np.inf, None)
        for hl in (3.0, 5.0, 7.0, 10.0, 15.0):
            SL = short_lam(hl)
            lv = np.array([np.mean((SL[int(t)] @ P[t] - Y[t]) ** 2) for t in va])
            v = float(np.mean(np.minimum(lv, caps[va])))
            if v < best[0]:
                best = (v, hl)
            # keep the last for reuse below
            if hl == 5.0:
                SL5 = SL
        hl = best[1]
        SL = short_lam(hl)

        def loss_of(LAM, days):
            lo = np.array([np.mean((LAM[int(t)] @ P[t] - Y[t]) ** 2) for t in days])
            return np.minimum(lo, caps[days]), lo

        cl, _ = loss_of(LL, te)
        cs, _ = loss_of(SL, te)
        s_long, s_short = float(cl.mean()), float(cs.mean())
        s_or = float(np.minimum(cl, cs).mean())
        frac = float(np.mean(cs < cl))

        # ---- predictable?  ridge on  sign/shape of the differential
        # feature: fingerprint + recent loss history statistics (all causal)
        hist = []
        rec7 = np.zeros(M); rec28 = np.zeros(M)
        for t in range(len(P)):
            hist.append(np.concatenate([rec7 / (np.median(lm[t]) + 1e-9),
                                        rec28 / (np.median(lm[t]) + 1e-9),
                                        [float(np.mean(np.abs(P[t] - P[t].mean(0))))]]))
            rec7 = 0.85 * rec7 + 0.15 * lm[t]
            rec28 = 0.96 * rec28 + 0.04 * lm[t]
        H = np.array(hist)
        F = np.concatenate([PHI, H], axis=1)
        d = cs - cl                       # >0 : long better
        fit = np.concatenate([tr, va])
        Atr, Btr = F[fit], d[fit] if False else None
        # build the differential target for every day using causal lams
        dall = np.array([np.mean((SL[int(t)] @ P[t] - Y[t]) ** 2)
                         - np.mean((LL[int(t)] @ P[t] - Y[t]) ** 2)
                         for t in range(len(P))])
        dall = np.minimum(np.abs(dall), caps) * np.sign(dall)
        bestv = np.inf
        for lam_ in (1e-2, 1e-1, 1.0, 10.0, 100.0):
            d_ = F.shape[1]
            Wm = np.linalg.solve(F[fit].T @ F[fit] + lam_ * np.eye(d_),
                                 F[fit].T @ dall[fit])
            pr = F[te] @ Wm
            pick = pr < 0                      # predict short better
            v = float(np.mean(np.where(pick, cs, cl)))
            bestv = min(bestv, v)
        harvest = 100 * (min(s_long, s_short) - bestv) / min(s_long, s_short)

        HR.append(harvest)
        print(f"{dom:<9} {s_long:9.5f} {s_short:9.5f} {s_or:9.5f} "
              f"{harvest:8.2f}% {bestv:10.5f} {frac:18.3f}", flush=True)
    print(f"\nmedian harvestable by a perfect-feature gate: {np.median(HR):+.2f}%")


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    d = tuple(sys.argv[2:]) or DOMAIN_ORDER
    main(s, d)

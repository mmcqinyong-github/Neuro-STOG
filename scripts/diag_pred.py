"""How much of the per-day selection headroom is actually *learnable* from
features that are visible before the day's outcome is known?

For every domain we train, on the training split only, a predictor

    phi_t  ->  l_{t,m}   (the loss expert m would incur on day t)

and then route on the test split with the predicted losses.  Comparing

    oracle (uses the true l_t)   >=   learned   >=   best fixed expert

tells us whether the architecture's forward model has anything to bite on.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.config import SystemConfig  # noqa: E402
from cerebrocast.data import DOMAIN_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.protocol import build_envelope  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def ridge_fit(A, B, lam):
    """Solve A W = B in the least-squares sense with Tikhonov regularisation."""
    d = A.shape[1]
    G = A.T @ A + lam * np.eye(d)
    return np.linalg.solve(G, A.T @ B)


def main(seed=7, doms=DOMAIN_ORDER):
    print(f"{'domain':<10} {'best1':>8} {'top1orac':>9} {'oracle':>8} "
          f"{'ridge':>8} {'gbm':>8} | gain_ridge gain_gbm  (vs best1)")
    gr, gg = [], []
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
        scale = np.median(lm, axis=1) + 1e-9
        # predict the *relative* loss: within-day centred, scaled by day difficulty
        T = (lm - lm.mean(axis=1, keepdims=True)) / scale[:, None]

        fitidx = np.concatenate([tr, va])
        Atr, Btr = PHI[fitidx], T[fitidx]
        Ate = PHI[te]

        # ---- baselines on the test split
        def skill_of(pick):
            lo = np.array([lm[te[i]][pick[i]] for i in range(len(te))])
            return float(np.mean(np.minimum(lo, caps[te]) / scale[te]))

        fixed = np.array([np.mean(np.minimum(lm[tr][:, m], caps[tr]) / scale[tr])
                          for m in range(M)])
        best1 = int(np.argmin(fixed))
        s_fixed = float(np.mean(np.minimum(lm[te][:, best1], caps[te]) / scale[te]))
        s_orac = skill_of(np.argmin(lm[te], axis=1))

        res = {}
        for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            W = ridge_fit(Atr, Btr, lam)
            res[lam] = skill_of(np.argmin(Ate @ W, axis=1))
        s_ridge = min(res.values())

        s_gbm = np.inf
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            g = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                              max_leaf_nodes=15,
                                              random_state=seed)
            g.fit(Atr, Btr)
            s_gbm = skill_of(np.argmin(g.predict(Ate), axis=1))
        except Exception as e:  # pragma: no cover
            print("   gbm failed:", e)

        g1 = 100 * (s_fixed - s_ridge) / s_fixed
        g2 = 100 * (s_fixed - s_gbm) / s_fixed
        gr.append(g1); gg.append(g2)
        print(f"{dom:<10} {s_fixed:8.4f} {s_orac:9.4f} {s_orac:8.4f} "
              f"{s_ridge:8.4f} {s_gbm:8.4f} | {g1:+7.2f}% {g2:+7.2f}%", flush=True)
    print(f"\nmedian gain ridge {np.median(gr):+.2f}%   gbm {np.median(gg):+.2f}%")


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    d = tuple(sys.argv[2:]) or DOMAIN_ORDER
    main(s, d)

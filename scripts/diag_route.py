"""How much of the 46-73% selection headroom is reachable *causally*?

The per-day oracle is not, but a router that (i) normalises each day's expert
losses by the day's own difficulty, (ii) remembers which experts have recently
been good, and (iii) uses the spectral fingerprint as a static route should get
a useful fraction.  Everything here is calibrated on the validation split only
and scored under the shared winsorised envelope.

Fully vectorised: the score matrix is assembled for all days at once, so a whole
grid costs one einsum per configuration rather than one Python loop per day.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.config import SystemConfig  # noqa: E402
from cerebrocast.data import DOMAIN_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.protocol import build_envelope, score  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def softmax_rows(X):
    X = X - X.max(axis=1, keepdims=True)
    E = np.exp(X)
    return E / (E.sum(axis=1, keepdims=True) + 1e-12)


def topk_norm_rows(L, k, eps=0.02):
    L = np.maximum(L, 0.0)
    if k is not None and k < L.shape[1]:
        idx = np.argpartition(-L, k - 1, axis=1)[:, :k]
        m = np.zeros_like(L)
        np.put_along_axis(m, idx, np.take_along_axis(L, idx, axis=1), axis=1)
        L = m
    s = L.sum(axis=1, keepdims=True)
    lam = np.where(s > 1e-12, L / np.maximum(s, 1e-12), 1.0 / L.shape[1])
    return (1 - eps) * lam + eps / L.shape[1]


def main(seed=7):
    TAUS = (0.05, 0.1, 0.2, 0.4, 0.8)
    HLS = (3, 7, 15, 30, 60, 120)
    ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
    KS = (None, 5, 9)
    RIDGES = (0.1, 1.0, 10.0)

    summary = []
    for dom in DOMAIN_ORDER:
        z = np.load(os.path.join(POOL, f"{dom}__seed{seed}.npz"), allow_pickle=True)
        P, Y, X, dow = z["P"], z["Y"], z["X"], z["dow"]
        idx = {k: z[f"idx_{k}"] for k in ("train", "val", "test")}
        M, n = int(P.shape[1]), len(P)
        tr, va, te = idx["train"], idx["val"], idx["test"]
        cfg = SystemConfig(seed=seed)
        fp = Fingerprint(K=cfg.n_spec_bins)
        fp.fit(X[tr], dow[tr])
        PHI = fp.transform(X, dow)
        caps = build_envelope(P, Y, np.concatenate([tr, va, te]))
        capv = np.array([caps[int(t)] for t in range(n)])

        lm = np.array([[np.mean((P[t][m] - Y[t]) ** 2) for m in range(M)]
                       for t in range(n)])
        rel = lm / (lm.mean(axis=1, keepdims=True) + 1e-12)

        Z = np.concatenate([PHI, np.ones((n, 1))], axis=1)
        STAT = {}
        for rg in RIDGES:
            A = Z[tr].T @ Z[tr] + rg * np.eye(Z.shape[1])
            V = np.linalg.solve(A, Z[tr].T @ rel[tr]).T
            STAT[rg] = Z @ V.T                                   # (n, M)

        REC = {}
        for hl in HLS:
            rho = 0.5 ** (1.0 / hl)
            A = np.zeros((n, M))
            cur = np.zeros(M)
            for t in range(n):
                A[t] = cur
                cur = rho * cur + (1 - rho) * rel[t]
            REC[hl] = A

        def losses(sc):
            LAM = topk_norm_rows(softmax_rows(-sc / 1.0), None)
            return LAM

        def eval_cfg(tau, hl, alpha, rg, k):
            sc = alpha * STAT[rg] + (1 - alpha) * REC[hl]
            LAM = topk_norm_rows(softmax_rows(-sc / tau), k)
            yhat = np.einsum("tm,tmh->th", LAM, P)
            return np.mean((yhat - Y) ** 2, axis=1)

        def pick(alphas):
            best = (np.inf, None)
            for tau in TAUS:
                for hl in HLS:
                    for a in alphas:
                        for rg in RIDGES:
                            for k in KS:
                                o = eval_cfg(tau, hl, a, rg, k)
                                v = np.minimum(o[va], capv[va]).mean()
                                if v < best[0]:
                                    best = (v, (tau, hl, a, rg, k))
            return best

        bh = pick(ALPHAS)
        oh = eval_cfg(*bh[1])
        sh = score({int(t): oh[int(t)] for t in te}, caps)
        br = pick((0.0,))
        orc = eval_cfg(*br[1])
        sr = score({int(t): orc[int(t)] for t in te}, caps)
        bs = pick((1.0,))
        ost = eval_cfg(*bs[1])
        ss = score({int(t): ost[int(t)] for t in te}, caps)
        summary.append((dom, sh["win"], bh[1], sr["win"], br[1], ss["win"], bs[1]))
        print(f"{dom:<9} hybrid={sh['win']:.5f} {bh[1]} | rec={sr['win']:.5f} {br[1]} "
              f"| stat={ss['win']:.5f} {bs[1]}", flush=True)

    print("\n=== summary (winsorised test loss) ===")
    print(f"{'domain':<9} {'hybrid':>9} {'recency':>9} {'static':>9}")
    for d, h, c1, r, c2, s, c3 in summary:
        print(f"  {d:<9} {h:>9.5f} {r:>9.5f} {s:>9.5f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

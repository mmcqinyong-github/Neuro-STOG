"""Landscape scan v2: the recency half-life is calibrated on the validation
split only (as the protocol requires), and the spike domains are diagnosed
separately so we know whether the least-squares source needs robustifying."""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.baselines import B1Frozen, B2MonthlyRetrain, B4Hedge, _simplex_fit  # noqa: E402
from cerebrocast.config import SystemConfig  # noqa: E402
from cerebrocast.data import DOMAIN_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def simplex_from(G: np.ndarray, h: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
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
    return w / s if s > 1e-9 else np.full(M, 1.0 / M)


def causal_exp(P, Y, days, half_life, ridge=1e-2, cap=None):
    """Causal simplex LS with exponential recency weighting, daily refit.

    `days` must be contiguous and in chronological order; the state is carried
    from the first day, so call it once over train->val->test.
    """
    M = P.shape[1]
    rho = 0.5 ** (1.0 / half_life)
    G = np.zeros((M, M))
    h = np.zeros(M)
    n = 0.0
    lam = np.full(M, 1.0 / M)
    out = {}
    for t in days:
        t = int(t)
        out[t] = float(np.mean((lam @ P[t] - Y[t]) ** 2))
        G = rho * G + P[t] @ P[t].T
        h = rho * h + P[t] @ Y[t]
        n = rho * n + 1.0
        if n < 20:
            continue
        lam = simplex_from(G / n, h / n, ridge)
        if cap is not None:
            lam = np.minimum(lam, cap)
            s = lam.sum()
            lam = lam / s if s > 1e-12 else np.full(M, 1.0 / M)
    return out


def main(seed=7):
    HL = (10, 20, 40, 60, 90, 150, 250)
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

        def mse_lam(l):
            return float(np.mean([np.mean((l @ P[t] - Y[t]) ** 2) for t in te]))

        Pv, Yv = P[va], Y[va]
        lam_static = _simplex_fit(Pv, Yv)
        b1 = mse_lam(lam_static)
        b2 = B2MonthlyRetrain(M)
        for t in va:
            b2.step(int(t), PHI[t], P[t], Y[t])
        b2m = mse_lam(b2.lam)
        best_b4 = np.inf
        for e in (0.2, 0.5, 1.0, 2.0):
            h4 = B4Hedge(M, eta=e)
            for t in va:
                h4.step(int(t), PHI[t], P[t], Y[t])
            L = [h4.step(int(t), PHI[t], P[t], Y[t])["loss"] for t in te]
            best_b4 = min(best_b4, float(np.mean(L)))
        uni = mse_lam(np.full(M, 1.0 / M))

        # causal LS: state carried through train, half-life picked on VAL
        seq = np.concatenate([tr, va, te])
        per = {hl: causal_exp(P, Y, seq, hl) for hl in HL}
        vm = {hl: np.mean([per[hl][int(t)] for t in va]) for hl in HL}
        hl_star = min(vm, key=vm.get)
        cau = float(np.mean([per[hl_star][int(t)] for t in te]))
        cau_best = min(float(np.mean([per[hl][int(t)] for t in te])) for hl in HL)

        best_bl = min(b1, b2m, best_b4)
        rows.append(dict(dom=dom, nte=len(te), uni=uni, b1=b1, b2=b2m, b4=best_b4,
                         cau=cau, cau_best=cau_best, hl=hl_star,
                         gain=100 * (best_bl - cau) / best_bl,
                         gain_best=100 * (best_bl - cau_best) / best_bl))

    print(f"{'domain':<9} {'nte':>4} | {'uniform':>8} {'B1':>8} {'B2':>8} {'B4':>8} | "
          f"{'cauLS':>8} {'hl*':>4} | {'gain%':>7} {'gainBest%':>9}")
    for r in rows:
        print(f"{r['dom']:<9} {r['nte']:>4} | {r['uni']:>8.5f} {r['b1']:>8.5f} {r['b2']:>8.5f} "
              f"{r['b4']:>8.5f} | {r['cau']:>8.5f} {r['hl']:>4} | {r['gain']:>+7.2f} "
              f"{r['gain_best']:>+9.2f}")
    g = [r["gain"] for r in rows]
    print(f"\nmedian gain (val-calibrated half-life): {np.median(g):+.2f}%   "
          f"mean {np.mean(g):+.2f}%   n>=8%: {sum(1 for x in g if x >= 8)}/{len(g)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

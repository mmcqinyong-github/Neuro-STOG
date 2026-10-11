"""Landscape scan: for every domain, measure the baseline floor, the causal
achievable floor and the non-causal oracle ceiling.  This tells us where the
headroom actually is before we spend time on mechanism design."""
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
from cerebrocast.memory import ClusterField  # noqa: E402

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


def moment(Ps: np.ndarray, Ys: np.ndarray, w: np.ndarray):
    G = np.tensordot(w, Ps @ Ps.transpose(0, 2, 1), axes=(0, 0))
    h = np.tensordot(w, np.einsum("nmh,nh->nm", Ps, Ys), axes=(0, 0))
    return G, h


def causal_exp(P, Y, days_hist, days_eval, half_life, ridge=1e-2,
               topk=None, shrink_uniform=0.0, adapt=None):
    """Causal simplex LS with exponential recency weighting, daily refit."""
    M = P.shape[1]
    rho = 0.5 ** (1.0 / half_life)
    G = np.zeros((M, M))
    h = np.zeros(M)
    n = 0.0
    lam = np.full(M, 1.0 / M)
    ev = set(int(x) for x in days_eval)
    out = []
    for t in range(int(days_hist[0]), int(days_eval[-1]) + 1):
        if t in ev:
            l = lam.copy()
            if topk is not None and topk < M:
                k = np.argpartition(-l, topk - 1)[:topk]
                m = np.zeros(M)
                m[k] = l[k]
                l = m / (m.sum() + 1e-12)
            if shrink_uniform > 0:
                l = (1 - shrink_uniform) * l + shrink_uniform / M
            out.append(float(np.mean((l @ P[t] - Y[t]) ** 2)))
        G = rho * G + P[t] @ P[t].T
        h = rho * h + P[t] @ Y[t]
        n = rho * n + 1.0
        if n < 20:
            continue
        lam = simplex_from(G / max(n, 1e-9), h / max(n, 1e-9), ridge)
    return float(np.mean(out))


def main(seed=7):
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

        f = ClusterField(tau_kappa=cfg.tau_kappa, graph_alpha=cfg.graph_alpha,
                         cluster_momentum=cfg.cluster_momentum, grow_r_cov=cfg.grow_r_cov,
                         grow_persist=cfg.grow_persist, prune_mass=cfg.prune_mass,
                         prune_idle=cfg.prune_idle, k0=cfg.k0_clusters)
        f.init_from_train(PHI[tr], M, seed)
        assign = {t: int(np.argmax(f.affinity(PHI[t]))) for t in range(len(P))}
        C = f.n_clusters()

        def mse_lam(l):
            return float(np.mean([np.mean((l @ P[t] - Y[t]) ** 2) for t in te]))

        # baselines
        Pv, Yv = P[va], Y[va]
        lam_static = _simplex_fit(Pv, Yv)
        b1 = mse_lam(B1Frozen(lam_static).lam)
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

        # causal exponential-forgetting global LS, half-life grid
        hist = np.concatenate([tr, va])
        cands = {}
        for hl in (20, 40, 60, 90, 150):
            cands[hl] = causal_exp(P, Y, hist, np.concatenate([va, te]), hl)
        cbest = min(cands.values())
        chl = min(cands, key=cands.get)

        # oracle global on test
        G, h = moment(P[te], Y[te], np.ones(len(te)))
        lam_or = simplex_from(G / len(te), h / len(te))
        orc = mse_lam(lam_or)

        # oracle per-cluster on test
        lam_pc = {}
        for c in range(C):
            d = [t for t in te if assign[t] == c]
            if len(d) < 8:
                lam_pc[c] = lam_or
                continue
            Gc, hc = moment(P[d], Y[d], np.ones(len(d)))
            lam_pc[c] = simplex_from(Gc / len(d), hc / len(d))
        orc_pc = float(np.mean([np.mean((lam_pc[assign[t]] @ P[t] - Y[t]) ** 2) for t in te]))

        rows.append(dict(dom=dom, M=M, C=C, nte=len(te), uni=uni, b1=b1, b2=b2m,
                         b4=best_b4, cau=cbest, chl=chl, orc=orc, orc_pc=orc_pc))

    print(f"{'domain':<10} {'C':>2} {'nte':>4} | {'uniform':>8} {'B1':>8} {'B2':>8} "
          f"{'B4':>8} | {'causalLS':>9} {'hl':>4} | {'oracG':>8} {'oracPC':>8} | "
          f"{'B2-CAU%':>8} {'CAU-ORC%':>9} {'ORCPC/B2':>9}")
    for r in rows:
        b2_cau = 100 * (r['b2'] - r['cau']) / r['b2']
        cau_orc = 100 * (r['cau'] - r['orc']) / r['cau']
        pc_b2 = 100 * (r['b2'] - r['orc_pc']) / r['b2']
        print(f"{r['dom']:<10} {r['C']:>2} {r['nte']:>4} | {r['uni']:>8.5f} {r['b1']:>8.5f} "
              f"{r['b2']:>8.5f} {r['b4']:>8.5f} | {r['cau']:>9.5f} {r['chl']:>4} | "
              f"{r['orc']:>8.5f} {r['orc_pc']:>8.5f} | {b2_cau:>+8.2f} {cau_orc:>+9.2f} "
              f"{pc_b2:>+9.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

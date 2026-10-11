"""How much is there to win from *selection* at all?

If the identity of the best expert is stable, no router -- brain-inspired or
otherwise -- can beat a frozen combination, and the whole experiment is
unfalsifiable in the sense of F12.  So measure the selection headroom first:

  best-fixed-expert   : min_m  mean_t  l(t,m)
  per-day-oracle      : mean_t min_m   l(t,m)
  static-oracle       : simplex LS fitted on the whole test split
  per-cluster-oracle  : simplex LS fitted per fingerprint cluster on test

and, crucially, how concentrated the day-wise argmin is.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.config import SystemConfig  # noqa: E402
from cerebrocast.data import DOMAIN_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.memory import ClusterField  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def simplex_from(G, h, ridge=1e-2):
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


def main(seed=7):
    print(f"{'domain':<9} {'M':>3} | {'uniform':>8} {'bestFix':>8} {'dayOrac':>8} "
          f"{'statOrac':>9} {'clusOrac':>9} | {'SELhead%':>9} {'COMBhead%':>9} "
          f"{'CLUSgain%':>9} | top3 share")
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

        Pt, Yt = P[te], Y[te]
        lm = np.array([[np.mean((Pt[i][m] - Yt[i]) ** 2) for m in range(M)]
                       for i in range(len(te))])
        uniform = float(np.mean((Pt.mean(axis=1) - Yt) ** 2))
        best_fix = float(lm.mean(axis=0).min())
        day_orac = float(lm.min(axis=1).mean())
        G = np.tensordot(np.ones(len(te)), Pt @ Pt.transpose(0, 2, 1), axes=(0, 0)) / len(te)
        h = np.tensordot(np.ones(len(te)), np.einsum("nmh,nh->nm", Pt, Yt), axes=(0, 0)) / len(te)
        lam_st = simplex_from(G, h)
        stat_orac = float(np.mean([np.mean((lam_st @ Pt[i] - Yt[i]) ** 2)
                                   for i in range(len(te))]))

        f = ClusterField(tau_kappa=cfg.tau_kappa, graph_alpha=cfg.graph_alpha,
                         cluster_momentum=cfg.cluster_momentum, grow_r_cov=cfg.grow_r_cov,
                         grow_persist=cfg.grow_persist, prune_mass=cfg.prune_mass,
                         prune_idle=cfg.prune_idle, k0=cfg.k0_clusters)
        f.init_from_train(PHI[tr], M, seed)
        assign = np.array([int(np.argmax(f.affinity(PHI[t]))) for t in te])
        C = f.n_clusters()
        lam_pc = {}
        for c in range(C):
            d = np.where(assign == c)[0]
            if len(d) < 8:
                lam_pc[c] = lam_st
                continue
            Gc = np.tensordot(np.ones(len(d)), Pt[d] @ Pt[d].transpose(0, 2, 1),
                              axes=(0, 0)) / len(d)
            hc = np.tensordot(np.ones(len(d)), np.einsum("nmh,nh->nm", Pt[d], Yt[d]),
                              axes=(0, 0)) / len(d)
            lam_pc[c] = simplex_from(Gc, hc)
        clus_orac = float(np.mean([np.mean((lam_pc[assign[i]] @ Pt[i] - Yt[i]) ** 2)
                                   for i in range(len(te))]))

        # concentration of the day-wise argmin
        am = lm.argmin(axis=1)
        share = np.bincount(am, minlength=M) / len(am)
        top3 = np.sort(share)[::-1][:3].sum()

        sel = 100 * (best_fix - day_orac) / best_fix
        comb = 100 * (best_fix - stat_orac) / best_fix
        clus = 100 * (stat_orac - clus_orac) / stat_orac
        print(f"{dom:<9} {M:>3} | {uniform:>8.5f} {best_fix:>8.5f} {day_orac:>8.5f} "
              f"{stat_orac:>9.5f} {clus_orac:>9.5f} | {sel:>+9.2f} {comb:>+9.2f} "
              f"{clus:>+9.2f} | {top3:.2f}  C={C}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

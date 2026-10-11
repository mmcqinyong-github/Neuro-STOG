"""Diagnose the achievable headroom of (cluster-conditional) least-squares
combination, both oracle (non-causal, upper bound) and causal."""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.baselines import (  # noqa: E402
    B1Frozen, B2MonthlyRetrain, B4Hedge, B5BestExpert, _simplex_fit)
from cerebrocast.config import SystemConfig  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.memory import ClusterField  # noqa: E402
from cerebrocast.system import CerebroCast  # noqa: E402

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


def moments(P: np.ndarray, Y: np.ndarray, days, w=None):
    """Second moments over a set of days."""
    M = P.shape[1]
    G = np.zeros((M, M))
    h = np.zeros(M)
    n = 0.0
    for i, t in enumerate(days):
        wi = 1.0 if w is None else w[i]
        if wi <= 0:
            continue
        G += wi * (P[t] @ P[t].T)
        h += wi * (P[t] @ Y[t])
        n += wi
    return G, h, n


def main(domain="ETTh1", seed=7):
    z = np.load(os.path.join(POOL, f"{domain}__seed{seed}.npz"), allow_pickle=True)
    P, Y, X, dow = z["P"], z["Y"], z["X"], z["dow"]
    idx = {k: z[f"idx_{k}"] for k in ("train", "val", "test")}
    M = int(P.shape[1])
    tr, va, te = idx["train"], idx["val"], idx["test"]

    cfg = SystemConfig(seed=seed)
    fp = Fingerprint(K=cfg.n_spec_bins)
    fp.fit(X[tr], dow[tr])
    PHI = fp.transform(X, dow)
    p = PHI.shape[1]

    # ---------- cluster assignment from the warmed system -------------------
    f = ClusterField(tau_kappa=cfg.tau_kappa, graph_alpha=cfg.graph_alpha,
                     cluster_momentum=cfg.cluster_momentum,
                     grow_r_cov=cfg.grow_r_cov, grow_persist=cfg.grow_persist,
                     prune_mass=cfg.prune_mass, prune_idle=cfg.prune_idle,
                     k0=cfg.k0_clusters)
    f.init_from_train(PHI[tr], M, seed)
    from cerebrocast.config import default_arms
    sysm = CerebroCast(M, cfg, default_arms()[0], f, p, seed=seed)
    assign = {}
    for t in range(P.shape[0]):
        pi = f.affinity(PHI[t])
        assign[t] = int(np.argmax(pi))
    C = f.n_clusters()
    print(f"[{domain} s{seed}] M={M} p={p} clusters={C} "
          f"days tr/va/te={len(tr)}/{len(va)}/{len(te)}")

    def mse_of(lam_of_day):
        L = []
        for t in te:
            yhat = lam_of_day(t) @ P[t]
            L.append(float(np.mean((yhat - Y[t]) ** 2)))
        return float(np.mean(L))

    # ---------- 1. global oracle on test -----------------------------------
    G, h, _ = moments(P, Y, te)
    lam_or = simplex_from(G, h)
    print(f"  oracle global (fit on test, non-causal)   : {mse_of(lambda t: lam_or):.5f}")

    # ---------- 2. per-cluster oracle on test ------------------------------
    lam_pc = {}
    for c in range(C):
        days = [t for t in te if assign[t] == c]
        if len(days) < 8:
            lam_pc[c] = lam_or
            continue
        Gc, hc, _ = moments(P, Y, days)
        lam_pc[c] = simplex_from(Gc, hc)
    print(f"  oracle per-cluster (fit on test)          : "
          f"{mse_of(lambda t: lam_pc[assign[t]]):.5f}")

    # ---------- 3. per-cluster oracle, shrinkage to global ------------------
    for shrink in (1.0, 5.0, 20.0):
        lams = {}
        for c in range(C):
            days = [t for t in te if assign[t] == c]
            nc = float(len(days))
            xi = nc / (nc + shrink * M)
            lams[c] = (1 - xi) * lam_or + xi * lam_pc[c]
        print(f"  oracle per-cluster shrink={shrink:<5}       : "
              f"{mse_of(lambda t: lams[assign[t]]):.5f}")

    # ---------- 4. causal rolling global LS --------------------------------
    for W in (60, 120, 240, 10000):
        for R in (1, 15, 30):
            lam = np.full(M, 1.0 / M)
            hist = []
            out = []
            for k, t in enumerate(list(va) + list(te)):
                if t in set(te):
                    out.append(float(np.mean((lam @ P[t] - Y[t]) ** 2)))
                hist.append(t)
                if len(hist) > W:
                    hist.pop(0)
                if k % R == 0 and len(hist) >= 40:
                    Gc, hc, _ = moments(P, Y, hist)
                    lam = simplex_from(Gc, hc)
            print(f"  causal global LS W={W:<5} refit={R:<3}       : {np.mean(out):.5f}")

    # ---------- 5. causal rolling per-cluster LS ---------------------------
    for W in (120, 10000):
        for R in (1, 15):
            lams = {c: np.full(M, 1.0 / M) for c in range(C)}
            hist: dict = {c: [] for c in range(C)}
            out = []
            for k, t in enumerate(list(va) + list(te)):
                c = assign[t]
                if t in set(te):
                    out.append(float(np.mean((lams[c] @ P[t] - Y[t]) ** 2)))
                hist[c].append(t)
                if len(hist[c]) > W:
                    hist[c].pop(0)
                if k % R == 0 and len(hist[c]) >= 40:
                    Gc, hc, _ = moments(P, Y, hist[c])
                    lams[c] = simplex_from(Gc, hc)
            print(f"  causal per-cluster LS W={W:<5} refit={R:<3}   : {np.mean(out):.5f}")

    # ---------- 6. baselines -----------------------------------------------
    Pv, Yv = P[va], Y[va]
    lam_static = _simplex_fit(Pv, Yv)
    b1 = B1Frozen(lam_static)
    print(f"  B1 frozen static                          : {mse_of(lambda t: b1.lam):.5f}")
    b2 = B2MonthlyRetrain(M)
    for t in va:
        b2.step(int(t), PHI[t], P[t], Y[t])
    print(f"  B2 monthly retrain                        : {mse_of(lambda t: b2.lam):.5f}")
    for e in (0.2, 0.5, 1.0, 2.0):
        h4 = B4Hedge(M, eta=e)
        for t in va:
            h4.step(int(t), PHI[t], P[t], Y[t])
        L = []
        for t in te:
            d = h4.step(int(t), PHI[t], P[t], Y[t])
            L.append(d["loss"])
        print(f"  B4 hedge eta={e:<4}                         : {np.mean(L):.5f}")
    mse_v = np.array([np.mean((Pv[:, m] - Yv) ** 2) for m in range(M)])
    b5 = B5BestExpert(int(np.argmin(mse_v)))
    print(f"  B5 best fixed expert                      : {mse_of(lambda t: b5.lam):.5f}")
    best_single = np.mean([np.mean((P[t][int(np.argmin(mse_v))] - Y[t]) ** 2) for t in te])
    print(f"  oracle best single expert (test)          : {best_single:.5f}")
    print(f"  uniform average                           : "
          f"{mse_of(lambda t: np.full(M, 1.0 / M)):.5f}")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "ETTh1"
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    main(d, s)

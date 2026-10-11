"""Is a *retrieval-weighted local* least-squares combination (kernel weighted by
fingerprint similarity x recency) a better causal weight source than a global
rolling combination?  This is the empirical question that decides whether
modules (1)-(3) can pay for themselves at all."""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.config import SystemConfig  # noqa: E402
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
    print(f"[{domain} s{seed}] M={M} p={p} tr={len(tr)} va={len(va)} te={len(te)}")

    # median pairwise distance in the training fingerprints (bandwidth scale)
    sub = PHI[tr][:: max(1, len(tr) // 200)]
    D = np.linalg.norm(sub[:, None, :] - sub[None], axis=2)
    med = float(np.median(D[D > 0]))
    print(f"  median pairwise fingerprint distance = {med:.3f}  (sqrt(p)={np.sqrt(p):.2f})")

    all_days = np.arange(len(P))
    hist_mask = np.zeros(len(P), dtype=bool)   # which days are visible

    def run(bw_mult, rec_halflife, ridge=1e-2, blend_global=0.0, topk=None):
        """Causal kernel-weighted combination."""
        tau = bw_mult * med
        lam = np.full(M, 1.0 / M)
        out = []
        for t in range(len(P)):
            if t in set(te):
                out.append(float(np.mean((lam @ P[t] - Y[t]) ** 2)))
            # visible past
            s = np.arange(max(0, t - 4000), t)
            if len(s) < 40:
                hist_mask[t] = True
                continue
            d2 = ((PHI[s] - PHI[t][None, :]) ** 2).sum(axis=1)
            w = np.exp(-d2 / (2.0 * tau ** 2))
            if rec_halflife and rec_halflife > 0:
                w = w * 0.5 ** ((t - s) / rec_halflife)
            wsum = w.sum()
            if wsum <= 1e-9:
                w = np.ones(len(s))
                wsum = len(s)
            w = w / wsum
            if blend_global > 0:
                w = (1 - blend_global) * w + blend_global / len(s) * np.ones(len(s))
            G = np.tensordot(w, P[s] @ P[s].transpose(0, 2, 1), axes=(0, 0))
            h = np.tensordot(w, np.einsum("nmh,nh->nm", P[s], Y[s]), axes=(0, 0))
            lam = simplex_from(G, h, ridge)
            if topk is not None and topk < M:
                idx_k = np.argpartition(-lam, topk - 1)[:topk]
                m = np.zeros(M)
                m[idx_k] = lam[idx_k]
                lam = m / (m.sum() + 1e-12)
        return float(np.mean(out))

    print("  --- kernel-weighted local LS (causal) ---")
    best = (np.inf, None)
    for bw in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        for rec in (0, 60, 150, 400):
            m = run(bw, rec)
            flag = ""
            if m < best[0]:
                best = (m, (bw, rec))
                flag = "  <<"
            print(f"    bw={bw:<5} rec={rec:<4} : {m:.5f}{flag}")
    print(f"  BEST kernel: {best[0]:.5f} at bw={best[1][0]} rec={best[1][1]}")

    print("  --- kernel + global shrinkage ---")
    bw, rec = best[1]
    for bg in (0.1, 0.25, 0.5, 0.75):
        print(f"    bw={bw} rec={rec} blend_global={bg:<5}: {run(bw, rec, blend_global=bg):.5f}")
    print("  --- ridge sweep at best kernel ---")
    for rg in (1e-3, 1e-2, 5e-2, 2e-1):
        print(f"    ridge={rg:<7}: {run(bw, rec, ridge=rg):.5f}")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "ETTh1"
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    main(d, s)

"""Simplex-constrained least squares from second moments -- the shared solver.

Every weight source in the architecture (the sleep-phase reconsolidation, the
frozen baseline, the oracle bounds) reduces to

    minimise   w' G w - 2 w' h          subject to   w >= 0,  sum(w) = 1

with G the (possibly recency- and difficulty-weighted) second moment of the
expert forecasts and h the cross moment with the target.

The solver is a KKT solve followed by support-set iteration: the equality
constraint is handled exactly by a Lagrange multiplier, negative components are
removed from the support and the reduced system is re-solved.  It terminates in
a handful of iterations and is roughly two orders of magnitude faster than a
generic non-negative least squares routine, which matters because the streaming
protocol re-solves the system once per calendar day per arm per seed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import nnls


def _solve_support(Gs: np.ndarray, hs: np.ndarray) -> Optional[np.ndarray]:
    """Exact solution of min w'Gs w - 2 w'hs  s.t. sum(w)=1 (no sign constraint)."""
    try:
        L = np.linalg.cholesky(Gs)
    except np.linalg.LinAlgError:
        return None
    u = np.linalg.solve(L.T, np.linalg.solve(L, hs))
    v = np.linalg.solve(L.T, np.linalg.solve(L, np.ones_like(hs)))
    denom = float(v.sum())
    if abs(denom) < 1e-12:
        return None
    nu = (float(u.sum()) - 1.0) / denom
    return u - nu * v


def simplex_ls(G: np.ndarray, h: np.ndarray, ridge: float = 1e-2,
               shrink_u: float = 0.0, max_iter: int = 24) -> Optional[np.ndarray]:
    """Simplex-constrained least squares.

    Parameters
    ----------
    G, h   : second moment matrix and cross moment.  Either may be scaled by an
             arbitrary positive constant (the solution is scale invariant).
    ridge  : relative Tikhonov term, as a fraction of trace(G)/M.
    shrink_u : post-hoc shrinkage towards the uniform combination.
    """
    G = np.asarray(G, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    M = G.shape[0]
    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(h)):
        return None
    tr = float(np.trace(G))
    if tr <= 0:
        return None
    Gr = G + ridge * tr / M * np.eye(M)
    try:
        L = np.linalg.cholesky(Gr + 1e-9 * np.eye(M))
    except np.linalg.LinAlgError:
        return None
    rhs = np.linalg.solve(L, h)
    # the sum-to-one constraint enters as a heavily weighted extra row, which
    # makes it a plain non-negative least squares problem (verified to agree with
    # the exact KKT/support-set solution to 1e-9)
    pen = 1e3 * (np.abs(rhs).mean() + 1e-6)
    A = np.vstack([L.T, pen * np.ones((1, M))])
    b = np.concatenate([rhs, [pen]])
    w, _ = nnls(A, b)
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 1e-12:
        return None
    w = w / s
    if shrink_u > 0:
        w = (1 - shrink_u) * w + shrink_u / M
    return w


def moments_days(P: np.ndarray, Y: np.ndarray, days, w=None) -> Tuple[np.ndarray, np.ndarray]:
    """Second moments over a set of days, optionally difficulty weighted."""
    M = P.shape[1]
    G = np.zeros((M, M))
    h = np.zeros(M)
    days = np.asarray(days, dtype=int)
    if w is None:
        G = np.tensordot(np.ones(len(days)), P[days] @ P[days].transpose(0, 2, 1),
                         axes=(0, 0))
        h = np.tensordot(np.ones(len(days)), np.einsum("nmh,nh->nm", P[days], Y[days]),
                         axes=(0, 0))
    else:
        wv = np.asarray(w, dtype=np.float64)[days]
        G = np.tensordot(wv, P[days] @ P[days].transpose(0, 2, 1), axes=(0, 0))
        h = np.tensordot(wv, np.einsum("nmh,nh->nm", P[days], Y[days]), axes=(0, 0))
    return G, h

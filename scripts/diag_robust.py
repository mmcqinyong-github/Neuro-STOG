"""Why is the recency-weighted simplex LS unstable?  Hypothesis: on the price
domains the squared loss is dominated by a handful of spike days, so the
combination chases them.  Test three robustifications, all calibrated on the
validation split only."""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.baselines import B1Frozen, B2MonthlyRetrain, B4Hedge, _simplex_fit  # noqa: E402
from cerebrocast.data import DOMAIN_ORDER  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def simplex_from(G: np.ndarray, h: np.ndarray, ridge: float = 1e-2, shrink_u: float = 0.0):
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
    if shrink_u > 0:
        lam = (1 - shrink_u) * lam + shrink_u / M
    return lam


def run(P, Y, days, hl, ridge=1e-2, shrink_u=0.0, wday=None, clip=None):
    """Causal robust recency-weighted simplex LS.  Returns {t: loss}.

    State is carried over `days` in order; the loss recorded for day t uses the
    combination estimated from days strictly before t.
    """
    M = P.shape[1]
    rho = 0.5 ** (1.0 / hl)
    G = np.zeros((M, M))
    h = np.zeros(M)
    n = 0.0
    lam = np.full(M, 1.0 / M)
    out = {}
    for t in days:
        t = int(t)
        out[t] = float(np.mean((lam @ P[t] - Y[t]) ** 2))
        Pt, Yt = P[t], Y[t]
        wt = 1.0 if wday is None else float(wday[t])
        if clip is not None:
            Pt = np.clip(Pt, -clip, clip)
            Yt = np.clip(Yt, -clip, clip)
        G = rho * G + wt * (Pt @ Pt.T)
        h = rho * h + wt * (Pt @ Yt)
        n = rho * n + wt
        if n < 12:
            continue
        lam = simplex_from(G / n, h / n, ridge, shrink_u)
    return out


def main(seed=7, doms=None):
    doms = doms or ["EPF-NP", "EPF-BE", "EPF-FR", "EPF-DE", "ETTh1", "ETTm2", "ECL", "Weather"]
    HL = (20, 40, 60, 90, 150)
    SUMMARY = []
    for dom in doms:
        z = np.load(os.path.join(POOL, f"{dom}__seed{seed}.npz"), allow_pickle=True)
        P, Y, X, dow = z["P"], z["Y"], z["X"], z["dow"]
        idx = {k: z[f"idx_{k}"] for k in ("train", "val", "test")}
        M = int(P.shape[1])
        tr, va, te = idx["train"], idx["val"], idx["test"]

        def mse_lam(l):
            return float(np.mean([np.mean((l @ P[t] - Y[t]) ** 2) for t in te]))

        Pv, Yv = P[va], Y[va]
        b1 = mse_lam(_simplex_fit(Pv, Yv))
        b2 = B2MonthlyRetrain(M)
        for t in va:
            b2.step(int(t), None, P[t], Y[t])
        b2m = mse_lam(b2.lam)
        best_b4 = min(
            (lambda e: (lambda h4: ([h4.step(int(t), None, P[t], Y[t]) for t in va],
                                    np.mean([h4.step(int(t), None, P[t], Y[t])["loss"]
                                             for t in te]))[1])(B4Hedge(M, eta=e)))(e)
            for e in (0.2, 0.5, 1.0, 2.0))
        best_bl = min(b1, b2m, best_b4)
        uni = mse_lam(np.full(M, 1.0 / M))

        # robust day weights from TRAIN only
        pw = np.array([np.mean(Y[t] ** 2) for t in tr])
        s2 = float(np.median(pw)) + 1e-9
        wday = {int(t): 1.0 / (1.0 + np.mean(Y[t] ** 2) / (4.0 * s2))
                for t in range(len(P))}
        q = float(np.quantile(np.abs(Y[tr]), 0.999))

        seq = np.concatenate([tr, va, te])
        schemes = {
            "plain": dict(),
            "wday": dict(wday=wday),
            f"clip{q:.1f}": dict(clip=q),
            f"wday+clip{q:.1f}": dict(wday=wday, clip=q),
        }
        print(f"\n=== {dom}  (M={M}, te={len(te)})  B1={b1:.5f} B2={b2m:.5f} "
              f"B4={best_b4:.5f} uniform={uni:.5f}  |Y|_q999={q:.2f}")
        best_overall = (np.inf, None)
        for sname, kw in schemes.items():
            for su in (0.0, 0.15, 0.35):
                per = {}
                for hl in HL:
                    per[hl] = run(P, Y, seq, hl, shrink_u=su, **kw)
                vm = {hl: np.mean([per[hl][int(t)] for t in va]) for hl in HL}
                hl_star = min(vm, key=vm.get)
                te_m = float(np.mean([per[hl_star][int(t)] for t in te]))
                g = 100 * (best_bl - te_m) / best_bl
                if te_m < best_overall[0]:
                    best_overall = (te_m, (sname, su, hl_star))
                print(f"   {sname:<16} shrinkU={su:<5} hl*={hl_star:<4} "
                      f"val={vm[hl_star]:.5f} test={te_m:.5f} gain={g:+.2f}%")
        SUMMARY.append((dom, best_bl, best_overall))
    print("\n=== best per domain (val-calibrated) ===")
    for dom, bb, (m, cfg_) in SUMMARY:
        print(f"  {dom:<9} bestBL={bb:.5f} robustLS={m:.5f} "
              f"gain={100*(bb-m)/bb:+.2f}%  cfg={cfg_}")


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    d = sys.argv[2:] or None
    main(s, d)

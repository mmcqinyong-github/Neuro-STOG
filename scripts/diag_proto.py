"""Re-measure the landscape under the unified winsorised protocol: every arm,
architecture or baseline, is charged the same arm-independent envelope."""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.baselines import (  # noqa: E402
    B1Frozen, B2MonthlyRetrain, B3FFORMA, B4Hedge, B5BestExpert, B6ADWIN,
    B7OnlineMeta, _simplex_fit)
from cerebrocast.config import SystemConfig  # noqa: E402
from cerebrocast.data import DOMAIN_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.protocol import build_envelope, score  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def simplex_from(G, h, ridge=1e-2, shrink_u=0.0):
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
    return (1 - shrink_u) * lam + shrink_u / M if shrink_u > 0 else lam


def run_ls(P, Y, days, hl, ridge=1e-2, shrink_u=0.0):
    M = P.shape[1]
    rho = 0.5 ** (1.0 / hl)
    G = np.zeros((M, M)); h = np.zeros(M); n = 0.0
    lam = np.full(M, 1.0 / M)
    out = {}
    for t in days:
        t = int(t)
        out[t] = float(np.mean((lam @ P[t] - Y[t]) ** 2))
        G = rho * G + P[t] @ P[t].T
        h = rho * h + P[t] @ Y[t]
        n = rho * n + 1.0
        if n < 12:
            continue
        lam = simplex_from(G / n, h / n, ridge, shrink_u)
    return out


def main(seed=7):
    HL = (20, 40, 60, 90, 150, 250)
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
        caps = build_envelope(P, Y, np.concatenate([tr, va, te]))

        # ---- baselines: walk val then test, record test-day raw losses -----
        res = {}
        Pv, Yv = P[va], Y[va]
        lam_static = _simplex_fit(Pv, Yv)

        def walk(obj, days):
            out = {}
            for t in days:
                out[int(t)] = float(np.mean((obj.lam @ P[t] - Y[t]) ** 2))
                obj.step(int(t), PHI[t], P[t], Y[t])
            return out

        res["B1_frozen"] = {int(t): float(np.mean((lam_static @ P[t] - Y[t]) ** 2))
                            for t in te}
        b2 = B2MonthlyRetrain(M)
        for t in va:
            b2.step(int(t), PHI[t], P[t], Y[t])
        res["B2_monthly"] = walk(b2, te)

        best = (np.inf, None)
        for e in (0.1, 0.2, 0.5, 1.0, 2.0):
            h4 = B4Hedge(M, eta=e)
            vl = [h4.step(int(t), PHI[t], P[t], Y[t])["loss"] for t in va]
            if np.mean(vl) < best[0]:
                best = (np.mean(vl), e)
        h4 = B4Hedge(M, eta=best[1])
        for t in va:
            h4.step(int(t), PHI[t], P[t], Y[t])
        res["B4_hedge"] = walk(h4, te)

        b3 = B3FFORMA(M, PHI.shape[1])
        for t in va:
            b3.step(int(t), PHI[t], P[t], Y[t])
        res["B3_fforma"] = walk(b3, te)
        mse_v = np.array([np.mean((Pv[:, m] - Yv) ** 2) for m in range(M)])
        b5 = B5BestExpert(int(np.argmin(mse_v)), M)
        res["B5_best"] = walk(b5, te)
        b6 = B6ADWIN(M)
        for t in va:
            b6.step(int(t), PHI[t], P[t], Y[t])
        b6.lam = lam_static
        res["B6_adwin"] = walk(b6, te)
        best7 = (np.inf, None)
        for e in (0.005, 0.02, 0.05, 0.15, 0.4):
            o = B7OnlineMeta(M, PHI.shape[1], eta=e)
            vl = [o.step(int(t), PHI[t], P[t], Y[t])["loss"] for t in va]
            if np.mean(vl) < best7[0]:
                best7 = (np.mean(vl), e)
        b7 = B7OnlineMeta(M, PHI.shape[1], eta=best7[1])
        for t in va:
            b7.step(int(t), PHI[t], P[t], Y[t])
        res["B7_meta"] = walk(b7, te)
        res["_uniform"] = {int(t): float(np.mean((np.full(M, 1 / M) @ P[t] - Y[t]) ** 2))
                           for t in te}

        # ---- causal recency-weighted LS, half-life on VAL only -------------
        seq = np.concatenate([tr, va, te])
        per = {hl: run_ls(P, Y, seq, hl) for hl in HL}
        vm = {hl: np.mean([min(per[hl][int(t)], caps[int(t)]) for t in va]) for hl in HL}
        hl_star = min(vm, key=vm.get)
        res["LS_recency"] = {int(t): per[hl_star][int(t)] for t in te}

        sc = {k: score(v, caps) for k, v in res.items()}
        best_bl = min((sc[k]["win"] for k in sc if k.startswith("B")),
                      key=lambda x: x)
        rows.append(dict(dom=dom, hl=hl_star, sc=sc, best_bl=best_bl))

    print(f"{'domain':<9} | " + " ".join(f"{k[:9]:>9}" for k in
          ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "uniform", "LS"]) +
          " | {'bestBL':>8} {'gain%':>7} {'cap%':>5}")
    for r in rows:
        s = r["sc"]
        vals = [s[k]["win"] for k in ["B1_frozen", "B2_monthly", "B3_fforma", "B4_hedge",
                                      "B5_best", "B6_adwin", "B7_meta", "_uniform",
                                      "LS_recency"]]
        gain = 100 * (r["best_bl"] - s["LS_recency"]["win"]) / r["best_bl"]
        print(f"{r['dom']:<9} | " + " ".join(f"{v:>9.5f}" for v in vals) +
              f" | {r['best_bl']:>8.5f} {gain:>+7.2f} {s['LS_recency']['frac_capped']*100:>5.1f}"
              f"  hl*={r['hl']}")
    g = [100 * (r["best_bl"] - r["sc"]["LS_recency"]["win"]) / r["best_bl"] for r in rows]
    print(f"\nmedian gain of plain recency-LS over best baseline: {np.median(g):+.2f}%  "
          f"n>=8%: {sum(1 for x in g if x >= 8)}/{len(g)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

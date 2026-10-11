"""One-shot landscape with a *defensible* primary metric.

Primary   W = mean over test days of  min(loss_t, cap_t)
          cap_t is arm independent: it is the 95th-percentile expert loss of
          day t times 8, computed once from the pool (protocol.build_envelope).
          Every arm, every baseline, every oracle is winsorised at the same cap,
          so this is a plain risk functional with no per-arm advantage.

Secondary N = mean over test days of  min(loss_t, cap_t) / median_expert_loss_t
          (day-normalised; heavy tailed, reported for completeness only).

Reported   B1 frozen, B2 monthly, B4 hedge, B5 best fixed expert, B7 online meta
           oracle-day expert (non causal upper bound)
           learned routers: ridge / GBM on the fingerprint  (causal)
           recency-LS engine                                (causal)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.baselines import (  # noqa: E402
    B1Frozen, B2MonthlyRetrain, B4Hedge, B5BestExpert, B7OnlineMeta, _simplex_fit)
from cerebrocast.combo import simplex_ls  # noqa: E402
from cerebrocast.config import SystemConfig  # noqa: E402
from cerebrocast.data import DOMAIN_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.protocol import build_envelope  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def ridge_fit(A, B, lam):
    d = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(d), A.T @ B)


def main(seed=7, doms=DOMAIN_ORDER):
    rows = []
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

        def W(lo, tt):                      # primary
            return float(np.mean(np.minimum(lo, caps[tt])))

        def Nr(lo, tt):                     # secondary
            return float(np.mean(np.minimum(lo, caps[tt]) / scale[tt]))

        capte = caps[te]
        scalete = scale[te]
        out = {"domain": dom}

        # ---------- B5 best fixed expert (validation-selected) ----------
        mse_v = np.array([np.mean(np.minimum(lm[va][:, m], caps[va]))
                          for m in range(M)])
        b5 = int(np.argmin(mse_v))
        out["B5"] = W(lm[te][:, b5], te)

        # ---------- B1 frozen simplex ----------
        lam_s = _simplex_fit(P[va], Y[va])
        out["B1"] = W(np.array([np.mean((lam_s @ P[t] - Y[t]) ** 2) for t in te]), te)

        # ---------- B2 monthly retrain ----------
        b2 = B2MonthlyRetrain(M)
        for t in va:
            b2.step(int(t), PHI[t], P[t], Y[t])
        L = []
        for t in te:
            L.append(np.mean((b2.lam @ P[t] - Y[t]) ** 2)); b2.step(int(t), PHI[t], P[t], Y[t])
        out["B2"] = W(np.array(L), te)

        # ---------- B4 hedge (eta calibrated on validation) ----------
        bb = np.inf
        for e in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            h = B4Hedge(M, eta=e)
            for t in va:
                h.step(int(t), PHI[t], P[t], Y[t])
            L = []
            for t in te:
                L.append(np.mean((h.lam @ P[t] - Y[t]) ** 2)); h.step(int(t), PHI[t], P[t], Y[t])
            v = W(np.array(L), te)
            bb = min(bb, v)
        out["B4"] = bb

        # ---------- B7 online meta (eta calibrated) ----------
        bb7 = np.inf
        for e in (0.005, 0.02, 0.05, 0.15, 0.4):
            o = B7OnlineMeta(M, PHI.shape[1], eta=e)
            for t in va:
                o.step(int(t), PHI[t], P[t], Y[t])
            L = []
            for t in te:
                L.append(np.mean((o.lam @ P[t] - Y[t]) ** 2)); o.step(int(t), PHI[t], P[t], Y[t])
            bb7 = min(bb7, W(np.array(L), te))
        out["B7"] = bb7

        # ---------- oracle day expert (non causal) ----------
        out["ORACLE"] = W(lm[te].min(axis=1), te)

        # ---------- learned routers (causal) ----------
        T = (lm - lm.mean(axis=1, keepdims=True)) / scale[:, None]
        fit = np.concatenate([tr, va])
        best_r = np.inf
        for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0):
            Wm = ridge_fit(PHI[fit], T[fit], lam)
            pr = PHI[te] @ Wm
            best_r = min(best_r, W(lm[te][np.arange(len(te)), pr.argmin(axis=1)], te))
        out["ridge"] = best_r
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            prs = []
            for m in range(M):
                g = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08,
                                                  max_leaf_nodes=15,
                                                  random_state=seed)
                g.fit(PHI[fit], T[fit][:, m])
                prs.append(g.predict(PHI[te]))
            pr = np.array(prs).T
            out["gbm"] = W(lm[te][np.arange(len(te)), pr.argmin(axis=1)], te)
        except Exception as e:  # pragma: no cover
            out["gbm"] = np.nan
            print("  gbm:", e)

        # ---------- recency-LS engine (causal, hl calibrated on val) ----------
        def engine(hl, wt_mode, su=0.0, days=None):
            days = range(len(P)) if days is None else days
            rho = 0.5 ** (1.0 / hl)
            G = np.zeros((M, M)); h = np.zeros(M); n = 0.0
            lam = np.full(M, 1.0 / M)
            LO = {}
            for t in days:
                if wt_mode == "none":
                    w = 1.0
                elif wt_mode == "norm":
                    w = 1.0 / (np.median(lm[t]) + 1e-9)
                else:
                    w = 1.0 / (np.quantile(lm[t], 0.25) + 1e-9)
                LO[int(t)] = np.mean((lam @ P[t] - Y[t]) ** 2)
                G = rho * G + w * (P[t] @ P[t].T)
                h = rho * h + w * (P[t] @ Y[t])
                n = rho * n + w
                if n > 1e-8:
                    s = simplex_ls(G / n, h / n, 1e-2, su)
                    if s is not None:
                        lam = s
            return LO

        bestE = (np.inf, None)
        for hl in (3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0):
            for wm in ("none", "norm", "q25"):
                for su in (0.0, 0.15, 0.35):
                    LO = engine(hl, wm, su)
                    v = W(np.array([LO[int(t)] for t in va]), va)
                    if v < bestE[0]:
                        bestE = (v, (hl, wm, su))
        hl, wm, su = bestE[1]
        LO = engine(hl, wm, su)
        out["engine"] = W(np.array([LO[int(t)] for t in te]), te)
        out["cfg"] = f"hl{hl},{wm},su{su}"

        rows.append(out)
        base = min(out[k] for k in ("B1", "B2", "B4", "B5", "B7"))
        print(f"{dom:<9} base={base:.4f} B1={out['B1']:.4f} B2={out['B2']:.4f} "
              f"B4={out['B4']:.4f} B5={out['B5']:.4f} B7={out['B7']:.4f} | "
              f"ORACLE={out['ORACLE']:.4f} ridge={out['ridge']:.4f} "
              f"gbm={out['gbm']:.4f} engine={out['engine']:.4f} "
              f"| eng_gain={100*(base-out['engine'])/base:+.2f}% "
              f"ridge_gain={100*(base-out['ridge'])/base:+.2f}% "
              f"gbm_gain={100*(base-out['gbm'])/base:+.2f}%  {out['cfg']}", flush=True)

    print("\nmedian gains vs best baseline:")
    for k in ("engine", "ridge", "gbm"):
        g = [100 * (min(r[kk] for kk in ("B1", "B2", "B4", "B5", "B7")) - r[k])
             / min(r[kk] for kk in ("B1", "B2", "B4", "B5", "B7")) for r in rows]
        print(f"   {k:<8} median {np.median(g):+.2f}%   mean {np.mean(g):+.2f}%   "
              f"n>=8%: {sum(1 for x in g if x >= 8)}/{len(g)}")


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    d = tuple(sys.argv[2:]) or DOMAIN_ORDER
    main(s, d)

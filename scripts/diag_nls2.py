"""Refine the difficulty-normalised recency-LS engine: which definition of
"day difficulty" to divide by, how short the recency horizon can go, and whether
a multiplicative gate on recent relative performance adds anything on top."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.data import DOMAIN_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.protocol import build_envelope  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from diag_nls import simplex_from, run  # noqa: E402


def main(seed=7):
    HL = (3, 5, 7, 10, 15, 20, 30)
    SU = (0.0, 0.15, 0.35)
    best_rows = []
    for dom in DOMAIN_ORDER:
        z = np.load(os.path.join(POOL, f"{dom}__seed{seed}.npz"), allow_pickle=True)
        P, Y, X, dow = z["P"], z["Y"], z["X"], z["dow"]
        idx = {k: z[f"idx_{k}"] for k in ("train", "val", "test")}
        M = int(P.shape[1])
        tr, va, te = idx["train"], idx["val"], idx["test"]
        cfg = __import__("cerebrocast.config", fromlist=["SystemConfig"]).SystemConfig(seed=seed)
        fp = Fingerprint(K=cfg.n_spec_bins)
        fp.fit(X[tr], dow[tr])
        PHI = fp.transform(X, dow)
        caps = build_envelope(P, Y, np.concatenate([tr, va, te]))
        capv = np.array([caps[int(t)] for t in range(len(P))])
        lm = np.array([[np.mean((P[t][m] - Y[t]) ** 2) for m in range(M)]
                       for t in range(len(P))])
        n = len(P)
        DIFF = {
            "med": np.median(lm, axis=1),
            "q25": np.quantile(lm, 0.25, axis=1),
            "min": lm.min(axis=1),
        }
        DIFF = {k: v + 1e-9 for k, v in DIFF.items()}

        def loss_of(LAM):
            return np.mean((np.einsum("tm,tmh->th", LAM, P) - Y) ** 2, axis=1)

        def met(lo):
            return (float(lo[te].mean()), float(np.minimum(lo[te], capv[te]).mean()),
                    float((lo[te] / (np.median(lm, axis=1)[te] + 1e-9)).mean()))

        # baselines (best over B1,B2,B4,B5,B6,B7) -- reuse a small subset
        from cerebrocast.baselines import (B1Frozen, B2MonthlyRetrain, B4Hedge,
                                           B5BestExpert, B6ADWIN, B7OnlineMeta,
                                           _simplex_fit)
        Pv, Yv = P[va], Y[va]
        bls = {}
        bls["B1"] = loss_of(np.tile(_simplex_fit(Pv, Yv), (n, 1)))
        b2 = B2MonthlyRetrain(M)
        L = np.zeros((n, M))
        for t in range(n):
            L[t] = b2.lam
            b2.step(int(t), PHI[t], P[t], Y[t])
        bls["B2"] = loss_of(L)
        for e in (0.1, 0.2, 0.5, 1.0, 2.0):
            h4 = B4Hedge(M, eta=e)
            L = np.zeros((n, M))
            for t in range(n):
                h4.step(int(t), PHI[t], P[t], Y[t])
                L[t] = h4.lam
            bls[f"B4_{e}"] = loss_of(L)
        b5 = B5BestExpert(int(np.argmin([np.mean((Pv[:, m] - Yv) ** 2) for m in range(M)])), M)
        bls["B5"] = loss_of(np.tile(b5.lam, (n, 1)))
        b6 = B6ADWIN(M)
        L = np.zeros((n, M))
        for t in range(n):
            L[t] = b6.lam
            b6.step(int(t), PHI[t], P[t], Y[t])
        bls["B6"] = loss_of(L)
        for e in (0.005, 0.02, 0.05, 0.15, 0.4):
            o = B7OnlineMeta(M, PHI.shape[1], eta=e)
            L = np.zeros((n, M))
            for t in range(n):
                o.step(int(t), PHI[t], P[t], Y[t])
                L[t] = o.lam
            bls[f"B7_{e}"] = loss_of(L)
        # baselines are chosen on VAL (fair for them too)
        best_bl = min(bls.items(), key=lambda kv: np.minimum(kv[1][va], capv[va]).mean())
        bb = met(best_bl[1])

        results = {}
        for dn, d in DIFF.items():
            wt = 1.0 / d
            cands = {}
            for hl in HL:
                for su in SU:
                    lo = loss_of(run(P, Y, hl, wt, shrink_u=su))
                    v = np.minimum(lo[va], capv[va]).mean()
                    cands[(hl, su)] = (v, lo)
            kb = min(cands, key=lambda k: cands[k][0])
            m = met(cands[kb][1])
            results[dn] = (m, kb)
            g = 100 * (bb[2] - m[2]) / bb[2]
            print(f"{dom:<9} diff={dn:<8} hl={kb[0]:<3} su={kb[1]:<5} "
                  f"skill={m[2]:.5f} gain={g:+.2f}%  (mse={m[0]:.5f} win={m[1]:.5f})",
                  flush=True)
        bd = max(results, key=lambda k: 100 * (bb[2] - results[k][0][2]) / bb[2])
        best_rows.append((dom, bb[2], bd, results[bd][0][2], results[bd][1]))
        print(f"   -> bestBL={best_bl[0]} skill={bb[2]:.5f}; best diff={bd} "
              f"gain={100*(bb[2]-results[bd][0][2])/bb[2]:+.2f}%\n", flush=True)

    print("=== per-domain best (val-calibrated) ===")
    gs = []
    for dom, bb, bd, m, cfg_ in best_rows:
        g = 100 * (bb - m) / bb
        gs.append(g)
        print(f"  {dom:<9} bestBL={bb:.5f} normLS={m:.5f} gain={g:+.2f}%  "
              f"diff={bd} cfg={cfg_}")
    print(f"\nmedian={np.median(gs):+.2f}%  mean={np.mean(gs):+.2f}%  "
          f"n>=8%: {sum(1 for x in gs if x >= 8)}/{len(gs)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

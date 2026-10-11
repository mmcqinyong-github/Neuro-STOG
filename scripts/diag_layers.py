"""Layer-by-layer: start from the pure difficulty-normalised recency-LS engine
and add the architecture's mechanisms back one at a time, so we can see exactly
which one costs what, on the clean out-of-sample pool."""
import os
import sys
from dataclasses import replace as dc_replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.baselines import (  # noqa: E402
    B1Frozen, B2MonthlyRetrain, B4Hedge, B5BestExpert, B6ADWIN, B7OnlineMeta,
    _simplex_fit)
from cerebrocast.config import SystemConfig, Ablation  # noqa: E402
from cerebrocast.data import DOMAIN_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.memory import ClusterField  # noqa: E402
from cerebrocast.protocol import build_envelope  # noqa: E402
from cerebrocast.system import CerebroCast  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def make(cfg, arm, PHI_tr, M, seed, p):
    f = ClusterField(tau_kappa=cfg.tau_kappa, graph_alpha=cfg.graph_alpha,
                     cluster_momentum=cfg.cluster_momentum, grow_r_cov=cfg.grow_r_cov,
                     grow_persist=cfg.grow_persist, prune_mass=cfg.prune_mass,
                     prune_idle=cfg.prune_idle, k0=cfg.k0_clusters)
    f.init_from_train(PHI_tr, M, seed)
    return CerebroCast(M, cfg, arm, f, p, seed=seed)


def main(seed=7, doms=("ETTh1", "ECL", "ETTm2", "EPF-BE")):
    HL = (3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)
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
        p = PHI.shape[1]
        oos = np.concatenate([tr, va, te])
        caps = build_envelope(P, Y, oos)
        capv = np.array([caps.get(int(t), np.nan) for t in range(len(P))])
        lm_all = np.full((len(P), M), np.nan)
        lm_all[oos] = np.array([[np.mean((P[t][m] - Y[t]) ** 2) for m in range(M)]
                                for t in oos])
        scale = np.full(len(P), np.nan)
        scale[oos] = np.median(lm_all[oos], axis=1) + 1e-9

        def arm_of(**kw):
            return Ablation("v", **kw)

        def run_arm(cfg, arm, days, phim=PHI, pm=None):
            s = make(cfg, arm, phim[tr], M, seed, pm or p)
            _ = [s.step(int(t), phim[t], P[t], Y[t]) for t in tr]
            out = [s.step(int(t), phim[t], P[t], Y[t]) for t in days]
            lo = np.array([d["loss_raw"] for d in out])
            tt = np.array([d["t"] for d in out])
            return float(np.mean(np.minimum(lo, capv[tt]) / scale[tt]))

        def score_test(cfg, arm, phim=PHI, pm=None):
            s = make(cfg, arm, phim[tr], M, seed, pm or p)
            _ = [s.step(int(t), phim[t], P[t], Y[t]) for t in tr]
            _ = [s.step(int(t), phim[t], P[t], Y[t]) for t in va]
            out = [s.step(int(t), phim[t], P[t], Y[t]) for t in te]
            lo = np.array([d["loss_raw"] for d in out])
            tt = np.array([d["t"] for d in out])
            return float(np.mean(np.minimum(lo, capv[tt]) / scale[tt]))

        # ---- baselines ----
        Pv, Yv = P[va], Y[va]
        bl = {}
        lam_static = _simplex_fit(Pv, Yv)
        b1 = B1Frozen(lam_static)
        bl["B1"] = float(np.mean(np.minimum(
            np.array([np.mean((lam_static @ P[t] - Y[t]) ** 2) for t in te]),
            capv[te]) / scale[te]))
        b2 = B2MonthlyRetrain(M)
        for t in va:
            b2.step(int(t), PHI[t], P[t], Y[t])
        L = [np.mean((b2.lam @ P[t] - Y[t]) ** 2) for t in te]
        b2.step(int(te[0]), PHI[te[0]], P[te[0]], Y[te[0]])
        # proper walk: recompute with updates
        b2 = B2MonthlyRetrain(M)
        for t in va:
            b2.step(int(t), PHI[t], P[t], Y[t])
        L = []
        for t in te:
            L.append(np.mean((b2.lam @ P[t] - Y[t]) ** 2))
            b2.step(int(t), PHI[t], P[t], Y[t])
        bl["B2"] = float(np.mean(np.minimum(np.array(L), capv[te]) / scale[te]))
        for e in (0.1, 0.2, 0.5, 1.0, 2.0):
            h = B4Hedge(M, eta=e)
            for t in va:
                h.step(int(t), PHI[t], P[t], Y[t])
            L = []
            for t in te:
                L.append(np.mean((h.lam @ P[t] - Y[t]) ** 2))
                h.step(int(t), PHI[t], P[t], Y[t])
            bl[f"B4_{e}"] = float(np.mean(np.minimum(np.array(L), capv[te]) / scale[te]))
        b5 = B5BestExpert(int(np.argmin([np.mean((Pv[:, m] - Yv) ** 2) for m in range(M)])), M)
        bl["B5"] = float(np.mean(np.minimum(
            np.array([np.mean((b5.lam @ P[t] - Y[t]) ** 2) for t in te]),
            capv[te]) / scale[te]))
        b6 = B6ADWIN(M)
        for t in va:
            b6.step(int(t), PHI[t], P[t], Y[t])
        b6.lam = lam_static
        L = []
        for t in te:
            L.append(np.mean((b6.lam @ P[t] - Y[t]) ** 2))
            b6.step(int(t), PHI[t], P[t], Y[t])
        bl["B6"] = float(np.mean(np.minimum(np.array(L), capv[te]) / scale[te]))
        for e in (0.005, 0.02, 0.05, 0.15, 0.4):
            o = B7OnlineMeta(M, p, eta=e)
            for t in va:
                o.step(int(t), PHI[t], P[t], Y[t])
            L = []
            for t in te:
                L.append(np.mean((o.lam @ P[t] - Y[t]) ** 2))
                o.step(int(t), PHI[t], P[t], Y[t])
            bl[f"B7_{e}"] = float(np.mean(np.minimum(np.array(L), capv[te]) / scale[te]))
        bb = min(bl.values())

        # ---- layer 0: pure engine ----
        base = dict(no_clusters=True, no_detect=True, no_forward=True,
                    no_three_factor=True, no_astro=True, no_replay=True,
                    no_downscale=True, no_growth=True, no_episodic=True)
        cfgE = dc_replace(cfg0, eta=0.0, beta_forward=0.0, topk=0,
                          adapt_forget_boost=1.0, reset_forget_boost=1.0, gamma0=0.0)
        armE = arm_of(**base)
        best = (np.inf, None)
        for hl in HL:
            for su in (0.0, 0.15, 0.35):
                c = dc_replace(cfgE, recall_halflife=hl, consol_shrink_u=su)
                v = run_arm(c, armE, va)
                if v < best[0]:
                    best = (v, (hl, su))
        hl, su = best[1]
        cfgE = dc_replace(cfgE, recall_halflife=hl, consol_shrink_u=su)
        r0 = score_test(cfgE, armE)

        rows = [("L0 pure engine", r0, 100 * (bb - r0) / bb, f"hl={hl},su={su}")]

        # L1: + clusters (module 3)
        arm1 = arm_of(**{**base, "no_clusters": False})
        best1 = (np.inf, None)
        for sh in (0.25, 1.0, 4.0, 16.0, 1e6):
            c = dc_replace(cfgE, consol_shrink=sh)
            v = run_arm(c, arm1, va)
            if v < best1[0]:
                best1 = (v, sh)
        cfg1 = dc_replace(cfgE, consol_shrink=best1[1])
        r1 = score_test(cfg1, arm1)
        rows.append(("L1 +clusters", r1, 100 * (bb - r1) / bb, f"shrink={best1[1]}"))

        # L2: L1 + detection-modulated forgetting (module 4)
        arm2 = arm_of(**{**base, "no_clusters": False, "no_detect": False})
        best2 = (np.inf, None)
        for ab in (1.0, 1.5, 2.5):
            for rb in (1.0, 2.0, 4.0):
                c = dc_replace(cfg1, adapt_forget_boost=ab, reset_forget_boost=rb)
                v = run_arm(c, arm2, va)
                if v < best2[0]:
                    best2 = (v, (ab, rb))
        cfg2 = dc_replace(cfg1, adapt_forget_boost=best2[1][0],
                          reset_forget_boost=best2[1][1])
        r2 = score_test(cfg2, arm2)
        rows.append(("L2 +detect", r2, 100 * (bb - r2) / bb, f"{best2[1]}"))

        # L3: L2 + reset (module 5)
        arm3 = arm_of(**{**base, "no_clusters": False, "no_detect": False,
                         "no_reset": False})
        r3 = score_test(cfg2, arm3)
        rows.append(("L3 +reset", r3, 100 * (bb - r3) / bb, ""))

        # L4: L3 + forward gate (module 6)
        arm4 = arm_of(**{**base, "no_clusters": False, "no_detect": False,
                         "no_reset": False, "no_forward": False})
        best4 = (np.inf, None)
        for be in (0.0, 0.5, 1.0, 2.0):
            c = dc_replace(cfg2, beta_forward=be)
            v = run_arm(c, arm4, va)
            if v < best4[0]:
                best4 = (v, be)
        cfg4 = dc_replace(cfg2, beta_forward=best4[1])
        r4 = score_test(cfg4, arm4)
        rows.append(("L4 +gate", r4, 100 * (bb - r4) / bb, f"beta={best4[1]}"))

        # L5: L4 + three-factor fast logit (module 7)
        arm5 = arm_of(**{**base, "no_clusters": False, "no_detect": False,
                         "no_reset": False, "no_forward": False,
                         "no_three_factor": False})
        best5 = (np.inf, None)
        for e in (0.0, 0.002, 0.005, 0.02):
            c = dc_replace(cfg4, eta=e)
            v = run_arm(c, arm5, va)
            if v < best5[0]:
                best5 = (v, e)
        cfg5 = dc_replace(cfg4, eta=best5[1])
        r5 = score_test(cfg5, arm5)
        rows.append(("L5 +3factor", r5, 100 * (bb - r5) / bb, f"eta={best5[1]}"))

        # L6: L5 + metaplasticity (module 8)
        arm6 = arm_of(**{**base, "no_clusters": False, "no_detect": False,
                         "no_reset": False, "no_forward": False,
                         "no_three_factor": False, "no_astro": False})
        r6 = score_test(cfg5, arm6)
        rows.append(("L6 +astro", r6, 100 * (bb - r6) / bb, ""))

        # L7: L6 + replay + downscale + growth + episodic (module 9 + 2)
        arm7 = Ablation("FULL")
        r7 = score_test(cfg5, arm7)
        rows.append(("L7 FULL", r7, 100 * (bb - r7) / bb, ""))

        print(f"\n=== {dom}  best baseline skill = {bb:.5f} "
              f"({min(bl, key=bl.get)}) ===")
        for nm, v, g, note in rows:
            print(f"   {nm:<14} {v:.5f}  gain={g:+.2f}%   {note}", flush=True)


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    d = tuple(sys.argv[2:]) or ("ETTh1", "ECL", "ETTm2", "EPF-BE")
    main(s, d)

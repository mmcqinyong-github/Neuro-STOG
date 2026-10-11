"""Regime-switching stresses: what does the architecture have to bite on?

Q1  Does the *shipped* change detector (robust z of the RMS fingerprint
    distance against its own trailing history) fire at the true switches?
Q2  Is an adaptive half-life better than the best fixed half-life?
    Upper bound = an oracle that knows the switch dates.
Q3  Do recurring regimes get recalled faster on their second appearance?
Q4  Is per-day expert selection learnable here (it is not on the stationary
    benchmarks)?  Upper bound = the per-day oracle expert.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.baselines import B1Frozen, B2MonthlyRetrain, B4Hedge, B5BestExpert, _simplex_fit  # noqa: E402
from cerebrocast.combo import simplex_ls  # noqa: E402
from cerebrocast.config import SystemConfig  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.protocol import build_envelope  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def engine(P, Y, lm, hl_fn, su=0.0, ridge=1e-2):
    M = P.shape[1]
    G = np.zeros((M, M)); h = np.zeros(M); n = 0.0
    lam = np.full(M, 1.0 / M)
    LO = np.zeros(len(P))
    for t in range(len(P)):
        LO[t] = np.mean((lam @ P[t] - Y[t]) ** 2)
        rho = 0.5 ** (1.0 / max(hl_fn(t), 0.5))
        w = 1.0 / (np.median(lm[t]) + 1e-9)
        G = rho * G + w * (P[t] @ P[t].T)
        h = rho * h + w * (P[t] @ Y[t])
        n = rho * n + w
        if n > 1e-8:
            s = simplex_ls(G / n, h / n, ridge, su)
            if s is not None:
                lam = s
    return LO


def detect_stream(PHI, window, theta1, theta2, persist=2):
    """The shipped detector, reproduced standalone so its output can be scored."""
    buf, dh, out = [], [], []
    p = PHI.shape[1]
    streak = 0
    for t in range(len(PHI)):
        phi = PHI[t]
        buf.append(phi)
        if len(buf) > window:
            buf.pop(0)
        mu = np.asarray(buf).mean(axis=0)
        delta = float(np.linalg.norm(phi - mu) / np.sqrt(p))
        if len(dh) >= 30:
            hh = np.asarray(dh[-200:])
            med = np.median(hh)
            mad = np.median(np.abs(hh - med)) * 1.4826 + 1e-9
            z = (delta - med) / mad
        else:
            z = 0.0
        dh.append(delta)
        if z >= theta2:
            streak += 1
            tier = "RESET" if streak >= persist else "ADAPT"
        elif z >= theta1:
            streak = 0; tier = "ADAPT"
        else:
            streak = 0; tier = "HOLD"
        out.append((z, tier))
    return out


def main(seed=7, doms=("STRESS-A", "STRESS-B", "STRESS-C", "STRESS-0")):
    for dom in doms:
        z = np.load(os.path.join(POOL, f"{dom}__seed{seed}.npz"), allow_pickle=True)
        P, Y, X, dow = z["P"], z["Y"], z["X"], z["dow"]
        tr, va, te = z["idx_train"], z["idx_val"], z["idx_test"]
        M = int(P.shape[1])
        lm = np.array([[np.mean((P[t][m] - Y[t]) ** 2) for m in range(M)]
                       for t in range(len(P))])
        capd = build_envelope(P, Y, np.concatenate([tr, va, te]))
        caps = np.array([capd[int(t)] for t in range(len(P))])
        sw = np.asarray(z["switches"], dtype=int)
        blen = int(z["block_len"])
        srcs = list(z["block_sources"])

        def W(lo, tt):
            return float(np.mean(np.minimum(lo, caps[tt])))

        cfg0 = SystemConfig(seed=seed)
        fp = Fingerprint(K=cfg0.n_spec_bins)
        fp.fit(X[tr], dow[tr])
        PHI = fp.transform(X, dow)

        print(f"\n=== {dom}  blocks {blen}d x {len(srcs)}  "
              f"sources {sorted(set(srcs))}  test switches {sw.tolist()} ===")

        # ---- Q1 detector quality (calibrated on the validation split)
        best = None
        for th1, th2 in ((1.0, 2.0), (1.5, 2.5), (1.5, 3.0), (2.0, 3.5), (2.5, 4.0)):
            out = detect_stream(PHI, cfg0.detect_window, th1, th2,
                                cfg0.detect_persist)
            zz = np.array([o[0] for o in out])
            fired_v = [t for t in va if zz[t] >= th2]
            tp_v = sum(1 for s in sw if any(abs(f - s) <= 3 for f in fired_v))
            fp_v = sum(1 for f in fired_v if not any(abs(f - s) <= 3 for s in sw))
            prec = tp_v / (tp_v + fp_v + 1e-9)
            rec = tp_v / max(len(sw), 1)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            if best is None or f1 > best[0]:
                best = (f1, th1, th2, prec, rec)
        print(f"  Q1 detector (val-calibrated) theta1={best[1]} theta2={best[2]} "
              f"prec={best[3]:.2f} rec={best[4]:.2f} F1={best[0]:.2f}")

        # ---- Q2 fixed vs oracle-adaptive half-life
        bestf = (np.inf, None)
        for hl in (2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 60.0):
            LO = engine(P, Y, lm, lambda t, hl=hl: hl)
            v = W(LO[va], va)
            if v < bestf[0]:
                bestf = (v, hl)
        hl_fix = bestf[1]
        LO_fix = engine(P, Y, lm, lambda t, hl=hl_fix: hl)

        def oracle_hl(t):
            return 2.0 if any(0 <= t - s < 15 for s in sw) else hl_fix
        LO_or = engine(P, Y, lm, oracle_hl)
        wf, wo = W(LO_fix[te], te), W(LO_or[te], te)
        print(f"  Q2 fixed hl={hl_fix:<5} W(te)={wf:.5f}   oracle-adaptive "
              f"W(te)={wo:.5f}   gain {100*(wf-wo)/wf:+.2f}%")

        # ---- baselines
        mse_v = np.array([np.mean(np.minimum(lm[va][:, m], caps[va]))
                          for m in range(M)])
        b5 = int(np.argmin(mse_v))
        b1 = W(lm[te][:, b5], te)
        lam_s = _simplex_fit(P[va], Y[va])
        b1v = W(np.array([np.mean((lam_s @ P[t] - Y[t]) ** 2) for t in te]), te)
        bb = np.inf
        for e in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            hh = B4Hedge(M, eta=e)
            for t in va:
                hh.step(int(t), PHI[t], P[t], Y[t])
            L = []
            for t in te:
                L.append(np.mean((hh.lam @ P[t] - Y[t]) ** 2)); hh.step(int(t), PHI[t], P[t], Y[t])
            bb = min(bb, W(np.array(L), te))
        b2 = B2MonthlyRetrain(M)
        for t in va:
            b2.step(int(t), PHI[t], P[t], Y[t])
        L = []
        for t in te:
            L.append(np.mean((b2.lam @ P[t] - Y[t]) ** 2)); b2.step(int(t), PHI[t], P[t], Y[t])
        b2v = W(np.array(L), te)
        base = min(b1, b1v, bb, b2v)
        print(f"  Q2' best baseline W(te)={base:.5f}  (B1={b1v:.5f} "
              f"B2={b2v:.5f} B4={bb:.5f} B5={b1:.5f})  "
              f"engine gain {100*(base-wf)/base:+.2f}%   "
              f"oracle-expert {W(lm[te].min(axis=1), te):.5f}")

        # ---- Q3 first vs second appearance of a recurring regime
        occ = {}
        order = []
        for b, s in enumerate(srcs):
            occ[s] = occ.get(s, 0) + 1
            order.append((s, occ[s], b * blen))
        bysrc = {}
        for s, k, start in order:
            win = np.arange(start, min(start + 20, len(P)))
            win = win[win >= int(te[0])]
            if len(win) < 8:
                continue
            bysrc.setdefault(s, {})[k] = W(LO_fix[win], win)
        shown = 0
        for s, d in bysrc.items():
            if 1 in d and 2 in d:
                print(f"  Q3 {s:<8} 1st={d[1]:.5f} 2nd={d[2]:.5f} "
                      f"({100*(d[1]-d[2])/d[1]:+.1f}%)")
                shown += 1
        if not shown:
            print("  Q3 no regime recurs inside the test split of this stress")


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    d = tuple(sys.argv[2:]) or ("STRESS-A", "STRESS-B", "STRESS-C", "STRESS-0")
    main(s, d)

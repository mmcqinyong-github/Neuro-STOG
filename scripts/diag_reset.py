"""Do the *mechanisms* add anything over the plain short-memory engine?

On STRESS-A the plain engine (hl = 3) gains +40% over the best baseline, and an
oracle that switches the half-life only adds +1%.  So the value of the gating /
reset / memory machinery has to be tested differently: let the system normally
run with a LONG half-life (which is better in stable periods) and ask what
should happen when a switch is signalled.

V0  fixed long half-life                          (the "no mechanism" arm)
V1  fixed short half-life                         (the plain engine)
V2  long half-life + flush   at oracle switches   (module 5, upper bound)
V3  long half-life + flush   at detected switches (module 5, as shipped)
V4  per-cluster statistics, affinity routing      (modules 2/3/8)
V5  V4 + flush the routed cluster at detection    (modules 4+5+2/3/8)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.combo import simplex_ls  # noqa: E402
from cerebrocast.config import SystemConfig  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.protocol import build_envelope  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def run(P, Y, lm, hl, flush_at=None, decay=0.05, ridge=1e-2):
    M = P.shape[1]
    G = np.zeros((M, M)); h = np.zeros(M); n = 0.0
    lam = np.full(M, 1.0 / M)
    LO = np.zeros(len(P))
    for t in range(len(P)):
        LO[t] = np.mean((lam @ P[t] - Y[t]) ** 2)
        if flush_at is not None and flush_at[t]:
            G *= decay; h *= decay; n *= decay
        rho = 0.5 ** (1.0 / max(hl, 0.5))
        w = 1.0 / (np.median(lm[t]) + 1e-9)
        G = rho * G + w * (P[t] @ P[t].T)
        h = rho * h + w * (P[t] @ Y[t])
        n = rho * n + w
        if n > 1e-8:
            s = simplex_ls(G / n, h / n, ridge, 0.0)
            if s is not None:
                lam = s
    return LO


def run_clusters(P, Y, lm, PHI, hl, K, flush_at=None, decay=0.05,
                 ridge=1e-2, tau=None, seed=7):
    """Per-cluster second-moment statistics with soft affinity routing."""
    from sklearn.cluster import KMeans
    n, M = len(P), P.shape[1]
    km = KMeans(n_clusters=K, n_init=5, random_state=seed).fit(PHI[:n // 2])
    cen = km.cluster_centers_
    d = np.linalg.norm(PHI[:n // 2] - cen[km.labels_], axis=1)
    tau = tau if tau is not None else float(max(0.75 * np.median(d), 1e-6))
    G = np.zeros((K, M, M)); h = np.zeros((K, M)); nn = np.zeros(K)
    LAM = np.full((K, M), 1.0 / M)
    LO = np.zeros(n)
    for t in range(n):
        dd = np.linalg.norm(cen - PHI[t], axis=1)
        pi = np.exp(-(dd ** 2) / (2 * tau ** 2))
        pi /= pi.sum()
        lam = pi @ LAM
        LO[t] = np.mean((lam @ P[t] - Y[t]) ** 2)
        if flush_at is not None and flush_at[t]:
            c = int(np.argmax(pi))
            G[c] *= decay; h[c] *= decay; nn[c] *= decay
        rho = 0.5 ** (1.0 / max(hl, 0.5))
        w = 1.0 / (np.median(lm[t]) + 1e-9)
        PP = P[t] @ P[t].T
        G = rho * G + (pi * w)[:, None, None] * PP[None]
        h = rho * h + (pi * w)[:, None] * (P[t] @ Y[t])[None]
        nn = rho * nn + pi * w
        for c in range(K):
            if nn[c] > 1e-8:
                s = simplex_ls(G[c] / nn[c], h[c] / nn[c], ridge, 0.0)
                if s is not None:
                    LAM[c] = s
    return LO


def detect(PHI, window, theta, persist=1):
    buf, dh, out = [], [], []
    p = PHI.shape[1]
    for t in range(len(PHI)):
        phi = PHI[t]; buf.append(phi)
        if len(buf) > window: buf.pop(0)
        mu = np.asarray(buf).mean(0)
        delta = float(np.linalg.norm(phi - mu) / np.sqrt(p))
        if len(dh) >= 30:
            hh = np.asarray(dh[-200:]); med = np.median(hh)
            mad = np.median(np.abs(hh - med)) * 1.4826 + 1e-9
            z = (delta - med) / mad
        else:
            z = 0.0
        dh.append(delta)
        out.append(z)
    return np.array(out)


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
        allsw = [int(i * int(z["block_len"])) for i in range(1, len(z["block_sources"]))]
        cfg0 = SystemConfig(seed=seed)
        fp = Fingerprint(K=cfg0.n_spec_bins)
        fp.fit(X[tr], dow[tr])
        PHI = fp.transform(X, dow)

        def W(lo, tt):
            return float(np.mean(np.minimum(lo, caps[tt])))

        # ---- pick hl and theta on validation
        bestL = (np.inf, None)
        for hl in (5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0):
            v = W(run(P, Y, lm, hl)[va], va)
            if v < bestL[0]:
                bestL = (v, hl)
        hlL = bestL[1]
        bestS = (np.inf, None)
        for hl in (1.5, 2.0, 3.0, 5.0, 7.0):
            v = W(run(P, Y, lm, hl)[va], va)
            if v < bestS[0]:
                bestS = (v, hl)
        hlS = bestS[1]
        dz = detect(PHI, cfg0.detect_window, None)
        bestT = (np.inf, None)
        for th in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
            fa = dz >= th
            v = W(run(P, Y, lm, hlL, fa)[va], va)
            if v < bestT[0]:
                bestT = (v, th)
        th = bestT[1]
        for dec in (0.05, 0.15, 0.35):
            pass
        bestD = (np.inf, None)
        for dec in (0.02, 0.05, 0.15, 0.35):
            v = W(run(P, Y, lm, hlL, dz >= th, dec)[va], va)
            if v < bestD[0]:
                bestD = (v, dec)
        dec = bestD[1]
        bestK = (np.inf, None)
        for K in (3, 5, 8):
            v = W(run_clusters(P, Y, lm, PHI, hlL, K)[va], va)
            if v < bestK[0]:
                bestK = (v, K)
        K = bestK[1]

        orc = np.zeros(len(P), bool)
        for s in allsw:
            if s < len(P):
                orc[s] = True
        res = {
            "V0 long hl": run(P, Y, lm, hlL),
            "V1 short hl": run(P, Y, lm, hlS),
            "V2 oracle flush": run(P, Y, lm, hlL, orc, dec),
            "V3 detected flush": run(P, Y, lm, hlL, dz >= th, dec),
            "V4 clusters": run_clusters(P, Y, lm, PHI, hlL, K),
            "V5 clusters+flush": run_clusters(P, Y, lm, PHI, hlL, K, dz >= th, dec),
        }
        print(f"\n=== {dom}  hl_long={hlL} hl_short={hlS} theta={th} "
              f"decay={dec} K={K} ===")
        for k, LO in res.items():
            print(f"   {k:<18} val={W(LO[va], va):.5f}  test={W(LO[te], te):.5f}")


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    d = tuple(sys.argv[2:]) or ("STRESS-A", "STRESS-B", "STRESS-C", "STRESS-0")
    main(s, d)

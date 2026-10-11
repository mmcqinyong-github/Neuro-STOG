"""Build the non-stationarity ladder out of the real expert pools.

Motivation
----------
On the eleven individual benchmarks a monthly-refit simplex combination already
sits on the causal frontier: the per-day *oracle* expert is far better than any
causal method we could build, but predicting which expert will be good from
features that are visible before the day is not learnable on these data
(measured: ridge predictions are anti-correlated with the truth, corr = -0.5).
Those benchmarks are simply too close to stationary for adaptivity to pay, and
we report that as a null result.

The ladder below is the honest complement.  Every rung is the *same* stream
length (12 blocks x 60 days), the *same* block length and the *same* expert
pool, and differs only in how much the data-generating process actually changes
along the stream:

  STAT-0  one contiguous out-of-sample window of a single series -- no splice,
          so the stream is as stationary as the data allow
  STAT-1  12 random windows of the *same* series -- discontinuities in level and
          season, but one DGP
  SW-2    alternating windows of two series of the same family
  SW-4    cycling windows of four series of the same family
  SW-X    cycling windows of three series from *different* families

If the architecture's advantage is caused by genuine distribution change then
the gain over the best causal baseline should increase along the ladder; if it
is an artefact of the splicing procedure, STAT-0 and STAT-1 would show it too.

Blocks are drawn only from the out-of-sample region of each pool (the fit block
has no forecasts by construction) and are rescaled by the standard deviation of
their own targets so that the combination problem stays well posed when the
sources live on different scales.  Single-source rungs use one global scale so
that the control is not itself given a per-block variance shift.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "pool")

N_BLOCKS = 12
BLEN = 60
POOL_SEED = 7                 # the pool is one realisation, shared by all seeds
STREAM_SPLIT = (0.20, 0.10, 0.70)       # warm / val / test inside the ladder
LADDER = {
    "STAT-0": (["EPF-NP"], "contiguous"),
    "STAT-1": (["EPF-NP"], "random"),
    "SW-2": (["EPF-NP", "EPF-FR"], "random"),
    "SW-4": (["EPF-NP", "EPF-FR", "EPF-DE", "EPF-PJM"], "random"),
    "SW-X": (["ETTh1", "ECL", "Weather"], "random"),
}
DEGENERATE = 0.05          # min block sd, as a fraction of the source sd


def build(seed: int, name: str, srcs: list[str], mode: str) -> dict:
    zs = {d: np.load(os.path.join(POOL, f"{d}__seed{POOL_SEED}.npz"), allow_pickle=True)
          for d in srcs}
    M = min(int(z["P"].shape[1]) for z in zs.values())
    span = N_BLOCKS * BLEN
    rng = np.random.default_rng(seed * 1000 + abs(hash(name)) % 997)

    def start_bounds(z):
        """Valid block starts: inside the out-of-sample region, room for BLEN."""
        lo = int(z["idx_fit"][-1]) + 1 if len(z["idx_fit"]) else 0
        hi = int(z["Y"].shape[0]) - BLEN - 1
        if hi <= lo:
            raise ValueError(f"{name}: source has {int(z['Y'].shape[0])} days, "
                             f"out-of-sample region too short for {BLEN}-day blocks")
        return lo, hi

    Ps, Ys, Xs, Ds, cuts, sw = [], [], [], [], [], []
    off = 0
    if mode == "contiguous":
        z0 = zs[srcs[0]]
        n0 = int(z0["Y"].shape[0])
        lo, hi = int(z0["idx_fit"][-1]) + 1, n0 - span - 1
        if hi <= lo:
            raise ValueError(f"{name}: need {span} contiguous out-of-sample days, "
                             f"only {n0 - int(z0['idx_fit'][-1]) - 1} available")
        base = int(rng.integers(lo, hi))
        sd_all = float(np.std(np.asarray(z0["Y"])))
        for b in range(N_BLOCKS):
            sl = slice(base + b * BLEN, base + (b + 1) * BLEN)
            Ps.append(np.array(z0["P"][sl][:, :M, :], dtype=float) / sd_all)
            Ys.append(np.array(z0["Y"][sl], dtype=float) / sd_all)
            Xs.append(np.array(z0["X"][sl], dtype=float))
            Ds.append(np.array(z0["dow"][sl]))
            off += BLEN
            cuts.append(off)
            sw.append(srcs[0])
    else:
        single = len(set(srcs)) == 1
        for b in range(N_BLOCKS):
            src = srcs[b % len(srcs)]
            z = zs[src]
            lo, hi = start_bounds(z)
            lo_sd = float(np.std(np.asarray(z["Y"]))) * DEGENERATE
            Yb = None
            sl = slice(lo, lo + BLEN)
            for _ in range(80):
                start = int(rng.integers(lo, hi))
                sl = slice(start, start + BLEN)
                Yb = np.asarray(z["Y"][sl], dtype=float)
                if float(np.std(Yb)) >= lo_sd:
                    break
            P = np.array(z["P"][sl][:, :M, :], dtype=float)
            X = np.array(z["X"][sl], dtype=float)
            D = np.array(z["dow"][sl])
            sd = float(np.std(np.asarray(z["Y"]))) if single else float(np.std(Yb))
            if not np.isfinite(sd) or sd < 1e-12:
                sd = 1.0
            Ps.append(P / sd)
            Ys.append(Yb / sd)
            Xs.append(X)
            Ds.append(D)
            off += BLEN
            cuts.append(off)
            sw.append(src)
    P = np.concatenate(Ps, axis=0)
    Y = np.concatenate(Ys, axis=0)
    X = np.concatenate(Xs, axis=0)
    dow = np.concatenate(Ds, axis=0)
    n = len(Y)
    a, b_, c = STREAM_SPLIT
    i0 = int(n * a)
    i1 = i0 + int(n * b_)
    idx = dict(warm=np.arange(0, i0), val=np.arange(i0, i1), test=np.arange(i1, n))
    return dict(
        P=P, Y=Y, X=X, dow=dow, hour=np.array(next(iter(zs.values()))["hour"]),
        names=np.array(next(iter(zs.values()))["names"][:M]), mu=0.0, sd=1.0,
        idx_fit=np.array([], dtype=int), idx_warm=idx["warm"],
        idx_val=idx["val"], idx_test=idx["test"], idx_train=idx["warm"],
        family="STRESS",
        switches=np.array([x for x in cuts[:-1] if x >= i1], dtype=int),
        block_sources=np.array(sw),
        block_len=BLEN,
        ladder_rung=name,
    )


def main(seeds=(7,)):
    for seed in seeds:
        for name, (srcs, mode) in LADDER.items():
            d = build(seed, name, srcs, mode)
            path = os.path.join(OUT, f"{name}__seed{seed}.npz")
            np.savez_compressed(path, **d)
            te = d["idx_test"]
            lm = np.array([[np.mean((d["P"][t][m] - d["Y"][t]) ** 2)
                            for m in range(d["P"].shape[1])] for t in te])
            print(f"{name:<8} seed{seed}: n={len(d['Y'])} M={d['P'].shape[1]} "
                  f"test={len(te)} switches_in_test={len(d['switches'])} "
                  f"zero-oracle={int((lm.min(1) < 1e-12).sum())}/{len(te)}",
                  flush=True)


if __name__ == "__main__":
    ss = tuple(int(x) for x in sys.argv[1:]) or (7,)
    main(ss)

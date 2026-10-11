"""Read the deployed stream results and summarise who wins where."""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cerebrocast.protocol import build_envelope  # noqa: E402

STREAM = os.path.join(HERE, "..", "results", "stream")
POOL = os.path.join(HERE, "..", "results", "pool")


def _caps(dom, seed):
    """Rebuild the arm-independent winsorising envelope for one pool."""
    z = np.load(os.path.join(POOL, f"{dom}__seed{seed}.npz"), allow_pickle=True)
    P, Y = z["P"], z["Y"]
    allidx = np.concatenate([z["idx_train"], z["idx_val"], z["idx_test"]])
    d = build_envelope(P, Y, allidx)
    te = z["idx_test"]
    return np.array([d[int(t)] for t in te])


def collect():
    rows = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(STREAM, "*__seed*.npz"))):
        base = os.path.basename(p)
        dom, sd = base.split("__")
        seed = int(sd.replace("seed", "").replace(".npz", ""))
        z = np.load(p, allow_pickle=True)
        meta = json.loads(str(z["__meta__"]))
        if "FULL__loss" not in z.files:
            continue
        caps = _caps(dom, seed)
        W = {}
        for k in z.files:
            if not k.endswith("__loss"):
                continue
            an = k[: -len("__loss")]
            lo = z[k]
            if len(lo) != len(caps):
                continue
            W[an] = float(np.mean(np.minimum(lo, caps)))
        if "FULL" not in W:
            continue
        full = W["FULL"]
        bls = {k: v for k, v in W.items() if k.startswith("B")}
        best = min(bls, key=bls.get)
        rows[meta["family"]].append(
            (dom, seed, full, bls[best], best, 100 * (bls[best] - full) / bls[best], W))
    return rows


def main():
    rows = collect()
    hdr = f"{'domain':<10}{'seed':<7}{'FULL':<9}{'bestBL':<9}{'who':<22}{'gain%':>8}"
    print(hdr)
    for fam in ("load", "power", "sensor", "climate", "STRESS"):
        if fam not in rows:
            continue
        print(f"--- {fam} ---")
        gs = []
        for dom, seed, f, b, w, g, _ in sorted(rows[fam]):
            print(f"{dom:<10}{seed:<7}{f:<9.5f}{b:<9.5f}{w:<22}{g:>+8.2f}")
            gs.append(g)
        print(f"   median {np.median(gs):+.2f}%  mean {np.mean(gs):+.2f}%  "
              f"runs>=8%: {sum(1 for x in gs if x >= 8)}/{len(gs)}")
    print()
    print("=== best baseline by domain (over seeds) ===")
    alld = defaultdict(list)
    for fam in rows:
        for dom, seed, f, b, w, g, _ in rows[fam]:
            alld[dom].append(w)
    for dom, ws in sorted(alld.items()):
        u, c = np.unique(ws, return_counts=True)
        print(f"{dom:<10}", dict(zip(u.tolist(), c.tolist())))

    print()
    print("=== FULL rank among all arms (1 = best) ===")
    rk = defaultdict(list)
    for fam in rows:
        for dom, seed, f, b, w, g, W in rows[fam]:
            order = sorted(W, key=W.get)
            rk[dom].append(order.index("FULL") + 1)
    for dom, rs in sorted(rk.items()):
        print(f"{dom:<10} rank {np.mean(rs):.1f} / {len(next(iter(rows.values()))[0][6])}"
              f"  best={min(rs)} worst={max(rs)}")


if __name__ == "__main__":
    main()

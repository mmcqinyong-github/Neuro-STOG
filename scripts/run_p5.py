"""P5 -- measure the mechanism costs on the five directed stress families.

For each family the pre-registration predicts *which* mechanism the family should
expose: the family is a counterexample for that mechanism, so removing it must
cost strictly more there than elsewhere.  This script runs the full architecture
and every ablation on each family and reports the realised cost ordering, so the
prediction is testable rather than asserted.

Output: `results/synth/p5_families.json`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace as dc_replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.config import SystemConfig, default_arms  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.memory import ClusterField  # noqa: E402
from cerebrocast.synth_p5 import FAMILIES, FAMILY_TARGET, generate_family  # noqa: E402
from cerebrocast.system import CerebroCast  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "synth")

GROUPS = {
    "detection": ["A4_no_change_detection"],
    "identification": ["A2_no_episodic", "A3_no_clusters"],
    "adaptation": ["A5_no_reset", "A6_no_forward_model"],
    "maintenance": ["A7_no_three_factor", "A8_no_metaplasticity", "A9_no_replay",
                    "A10_no_downscaling", "A11_no_growth"],
}
# which group each family is *supposed* to indict
PREDICTED = {
    "periodic_jump": "detection",
    "rare_high_loss": "detection",
    "heavy_tail": "maintenance",
    "regime_cycle": "identification",
    "linear_drift": "adaptation",
}

ABL = [a for a in GROUPS["detection"] + GROUPS["identification"]
       + GROUPS["adaptation"] + GROUPS["maintenance"]]


def _run(meta, cfg, arm, seed, n_warm=60):
    X, Y, P = meta["X"], meta["Y"], meta["P"]
    M = P.shape[1]
    n = len(X)
    ntr = int(n * 0.5)
    fp = Fingerprint(K=cfg.n_spec_bins)
    fp.fit(X[:ntr])
    PHI = fp.transform(X)
    field = ClusterField(
        tau_kappa=cfg.tau_kappa, graph_alpha=cfg.graph_alpha,
        cluster_momentum=cfg.cluster_momentum, grow_r_cov=cfg.grow_r_cov,
        grow_persist=cfg.grow_persist, prune_mass=cfg.prune_mass,
        prune_idle=cfg.prune_idle, k0=cfg.k0_clusters).init_from_train(PHI[:ntr], M, seed)
    s = CerebroCast(M, cfg, arm, field, PHI.shape[1], seed=seed)
    out = []
    for t in range(n):
        d = s.step(t, PHI[t], P[t], Y[t])
        if t >= n_warm:
            out.append(d["loss_raw"])
    return np.asarray(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 42, 2021])
    ap.add_argument("--n-days", type=int, default=900)
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    arms = {x.name: x for x in default_arms()}

    rows = []
    for fam in a.families:
        for sd in a.seeds:
            meta = generate_family(fam, n_days=a.n_days, seed=sd)
            cfg = SystemConfig(seed=sd)
            base = float(np.mean(_run(meta, cfg, arms["FULL"], sd)))
            costs = {}
            for nm in ABL:
                v = float(np.mean(_run(meta, cfg, arms[nm], sd)))
                costs[nm] = 100.0 * (v - base) / max(1e-12, base)
            group = {}
            for g, members in GROUPS.items():
                vals = [costs[m] for m in members if m in costs]
                group[g] = float(np.mean(vals)) if vals else np.nan
            pred = PREDICTED[fam]
            got = max(group, key=lambda k: group[k])
            rows.append(dict(family=fam, seed=sd, target=FAMILY_TARGET[fam],
                             FULL_loss=base, **{f"cost_{k}": v for k, v in costs.items()},
                             **{f"group_{k}": v for k, v in group.items()},
                             predicted_group=pred, realised_top_group=got,
                             prediction_holds=bool(got == pred)))
            print(f"[{fam} s{sd}] top group = {got} "
                  f"(predicted {pred})  holds={got == pred}", flush=True)

    path = os.path.join(a.out, "p5_families.json")
    json.dump(rows, open(path, "w", encoding="utf-8"), indent=1, default=str)

    agg = {}
    for r in rows:
        agg.setdefault(r["family"], []).append(r)
    print("\nfamily            predicted      realised top   holds  group costs "
          "(det/id/adapt/maint, pct)")
    for fam, rs in agg.items():
        holds = np.mean([r["prediction_holds"] for r in rs])
        g = {k: np.mean([r[f"group_{k}"] for r in rs]) for k in GROUPS}
        top = max(g, key=lambda k: g[k])
        print(f"{fam:<17} {PREDICTED[fam]:<14} {top:<14} {holds:.2f}   "
              f"{g['detection']:+.2f}/{g['identification']:+.2f}/"
              f"{g['adaptation']:+.2f}/{g['maintenance']:+.2f}")
    n_hold = np.mean([r["prediction_holds"] for r in rows])
    print(f"\nP5 ordering prediction holds on {n_hold:.2%} of "
          f"{len(rows)} family-seed runs")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

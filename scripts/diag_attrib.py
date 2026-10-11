"""C1 -- attribution regression: does the environment explain the mechanism costs?

Pre-registration (Section 5, conjecture C1) states that the total regret of the
architecture decomposes into four attributable terms -- detection, identification,
adaptation, maintenance -- and gives an explicit falsification criterion: an
*attribution regression* below threshold on the synthetic grid.

The test implemented here is deliberately non-circular.  The **predictors** are
statistics that can be computed from the stream itself, with no access to any
arm (robust change-signal level, gate tier frequencies, fingerprint drift,
difficulty dispersion, irreducible noise floor).  The **targets** are the
realised costs of the four mechanism groups, measured by the ablation arms on
the identical days.  If the decomposition is meaningful, the environment
statistics should predict the group costs out of sample.

Validation is leave-one-domain-out, so a positive reading is not the in-sample
fit of a saturated model.  Every number is written to
`results/synth/c1_attribution.json`; the pre-registered threshold is R^2 >= 0.6.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.data import ALL_ORDER  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")
STREAM = os.path.join(os.path.dirname(__file__), "..", "results", "stream")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "synth")

R2_THRESHOLD = 0.6

GROUPS = {
    "detection": ["A4_no_change_detection"],
    "identification": ["A2_no_episodic", "A3_no_clusters"],
    "adaptation": ["A5_no_reset", "A6_no_forward_model"],
    "maintenance": ["A7_no_three_factor", "A8_no_metaplasticity",
                    "A9_no_replay", "A10_no_downscaling", "A11_no_growth"],
}
FEATURES = ["mean_delta_z", "frac_adapt", "frac_hold", "frac_reset",
            "fp_drift", "difficulty_iqr", "noise_floor", "pool_median"]


def features_for(dom: str, seed: int, pool_seed: int = 7) -> Dict[str, float]:
    """Observable environment statistics -- computed without touching any arm."""
    zp = os.path.join(POOL, f"{dom}__seed{pool_seed}.npz")
    zs = os.path.join(STREAM, f"{dom}__seed{seed}.npz")
    if not (os.path.exists(zp) and os.path.exists(zs)):
        return {}
    zp, zs = np.load(zp, allow_pickle=True), np.load(zs, allow_pickle=True)
    X = zp["X"]
    te = zp["idx_test"]
    trial = zs["ref__pool_median"]
    core = zs["ref__oracle_expert"]

    # robust change-signal level and gate tier frequencies, from the FULL arm
    z = np.asarray(zs["FULL__z"], dtype=float)
    tier = np.asarray([str(t) for t in zs["FULL__tier"]])
    lm = np.asarray(zs["FULL__lm"], dtype=float)
    dmed = np.median(lm, axis=1)

    # fingerprint drift: mean distance between consecutive days' raw windows
    W = X[te]
    fp_drift = float(np.mean(np.linalg.norm(np.diff(W, axis=0), axis=1)
                             / max(1.0, np.mean(np.linalg.norm(W, axis=1)))))

    return dict(
        mean_delta_z=float(np.mean(z)),
        frac_adapt=float(np.mean(tier == "ADAPT")),
        frac_hold=float(np.mean(tier == "HOLD")),
        frac_reset=float(np.mean(tier == "RESET")),
        fp_drift=fp_drift,
        difficulty_iqr=float(np.percentile(dmed, 75) - np.percentile(dmed, 25)),
        noise_floor=float(np.median(core)),
        pool_median=float(np.median(trial)),
    )


def group_costs(dom: str, seed: int) -> Dict[str, float]:
    """Realised cost of each mechanism group = mean relative ablation penalty."""
    zs = os.path.join(STREAM, f"{dom}__seed{seed}.npz")
    if not os.path.exists(zs):
        return {}
    z = np.load(zs, allow_pickle=True)
    caps = z["__caps__"]

    def W(name):
        return float(np.mean(np.minimum(np.asarray(z[name + "__loss"]), caps)))

    wf = W("FULL")
    out = {}
    for g, arms in GROUPS.items():
        vals = []
        for a in arms:
            try:
                vals.append(100.0 * (W(a) - wf) / max(1e-12, wf))
            except KeyError:
                continue
        out[g] = float(np.mean(vals)) if vals else np.nan
    return out


def ridge_fit(X, y, lam=1.0):
    X1 = np.column_stack([X, np.ones(len(X))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    return np.linalg.solve(A, X1.T @ y)


def cv_r2(X, y, groups, lam=1.0):
    """Leave-one-GROUP-out cross-validated R^2 (groups = domains)."""
    yhat = np.full(len(y), np.nan)
    for g in np.unique(groups):
        tr = groups != g
        te = ~tr
        if tr.sum() < X.shape[1] + 2 or te.sum() == 0:
            continue
        w = ridge_fit(X[tr], y[tr], lam)
        X1 = np.column_stack([X[te], np.ones(te.sum())])
        yhat[te] = X1 @ w
    ok = np.isfinite(yhat)
    if ok.sum() < 3:
        return float("nan"), 0
    yv, yh = y[ok], yhat[ok]
    ss_res = float(np.sum((yv - yh) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    return (1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=ALL_ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 42, 2021, 3407, 12345])
    ap.add_argument("--pool-seed", type=int, default=7)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    Xs, doms, tg = [], [], {g: [] for g in GROUPS}
    for dom in a.domains:
        for s in a.seeds:
            f = features_for(dom, s, a.pool_seed)
            c = group_costs(dom, s)
            if not f or not c or not all(np.isfinite(list(c.values()))):
                continue
            Xs.append([f[k] for k in FEATURES])
            doms.append(dom)
            for g in GROUPS:
                tg[g].append(c[g])
    if len(Xs) < 8:
        print(f"not enough streams for the regression ({len(Xs)})")
        return
    X = np.asarray(Xs, dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)      # standardise, incl. the ladder
    doms = np.asarray(doms)
    print(f"n = {len(X)} streams over {len(set(doms))} domains, "
          f"{X.shape[1]} environment features\n")
    res = {"n_streams": int(len(X)), "n_domains": int(len(set(doms))),
           "features": FEATURES, "threshold": R2_THRESHOLD, "groups": {}}
    r2s = []
    for g in GROUPS:
        y = np.asarray(tg[g], dtype=float)
        r2, n = cv_r2(X, y, doms)
        res["groups"][g] = dict(r2_out_of_sample=float(r2), n=int(n),
                                mean_cost_pct=float(np.mean(y)))
        r2s.append(r2)
        print(f"  {g:<16} out-of-sample R^2 = {r2:+.3f}   "
              f"(mean cost {np.mean(y):+.3f}%)")
    # pooled: predict every group cost with the group as a dummy
    ypool, gp = [], []
    for g in GROUPS:
        ypool.extend(tg[g])
        gp.extend([g] * len(tg[g]))
    Xpool = np.repeat(X, len(GROUPS), axis=0)
    dummy = np.zeros((len(gp), len(GROUPS)))
    for i, g in enumerate(gp):
        dummy[i, list(GROUPS).index(g)] = 1.0
    Xa = np.column_stack([Xpool, dummy[:, 1:]])
    r2p, np_ = cv_r2(Xa, np.asarray(ypool, dtype=float), np.repeat(doms, len(GROUPS)))
    res["pooled"] = dict(r2_out_of_sample=float(r2p), n=int(np_))
    res["supported"] = bool(np.all(np.array(r2s) >= R2_THRESHOLD))
    res["verdict"] = ("supported: every group is predictable from the environment "
                      "out of sample" if res["supported"] else
                      "not supported: at least one mechanism group is not "
                      "predictable from observable environment statistics")
    print(f"\n  pooled out-of-sample R^2 = {r2p:+.3f}")
    print(f"  C1 (threshold R^2 >= {R2_THRESHOLD}): {res['verdict']}")
    path = os.path.join(a.out, "c1_attribution.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

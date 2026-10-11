"""H6 -- expert onboarding: does the responsibility signal find a new expert?

Review item A1 (compliance review) / item 1 (independent evaluation).  The
pre-registered threshold is

    H6: within 30 days of an expert being released into the pool, the
        responsibility signal converges to the post-hoc top-5 ranking.

Design
------
The pool already contains ``M`` experts; we *reserve* three of them and declare
them unavailable until calendar days inside the test split, then release them:

    release point 1 -- at 1/3 of the test split, one expert becomes available
    release point 2 -- at 2/3 of the test split, two more become available

The reservation is enforced inside ``CerebroCast`` (per-day availability mask):
an unreleased expert contributes nothing to the mixture *and* accumulates no
second-moment history, so onboarding is not free -- the system really has to
notice the newcomer after the fact.  This is the "cheapest missing experiment"
the review recommended, and it needs no new data.

The reserved experts are chosen to be genuinely good: their post-hoc rank on the
test period is inside the top 5, so the convergence target is attainable and the
test is not rigged to fail.

Protocol matches the main streaming experiment: warm-up on the training split,
restore, walk the validation split, then the test split; the calibration stored
by ``run_stream.py`` is reused so H6 is measured on the same configuration as
FULL.
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
from cerebrocast.data import ALL_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.memory import ClusterField  # noqa: E402
from cerebrocast.system import CerebroCast  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")
STREAM = os.path.join(os.path.dirname(__file__), "..", "results", "stream")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "synth")

N_LATE = 3           # experts held back
TOP_K = 5            # the pre-registered ranking target
WINDOW = 5           # trailing window used to read the responsibility mass
CONVERGE_DAYS = 30   # pre-registered budget


def load_calib(dom: str, pool_seed: int) -> dict:
    p = os.path.join(STREAM, f"{dom}__seed{pool_seed}.npz")
    if not os.path.exists(p):
        return {}
    try:
        z = np.load(p, allow_pickle=True)
        return json.loads(str(z["__meta__"])).get("calib", {})
    except Exception:  # noqa: BLE001 -- file may be mid-write by a parallel run
        return {}


def pick_late(P, Y, te, M, n_late=N_LATE, top_k=TOP_K):
    """Reserve experts whose post-hoc test-period rank is inside the top `top_k`."""
    mse = np.array([np.mean((P[te, m] - Y[te]) ** 2) for m in range(M)])
    order = np.argsort(mse)                       # best first
    cand = order[:max(n_late, top_k)]             # genuinely good experts
    # spread the picks over the good band so they are not all the same expert
    idx = np.linspace(1, len(cand) - 1, n_late).round().astype(int)
    late = [int(cand[i]) for i in idx]
    return sorted(set(late)), mse, order


def run_domain(dom: str, seed: int, pool_seed: int = 7):
    z = np.load(os.path.join(POOL, f"{dom}__seed{pool_seed}.npz"), allow_pickle=True)
    P, Y, X, dow = z["P"], z["Y"], z["X"], z["dow"]
    tr, va, te = z["idx_train"], z["idx_val"], z["idx_test"]
    M = int(P.shape[1])

    cfg0 = SystemConfig(seed=seed)
    cal = load_calib(dom, pool_seed)
    if cal:
        cfg = dc_replace(cfg0,
                         recall_halflife=float(cal["recall_halflife"]),
                         consol_shrink=float(cal["consol_shrink"]),
                         consol_shrink_u=float(cal["consol_shrink_u"]),
                         reset_decay=float(cal["reset_decay"]),
                         beta_forward=float(cal["beta_forward"]),
                         eta=float(cal["eta"]), topk=int(cal["topk"]),
                         theta1=float(cal["theta1"]), theta2=float(cal["theta2"]),
                         gamma0=float(cal["gamma0"]))
    else:
        cfg = cfg0

    late, mse_all, order = pick_late(P, Y, te, M)
    if not late:
        return []
    rank_true = {m: int(np.where(order == m)[0][0]) + 1 for m in late}

    # release schedule inside the test split
    n_te = len(te)
    cut1 = te[max(0, n_te // 3)]
    cut2 = te[max(0, 2 * n_te // 3)]
    sched = {int(cut1): [late[0]], int(cut2): late[1:]}
    release_day = {late[0]: int(cut1)}
    for m in late[1:]:
        release_day[m] = int(cut2)

    avail = np.ones(M, dtype=bool)
    avail[late] = False

    fp = Fingerprint(K=cfg0.n_spec_bins)
    fp.fit(X[tr], dow[tr])
    PHI = fp.transform(X, dow)
    p = PHI.shape[1]

    def mk(av, sch):
        f = ClusterField(tau_kappa=cfg.tau_kappa, graph_alpha=cfg.graph_alpha,
                         cluster_momentum=cfg.cluster_momentum,
                         grow_r_cov=cfg.grow_r_cov, grow_persist=cfg.grow_persist,
                         prune_mass=cfg.prune_mass, prune_idle=cfg.prune_idle,
                         k0=cfg.k0_clusters)
        f.init_from_train(PHI[tr], M, seed)
        return CerebroCast(M, cfg, default_arms()[0], f, p, seed=seed,
                           avail=av, release_at=sch)

    # --- the onboarding arm -------------------------------------------------
    s = mk(avail.copy(), {k: list(v) for k, v in sched.items()})
    for t in tr:
        s.step(int(t), PHI[t], P[t], Y[t])
    snap = s.snapshot()
    s = mk(avail.copy(), {k: list(v) for k, v in sched.items()})
    s.restore(snap)
    for t in va:
        s.step(int(t), PHI[t], P[t], Y[t])
    days, lams, losses = [], [], []
    for t in te:
        d = s.step(int(t), PHI[t], P[t], Y[t])
        days.append(int(t))
        lams.append(np.asarray(d["lam"], dtype=float))
        losses.append(float(d["loss_raw"]))
    lams = np.asarray(lams)

    # --- the counterfactual arm: those experts never exist -----------------
    never = np.ones(M, dtype=bool)
    never[late] = False
    s2 = mk(never.copy(), {})
    for t in tr:
        s2.step(int(t), PHI[t], P[t], Y[t])
    snap2 = s2.snapshot()
    s2 = mk(never.copy(), {})
    s2.restore(snap2)
    for t in va:
        s2.step(int(t), PHI[t], P[t], Y[t])
    losses_never = np.array([s2.step(int(t), PHI[t], P[t], Y[t])["loss_raw"]
                             for t in te])

    out = []
    for m in late:
        rel_from = int(np.where(np.asarray(days) >= release_day[m])[0][0]) \
            if np.any(np.asarray(days) >= release_day[m]) else None
        if rel_from is None:
            continue
        mass = lams[:, m]
        # trailing mean responsibility of EVERY expert, so the ranking is over
        # the live pool rather than over a single expert's scalar series
        trail = np.array([lams[max(0, i - WINDOW + 1):i + 1].mean(axis=0)
                          for i in range(len(lams))])
        days_to_top5 = None
        for i in range(rel_from, len(days)):
            avail_now = [k for k in range(M) if k not in late or
                         release_day[k] <= days[i]]
            ranked = sorted(avail_now, key=lambda k: -trail[i][k])
            if m in ranked[:TOP_K]:
                days_to_top5 = int(i - rel_from)
                break
        out.append(dict(
            domain=dom, seed=seed, expert=int(m),
            true_rank_posthoc=rank_true[m],
            released_day=int(release_day[m]),
            release_frac=float(rel_from / max(1, len(days))),
            days_to_top5_rank=(days_to_top5 if days_to_top5 is not None else np.nan),
            converged_30d=bool(days_to_top5 is not None and days_to_top5 <= CONVERGE_DAYS),
            mass_at_release=float(mass[rel_from]),
            mass_30d_after=float(trail[min(rel_from + CONVERGE_DAYS, len(trail) - 1)][m]),
            expert_true_mse=float(mse_all[m]),
            pool_median_mse=float(np.median(mse_all)),
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=ALL_ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 42, 2021, 3407, 12345])
    ap.add_argument("--pool-seed", type=int, default=7)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    all_rows = []
    for dom in a.domains:
        for s in a.seeds:
            try:
                rs = run_domain(dom, s, a.pool_seed)
            except FileNotFoundError:
                continue
            all_rows.extend(rs)
            if rs:
                conv = np.mean([r["converged_30d"] for r in rs])
                print(f"[{dom} s{s}] events={len(rs)} converged30d={conv:.2f} "
                      f"days_to_top5={[r['days_to_top5_rank'] for r in rs]}", flush=True)
    path = os.path.join(a.out, "h6_onboarding.json")
    json.dump(all_rows, open(path, "w", encoding="utf-8"), indent=1, default=str)
    if all_rows:
        d = [r["days_to_top5_rank"] for r in all_rows
             if np.isfinite(r["days_to_top5_rank"])]
        print(f"\nH6: {len(all_rows)} onboarding events, "
              f"converged within 30d: "
              f"{np.mean([r['converged_30d'] for r in all_rows]):.2%}, "
              f"mean days to top-5: {np.mean(d) if d else float('nan'):.2f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

"""Stage 3 -- controlled-drift experiments (H2, H4, and the P2 boundary).

The synthetic generator is the only place where the true regime path is known,
so it is the only place where the three-tier gate can be scored and where the
"is RESET worth it" question can be answered by comparing against a counterfactual
that never resets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace as dc_replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.config import SystemConfig, default_arms  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.memory import ClusterField  # noqa: E402
from cerebrocast.stats import macro_f1, recovery_time  # noqa: E402
from cerebrocast.synth import SynthConfig, generate  # noqa: E402
from cerebrocast.system import CerebroCast  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "synth")
LABELS = ["HOLD", "ADAPT", "RESET"]


def _arm(name: str):
    for a in default_arms():
        if a.name == name:
            return a
    raise KeyError(name)


def run_stream(meta, cfg: SystemConfig, arm, seed, n_warm=250, p_reset=0.0, p_growth=0.0):
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
    s = CerebroCast(M, cfg, arm, field, PHI.shape[1], seed=seed,
                    placebo_reset_rate=p_reset, placebo_growth_rate=p_growth)
    tiers, losses, deltas = [], [], []
    for t in range(n):
        if t < 60:
            s.step(t, PHI[t], P[t], Y[t])
            continue
        d = s.step(t, PHI[t], P[t], Y[t])
        tiers.append(d["tier"])
        losses.append(d["loss"])
        deltas.append(d["delta"])
    return dict(tiers=np.array(tiers), losses=np.array(losses),
                delta=np.array(deltas), truth=np.asarray(meta["labels"][60:]))


def _forced(meta, sc, seed, mode, p_reset=0.0, p_growth=0.0):
    """Run the architecture with the gate pinned to one tier for every day."""
    from dataclasses import replace as _rep
    arm = _arm("FULL")
    if mode == "ADAPT":
        arm = _arm("A4_no_change_detection")
        cfg = sc
    elif mode == "HOLD":
        cfg = _rep(sc, theta1=1e9, theta2=1e9)
    else:                                   # RESET every day
        cfg = _rep(sc, theta1=-1e9, theta2=-1e9, detect_persist=1)
    return run_stream(meta, cfg, arm, seed, p_reset=p_reset, p_growth=p_growth)


def experiment_h4(seeds=(7, 42, 2021), n_days=1200, label_window=5):
    """Three-tier gate versus a decision-theoretic ground truth.

    The tier that *ought* to fire on day t is the one whose policy would have
    been best: we run the architecture three times, once with the gate pinned to
    HOLD, once to ADAPT and once to RESET, and take the argmin of their losses.
    Because a single day's loss is noisy the label is the argmin over a short
    trailing-averaged window, which makes it an oracle action label: it uses
    `label_window` days of future data and is never available to the method.
    """
    rows = []
    for sd in seeds:
        for p in (0.02, 0.05, 0.10):
            for sp in (0.0, 0.05):
                for g in (False, True):
                    cfg = SynthConfig(n_days=n_days, p_switch=p, spike=sp,
                                      gradual=g, seed=sd)
                    meta = generate(cfg)
                    sc = SystemConfig(seed=sd)
                    base = run_stream(meta, sc, _arm("FULL"), sd)
                    L = {}
                    for mode in ("HOLD", "ADAPT", "RESET"):
                        L[mode] = _forced(meta, sc, sd, mode)["losses"]
                    n = min(len(v) for v in L.values())
                    stack = np.stack([L[m][:n] for m in ("HOLD", "ADAPT", "RESET")])
                    # trailing mean over label_window days, aligned to the day
                    kern = np.ones(label_window) / label_window
                    sm = np.stack([np.convolve(v, kern, mode="full")[:n]
                                   for v in stack])
                    truth = np.array(["HOLD", "ADAPT", "RESET"])[np.argmin(sm, axis=0)]
                    tiers = base["tiers"][:n]
                    f = macro_f1(truth, tiers, LABELS)
                    rows.append(dict(seed=sd, p_switch=p, spike=sp, gradual=g,
                                     n_hold=int((truth == "HOLD").sum()),
                                     n_adapt=int((truth == "ADAPT").sum()),
                                     n_reset=int((truth == "RESET").sum()),
                                     macro_f1=f["macro_f1"], accuracy=f["accuracy"],
                                     **{f"f1_{k}": v for k, v in f["per_class"].items()}))
    return rows


def _recovery(losses: np.ndarray, b: int, window: int = 40,
              tol: float = 1.05) -> float:
    """Days until the post-break running mean loss returns to the pre-break
    median (times tol).  Returns `window` if it never does inside the window."""
    pre = losses[max(0, b - 30):b]
    base = float(np.median(pre)) if len(pre) else float(np.median(losses))
    l = losses[b:b + window]
    if len(l) == 0:
        return float("nan")
    run = np.cumsum(l) / np.arange(1, len(l) + 1)
    idx = np.where(run <= base * tol)[0]
    return float(idx[0]) if len(idx) else float(window)


def experiment_h2_p2(seeds=(7, 42, 2021, 3407, 12345), n_days=1200,
                     jumps=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Recovery time after an annotated break, and the RESET/ADAPT boundary."""
    rows = []
    for sd in seeds:
        for js in jumps:
            cfg = SynthConfig(n_days=n_days, p_switch=0.03, jump_scale=js, seed=sd)
            meta = generate(cfg)
            sc = SystemConfig(seed=sd)
            full = run_stream(meta, sc, _arm("FULL"), sd)
            nores = run_stream(meta, sc, _arm("A5_no_reset"), sd)
            breaks = meta["breaks"]
            breaks = breaks[(breaks >= 60) & (breaks < len(full["losses"]) - 15)]
            rec_full, rec_nores = [], []
            post_full, post_nores = [], []
            tier_full = []
            tiers_f = list(full["tiers"])
            for b in breaks:
                b = int(b) - 60
                if b < 5:
                    continue
                r1 = _recovery(full["losses"], b)
                r2 = _recovery(nores["losses"], b)
                rec_full.append(r1)
                rec_nores.append(r2)
                w = slice(b, min(b + 10, len(full["losses"])))
                post_full.append(np.mean(full["losses"][w]))
                tier_full.append(tiers_f[b] if b < len(tiers_f) else "ADAPT")
                post_nores.append(np.mean(nores["losses"][w]))
            rows.append(dict(seed=sd, jump_scale=js, n_breaks=len(rec_full),
                             recovery_full=float(np.mean(rec_full)) if rec_full else np.nan,
                             recovery_noreset=float(np.mean(rec_nores)) if rec_nores else np.nan,
                             post_loss_full=float(np.mean(post_full)) if post_full else np.nan,
                             post_loss_noreset=float(np.mean(post_nores)) if post_nores else np.nan,
                             frac_reset_at_break=(float(np.mean([t == "RESET" for t in tier_full]))
                                                  if tier_full else np.nan)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-days", type=int, default=900)
    ap.add_argument("--skip-h4", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    if not a.skip_h4:
        rows = experiment_h4(n_days=a.n_days)
        with open(os.path.join(a.out, "h4_detection.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
        f1 = [r["macro_f1"] for r in rows]
        print(f"H4: macro-F1 mean={np.mean(f1):.3f} median={np.median(f1):.3f} "
              f"min={np.min(f1):.3f} (n={len(f1)})  [{time.time()-t0:.0f}s]", flush=True)

    rows = experiment_h2_p2(n_days=a.n_days)
    with open(os.path.join(a.out, "h2_p2_recovery.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    for r in rows[:5]:
        print(r, flush=True)
    agg = {}
    for r in rows:
        agg.setdefault(r["jump_scale"], []).append(r)
    print("\njump_scale | recovery_full | recovery_noreset | post_full | post_noreset")
    for js, rs in sorted(agg.items()):
        print(f"{js:>10} | {np.mean([r['recovery_full'] for r in rs]):13.2f} | "
              f"{np.mean([r['recovery_noreset'] for r in rs]):16.2f} | "
              f"{np.mean([r['post_loss_full'] for r in rs]):9.4f} | "
              f"{np.mean([r['post_loss_noreset'] for r in rs]):13.4f}")
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

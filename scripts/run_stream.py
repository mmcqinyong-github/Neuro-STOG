"""Stage 2 -- the calendar-day streaming experiment.

For every (domain, seed):
  1. encode fingerprints with statistics fitted on the training split only;
  2. warm the architecture up on the training days (online, no future data);
  3. calibrate the free scalars on the validation split (coordinate descent,
     then frozen);
  4. deploy: run every ablation / placebo arm and every baseline through the
     validation days and then the test days, recording only test-day losses.

Nothing after step 3 touches the test split for tuning.

Endpoints
---------
Every arm is recorded as a RAW per-day mean squared error.  Two arm-independent
transformations are applied afterwards, identically to every arm:

  W  = mean_t min(loss_t, cap_t)                 primary
       cap_t is the 95th-percentile expert loss of day t times 8, computed once
       from the pool (protocol.build_envelope).  This is a plain risk
       functional; no arm can gain by clipping itself.
  N  = mean_t min(loss_t, cap_t) / median_expert_loss_t     secondary
       day-normalised; heavy tailed, reported for completeness.

Calibration (step 3) optimises W on the validation split.
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

from cerebrocast.baselines import (  # noqa: E402
    B1Frozen, B2MonthlyRetrain, B3FFORMA, B4Hedge, B5BestExpert, B6ADWIN,
    B7OnlineMeta, _simplex_fit)
from cerebrocast.config import SEEDS, SystemConfig, Ablation, default_arms  # noqa: E402
from cerebrocast.data import ALL_ORDER  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.memory import ClusterField  # noqa: E402
from cerebrocast.protocol import build_envelope  # noqa: E402
from cerebrocast.system import CerebroCast  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "stream")

# The bare engine (L0): every brain module switched off, leaving only the
# difficulty-normalised, recency-weighted, population-level simplex least squares
# read-out.  This is baseline B8 -- see the note in the baseline block below.
L0_BASE = dict(no_clusters=True, no_detect=True, no_forward=True,
               no_three_factor=True, no_astro=True, no_replay=True,
               no_downscale=True, no_growth=True, no_episodic=True)


def _new_field(cfg: SystemConfig, PHI_tr: np.ndarray, M: int, seed: int) -> ClusterField:
    f = ClusterField(
        tau_kappa=cfg.tau_kappa, graph_alpha=cfg.graph_alpha,
        cluster_momentum=cfg.cluster_momentum, grow_r_cov=cfg.grow_r_cov,
        grow_persist=cfg.grow_persist, prune_mass=cfg.prune_mass,
        prune_idle=cfg.prune_idle, k0=cfg.k0_clusters)
    return f.init_from_train(PHI_tr, M, seed)


def _make_system(cfg, arm, PHI_tr, M, seed, p, pr=0.0, pg=0.0) -> CerebroCast:
    field = _new_field(cfg, PHI_tr, M, seed)
    return CerebroCast(M, cfg, arm, field, p, seed=seed,
                       placebo_reset_rate=pr, placebo_growth_rate=pg)


def _run_days(sysm: CerebroCast, days, PHI, P, Y, record=True):
    out = []
    for t in days:
        d = sysm.step(int(t), PHI[t], P[t], Y[t])
        if record:
            out.append(d)
    return out


def run_one(domain: str, seed: int, pool_dir: str, out_dir: str,
            quick: bool = False, calib_rounds: int = 2, pool_seed: int = 7) -> dict:
    # the expert pool is a single realisation shared by every architecture seed,
    # which makes the architecture comparisons paired over identical days
    z = np.load(os.path.join(pool_dir, f"{domain}__seed{pool_seed}.npz"), allow_pickle=True)
    P, Y, X, dow = z["P"], z["Y"], z["X"], z["dow"]
    idx = {k: z[f"idx_{k}"] for k in ("train", "val", "test")}
    M = int(P.shape[1])
    fam = str(z["family"])
    tr, va, te = idx["train"], idx["val"], idx["test"]

    cfg0 = SystemConfig(seed=seed)
    fp = Fingerprint(K=cfg0.n_spec_bins)
    fp.fit(X[tr], dow[tr])
    PHI = fp.transform(X, dow)
    p = PHI.shape[1]

    # arm-independent endpoint transforms (fit-block days have no forecasts)
    oos = np.concatenate([tr, va, te])
    capd = build_envelope(P, Y, oos)
    capv = np.array([capd.get(int(t), np.nan) for t in range(len(P))])
    lm_oos = np.array([[np.mean((P[t][m] - Y[t]) ** 2) for m in range(M)]
                       for t in oos])
    lm_all = np.full((len(P), M), np.nan)
    lm_all[oos] = lm_oos
    scale = np.full(len(P), np.nan)
    scale[oos] = np.median(lm_oos, axis=1) + 1e-9

    arms = default_arms()
    full = arms[0]

    # ---------------- warmup on the training split -------------------------
    t0 = time.time()
    sysw = _make_system(cfg0, full, PHI[tr], M, seed, p)
    _run_days(sysw, tr, PHI, P, Y, record=False)
    snap = sysw.snapshot()
    warm_time = time.time() - t0

    # ---------------- placebo rates from the FULL arm on val ---------------
    sysr = _make_system(cfg0, full, PHI[tr], M, seed, p)
    sysr.restore(snap)
    dv = _run_days(sysr, va, PHI, P, Y, record=True)
    p_reset = sum(1 for d in dv if d["reset"]) / max(1, len(dv))
    p_growth = sum(d["grew"] for d in dv) / max(1, len(dv))

    # ---------------- calibration on the validation split ------------------
    def val_W(cfg, arm=full, PHIm=PHI):
        s = _make_system(cfg, arm, PHI[tr], M, seed, p,
                         p_reset if arm.placebo == "reset" else 0.0,
                         p_growth if arm.placebo == "growth" else 0.0)
        s.restore(snap)
        out = _run_days(s, va, PHIm, P, Y, record=True)
        lo = np.array([d["loss_raw"] for d in out])
        cp = np.array([capv[int(d["t"])] for d in out])
        return float(np.mean(np.minimum(lo, cp)))

    bcfg = cfg0
    best = {"w": val_W(cfg0)}
    trace = [dict(stage="init", w=best["w"])]

    def sweep(name, values, make):
        nonlocal bcfg, best
        b = None
        for v in values:
            c = make(bcfg, v)
            m = val_W(c)
            trace.append(dict(stage=name, value=v, w=m))
            if b is None or m < b[0]:
                b = (m, c, v)
        if b is not None and b[0] < best["w"] - 1e-12:
            best["w"], bcfg = b[0], b[1]

    if not quick:
        for _ in range(calib_rounds):
            sweep("recall_halflife", (2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0,
                                      45.0, 60.0, 90.0, 1e6),
                  lambda c, v: dc_replace(c, recall_halflife=v))
            sweep("consol_shrink", (0.25, 1.0, 4.0, 16.0, 1e6),
                  lambda c, v: dc_replace(c, consol_shrink=v))
            sweep("consol_shrink_u", (0.0, 0.15, 0.35),
                  lambda c, v: dc_replace(c, consol_shrink_u=v))
            sweep("reset_decay", (0.02, 0.05, 0.15, 0.35, 1.0),
                  lambda c, v: dc_replace(c, reset_decay=v))
            sweep("theta1", (1.0, 1.5, 2.0, 2.5),
                  lambda c, v: dc_replace(c, theta1=v))
            sweep("theta2", (2.5, 3.0, 3.5, 4.0, 4.5),
                  lambda c, v: dc_replace(c, theta2=v))
            sweep("beta_forward", (0.0, 0.5, 1.0, 2.0),
                  lambda c, v: dc_replace(c, beta_forward=v))
            sweep("eta", (0.0, 0.005, 0.02, 0.05),
                  lambda c, v: dc_replace(c, eta=v))
            sweep("topk", (0, 5, 9, 15),
                  lambda c, v: dc_replace(c, topk=v))
            sweep("gamma0", (0.0, 1e-3, 3e-3),
                  lambda c, v: dc_replace(c, gamma0=v))

    res = {"domain": domain, "seed": seed, "family": fam, "M": M, "p": int(p),
           "days": {k: int(len(v)) for k, v in idx.items()},
           "calib": dict(recall_halflife=bcfg.recall_halflife,
                         consol_shrink=bcfg.consol_shrink,
                         consol_shrink_u=bcfg.consol_shrink_u,
                         reset_decay=bcfg.reset_decay,
                         beta_forward=bcfg.beta_forward, eta=bcfg.eta,
                         topk=bcfg.topk, theta1=bcfg.theta1, theta2=bcfg.theta2,
                         gamma0=bcfg.gamma0, val_W=best["w"]),
           "trace": trace, "p_reset": p_reset, "p_growth": p_growth,
           "warm_time_s": warm_time, "arms": {}}

    # ---------------- deploy: arms on val + test ---------------------------
    fp_ns = None
    PHI_ns = None
    snap_ns = None
    for arm in arms:
        if arm.no_spectral and fp_ns is None:
            # a different encoder means a different state space, so the A1 arm
            # needs its own warm-up rather than the FULL snapshot
            fp_ns = Fingerprint(K=cfg0.n_spec_bins, no_spectral=True)
            fp_ns.fit(X[tr], dow[tr])
            PHI_ns = fp_ns.transform(X, dow)
            w = _make_system(cfg0, full, PHI_ns[tr], M, seed, fp_ns.p)
            _run_days(w, tr, PHI_ns, P, Y, record=False)
            snap_ns = w.snapshot()
        PHIm = PHI_ns if arm.no_spectral else PHI
        pm = fp_ns.p if arm.no_spectral else p
        s = _make_system(bcfg, arm, PHIm[tr], M, seed, pm,
                         p_reset if arm.placebo == "reset" else 0.0,
                         p_growth if arm.placebo == "growth" else 0.0)
        s.restore(snap_ns if arm.no_spectral else snap)
        t1 = time.time()
        _run_days(s, va, PHIm, P, Y, record=False)
        te_d = _run_days(s, te, PHIm, P, Y, record=True)
        rt = time.time() - t1
        res["arms"][arm.name] = dict(
            loss=np.array([d["loss_raw"] for d in te_d]),
            loss_wins=np.array([d["loss"] for d in te_d]),
            tier=np.array([d["tier"] for d in te_d]),
            delta=np.array([d["delta"] for d in te_d]),
            z=np.array([d["z"] for d in te_d]),
            n_clusters=np.array([d["n_clusters"] for d in te_d]),
            entropy=np.array([d["entropy"] for d in te_d]),
            top1=np.array([d["top1"] for d in te_d]),
            lam=np.array([d["lam"] for d in te_d]),
            lm=np.array([d["lm"] for d in te_d]),
            reset=np.array([d["reset"] for d in te_d], dtype=int),
            n_replay=np.array([d["n_replay"] for d in te_d]),
            w_norm=np.array([d["w_norm"] for d in te_d]),
            runtime_s=float(rt),
            sec_per_day=float(rt / max(1, len(va) + len(te))),
            description=arm.description,
        )

    # ---------------- baselines -------------------------------------------
    Pv, Yv = P[va], Y[va]
    lam_static = _simplex_fit(Pv, Yv)

    def bl_val_test(obj):
        for t in va:
            obj.step(int(t), PHI[t], P[t], Y[t])
        return obj

    def raw_losses(obj, days):
        out = []
        for t in days:
            out.append(float(np.mean((obj.lam @ P[t] - Y[t]) ** 2)))
            obj.step(int(t), PHI[t], P[t], Y[t])
        return np.array(out)

    b1 = B1Frozen(lam_static)
    b2 = bl_val_test(B2MonthlyRetrain(M))
    b3 = bl_val_test(B3FFORMA(M, p))
    best_eta, best_m = 0.5, np.inf
    for e in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        h = B4Hedge(M, eta=e)
        m = np.mean([min(h.step(int(t), PHI[t], P[t], Y[t])["loss_raw"],
                         capv[int(t)]) for t in va])
        if m < best_m:
            best_eta, best_m = e, m
    b4 = bl_val_test(B4Hedge(M, eta=best_eta))
    b5 = B5BestExpert(int(np.argmin([np.mean((Pv[:, m] - Yv) ** 2) for m in range(M)])), M)
    b6 = bl_val_test(B6ADWIN(M, window=90))
    b6.lam = lam_static
    best_e7, best_m7 = 0.05, np.inf
    for e in (0.005, 0.02, 0.05, 0.15, 0.4):
        o = B7OnlineMeta(M, p, eta=e)
        m = np.mean([o.step(int(t), PHI[t], P[t], Y[t])["loss"] for t in va])
        if m < best_m7:
            best_e7, best_m7 = e, m
    b7 = bl_val_test(B7OnlineMeta(M, p, eta=best_e7))

    # ---------------- B8: the architecture's own engine, bare --------------
    # Review item C7/20.  The diagnostics found that the strongest *classical*
    # causal combiner was recency-weighted, difficulty-normalised simplex LS --
    # which is exactly the architecture's own consolidation engine (L0) -- yet
    # B1-B7 did not contain it.  Every "architecture vs best baseline" gain
    # therefore conflated the engine with the brain modules.  B8 is that engine
    # with every module switched off, calibrated on the validation split under
    # the same protocol, which yields a clean attribution split:
    #     engine contribution = B8       vs best of B1..B7
    #     module contribution = FULL     vs B8
    armE = Ablation("B8_rwls_engine", **L0_BASE)
    cfgE0 = dc_replace(cfg0, eta=0.0, beta_forward=0.0, topk=0,
                       adapt_forget_boost=1.0, reset_forget_boost=1.0, gamma0=0.0)
    bestE = (np.inf, None)
    for hl in (3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0):
        for su in (0.0, 0.15, 0.35):
            c = dc_replace(cfgE0, recall_halflife=hl, consol_shrink_u=su)
            m = val_W(c, arm=armE)
            trace.append(dict(stage="B8", value=(hl, su), w=m))
            if m < bestE[0]:
                bestE = (m, c)
    cfgE = bestE[1] if bestE[1] is not None else cfgE0
    t1 = time.time()
    sE = _make_system(cfgE, armE, PHI[tr], M, seed, p)
    sE.restore(snap)
    _run_days(sE, va, PHI, P, Y, record=False)
    teE = _run_days(sE, te, PHI, P, Y, record=True)
    rtE = time.time() - t1
    res["arms"]["B8_rwls_engine"] = dict(
        loss=np.array([d["loss_raw"] for d in teE]),
        lm=np.array([d["lm"] for d in teE]),
        lam=np.array([d["lam"] for d in teE]),
        tier=np.array([d["tier"] for d in teE]),
        reset=np.array([d["reset"] for d in teE], dtype=int),
        n_clusters=np.array([d["n_clusters"] for d in teE]),
        entropy=np.array([d["entropy"] for d in teE]),
        top1=np.array([d["top1"] for d in teE]),
        runtime_s=float(rtE),
        sec_per_day=float(rtE / max(1, len(va) + len(te))),
        calibrated=dict(recall_halflife=cfgE.recall_halflife,
                        consol_shrink_u=cfgE.consol_shrink_u, val_W=bestE[0]),
        description="bare recency-weighted difficulty-normalised simplex LS (L0)")

    extras = {"B4_hedge": best_eta, "B7_online_meta": best_e7}
    for name, obj in [("B1_frozen_static", b1), ("B2_monthly_retrain", b2),
                      ("B3_fforma", b3), ("B4_hedge", b4),
                      ("B5_best_expert", b5), ("B6_adwin_retrain", b6),
                      ("B7_online_meta", b7)]:
        t1 = time.time()
        loss = raw_losses(obj, te)
        rt = time.time() - t1
        res["arms"][name] = dict(
            loss=loss, lam=None, tier=None, runtime_s=float(rt),
            sec_per_day=float(rt / max(1, len(te))),
            calibrated=extras.get(name, None), description="control condition")
    # reference points that are diagnostic, not competitors
    res["ref"] = dict(
        oracle_expert=np.array([float(lm_all[int(t)].min()) for t in te]),
        pool_median=np.array([float(np.median(lm_all[int(t)])) for t in te]),
        switches=(np.asarray(z["switches"], dtype=int)
                  if "switches" in z.files else np.array([], dtype=int)),
        block_len=(int(z["block_len"]) if "block_len" in z.files else 0),
    )
    res["caps"] = np.array([capd[int(t)] for t in te])
    res["scale"] = np.array([scale[int(t)] for t in te])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=ALL_ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--pool-seed", type=int, default=7)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for dom in a.domains:
        for s in a.seeds:
            t0 = time.time()
            r = run_one(dom, s, a.pool, a.out, quick=a.quick,
                        calib_rounds=a.rounds, pool_seed=a.pool_seed)
            path = os.path.join(a.out, f"{dom}__seed{s}.npz")
            flat = {"__meta__": np.array(json.dumps(
                {k: v for k, v in r.items() if k not in ("arms", "ref")},
                default=str)),
                "__caps__": np.asarray(r["caps"]),
                "__scale__": np.asarray(r["scale"])}
            for an, ad in r["arms"].items():
                for k, v in ad.items():
                    if isinstance(v, np.ndarray):
                        flat[f"{an}__{k}"] = v
                    elif v is not None:
                        flat[f"{an}__{k}"] = np.array(v)
            for k, v in r["ref"].items():
                flat[f"ref__{k}"] = v
            np.savez_compressed(path, **flat)
            caps, scale = r["caps"], r["scale"]
            W = lambda lo: float(np.mean(np.minimum(lo, caps)))
            N = lambda lo: float(np.mean(np.minimum(lo, caps) / scale))
            f_w, f_n = W(r["arms"]["FULL"]["loss"]), N(r["arms"]["FULL"]["loss"])
            bl = {k: W(v["loss"]) for k, v in r["arms"].items() if k.startswith("B")}
            best_bl = min(bl.values())
            print(f"[{dom} s{s}] FULL_W={f_w:.5f} bestBL={best_bl:.5f} "
                  f"gain={100*(best_bl-f_w)/best_bl:+.2f}%  "
                  f"(N: {f_n:.3f} vs {N(r['arms'][min(bl, key=bl.get)]['loss']):.3f}; "
                  f"oracle={W(r['ref']['oracle_expert']):.5f}, {time.time()-t0:.1f}s)",
                  flush=True)


if __name__ == "__main__":
    main()

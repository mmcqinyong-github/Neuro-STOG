"""Secondary metrics M3 / M4 and the MASE standardisation.

The pre-registered metric list named M3 (selection-frequency entropy), M4 (the
forward-model calibration curve) and MASE.  None of them was produced in the
first round; M4 in particular is not optional, because the theory chapter's L3
(weak joint, eligibility-trace credit consistency) cites a calibration audit as
the premise of its statement.  This script produces all three.

* **M3** -- entropy of the daily responsibility vector, read straight out of the
  stored FULL arms (`FULL__entropy`); it answers "does the selection rule lock
  into a degenerate subset of experts".
* **MASE** -- mean absolute scaled error of the architecture's fused hourly
  forecast against a seasonal-naive reference (lag 24 h).  The fused forecast is
  reconstructable because the model stores its daily responsibility vector
  (`FULL__lam`) alongside the pool's expert forecasts, so no re-run is needed.
* **M4** -- the forward error model predicts tomorrow's loss before it is
  observed; a calibrated model has predicted and realised losses on the diagonal.
  This needs the internal pairs, so a small number of domains are re-run with the
  exact calibration stored by `run_stream.py`.
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


def m3_entropy(dom: str, seed: int):
    p = os.path.join(STREAM, f"{dom}__seed{seed}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    e = np.asarray(z["FULL__entropy"], dtype=float)
    rand = float(np.log(int(z["FULL__lam"].shape[1])))   # entropy of the uniform mix
    return dict(mean_entropy=float(np.mean(e)), median_entropy=float(np.median(e)),
                entropy_of_uniform=rand,
                normalised=float(np.mean(e) / rand if rand > 0 else np.nan),
                degenerate_frac=float(np.mean(e < 0.25 * rand)))


def mase(dom: str, seed: int, pool_seed: int = 7):
    zp = os.path.join(POOL, f"{dom}__seed{pool_seed}.npz")
    zs = os.path.join(STREAM, f"{dom}__seed{seed}.npz")
    if not (os.path.exists(zp) and os.path.exists(zs)):
        return None
    zp = np.load(zp, allow_pickle=True)
    zs = np.load(zs, allow_pickle=True)
    P, Y, te = zp["P"], zp["Y"], zp["idx_test"]
    lam = np.asarray(zs["FULL__lam"], dtype=float)
    if lam.shape[0] != len(te):
        return None
    yhat = np.einsum("tm,tmh->th", lam, P[te])
    mae = float(np.mean(np.abs(yhat - Y[te])))
    # seasonal-naive reference: same hour, previous day (24 h lag)
    naive = np.mean(np.abs(Y[te][1:] - Y[te][:-1]))
    return dict(mae=mae, mae_seasonal_naive=float(naive),
                mase=float(mae / naive if naive > 1e-12 else np.nan))


def m4_calibration(dom: str, seed: int, pool_seed: int = 7, n_bins: int = 10):
    zp = os.path.join(POOL, f"{dom}__seed{pool_seed}.npz")
    zs = os.path.join(STREAM, f"{dom}__seed{seed}.npz")
    if not (os.path.exists(zp) and os.path.exists(zs)):
        return None
    zp, zs = np.load(zp, allow_pickle=True), np.load(zs, allow_pickle=True)
    P, Y, X, dow = zp["P"], zp["Y"], zp["X"], zp["dow"]
    tr, va, te = zp["idx_train"], zp["idx_val"], zp["idx_test"]
    M = int(P.shape[1])
    cal = json.loads(str(zs["__meta__"])).get("calib", {})
    cfg0 = SystemConfig(seed=seed)
    cfg = dc_replace(cfg0, **{k: (int(v) if k == "topk" else float(v))
                              for k, v in cal.items() if k in {
                                  "recall_halflife", "consol_shrink", "consol_shrink_u",
                                  "reset_decay", "beta_forward", "eta", "topk",
                                  "theta1", "theta2", "gamma0"}}) if cal else cfg0
    fp = Fingerprint(K=cfg0.n_spec_bins)
    fp.fit(X[tr], dow[tr])
    PHI = fp.transform(X, dow)
    f = ClusterField(tau_kappa=cfg.tau_kappa, graph_alpha=cfg.graph_alpha,
                     cluster_momentum=cfg.cluster_momentum,
                     grow_r_cov=cfg.grow_r_cov, grow_persist=cfg.grow_persist,
                     prune_mass=cfg.prune_mass, prune_idle=cfg.prune_idle,
                     k0=cfg.k0_clusters)
    f.init_from_train(PHI[tr], M, seed)
    s = CerebroCast(M, cfg, default_arms()[0], f, PHI.shape[1], seed=seed)
    for t in tr:
        s.step(int(t), PHI[t], P[t], Y[t])
    snap = s.snapshot()
    f2 = ClusterField(tau_kappa=cfg.tau_kappa, graph_alpha=cfg.graph_alpha,
                      cluster_momentum=cfg.cluster_momentum,
                      grow_r_cov=cfg.grow_r_cov, grow_persist=cfg.grow_persist,
                      prune_mass=cfg.prune_mass, prune_idle=cfg.prune_idle,
                      k0=cfg.k0_clusters)
    f2.init_from_train(PHI[tr], M, seed)
    s = CerebroCast(M, cfg, default_arms()[0], f2, PHI.shape[1], seed=seed)
    s.restore(snap)
    for t in va:
        s.step(int(t), PHI[t], P[t], Y[t])
    for t in te:
        s.step(int(t), PHI[t], P[t], Y[t])
    cal_pairs = np.asarray(s.calib, dtype=float)          # (n, 2) = (pred, realised)
    if cal_pairs.size == 0:
        return None
    pred, real = cal_pairs[:, 0], cal_pairs[:, 1]
    qs = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    rows = []
    for i in range(len(qs) - 1):
        m = (pred >= qs[i]) & (pred <= qs[i + 1])
        if m.sum() == 0:
            continue
        rows.append(dict(bin=i, n=int(m.sum()),
                         pred_mean=float(pred[m].mean()),
                         realised_mean=float(real[m].mean())))
    if not rows:
        return None
    ece = float(np.mean([abs(r["pred_mean"] - r["realised_mean"]) for r in rows]))
    scale = float(np.mean(real)) + 1e-12
    # a constant predictor attains this MAE; a calibrated one should beat it
    base = float(np.mean(np.abs(pred - real)))
    return dict(bins=rows, ece=ece, ece_relative=float(ece / scale),
                mae_pred_vs_real=base, n_days=int(len(pred)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=ALL_ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 42, 2021, 3407, 12345])
    ap.add_argument("--m4-domains", nargs="*",
                    default=["ETTh1", "ECL", "EPF-BE", "EPF-NP", "SW-2", "SW-X"])
    ap.add_argument("--pool-seed", type=int, default=7)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m3 = {}
    ms = {}
    for dom in a.domains:
        e = [m3_entropy(dom, s) for s in a.seeds]
        e = [x for x in e if x]
        if e:
            m3[dom] = dict(mean_entropy=float(np.mean([x["mean_entropy"] for x in e])),
                           normalised=float(np.mean([x["normalised"] for x in e])),
                           degenerate_frac=float(np.mean([x["degenerate_frac"] for x in e])))
        k = [mase(dom, s, a.pool_seed) for s in a.seeds]
        k = [x for x in k if x]
        if k:
            ms[dom] = dict(mase=float(np.mean([x["mase"] for x in k])),
                           mae=float(np.mean([x["mae"] for x in k])))
    json.dump(dict(M3_entropy=m3, MASE=ms),
              open(os.path.join(a.out, "m3_mase.json"), "w", encoding="utf-8"),
              indent=1, default=str)
    if m3:
        print("M3 selection entropy (normalised to the uniform mix):")
        for d, v in sorted(m3.items(), key=lambda x: x[1]["normalised"]):
            print(f"  {d:<10} H/H_uniform={v['normalised']:.3f}  "
                  f"degenerate days={v['degenerate_frac']:.2%}")
    if ms:
        print("\nMASE vs seasonal naive (lag 24 h):")
        for d, v in sorted(ms.items(), key=lambda x: x[1]["mase"]):
            print(f"  {d:<10} MASE={v['mase']:.3f}")

    m4 = {}
    for dom in a.m4_domains:
        r = m4_calibration(dom, a.seeds[0], a.pool_seed)
        if r:
            m4[dom] = r
    json.dump(m4, open(os.path.join(a.out, "m4_calibration.json"), "w", encoding="utf-8"),
              indent=1, default=str)
    if m4:
        print(f"\nM4 forward-model calibration (L3 premise audit), n_bins={len(next(iter(m4.values()))['bins'])}:")
        for d, v in m4.items():
            print(f"  {d:<10} ECE={v['ece']:.4f} (rel {v['ece_relative']:.3f})  "
                  f"MAE(pred,real)={v['mae_pred_vs_real']:.4f}")
        ece = float(np.mean([v["ece_relative"] for v in m4.values()]))
        verdict = ("calibrated" if ece < 0.5 else
                   "poorly calibrated -- the L3 premise is NOT met by this "
                   "implementation")
        print(f"  mean relative ECE = {ece:.3f} -> {verdict}")
        json.dump(dict(mean_relative_ece=ece, verdict=verdict, per_domain=m4),
                  open(os.path.join(a.out, "m4_calibration.json"), "w", encoding="utf-8"),
                  indent=1, default=str)
    print(f"\nwrote {a.out}/m3_mase.json, {a.out}/m4_calibration.json")


if __name__ == "__main__":
    main()

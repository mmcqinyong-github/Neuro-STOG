"""Diagnostic: isolate the learning-rate scale of the responsibility update.

Runs the architecture on one domain with detection/sleep disabled and sweeps
eta over a wide log range, so that the calibration grid can be centred on the
right order of magnitude instead of guessed.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace as dc_replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.config import Ablation, SystemConfig  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.memory import ClusterField  # noqa: E402
from cerebrocast.system import CerebroCast  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def run(domain, seed, eta, no_detect=True, no_astro=True, no_sleep=True,
        k0=6, topk=12):
    z = np.load(os.path.join(POOL, f"{domain}__seed{seed}.npz"), allow_pickle=True)
    P, Y, X = z["P"], z["Y"], z["X"]
    idx = {k: z[f"idx_{k}"] for k in ("train", "val", "test")}
    M = int(P.shape[1])
    cfg = SystemConfig(seed=seed, eta=eta, k0_clusters=k0, topk=topk)
    arm = Ablation("diag", no_detect=no_detect, no_astro=no_astro,
                   no_replay=no_sleep, no_downscale=no_sleep, no_growth=no_sleep)
    fp = Fingerprint(K=cfg.n_spec_bins); fp.fit(X[idx["train"]])
    PHI = fp.transform(X)
    field = ClusterField(k0=cfg.k0_clusters, tau_kappa=cfg.tau_kappa,
                         graph_alpha=cfg.graph_alpha,
                         cluster_momentum=cfg.cluster_momentum,
                         grow_r_cov=cfg.grow_r_cov, grow_persist=cfg.grow_persist,
                         prune_mass=cfg.prune_mass, prune_idle=cfg.prune_idle
                         ).init_from_train(PHI[idx["train"]], M, seed)
    s = CerebroCast(M, cfg, arm, field, PHI.shape[1], seed=seed)
    for t in idx["train"]:
        s.step(int(t), PHI[t], P[t], Y[t])
    te = []
    for t in idx["test"]:
        te.append(s.step(int(t), PHI[t], P[t], Y[t])["loss"])
    lm = np.array([np.mean((P[t] - Y[t]) ** 2, axis=1) for t in idx["test"]])
    return dict(mse=float(np.mean(te)), best_expert=float(np.mean(lm.min(axis=1))),
                mean_expert=float(np.mean(lm.mean(axis=1))),
                wmax=float(np.abs(s.field.W).max()))


if __name__ == "__main__":
    domain = sys.argv[1] if len(sys.argv) > 1 else "ETTh1"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    lm0 = None
    print(f"domain={domain} seed={seed}")
    for eta in (0.01, 0.03, 0.1, 0.3, 0.9, 2.7, 8.0):
        r = run(domain, seed, eta)
        lm0 = r
        print(f"  eta={eta:>6}: mse={r['mse']:.5f} best_expert={r['best_expert']:.5f} "
              f"mean_expert={r['mean_expert']:.5f} |W|max={r['wmax']:.2f}", flush=True)
    print("\nwith detection + sleep enabled:")
    for eta in (0.01, 0.03, 0.1, 0.3, 0.9, 2.7, 8.0):
        r = run(domain, seed, eta, no_detect=False, no_astro=False, no_sleep=False)
        print(f"  eta={eta:>6}: mse={r['mse']:.5f} best_expert={r['best_expert']:.5f} "
              f"|W|max={r['wmax']:.2f}", flush=True)

"""Which module is hurting?  One-domain component diagnostic.

Runs the architecture with individual mechanisms switched off and compares each
variant against the plain online-learning floor (B4 Hedge) on the same pool.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace as dc_replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.baselines import B4Hedge  # noqa: E402
from cerebrocast.config import Ablation, SystemConfig  # noqa: E402
from cerebrocast.fingerprints import Fingerprint  # noqa: E402
from cerebrocast.memory import ClusterField  # noqa: E402
from cerebrocast.system import CerebroCast  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def run(domain, seed, cfg_over=None, arm=None, beta=1.0, k0=6, eta=0.1, topk=12):
    z = np.load(os.path.join(POOL, f"{domain}__seed{seed}.npz"), allow_pickle=True)
    P, Y, X = z["P"], z["Y"], z["X"]
    idx = {k: z[f"idx_{k}"] for k in ("train", "val", "test")}
    M = int(P.shape[1])
    cfg = SystemConfig(seed=seed, eta=eta, k0_clusters=k0, topk=topk,
                       beta_forward=beta)
    if cfg_over:
        cfg = dc_replace(cfg, **cfg_over)
    arm = arm or Ablation("full")
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
    te = [s.step(int(t), PHI[t], P[t], Y[t])["loss"] for t in idx["test"]]
    return float(np.mean(te)), s


def hedge(domain, seed, eta=0.5):
    z = np.load(os.path.join(POOL, f"{domain}__seed{seed}.npz"), allow_pickle=True)
    P, Y, X = z["P"], z["Y"], z["X"]
    idx = {k: z[f"idx_{k}"] for k in ("train", "val", "test")}
    M = int(P.shape[1])
    fp = Fingerprint(K=32); fp.fit(X[idx["train"]])
    PHI = fp.transform(X)
    h = B4Hedge(M, eta=eta)
    for t in idx["train"]:
        h.step(int(t), PHI[t], P[t], Y[t])
    return float(np.mean([h.step(int(t), PHI[t], P[t], Y[t])["loss"]
                          for t in idx["test"]]))


if __name__ == "__main__":
    domain = sys.argv[1] if len(sys.argv) > 1 else "ETTh1"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    print(f"domain={domain} seed={seed}")
    print(f"  {'B4 hedge':38s} {hedge(domain, seed):.5f}")
    variants = [
        ("full", dict(), None, 1.0, 6),
        ("beta=0 (no forward gating)", dict(), None, 0.0, 6),
        ("k0=1 (single cluster)", dict(), None, 1.0, 1),
        ("no detection", dict(), Ablation("x", no_detect=True), 1.0, 6),
        ("no sleep", dict(), Ablation("x", no_replay=True, no_downscale=True,
                                      no_growth=True), 1.0, 6),
        ("no episodic", dict(), Ablation("x", no_episodic=True), 1.0, 6),
        ("k0=1 + beta=0 + no sleep", dict(),
         Ablation("x", no_replay=True, no_downscale=True, no_growth=True), 0.0, 1),
    ]
    for name, ov, arm, beta, k0 in variants:
        m, s = run(domain, seed, ov, arm, beta, k0)
        print(f"  {name:38s} {m:.5f}   |W|max={np.abs(s.field.W).max():.2f} "
              f"clusters={s.field.n_clusters()}", flush=True)

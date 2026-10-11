"""P5 -- the five directed stress families.

The theory chapter (Section 5.4) states that each mechanism is necessary by
exhibiting a data-generating process on which removing it is strictly harmful,
and pre-commits to instantiating all five families.  The first round did not do
that: it built a splicing ladder (STAT-0/1, SW-2/4/X), which is a *different*
design.  This module supplies the five families so the claim becomes true, and
`scripts/run_p5.py` measures the cost of each ablation on each family.

The five families, each aimed at a specific failure mode of Appendix A:

| family | construction | targets |
|---|---|---|
| `periodic_jump`   | large abrupt regime switches at a fixed 30-day period | F3 (post-break old prior drag), detection module |
| `rare_high_loss`  | stationary body with rare, very large loss days | F1 (noise mistaken for drift -> over-updating) |
| `heavy_tail`      | Student-t innovations, heavy tails | F7 robustness / maintenance |
| `regime_cycle`    | regimes cycle deterministically, period 30 days | F9 (unseen regime combinations) |
| `linear_drift`    | smooth constant-rate parameter drift, no jumps | F2 (gradual tracking too slow) |

Every family returns the same `meta` layout as `synth.generate`, so the streaming
harness can consume it unchanged.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

H = 24
L = 168

FAMILIES = ["periodic_jump", "rare_high_loss", "heavy_tail", "regime_cycle",
            "linear_drift"]

FAMILY_TARGET = {
    "periodic_jump": "F3 post-break prior drag / detection",
    "rare_high_loss": "F1 noise mistaken for drift",
    "heavy_tail": "F7 robustness under heavy tails",
    "regime_cycle": "F9 unseen regime combinations",
    "linear_drift": "F2 gradual tracking too slow",
}


def _wrap(regime, Y, P, labels, breaks, med_jump):
    D = len(Y)
    X = np.stack([Y[t - 7:t].ravel() for t in range(7, D)])
    return dict(regime=regime[7:], labels=np.asarray(labels[7:], dtype=object),
                Y=Y[7:], P=P[7:], X=X, med_jump=med_jump,
                breaks=np.array([b for b in breaks if 0 <= b < D - 7], dtype=int))


def _pool(rng, Y, regime, n_regimes: int, n_experts: int,
          jump_scale: float = 1.0):
    """Regime-specialised expert pool, as in the main generator."""
    M, D = n_experts, len(Y)
    bias = np.full((M, n_regimes), 0.65)
    for m in range(M):
        bias[m, m % n_regimes] = 0.12
    bias += rng.uniform(0.0, 0.15, size=(M, n_regimes))
    sd = rng.uniform(0.25, 0.55, size=(M, n_regimes))
    amp = np.ones(n_regimes)
    P = np.zeros((D, M, H))
    for t in range(D):
        r = regime[t]
        common = 0.35 * rng.standard_normal(H)
        for m in range(M):
            P[t, m] = (Y[t] + amp[r] * bias[m, r] * jump_scale
                       + amp[r] * sd[m, r] * rng.standard_normal(H) + common)
    return P


def _hour_profile(rng, n_regimes):
    base = np.sin(2 * np.pi * (np.arange(H) - 6) / H)
    return np.stack([base + 0.35 * rng.standard_normal(H) for _ in range(n_regimes)])


def generate_family(family: str, n_days: int = 900, seed: int = 7,
                    n_regimes: int = 3, n_experts: int = 12,
                    period: int = 30) -> Dict:
    """Build one directed stress family."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; expected one of {FAMILIES}")
    rng = np.random.default_rng(seed)
    D, R = n_days, n_regimes
    prof = _hour_profile(rng, R)
    mean_r = rng.uniform(-1.0, 1.0, size=R)
    phi_r = rng.uniform(0.2, 0.8, size=R)

    regime = np.zeros(D, dtype=int)
    labels = np.full(D, "HOLD", dtype=object)
    breaks = []
    drift = np.zeros(D)
    med_jump = 0.0

    if family == "periodic_jump":
        for t in range(1, D):
            if t % period == 0:
                regime[t] = (regime[t - 1] + 1) % R
                breaks.append(t)
                for d in range(t, min(t + 5, D)):
                    labels[d] = "RESET"
            else:
                regime[t] = regime[t - 1]
        med_jump = float(np.median([abs(mean_r[(regime[t])] - mean_r[regime[t - 1]])
                                    for t in range(1, D) if t % period == 0] or [1.0]))

    elif family == "rare_high_loss":
        # stationary regime; only the *noise* is occasionally extreme
        regime[:] = 0
        for t in range(D):
            if rng.random() < 0.04:
                for d in range(t, min(t + 3, D)):
                    labels[d] = "ADAPT"

    elif family == "heavy_tail":
        regime[:] = 0

    elif family == "regime_cycle":
        for t in range(D):
            regime[t] = (t // period) % R
            if t % period == 0 and t > 0:
                breaks.append(t)
                for d in range(t, min(t + 5, D)):
                    labels[d] = "RESET"
        med_jump = float(np.mean([abs(mean_r[(i + 1) % R] - mean_r[i % R])
                                  for i in range(R)]))

    elif family == "linear_drift":
        regime[:] = 0
        drift = np.linspace(0.0, 3.0, D)          # smooth, jump-free drift
        for t in range(D):
            labels[t] = "ADAPT"

    # ---- innovations -----------------------------------------------------
    heavy = family == "heavy_tail"
    spike_p = 0.06 if family == "rare_high_loss" else 0.0
    Y = np.zeros((D, H))
    x = 0.0
    for t in range(D):
        r = regime[t]
        mu = mean_r[r] + (0.6 * drift[t] if family == "linear_drift" else 0.0)
        for h in range(H):
            if heavy:
                inn = rng.standard_t(3.0) / np.sqrt(3.0 / (3.0 - 2.0))
            else:
                inn = rng.standard_normal()
            x = phi_r[r] * x + (1 - phi_r[r]) * inn
            v = mu + (prof[r, h] + 0.8 * x)
            if spike_p > 0 and rng.random() < spike_p:
                v += rng.standard_normal() * 12.0
            Y[t, h] = v

    P = _pool(rng, Y, regime, R, n_experts,
              jump_scale=(1.0 if family != "periodic_jump" else 1.5))
    return _wrap(regime, Y, P, labels, breaks, med_jump)

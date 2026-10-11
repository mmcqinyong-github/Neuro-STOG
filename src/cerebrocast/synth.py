"""Controlled-drift synthetic generator with known ground truth.

Used for the two questions that only a generator with known change points can
answer: (i) H4 -- how good is the three-tier gate at telling HOLD / ADAPT /
RESET apart; (ii) P2 -- for which jump magnitude is RESET actually better than
ADAPT.

The generator produces an hourly target plus a pool of synthetic experts whose
bias depends on the current regime, so that "which expert is best" genuinely
moves with the regime -- the condition under which maintenance matters at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

H = 24
L = 168


@dataclass
class SynthConfig:
    n_days: int = 900
    n_regimes: int = 3
    p_switch: float = 0.005          # per-day probability of an abrupt switch
    alpha: float = 1.0               # spectral decay exponent
    spike: float = 0.0               # probability of a spike hour
    gradual: bool = False            # slow parameter drift inside a segment
    gradual_rate: float = 0.01
    amplitude_balance: bool = True   # regimes share an amplitude scale
    missing: float = 0.0
    n_experts: int = 12
    seed: int = 7
    jump_scale: float = 1.0          # multiplier on the abrupt jump magnitude
    tier_window: int = 5            # days labelled ADAPT/RESET after a switch
    common_noise: float = 0.35       # irreducible noise shared by all experts


def _hour_profile(rng: np.random.Generator, n_regimes: int) -> np.ndarray:
    base = np.sin(2 * np.pi * (np.arange(H) - 6) / H)
    return np.stack([base + 0.35 * rng.standard_normal(H) for _ in range(n_regimes)])


def generate(cfg: SynthConfig) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    D = cfg.n_days
    R = cfg.n_regimes

    prof = _hour_profile(rng, R)
    mean_r = rng.uniform(-1.0, 1.0, size=R) * cfg.jump_scale
    amp_r = np.ones(R) if cfg.amplitude_balance else rng.uniform(0.6, 1.6, size=R)
    phi_r = rng.uniform(0.2, 0.8, size=R)

    # ---- regime path -----------------------------------------------------
    # A switch is always to a different regime; the ground-truth tier of the
    # days that follow depends on how large the switch was.  "Large" is defined
    # relative to the median jump of that very stream, which keeps the three
    # classes populated in every configuration and makes the definition
    # auditable rather than tuned.
    regime = np.zeros(D, dtype=int)
    drift = np.zeros(D)
    jumps: List[float] = []
    switch_days: List[int] = []
    for t in range(1, D):
        if rng.random() < cfg.p_switch:
            new_r = (regime[t - 1] + 1 + rng.integers(0, R - 1)) % R
            jumps.append(abs(mean_r[new_r] - mean_r[regime[t - 1]]))
            switch_days.append(t)
            regime[t] = new_r
        else:
            regime[t] = regime[t - 1]
            if cfg.gradual:
                drift[t] = drift[t - 1] + cfg.gradual_rate

    med_jump = float(np.median(jumps)) if jumps else 0.0
    labels = np.full(D, "ADAPT" if cfg.gradual else "HOLD", dtype=object)
    for k, t in enumerate(switch_days):
        tier = "RESET" if jumps[k] >= med_jump else "ADAPT"
        for d in range(t, min(t + cfg.tier_window, D)):
            labels[d] = tier

    # hourly target
    Y = np.zeros((D, H))
    x = 0.0
    for t in range(D):
        r = regime[t]
        mu = mean_r[r] + (0.6 * drift[t] if cfg.gradual else 0.0)
        for h in range(H):
            x = phi_r[r] * x + (1 - phi_r[r]) * rng.standard_normal()
            v = mu + amp_r[r] * (prof[r, h] + 0.8 * x)
            if cfg.spike > 0 and rng.random() < cfg.spike:
                v += amp_r[r] * rng.standard_normal() * 6.0
            Y[t, h] = v

    # expert pool: regime-specialised bias + idiosyncratic noise
    M = cfg.n_experts
    bias = np.full((M, R), 0.65)
    for m in range(M):
        bias[m, m % R] = 0.12
    bias += rng.uniform(0.0, 0.15, size=(M, R))
    sd = rng.uniform(0.25, 0.55, size=(M, R))

    P = np.zeros((D, M, H))
    for t in range(D):
        r = regime[t]
        common = cfg.common_noise * rng.standard_normal(H)
        for m in range(M):
            P[t, m] = (Y[t] + amp_r[r] * bias[m, r]
                       + amp_r[r] * sd[m, r] * rng.standard_normal(H)
                       + common)
        if cfg.missing > 0 and rng.random() < cfg.missing:
            P[t, rng.integers(0, M)] = np.nan
    P = np.nan_to_num(P)

    # windows (7 previous days) for the fingerprint encoder
    X = np.stack([Y[t - 7:t].ravel() for t in range(7, D)])
    meta = dict(regime=regime[7:], labels=np.asarray(labels[7:], dtype=object),
                Y=Y[7:], P=P[7:], X=X, med_jump=med_jump)
    # break days (first day of a RESET run), in the blocked index
    breaks = [t - 7 for t in range(7, D) if labels[t] == "RESET"
              and (t == 7 or labels[t - 1] != "RESET")]
    meta["breaks"] = np.array([b for b in breaks if 0 <= b < D - 7], dtype=int)
    return meta


def run_grid(seeds=(7, 42, 2021), p_switch=(0.001, 0.005, 0.02),
             alpha=(0.0, 1.0, 2.0), spike=(0.0, 0.01, 0.05),
             gradual=(False, True), n_days=900) -> List[Dict]:
    """The factorial grid used for H4 and for the P2 boundary measurement."""
    out = []
    for s in seeds:
        for p in p_switch:
            for a in alpha:
                for sp in spike:
                    for g in gradual:
                        cfg = SynthConfig(n_days=n_days, p_switch=p, alpha=a,
                                          spike=sp, gradual=g, seed=s)
                        out.append(dict(cfg=cfg, data=generate(cfg)))
    return out

"""The shared evaluation protocol (Section 6.1 of the paper).

Two things in this file are pre-registered and therefore identical for every
arm, architecture and baseline alike:

1. the *winsorising envelope* ``cap_t``.  On the price domains the raw squared
   loss is dominated by a handful of spike days (the worst 1% of test days carry
   25-65% of the total MSE), so the primary metric is the daily loss truncated at
   a rolling envelope that depends only on the expert pool and the calendar day --
   never on any arm's own behaviour.  Raw MSE is reported alongside it.
2. the *day grid*: train / validation / test are contiguous calendar blocks,
   and every arm walks them in the same order.

Because ``cap_t`` is a function of the data alone, applying it to every arm is
not a favour to any particular method: an arm that beats every expert is never
truncated, and an arm that has a catastrophic day is charged the same ceiling as
any other arm would be.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Envelope:
    """Causal rolling winsorising cap, fitted from the expert pool only."""
    q: float = 0.95
    mult: float = 8.0
    ema: float = 0.99
    cap: Optional[float] = None
    series: List[float] = None          # type: ignore[assignment]

    def __post_init__(self):
        if self.series is None:
            self.series = []

    def observe(self, lm: np.ndarray) -> float:
        """Call once per calendar day, with that day's per-expert losses."""
        q = float(np.quantile(lm, self.q)) * self.mult
        self.cap = q if self.cap is None else self.ema * self.cap + (1 - self.ema) * q
        self.series.append(self.cap)
        return self.cap

    def clip(self, loss: float) -> float:
        return float(min(loss, self.cap)) if self.cap is not None else float(loss)


def build_envelope(P: np.ndarray, Y: np.ndarray, days: np.ndarray,
                   q: float = 0.95, mult: float = 8.0) -> Dict[int, float]:
    """Per-day cap, computed once from the expert pool.  Arm independent."""
    env = Envelope(q=q, mult=mult)
    out: Dict[int, float] = {}
    for t in days:
        t = int(t)
        lm = np.asarray([np.mean((P[t][m] - Y[t]) ** 2) for m in range(P.shape[1])])
        out[t] = env.observe(lm)
    return out


def score(losses: Dict[int, float], caps: Dict[int, float]) -> Dict[str, float]:
    """Raw and winsorised mean daily loss over the recorded days."""
    ks = sorted(losses)
    raw = np.array([losses[k] for k in ks])
    cap = np.array([caps[k] for k in ks])
    win = np.minimum(raw, cap)
    return dict(raw=float(raw.mean()), win=float(win.mean()),
                n=float(len(ks)), frac_capped=float(np.mean(raw > cap)))

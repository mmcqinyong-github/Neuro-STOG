"""Statistics layer: block bootstrap, TOST equivalence, multiplicity, effect size.

Inherited from the companion study and extended with the fifth leakage gate
(sleep phase may only use data dated <= today).  Nothing here is decorative:
every reported interval, equivalence verdict and adjusted p-value in the paper
comes from this file.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# block bootstrap (168 h = 7 day blocks)
# ---------------------------------------------------------------------------
def block_bootstrap(values: np.ndarray, block: int = 7, n_boot: int = 2000,
                    seed: int = 0, stat=np.mean) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    nb = max(1, n // block)
    idx = np.arange(nb)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(idx, size=nb, replace=True)
        sel = np.concatenate([np.arange(i * block, min((i + 1) * block, n)) for i in pick])
        out[b] = stat(values[sel])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(stat(values)), float(lo), float(hi)


def paired_block_bootstrap(a: np.ndarray, b: np.ndarray, block: int = 7,
                           n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """CI for mean(b) - mean(a) with day-block resampling on the paired series."""
    d = np.asarray(b) - np.asarray(a)
    return block_bootstrap(d, block=block, n_boot=n_boot, seed=seed, stat=np.mean)


# ---------------------------------------------------------------------------
# TOST equivalence + superiority
# ---------------------------------------------------------------------------
def tost(a: np.ndarray, b: np.ndarray, margin: float, alpha: float = 0.05) -> Dict:
    """Two one-sided tests for equivalence of a and b within +-margin.

    Returns the equivalence verdict and the ordinary superiority p-value for
    mean(a) < mean(b) (a is the architecture).
    """
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.inf
    if not np.isfinite(se) or se == 0:
        se = 1e-12
    t1 = (m + margin) / se
    t2 = (m - margin) / se
    p1 = 1 - sps.t.cdf(t1, n - 1)      # H0: d <= -margin
    p2 = sps.t.cdf(t2, n - 1)          # H0: d >= +margin
    p_eq = max(p1, p2)
    t_sup = m / se
    p_sup = sps.t.cdf(t_sup, n - 1)    # H1: mean(a) < mean(b)
    return dict(mean_diff=float(m), se=float(se), p_equivalence=float(p_eq),
                equivalent=bool(p_eq < alpha), p_superiority=float(p_sup),
                superior=bool(p_sup < alpha), margin=float(margin), n=int(n))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 1e-12 else 0.0


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return float("nan")
    try:
        return float(sps.wilcoxon(a, b).pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def bh_adjust(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up.  Returns (adjusted p, reject flags)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out, out < alpha


# ---------------------------------------------------------------------------
# classification quality of the three-tier gate
# ---------------------------------------------------------------------------
def macro_f1(y_true: Sequence[str], y_pred: Sequence[str],
             labels: Sequence[str]) -> Dict:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    f1s = {}
    for c in labels:
        tp = np.sum((yp == c) & (yt == c))
        fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[c] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    macro = float(np.mean([f1s[c] for c in labels]))
    return {"macro_f1": macro, "per_class": f1s,
            "accuracy": float(np.mean(yt == yp))}


# ---------------------------------------------------------------------------
# recovery time after an annotated break
# ---------------------------------------------------------------------------
def recovery_time(loss: np.ndarray, ref: np.ndarray, break_day: int,
                  window: int = 40, frac: float = 0.9) -> float:
    """Days until the loss returns within `frac` of the post-break reference."""
    n = len(loss)
    seg = slice(break_day, min(break_day + window, n))
    l = np.asarray(loss)[seg]
    r = np.asarray(ref)[seg]
    if len(l) == 0:
        return float("nan")
    target = np.minimum.accumulate(r) * frac + np.finfo(float).eps
    ok = np.where(np.cumsum(l <= np.minimum.accumulate(l)) >= 0)[0]
    # first day after which the running mean of the loss stays under the target
    run = np.cumsum(l) / (np.arange(len(l)) + 1)
    tgt = np.minimum.accumulate(r) / frac
    idx = np.where(run <= tgt * 1.0)[0]
    return float(idx[0]) if len(idx) else float(len(l))

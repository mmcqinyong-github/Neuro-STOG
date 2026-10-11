"""Module (1): the spectral fingerprint encoder  Phi : x -> (phi^S, phi^A, phi^R).

Three orthogonal factors, by construction:

* phi^S -- spectral *shape*: log-periodogram averaged over K log-spaced
  frequency bins and normalised to unit total power, so that amplitude is
  removed (this factorisation is the design constraint inherited from the
  falsified amplitude-balanced gating bound of the companion study);
* phi^A -- amplitude: one log-scale coordinate, carried separately;
* phi^R -- regime: robust shape statistics (skew, kurtosis, short/long
  autocorrelation, variance ratio, spike load, trend slope).

All normalisation statistics are fitted on training-split windows only.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _log_bin_edges(K: int, n: int) -> np.ndarray:
    """K+1 edges over FFT frequency index 1..n//2, log spaced."""
    lo, hi = 1.0, float(n // 2)
    edges = np.geomspace(lo, hi + 1e-9, K + 1)
    idx = np.unique(np.round(edges).astype(int))
    while len(idx) < K + 1:                      # degenerate for tiny K
        idx = np.unique(np.concatenate([idx, idx + 1]))[: K + 1]
    return idx[: K + 1]


def spectral_shape(X: np.ndarray, K: int = 32) -> np.ndarray:
    """(D, L) -> (D, K) log-periodogram shape, amplitude removed."""
    D, n = X.shape
    edges = _log_bin_edges(K, n)
    Xc = X - X.mean(axis=1, keepdims=True)
    F = np.fft.rfft(Xc, axis=1)
    psd = (np.abs(F) ** 2)[:, 1:]
    out = np.empty((D, K))
    for k in range(K):
        a, b = edges[k], max(edges[k + 1], edges[k] + 1)
        out[:, k] = psd[:, a - 1: b - 1].mean(axis=1) if b > a else psd[:, a - 1]
    tot = out.sum(axis=1, keepdims=True) + 1e-12
    return np.log(out / tot + 1e-9)


def amplitude(X: np.ndarray) -> np.ndarray:
    mad = np.median(np.abs(X - np.median(X, axis=1, keepdims=True)), axis=1)
    return np.log(mad + 1e-6)[:, None]


def regime(X: np.ndarray) -> np.ndarray:
    D, n = X.shape
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    Z = (X - mu) / sd
    skew = (Z ** 3).mean(axis=1)
    kurt = (Z ** 4).mean(axis=1) - 3.0
    zc = X - X.mean(axis=1, keepdims=True)
    denom = (zc ** 2).sum(axis=1) + 1e-12
    ac1 = (zc[:, :-1] * zc[:, 1:]).sum(axis=1) / denom
    m = min(24, n // 4)
    acm = (zc[:, :-m] * zc[:, m:]).sum(axis=1) / denom
    d1 = np.diff(X, axis=1)
    dm = X[:, m:] - X[:, :-m]
    vr = (dm ** 2).mean(axis=1) / ((d1 ** 2).mean(axis=1) + 1e-12) / m
    spike = (np.abs(Z) > 4).mean(axis=1)
    t = np.arange(n)
    tm = t - t.mean()
    slope = (X * tm).sum(axis=1) / (tm ** 2).sum() / (sd[:, 0] + 1e-8)
    rng = (X.max(axis=1) - X.min(axis=1)) / (sd[:, 0] + 1e-8)
    last = X[:, -m:] if m <= n else X
    level_z = (last.mean(axis=1) - mu[:, 0]) / (sd[:, 0] + 1e-8)
    ex = (X > (mu[:, 0] + 1.28 * sd[:, 0])[:, None]).mean(axis=1)
    # spectral entropy of the periodogram (concentration of power)
    Xc = X - mu
    psd = (np.abs(np.fft.rfft(Xc, axis=1)) ** 2)[:, 1:]
    ptot = psd.sum(axis=1) + 1e-12
    pent = -(psd / ptot[:, None] * np.log(psd / ptot[:, None] + 1e-12)).sum(axis=1)
    hstep = np.arange(n)
    half = n // 2
    trend2 = (X[:, half:].mean(axis=1) - X[:, :half].mean(axis=1)) / (sd[:, 0] + 1e-8)
    return np.column_stack([skew, kurt, ac1, acm, vr, spike, slope, rng,
                            level_z, ex, pent, trend2])


def calendar_features(dow: np.ndarray) -> np.ndarray:
    d = np.asarray(dow) % 7
    return np.column_stack([np.sin(2 * np.pi * d / 7.0),
                            np.cos(2 * np.pi * d / 7.0),
                            (d >= 5).astype(float)])


class Fingerprint:
    """Fitted encoder: raw window -> standardised (p,) fingerprint."""

    def __init__(self, K: int = 32, w_shape: float = 1.0,
                 w_amp: float = 1.0, w_reg: float = 1.0,
                 no_spectral: bool = False):
        self.K = K
        self.w = (w_shape, w_amp, w_reg)
        self.mu = None
        self.sd = None
        self.p = None
        # A1 ablation: the spectral-shape factor is dropped, so the encoder
        # carries amplitude and regime statistics only.
        self.no_spectral = no_spectral

    def fit(self, Xtr: np.ndarray, dow: np.ndarray = None) -> "Fingerprint":
        F = self._raw(Xtr, dow)
        self.mu = F.mean(axis=0)
        self.sd = F.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        self.p = F.shape[1]
        return self

    def _raw(self, X: np.ndarray, dow: np.ndarray = None) -> np.ndarray:
        if self.no_spectral:
            A = amplitude(X)
            R = regime(X)
            if dow is not None:
                R = np.column_stack([R, calendar_features(dow)])
            _, wa, wr = self.w
            return np.column_stack([wa * A, wr * R])
        S = spectral_shape(X, self.K)
        A = amplitude(X)
        R = regime(X)
        if dow is not None:
            R = np.column_stack([R, calendar_features(dow)])
        ws, wa, wr = self.w
        return np.column_stack([ws * S, wa * A, wr * R])

    def transform(self, X: np.ndarray, dow: np.ndarray = None) -> np.ndarray:
        return (self._raw(X, dow) - self.mu) / self.sd

    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))


def pairwise_spectral_distance(X: np.ndarray, K: int = 32) -> np.ndarray:
    """Symmetric matrix of shape-factor distances (used for the C2 monotonicity
    test and for the slow affinity graph initialisation)."""
    S = spectral_shape(X, K)
    S = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-8)
    G = S @ S.T
    sq = np.diag(G)
    D2 = sq[:, None] + sq[None, :] - 2 * G
    return np.sqrt(np.maximum(D2, 0.0))

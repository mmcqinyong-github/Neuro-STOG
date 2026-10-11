"""The seven control conditions B1-B7 (Section 6.3 of the paper).

B1 frozen static combination -- the "deploy it and walk away" reference
B2 monthly retraining      -- is retraining often enough?
B3 FFORMA-style weighting  -- the classical forecast-combination baseline
B4 normalised Hedge        -- the online-learning floor (P1 reduces to it)
B5 best fixed expert       -- single-model guardrail
B6 ADWIN + retraining      -- the best detector+retrain point solution
B7 online meta-learner     -- discriminative adaptation without the architecture

Every baseline sees exactly the same per-day expert forecasts as the
architecture and is calibrated on the validation split under the same protocol.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import nnls

from .config import SystemConfig
from .system import _softmax, _topk_normalize


def _simplex_fit(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least-squares non-negative combination with sum-to-one constraint."""
    D, M, H = P.shape
    A = P.transpose(0, 2, 1).reshape(D * H, M)
    b = Y.reshape(-1)
    C = 1e3 * max(1.0, np.abs(A).mean())
    Aa = np.vstack([A, C * np.ones((1, M))])
    ba = np.concatenate([b, [C]])
    w, _ = nnls(Aa, ba)
    s = w.sum()
    return w / s if s > 1e-12 else np.full(M, 1.0 / M)


class _Base:
    name = "base"

    def reset(self):
        pass

    def step(self, t, phi, P, y) -> Dict:
        raise NotImplementedError


class B1Frozen(_Base):
    name = "B1_frozen_static"

    def __init__(self, lam: np.ndarray):
        self.lam = lam

    def step(self, t, phi, P, y) -> Dict:
        yhat = self.lam @ P
        return dict(loss=float(np.mean((yhat - y) ** 2)), lam=self.lam.copy(),
                    lm=np.asarray([np.mean((P[m] - y) ** 2) for m in range(P.shape[0])]),
                    retrain=0)


class B2MonthlyRetrain(_Base):
    name = "B2_monthly_retrain"

    def __init__(self, M: int, period: int = 30, window: int = 180):
        self.M = M
        self.period = period
        self.window = window
        self.lam = np.full(M, 1.0 / M)
        self.hist: List[tuple] = []
        self.last = -10 ** 9
        self.n_fit = 0

    def step(self, t, phi, P, y) -> Dict:
        self.hist.append((P.copy(), y.copy()))
        if len(self.hist) > self.window:
            self.hist.pop(0)
        fit = 0
        if t - self.last >= self.period and len(self.hist) >= 30:
            Pa = np.stack([h[0] for h in self.hist])
            Ya = np.stack([h[1] for h in self.hist])
            self.lam = _simplex_fit(Pa, Ya)
            self.last = t
            fit = 1
            self.n_fit += 1
        lm = np.asarray([np.mean((P[m] - y) ** 2) for m in range(self.M)])
        return dict(loss=float(np.mean((self.lam @ P - y) ** 2)), lam=self.lam.copy(),
                    lm=lm, retrain=fit)


class B3FFORMA(_Base):
    """Feature-based forecast-model averaging, sliding re-estimation."""

    name = "B3_fforma"

    def __init__(self, M: int, p: int, period: int = 30, window: int = 240,
                 ridge: float = 1.0, tau: float = 0.5):
        self.M = M
        self.p = p
        self.period = period
        self.window = window
        self.ridge = ridge
        self.tau = tau
        self.lam = np.full(M, 1.0 / M)
        self.hist: List[tuple] = []
        self.last = -10 ** 9
        self.V = np.zeros((M, p + 1))
        self.n_fit = 0

    def _fit(self):
        if len(self.hist) < 40:
            return
        Z = np.asarray([h[0] for h in self.hist])
        Lm = np.asarray([h[1] for h in self.hist])
        A = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
        self.V = np.linalg.solve(A, Z.T @ Lm).T

    def step(self, t, phi, P, y) -> Dict:
        z = np.concatenate([phi, [1.0]])
        lm = np.asarray([np.mean((P[m] - y) ** 2) for m in range(self.M)])
        self.hist.append((z.copy(), lm.copy()))
        if len(self.hist) > self.window:
            self.hist.pop(0)
        fit = 0
        if t - self.last >= self.period:
            self._fit()
            self.last = t
            fit = 1
            self.n_fit += 1
        f = self.V @ z
        s = np.abs(f).mean() + 1e-9
        lam = _softmax(-f / (self.tau * s))
        self.lam = lam
        return dict(loss=float(np.mean((lam @ P - y) ** 2)), lam=lam, lm=lm, retrain=fit)


class B4Hedge(_Base):
    name = "B4_hedge"

    def __init__(self, M: int, eta: float = 0.5, cap_mult: float = 8.0):
        self.M = M
        self.eta = eta
        self.lam = None
        self.L = np.zeros(M)
        self.cap = None
        self.cap_mult = cap_mult

    def step(self, t, phi, P, y) -> Dict:
        lam = _softmax(-self.eta * self.L)
        self.lam = lam
        lm = np.asarray([np.mean((P[m] - y) ** 2) for m in range(self.M)])
        if self.cap is None:
            self.cap = float(np.quantile(lm, 0.95)) * self.cap_mult
        else:
            self.cap = 0.99 * self.cap + 0.01 * float(np.quantile(lm, 0.95)) * self.cap_mult
        lm_c = np.minimum(lm, self.cap)
        loss = float(np.mean((lam @ P - y) ** 2))
        self.L = self.L + lm_c
        self.L = self.L - self.L.min()
        # the internal cap only shapes B4's own weights; the reported loss is
        # raw so that the arm-independent protocol envelope applies to B4 too
        return dict(loss=loss, loss_raw=loss, loss_capped=min(loss, self.cap),
                    lam=lam, lm=lm_c, retrain=0)


class B5BestExpert(_Base):
    name = "B5_best_expert"

    def __init__(self, m: int, M: Optional[int] = None):
        self.m = m
        self.lam = np.zeros(M if M is not None else m + 1)
        self.lam[m] = 1.0

    def step(self, t, phi, P, y) -> Dict:
        lam = np.zeros(P.shape[0])
        lam[self.m] = 1.0
        self.lam = lam
        lm = np.asarray([np.mean((P[k] - y) ** 2) for k in range(P.shape[0])])
        return dict(loss=float(lm[self.m]), lam=lam, lm=lm, retrain=0)


class B6ADWIN(_Base):
    """Detector + full retraining: the strongest classical point solution."""

    name = "B6_adwin_retrain"

    def __init__(self, M: int, window: int = 90, delta: float = 0.002,
                 refit_window: int = 45):
        self.M = M
        self.window = window
        self.delta = delta
        self.refit_window = refit_window
        self.lam = np.full(M, 1.0 / M)
        self.buf: List[float] = []
        self.hist: List[tuple] = []
        self.n_drift = 0
        self.n_fit = 0

    def _detect(self) -> bool:
        n = len(self.buf)
        if n < 30:
            return False
        best = False
        for cut in range(15, n - 15, max(1, (n - 30) // 12 or 1)):
            w1 = np.asarray(self.buf[:cut])
            w2 = np.asarray(self.buf[cut:])
            m1, m2 = w1.mean(), w2.mean()
            n1, n2 = len(w1), len(w2)
            var = np.var(np.asarray(self.buf)) + 1e-12
            se = np.sqrt(var * (1.0 / n1 + 1.0 / n2))
            crit = np.sqrt(2 * var / n * np.log(2 / self.delta)) + 2 / 3 * np.log(2 / self.delta) / n
            if abs(m1 - m2) > max(se * 3.0, crit):
                best = True
                break
        return best

    def step(self, t, phi, P, y) -> Dict:
        lm = np.asarray([np.mean((P[m] - y) ** 2) for m in range(self.M)])
        loss = float(np.mean((self.lam @ P - y) ** 2))
        self.buf.append(loss)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        self.hist.append((P.copy(), y.copy()))
        if len(self.hist) > self.refit_window:
            self.hist.pop(0)
        fit = 0
        if self._detect() and len(self.hist) >= 20:
            Pa = np.stack([h[0] for h in self.hist])
            Ya = np.stack([h[1] for h in self.hist])
            self.lam = _simplex_fit(Pa, Ya)
            self.buf = self.buf[-10:]
            self.n_drift += 1
            fit = 1
            self.n_fit += 1
        return dict(loss=loss, lam=self.lam.copy(), lm=lm, retrain=fit)


class B7OnlineMeta(_Base):
    """End-to-end online gradient meta-learner: features -> expert weights."""

    name = "B7_online_meta"

    def __init__(self, M: int, p: int, eta: float = 0.05, temp: float = 1.0,
                 cap_mult: float = 8.0):
        self.M = M
        self.p = p
        self.eta = eta
        self.temp = temp
        self.V = np.zeros((M, p + 1))
        self.cap = None
        self.cap_mult = cap_mult
        self.scale = 1.0
        self.lam = np.full(M, 1.0 / M)

    def step(self, t, phi, P, y) -> Dict:
        z = np.concatenate([phi, [1.0]])
        lam = _softmax(self.V @ z / self.temp)
        self.lam = lam
        lm = np.asarray([np.mean((P[m] - y) ** 2) for m in range(self.M)])
        if self.cap is None:
            self.cap = float(np.quantile(lm, 0.95)) * self.cap_mult
        else:
            self.cap = 0.99 * self.cap + 0.01 * float(np.quantile(lm, 0.95)) * self.cap_mult
        lm_c = np.minimum(lm, self.cap)
        loss = float(np.mean((lam @ P - y) ** 2))
        self.scale = 0.98 * self.scale + 0.02 * (lm_c.mean() + 1e-9)
        lbar = float(lam @ lm_c)
        grad = np.outer(lam * (lm_c - lbar), z) / self.scale
        self.V = self.V - self.eta * grad
        # the internal cap only shapes B4's own weights; the reported loss is
        # raw so that the arm-independent protocol envelope applies to B4 too
        return dict(loss=loss, loss_raw=loss, loss_capped=min(loss, self.cap),
                    lam=lam, lm=lm_c, retrain=0)

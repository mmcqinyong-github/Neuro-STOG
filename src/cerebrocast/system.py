"""Modules (4)-(9) and the complete day loop (Algorithm 1 of the paper).

Wake phase  : encode -> retrieve -> gate -> select -> predict -> observe -> learn
Sleep phase : replay, cross-day linking, synaptic downscaling, growth/pruning,
              reconsolidation

The weight source is a *difficulty-normalised, recency-weighted, cluster-conditional
simplex least squares* re-estimated every night (module 9a) and read out the next
morning through the affinity field (module 3).  On top of that slow estimate sit
three fast mechanisms: the forward-model gate (module 6), the three-factor
eligibility logit (module 7) and the metaplastic gain (module 8).  The slow part is
statistically efficient; the fast parts buy adaptivity.  Every module can be
switched off independently, which is what makes the mechanism-pricing design a
factorial experiment rather than a demonstration.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np

from .combo import simplex_ls
from .config import Ablation, SystemConfig
from .memory import ClusterField, EpisodicStore


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def _topk_normalize(v: np.ndarray, k: Optional[int], eps: float) -> np.ndarray:
    """k <= 0 or k >= M means *no* sparsification: the full responsibility set."""
    v = np.maximum(v, 0.0)
    if k is not None and 0 < k < v.size:
        idx = np.argpartition(-v, k - 1)[:k]
        m = np.zeros_like(v)
        m[idx] = v[idx]
        v = m
    s = v.sum()
    if s <= 1e-12:
        return np.full_like(v, 1.0 / v.size)
    lam = v / s
    return (1 - eps) * lam + eps / v.size


class ForwardBank:
    """The M forward error models f_m (module 6), batched.

    One ridge per expert, all sharing the same feature layout, so the M linear
    systems are solved as a single stacked solve.  That is what keeps the
    streaming protocol cheap enough to run 15 arms x 11 domains x 5 seeds.
    """

    def __init__(self, M: int, d: int, lam: float = 1.0, forget: float = 0.985):
        self.M, self.d = M, d
        self.A = np.tile(lam * np.eye(d), (M, 1, 1))
        self.b = np.zeros((M, d))
        self.f = forget
        self.n = np.zeros(M, dtype=int)

    def features(self, phi: np.ndarray, e7: np.ndarray, e28: np.ndarray,
                 sc: float) -> np.ndarray:
        return np.column_stack([np.tile(phi, (self.M, 1)),
                                e7 / sc, e28 / sc, np.ones(self.M)])

    def update(self, X: np.ndarray, y: np.ndarray, forget: float = None):
        f = self.f if forget is None else forget
        self.A = f * self.A + X[:, :, None] * X[:, None, :]
        self.b = f * self.b + X * y[:, None]
        self.n += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.n.min() < 5:
            return np.zeros(self.M)
        try:
            W = np.linalg.solve(
                self.A + 1e-6 * np.eye(self.d)[None], self.b[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            return np.zeros(self.M)
        return np.einsum("md,md->m", X, W)


# ---------------------------------------------------------------------------
# the architecture
# ---------------------------------------------------------------------------
class CerebroCast:
    def __init__(self, M: int, cfg: SystemConfig, arm: Ablation,
                 field: ClusterField, fp_dim: int, seed: int = 0,
                 placebo_reset_rate: float = 0.0,
                 placebo_growth_rate: float = 0.0):
        self.M = M
        self.cfg = cfg
        self.arm = arm
        self.field = field
        self.rng = np.random.default_rng(seed)
        self.p = fp_dim
        C = field.n_clusters()

        # (2) episodic store
        self.epi = EpisodicStore(cap=cfg.episodic_cap, top_s=cfg.episodic_top_s)

        # (4) change detection state
        self.phi_buf: List[np.ndarray] = []
        self.delta_hist: List[float] = []
        self.reset_streak = 0
        self.reset_until = -1

        # (6) forward error models f_m
        d = fp_dim + 3
        self.fwd = ForwardBank(M, d, lam=1.0, forget=0.985)
        self.running_loss = np.zeros(M)
        self.ema7 = None
        self.ema28 = None
        self.seen = 0

        # (7) eligibility traces and noise floor
        self.E = np.zeros((C, M))
        self.sigma2 = 0.0
        self.lscale = 1.0
        self.w_cap = cfg.w_cap

        # (8) metaplastic state
        self.field.a = np.ones(C)

        # (9a) reconsolidation sufficient statistics.  One set per cluster plus a
        # population set; every entry is recency weighted and *difficulty
        # normalised*, so a spike day cannot dominate the fit.
        self.G = np.zeros((C, M, M))
        self.hh = np.zeros((C, M))
        self.gn = np.zeros(C)
        self.Gg = np.zeros((M, M))
        self.hg = np.zeros(M)
        self.gng = 0.0
        # the combination each cluster currently believes in
        self.LAM = np.full((C, M), 1.0 / M)

        # (9) replay buffer
        self.buffer: List[Dict] = []
        self.buffer_cap = 120

        # losses / calibration state
        self.loss_hist: List[float] = []
        self.cap = None
        self.t = 0
        self.tier_hist: List[str] = []
        self.reset_days: List[int] = []
        self.grow_events = 0
        self.prune_events = 0
        self.p_reset = placebo_reset_rate
        self.p_growth = placebo_growth_rate
        self.calib: List[np.ndarray] = []
        self.last_forget = 1.0

    # -- bookkeeping for a changing cluster field ---------------------------
    def _resize_stats(self, C: int):
        M = self.M
        if self.G.shape[0] < C:
            pad = C - self.G.shape[0]
            self.G = np.concatenate([self.G, np.zeros((pad, M, M))], axis=0)
            self.hh = np.concatenate([self.hh, np.zeros((pad, M))], axis=0)
            self.gn = np.concatenate([self.gn, np.zeros(pad)])
            self.LAM = np.concatenate(
                [self.LAM, np.full((pad, M), 1.0 / M)], axis=0)
        elif self.G.shape[0] > C:
            self.G = self.G[:C]
            self.hh = self.hh[:C]
            self.gn = self.gn[:C]
            self.LAM = self.LAM[:C]

    # -- introspection ------------------------------------------------------
    def snapshot(self) -> Dict:
        return dict(
            centers=self.field.centers.copy(), W=self.field.W.copy(),
            a=self.field.a.copy(), mass=self.field.mass.copy(),
            last_used=self.field.last_used.copy(), activity=self.field.activity.copy(),
            born=self.field.born.copy(), E=self.E.copy(),
            sigma2=self.sigma2, running_loss=self.running_loss.copy(),
            phi_buf=[p.copy() for p in self.phi_buf],
            delta_hist=list(self.delta_hist), reset_streak=self.reset_streak,
            fwd=(self.fwd.A.copy(), self.fwd.b.copy(), self.fwd.n.copy()),
            buffer=copy.deepcopy(self.buffer), loss_hist=list(self.loss_hist),
            t=self.t, tier_hist=list(self.tier_hist), reset_days=list(self.reset_days),
            G=self.G.copy(), hh=self.hh.copy(), gn=self.gn.copy(),
            Gg=self.Gg.copy(), hg=self.hg.copy(), gng=self.gng,
            LAM=self.LAM.copy(), ema7=None if self.ema7 is None else self.ema7.copy(),
            ema28=None if self.ema28 is None else self.ema28.copy(),
            seen=self.seen, cap=self.cap, lscale=self.lscale,
        )

    def restore(self, s: Dict):
        self.field.centers = s["centers"].copy()
        self.field.W = s["W"].copy()
        self.field.a = s["a"].copy()
        self.field.mass = s["mass"].copy()
        self.field.last_used = s["last_used"].copy()
        self.field.activity = s["activity"].copy()
        self.field.born = s["born"].copy()
        self.field.novel_streak = 0
        self.field.novel_buf = []
        self.E = s["E"].copy()
        if self.E.shape[0] != self.field.n_clusters():
            self.E = np.zeros((self.field.n_clusters(), self.M))
        self.sigma2 = s["sigma2"]
        self.running_loss = s["running_loss"].copy()
        self.phi_buf = [p.copy() for p in s["phi_buf"]]
        self.delta_hist = list(s["delta_hist"])
        self.reset_streak = s["reset_streak"]
        A, b, n = s["fwd"]
        self.fwd.A = A.copy(); self.fwd.b = b.copy(); self.fwd.n = n.copy()
        self.buffer = copy.deepcopy(s["buffer"])
        self.loss_hist = list(s["loss_hist"])
        self.t = s["t"]
        self.tier_hist = list(s["tier_hist"])
        self.reset_days = list(s["reset_days"])
        self.G = s["G"].copy(); self.hh = s["hh"].copy(); self.gn = s["gn"].copy()
        self.Gg = s["Gg"].copy(); self.hg = s["hg"].copy(); self.gng = s["gng"]
        self.LAM = s["LAM"].copy()
        self.ema7 = None if s["ema7"] is None else s["ema7"].copy()
        self.ema28 = None if s["ema28"] is None else s["ema28"].copy()
        self.seen = s["seen"]; self.cap = s["cap"]; self.lscale = s["lscale"]

    # -- (4) change detection + thalamic gating -----------------------------
    def _detect(self, phi: np.ndarray) -> Dict:
        cfg = self.cfg
        w = cfg.detect_window
        ref = np.asarray(self.phi_buf[-w:]) if len(self.phi_buf) else phi[None, :]
        mu = ref.mean(axis=0)
        delta = float(np.linalg.norm(phi - mu) / np.sqrt(max(self.p, 1)))
        if len(self.delta_hist) >= 30:
            h = np.asarray(self.delta_hist[-200:])
            med = np.median(h)
            mad = np.median(np.abs(h - med)) * 1.4826 + 1e-9
            z = (delta - med) / mad
        else:
            z = 0.0
        self.delta_hist.append(delta)

        if self.arm.no_detect:
            tier = "ADAPT"
        else:
            if z >= cfg.theta2:
                self.reset_streak += 1
                tier = "RESET" if self.reset_streak >= cfg.detect_persist else "ADAPT"
            elif z >= cfg.theta1:
                self.reset_streak = 0
                tier = "ADAPT"
            else:
                self.reset_streak = 0
                tier = "HOLD"

        if self.arm.placebo == "reset" and self.p_reset > 0:
            if self.rng.random() < self.p_reset:
                tier = "RESET"
            elif tier == "RESET":
                tier = "ADAPT"
        if self.arm.no_reset and tier == "RESET":
            tier = "ADAPT"
        if tier == "RESET":
            self.reset_streak = 0
        return {"delta": delta, "z": z, "tier": tier}

    def _forget_boost(self, tier: str) -> float:
        """Thalamic gating controls how fast the slow statistics are rewritten."""
        cfg = self.cfg
        if self.arm.no_detect:
            return cfg.adapt_forget_boost
        if tier == "RESET":
            return cfg.reset_forget_boost
        if tier == "ADAPT":
            return cfg.adapt_forget_boost
        return 1.0

    # -- (5) LC-NE reset ----------------------------------------------------
    def _reset(self, t: int, boost: float):
        cfg = self.cfg
        # the slow statistics are partly forgotten, not just the fast logits
        keep = cfg.reset_decay if self.arm.no_reset is False else 1.0
        self.G *= keep
        self.hh *= keep
        self.gn *= keep
        self.Gg *= keep
        self.hg *= keep
        self.gng *= keep
        self.field.W *= cfg.reset_decay
        self.field.a = np.minimum(cfg.astro_ceil, self.field.a * cfg.reset_lr_boost)
        self.E *= 0.2
        self.reset_until = t + cfg.reset_horizon
        self.reset_days.append(t)
        self.phi_buf = self.phi_buf[-7:] if len(self.phi_buf) > 7 else self.phi_buf
        self.delta_hist = self.delta_hist[-7:]

    # -- main step ----------------------------------------------------------
    def step(self, t: int, phi: np.ndarray, P: np.ndarray, y: np.ndarray) -> Dict:
        """One calendar day.  P is (M, H) expert forecasts, y is (H,) truth."""
        cfg, arm, M = self.cfg, self.arm, self.M
        C = self.field.n_clusters()
        if self.E.shape[0] != C:
            self.E = np.zeros((C, M))
        if self.LAM.shape[0] != C:
            self._resize_stats(C)

        det = self._detect(phi)
        tier = det["tier"]

        # ---- (2) retrieval boost
        boost = None
        if not arm.no_episodic:
            idx = self.epi.retrieve(phi)
            if len(idx):
                Pk = np.asarray([self.epi.phi[i] for i in idx])
                d = np.linalg.norm(self.field.centers[:, None, :] - Pk[None], axis=2)
                near = d.argmin(axis=0)
                b = np.bincount(near, minlength=C).astype(float)
                boost = b / (b.sum() + 1e-9)

        # ---- (3) affinities
        pi = self.field.affinity(phi, boost, growth_enabled=not arm.no_growth)
        if arm.no_clusters:
            pi = np.full(C, 1.0 / C)

        # ---- slow readout: the mixture of cluster combinations
        lam_slow = pi @ self.LAM

        # ---- (6) forward error models -> gate
        if self.ema7 is None:
            self.ema7 = np.zeros(M); self.ema28 = np.zeros(M)
        e7 = self.ema7 / (1 - 0.85 ** max(self.seen, 1))
        e28 = self.ema28 / (1 - 0.96 ** max(self.seen, 1))
        sc = self.lscale + 1e-9
        FX = self.fwd.features(phi, e7, e28, sc)
        fhat = self.fwd.predict(FX) if not arm.no_forward else np.zeros(M)
        fscale = np.mean(np.abs(fhat)) + 1e-9
        g = -cfg.beta_forward * fhat / fscale

        # ---- fast additive logit from three-factor plasticity
        temp = cfg.lambda_temp
        if t <= self.reset_until:
            temp *= cfg.reset_temp_boost
        logit = np.log(np.clip(lam_slow, 1e-6, None)) + pi @ self.field.W
        lam_raw = _softmax(logit / temp) * np.exp(np.clip(g, -8, 8))
        if arm.placebo == "responsibility":
            lam_raw = lam_raw[self.rng.permutation(M)]
        eps_expl = 0.05 if t <= self.reset_until else 0.02
        lam = _topk_normalize(lam_raw, cfg.topk if cfg.topk < M else None, eps_expl)

        # ---- prediction and losses
        yhat = lam @ P
        loss = float(np.mean((yhat - y) ** 2))
        lm = np.asarray([np.mean((P[m] - y) ** 2) for m in range(M)])

        if self.cap is None:
            self.cap = float(np.quantile(lm, cfg.winsor_q)) * cfg.loss_cap_mult
        else:
            q = float(np.quantile(lm, cfg.winsor_q)) * cfg.loss_cap_mult
            self.cap = 0.99 * self.cap + 0.01 * q
        lm_c = np.minimum(lm, self.cap)
        loss_c = min(loss, self.cap)
        self.loss_hist.append(loss_c)

        # ---- (6) RPE with the irreducible noise floor removed
        pred_loss = float(lam @ fhat) + self.sigma2 if not arm.no_forward \
            else float(lam @ lm_c)
        RPE = (pred_loss - loss_c) / (abs(pred_loss) + 1e-9)
        gain = float(np.clip(1.0 + RPE, 0.4, 2.2))
        self.sigma2 = 0.95 * self.sigma2 + 0.05 * float(lm_c.min())

        self.ema7 = 0.85 * self.ema7 + 0.15 * lm_c
        self.ema28 = 0.96 * self.ema28 + 0.04 * lm_c
        self.fwd.update(FX, lm_c)
        self.seen += 1
        self.running_loss = 0.98 * self.running_loss + 0.02 * lm_c
        self.calib.append(np.array([pred_loss, loss_c]))

        # ---- (5) reset before learning if the gate fired
        if tier == "RESET" and not arm.no_reset:
            self._reset(t, cfg.reset_forget_boost)

        # ---- credit: negative loss in level-normalised units
        lbar = float(lam @ lm_c)
        self.lscale = 0.99 * self.lscale + 0.01 * float(np.mean(lm_c)) if self.seen > 1 \
            else float(np.mean(lm_c))
        adv = np.clip(-(lm_c / (self.lscale + 1e-9)) + 1.0, -5.0, 5.0)

        # ---- (7) eligibility traces
        if arm.no_three_factor:
            act = np.full(C, 1.0 / C)
        else:
            act = pi * (pi > 1.0 / (2 * C))
        rho = 0.5 ** (1.0 / max(cfg.elig_halflife, 1))
        self.E = rho * self.E + act[:, None]
        Eeff = self.E[:, 0] * (1.0 - rho)

        # ---- (8) metaplastic gain
        if arm.no_astro:
            a = np.ones(C)
        else:
            surprise = np.clip(np.abs(lm_c - self.running_loss) /
                               (self.running_loss + 1e-9), 0, 3).mean()
            per_c = np.full(C, surprise)
            if tier == "HOLD":
                target = np.full(C, cfg.astro_floor)
            else:
                target = 1.0 + 0.8 * np.clip(per_c, 0, 1.5)
            a = self.field.a + cfg.astro_up * (target - self.field.a)
            a = np.clip(a, cfg.astro_floor, cfg.astro_ceil)
            self.field.a = a

        # ---- (7) fast weight update (an additive correction on the slow solve)
        eta = cfg.eta * (cfg.hold_gain if tier == "HOLD" else 1.0)
        if t <= self.reset_until:
            eta = cfg.eta * cfg.reset_lr_boost
        dW = eta * a[:, None] * gain * Eeff[:, None] * adv[None, :]
        self.field.W = cfg.trace_decay * self.field.W + dW
        self.field.W = self.field.W - self.field.W.mean(axis=1, keepdims=True)
        self.field.W = np.clip(self.field.W, -self.w_cap, self.w_cap)

        # ---- (9a) accumulate the reconsolidation statistics
        # difficulty normalisation: a day contributes in proportion to how hard it
        # was for the pool as a whole, so spike days cannot steer the fit.
        diff = float(np.quantile(lm_c, cfg.diff_q)) + 1e-9
        wt = 1.0 / diff
        boost_f = self._forget_boost(tier)
        rho_c = 0.5 ** (boost_f / max(cfg.recall_halflife, 1e-6))
        self.last_forget = rho_c
        self.G = rho_c * self.G + (pi * wt)[:, None, None] * (P @ P.T)[None, :, :]
        self.hh = rho_c * self.hh + (pi * wt)[:, None] * (P @ y)
        self.gn = rho_c * self.gn + pi * wt
        self.Gg = rho_c * self.Gg + wt * (P @ P.T)
        self.hg = rho_c * self.hg + wt * (P @ y)
        self.gng = rho_c * self.gng + wt

        # ---- bookkeeping for the sleep phase
        self.buffer.append(dict(t=t, phi=phi.copy(), lm=lm_c.copy(), lam=lam.copy(),
                                c=int(np.argmax(pi)), adv=adv.copy(), gain=gain,
                                tier=tier, wt=wt))
        if len(self.buffer) > self.buffer_cap:
            self.buffer.pop(0)
        self.epi.write(phi, lm_c, t, float(np.abs(loss_c - pred_loss)))
        self.field.touch(pi, t)
        self.field.drift(pi, phi)
        self.phi_buf.append(phi.copy())
        if len(self.phi_buf) > max(cfg.detect_window, 60):
            self.phi_buf.pop(0)
        self.tier_hist.append(tier)
        self.t += 1

        diag_sleep = self._sleep(t, phi, pi, lm_c)
        self._reset_transient(t)

        return dict(
            t=t, loss=loss_c, loss_raw=loss, tier=tier, delta=det["delta"], z=det["z"],
            lam=lam.copy(), pi=pi.copy(), n_clusters=C,
            entropy=float(-(np.where(lam > 0, lam, 1) * np.log(np.where(lam > 0, lam, 1))).sum()),
            top1=float(lam.max()), lbar=lbar, best_expert=float(lm_c.min()),
            oracle=float(lm_c.min()), fhat=fhat.copy(), lm=lm_c.copy(),
            reset=(tier == "RESET"), **diag_sleep,
        )

    # -- (9) sleep ----------------------------------------------------------
    def _sleep(self, t: int, phi: np.ndarray, pi: np.ndarray, lm_c: np.ndarray) -> Dict:
        cfg, arm = self.cfg, self.arm
        C = self.field.n_clusters()
        M = self.M
        if self.LAM.shape[0] != C:
            self._resize_stats(C)

        replayed = np.zeros(C, dtype=bool)
        n_replay = 0
        if not arm.no_replay and len(self.buffer) > 5:
            B = self.buffer
            k = max(1, int(np.ceil(cfg.replay_frac * len(B))))
            if arm.placebo == "replay":
                pick = self.rng.choice(len(B), size=k, replace=False)
            else:
                prio = np.array([abs(b["gain"] - 1.0) + 0.5 * np.abs(b["adv"]).mean()
                                 for b in B])
                prio = prio / (prio.sum() + 1e-9)
                pick = self.rng.choice(len(B), size=k, replace=False, p=prio)
            for j in pick:
                b = B[j]
                c_now = int(np.argmax(pi))
                c_rep = min(int(b["c"]), C - 1)
                replayed[c_rep] = True
                adv = np.clip((float(b["lam"] @ b["lm"]) - b["lm"])
                              / (self.lscale + 1e-9), -5.0, 5.0)
                act = np.zeros(C)
                act[c_rep] = 1.0
                self.field.W = np.clip(
                    self.field.W + (cfg.eta * cfg.replay_lr * self.field.a[:, None]
                                    * b["gain"] * act[:, None] * adv[None, :]),
                    -self.w_cap, self.w_cap)
                # (9b) cross-day linking
                if c_now != c_rep:
                    self.field.W[c_now] += 0.05 * (self.field.W[c_rep] - self.field.W[c_now])
                n_replay += 1

        # (9a) reconsolidation: re-estimate every cluster's combination from its
        # own difficulty-normalised, recency-weighted second moments.
        n_cons = 0
        lam_g = simplex_ls(self.Gg, self.hg, cfg.consol_ridge,
                           cfg.consol_shrink_u) if self.gng > 1e-8 else None
        for c in range(C):
            if self.gn[c] < 1e-8:
                self.LAM[c] = lam_g if lam_g is not None else np.full(M, 1.0 / M)
                continue
            lam_c = simplex_ls(self.G[c], self.hh[c], cfg.consol_ridge)
            if lam_c is None:
                self.LAM[c] = lam_g if lam_g is not None else np.full(M, 1.0 / M)
                continue
            if lam_g is not None and not arm.no_clusters:
                xi = self.gn[c] / (self.gn[c] + cfg.consol_shrink * self.gng + 1e-12)
                lam_c = (1 - xi) * lam_g + xi * lam_c
            self.LAM[c] = lam_c
            n_cons += 1
        if arm.no_clusters:
            # module (3) off: one shared combination for the whole population
            base = lam_g if lam_g is not None else np.full(M, 1.0 / M)
            self.LAM[:] = base
            n_cons = 1
        elif lam_g is not None:
            self.LAM = np.where(np.isfinite(self.LAM), self.LAM, lam_g)

        # (9c) synaptic downscaling with replay exemption
        if not arm.no_downscale:
            gt = cfg.gamma0 / np.sqrt(max(1.0, t / 50.0))
            exempt = replayed if not arm.no_replay else np.ones(C, dtype=bool)
            shrink = np.where(exempt, 1.0, 1.0 - gt)
            self.field.W = self.field.W * shrink[:, None]

        # (9d) growth and pruning
        grew = pruned = 0
        if not arm.no_growth and not arm.no_clusters:
            if arm.placebo == "growth":
                if self.p_growth > 0 and self.rng.random() < self.p_growth:
                    c0 = int(np.argmax(pi))
                    self.field.centers = np.vstack([self.field.centers, phi[None, :]])
                    self.field.W = np.vstack([self.field.W, self.field.W[c0][None, :]])
                    self.field.a = np.concatenate([self.field.a, [1.0]])
                    self.field.mass = np.concatenate([self.field.mass, [1.0 / (C + 1)]])
                    self.field.last_used = np.concatenate([self.field.last_used, [t]])
                    self.field.activity = np.concatenate([self.field.activity, [0.0]])
                    self.field.born = np.concatenate([self.field.born, [t]])
                    self.E = np.vstack([self.E, np.zeros((1, M))])
                    self._resize_stats(C + 1)
                    self.LAM[C] = self.LAM[c0]
                    grew = 1
            else:
                c0 = int(np.argmax(pi))
                if self.field.maybe_grow(phi, t, self.field.W[c0]):
                    self.E = np.vstack([self.E, np.zeros((1, M))])
                    self._resize_stats(self.field.n_clusters())
                    self.LAM[-1] = self.LAM[c0]
                    grew = 1
            pruned = self.field.maybe_prune(t)
            if pruned and self.E.shape[0] > self.field.n_clusters():
                self.E = self.E[: self.field.n_clusters()]
                self._resize_stats(self.field.n_clusters())
        self.grow_events += grew
        self.prune_events += pruned
        return dict(n_replay=n_replay, grew=grew, pruned=pruned, n_cons=n_cons,
                    w_norm=float(np.linalg.norm(self.field.W)))

    def _reset_transient(self, t: int):
        if t <= self.reset_until:
            return
        cfg = self.cfg
        self.field.a = self.field.a + 0.25 * (1.0 - self.field.a)
        self.field.a = np.clip(self.field.a, cfg.astro_floor, cfg.astro_ceil)

"""Modules (2) and (3): episodic memory and the slow like-to-like affinity graph.

Module (2) -- hippocampal episodic store.  One-shot write of the day's
fingerprint together with the day's loss vector; similarity-based retrieval of a
whole cluster of similar days (pattern completion), used both to modulate the
cluster affinities and to give replay its priority ordering.

Module (3) -- slow cortical affinity field.  Environment clusters carry their own
responsibility vector over experts; affinities come from a Gaussian kernel on the
fingerprint distance and are smoothed by one message-passing step on the
cluster-cluster graph (Lemma L2).  The field grows and prunes according to
module (9d).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# (2) episodic store
# ---------------------------------------------------------------------------
class EpisodicStore:
    def __init__(self, cap: int = 4000, top_s: int = 12):
        self.cap = cap
        self.top_s = top_s
        self.phi = []
        self.loss = []
        self.day = []
        self.priority = []

    def write(self, phi: np.ndarray, loss_vec: np.ndarray, day: int, prio: float):
        self.phi.append(np.asarray(phi, dtype=np.float64))
        self.loss.append(np.asarray(loss_vec, dtype=np.float64))
        self.day.append(int(day))
        self.priority.append(float(prio))
        if len(self.phi) > self.cap:
            self.phi.pop(0)
            self.loss.pop(0)
            self.day.pop(0)
            self.priority.pop(0)

    def retrieve(self, phi: np.ndarray, s: Optional[int] = None) -> np.ndarray:
        """Indices of the s most similar stored episodes (empty if none)."""
        if not self.phi:
            return np.array([], dtype=int)
        k = self.top_s if s is None else s
        P = np.asarray(self.phi)
        d = np.linalg.norm(P - phi[None, :], axis=1)
        k = min(k, len(d))
        return np.argpartition(d, k - 1)[:k]

    def __len__(self) -> int:
        return len(self.phi)


# ---------------------------------------------------------------------------
# (3) slow affinity field
# ---------------------------------------------------------------------------
@dataclass
class ClusterField:
    tau_kappa: float = 0.35
    graph_alpha: float = 0.30
    cluster_momentum: float = 0.05
    grow_r_cov: float = 1.60          # overwritten from data in init_from_train
    grow_persist: int = 3
    prune_mass: float = 1e-3
    prune_idle: int = 45
    k0: int = 6
    max_clusters: int = 20

    centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    W: np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    a: np.ndarray = field(default_factory=lambda: np.zeros(0))       # metaplastic rate
    mass: np.ndarray = field(default_factory=lambda: np.zeros(0))    # EMA responsibility
    last_used: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    activity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    born: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    novel_streak: int = 0
    novel_buf: List[np.ndarray] = field(default_factory=list)

    # -- construction ------------------------------------------------------
    def init_from_train(self, PHI: np.ndarray, M: int, seed: int = 0) -> "ClusterField":
        k = min(self.k0, len(PHI))
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(PHI)
        self.centers = km.cluster_centers_.astype(np.float64)
        # The fingerprint lives in p ~ 40 dimensions and is standardised, so the
        # typical radius is O(sqrt(p)).  Both the affinity temperature and the
        # growth radius are therefore calibrated from the training split rather
        # than hard-coded: a fixed radius is meaningless across domains.
        d = np.linalg.norm(PHI - self.centers[km.labels_], axis=1)
        self.tau_kappa = float(max(0.75 * np.median(d), 1e-6))
        self.grow_r_cov = float(max(np.quantile(d, 0.99) * 1.35, self.tau_kappa))
        self.W = np.zeros((k, M))
        self.a = np.ones(k)
        self.mass = np.full(k, 1.0 / k)
        self.last_used = np.zeros(k, dtype=int)
        self.activity = np.zeros(k)
        self.born = np.zeros(k, dtype=int)
        self.novel_streak = 0
        self.novel_buf = []
        return self

    # -- query -------------------------------------------------------------
    def distances(self, phi: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.centers - phi[None, :], axis=1)

    def affinity(self, phi: np.ndarray, episodic_boost: Optional[np.ndarray] = None,
                 growth_enabled: bool = True) -> np.ndarray:
        d = self.distances(phi)
        A = np.exp(-(d ** 2) / (2 * self.tau_kappa ** 2)) + 1e-9
        # one message-passing step on the cluster-cluster graph (L2)
        Dc = np.linalg.norm(self.centers[:, None, :] - self.centers[None, :, :], axis=2)
        G = np.exp(-(Dc ** 2) / (2 * self.tau_kappa ** 2))
        np.fill_diagonal(G, 0.0)
        rs = G.sum(axis=1, keepdims=True) + 1e-9
        A = A + self.graph_alpha * (G @ A) / rs[:, 0]
        if episodic_boost is not None and len(episodic_boost):
            A = A * (1.0 + 0.35 * episodic_boost)
        pi = A / A.sum()
        # record the novelty streak even when growth is disabled (logged)
        if d.min() > self.grow_r_cov:
            self.novel_streak += 1
            self.novel_buf.append(phi.copy())
        else:
            self.novel_streak = 0
            self.novel_buf = []
        return pi

    # -- plasticity of the field itself (9d) -------------------------------
    def maybe_grow(self, phi: np.ndarray, t: int, W_init: np.ndarray) -> bool:
        if self.novel_streak < self.grow_persist or not len(self.novel_buf):
            return False
        if len(self.mass) >= self.max_clusters:
            self.novel_streak = 0
            self.novel_buf = []
            return False
        new_c = np.mean(np.asarray(self.novel_buf[-self.grow_persist:]), axis=0)
        self.centers = np.vstack([self.centers, new_c[None, :]])
        self.W = np.vstack([self.W, W_init[None, :]])
        self.a = np.concatenate([self.a, [1.0]])
        self.mass = np.concatenate([self.mass, [1.0 / (len(self.mass) + 1)]])
        self.last_used = np.concatenate([self.last_used, [t]])
        self.activity = np.concatenate([self.activity, [0.0]])
        self.born = np.concatenate([self.born, [t]])
        self.novel_streak = 0
        self.novel_buf = []
        return True

    def maybe_prune(self, t: int) -> int:
        C = len(self.mass)
        if C <= 1:
            return 0
        dead = np.where((self.mass < self.prune_mass) &
                        (t - self.last_used > self.prune_idle))[0]
        if len(dead) == 0 or len(dead) >= C:
            return 0
        keep = np.setdiff1d(np.arange(C), dead)
        self.centers = self.centers[keep]
        self.W = self.W[keep]
        self.a = self.a[keep]
        self.mass = self.mass[keep]
        self.last_used = self.last_used[keep]
        self.activity = self.activity[keep]
        self.born = self.born[keep]
        self.mass = self.mass / self.mass.sum()
        return len(dead)

    def drift(self, pi: np.ndarray, phi: np.ndarray):
        """Slow like-to-like drift of the centres (module 3, 'slow' graph)."""
        self.centers += self.cluster_momentum * pi[:, None] * (phi[None, :] - self.centers)

    def touch(self, pi: np.ndarray, t: int):
        self.last_used = np.maximum(self.last_used, np.where(pi > 1e-3, t, self.last_used))
        self.mass = 0.97 * self.mass + 0.03 * pi
        self.mass = self.mass / (self.mass.sum() + 1e-12)
        self.activity = 0.95 * self.activity + 0.05 * pi

    def n_clusters(self) -> int:
        return len(self.mass)

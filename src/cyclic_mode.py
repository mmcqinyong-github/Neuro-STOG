"""循环模式访问预测器 v0.2：解耦版本（证伪规格 v0.1 §9 预注册）。

v0.1（已否决）：仅当前相位模式参与修正。
v0.2（本文件）：预测 = 骨干 + 全模式连续修正；相位隔离只保留在更新侧
（仅当前相位模式写入残差）。三个臂（循环/全局/反应式门控）使用同一
控制器拟合流程（特征标准化 + 相同 ridge），消除 v0.1 的对比不对称。

时序纪律（不可违反）：预测只用 t-1 状态；残差在观测到 x_t 之后写入；
测试阶段无任何梯度计算；骨干冻结。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    seed: int
    n_vars: int = 3
    n_train: int = 300
    n_calibrate: int = 400
    n_test: int = 500
    lookback: int = 24
    periods: tuple = (24, 168)
    season_amp: tuple = (1.0, 0.8)
    level_decay: float = 0.997
    drift_prob: float = 0.002
    drift_jump: float = 1.5
    noise_scale: float = 0.25
    env: str = "env1"
    pulse_period: int = 48
    pulse_phase: int = 7
    pulse_scale: float = 4.0
    ridge: float = 1e-4
    controller_ridge: float = 1.0
    rho: float = 0.9
    state_clip: float = 8.0
    n_modes: int = 3
    dwells: tuple = (24, 168, 12)
    p_cyc_structured: float = 0.95
    n_gate_features: int = 4


@dataclass(frozen=True)
class A1Result:
    base_wmse: float
    cyclic_wmse: float
    reactive_wmse: float
    base_mae: float
    cyclic_mae: float
    reactive_mae: float
    n_predictions: int


@dataclass(frozen=True)
class A2Result:
    structured_wmse: float
    unstructured_wmse: float
    structured_mae: float
    unstructured_mae: float
    n_predictions: int


@dataclass(frozen=True)
class A4Result:
    isolated_wmse: float
    global_wmse: float
    isolated_mae: float
    global_mae: float
    isolated_pollution: float
    global_pollution: float
    n_predictions: int


def make_series(cfg: EnvConfig, seed: int, n_obs: int) -> np.ndarray:
    """多季节 + 慢漂移 + 轻尾噪声；env3 叠加固定相位的外生脉冲。"""
    rng = np.random.default_rng(seed)
    level = np.zeros(cfg.n_vars)
    var_scale = 0.5 + rng.random(cfg.n_vars)
    phases = rng.uniform(0, 2 * np.pi, size=len(cfg.periods))
    out = np.empty((n_obs, cfg.n_vars))
    for t in range(n_obs):
        level = cfg.level_decay * level + (
            rng.random(cfg.n_vars) < cfg.drift_prob
        ) * rng.choice(np.array([-1.0, 1.0]), size=cfg.n_vars) * cfg.drift_jump
        season = sum(
            amp * np.sin(2 * np.pi * t / p + phases[j]) * var_scale
            for j, (amp, p) in enumerate(zip(cfg.season_amp, cfg.periods))
        )
        noise = cfg.noise_scale * rng.normal(size=cfg.n_vars)
        pulse = 0.0
        if cfg.env == "env3" and t % cfg.pulse_period == cfg.pulse_phase:
            pulse = cfg.pulse_scale * rng.normal(size=cfg.n_vars)
        out[t] = level + season + noise + pulse
    return out


def _windows(x, lookback):
    X = np.stack([x[t - lookback : t] for t in range(lookback, x.shape[0])])
    return X.reshape(X.shape[0], -1), x[lookback:]


def fit_dlinear(X2d, y, ridge):
    X2d = np.asarray(X2d, dtype=float)
    if X2d.ndim == 3:
        X2d = X2d.reshape(X2d.shape[0], -1)
    A = np.concatenate([X2d, np.ones((X2d.shape[0], 1))], axis=1)
    params = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y)
    return {"coef": params[:-1], "intercept": params[-1]}


def predict_dlinear(backbone, window):
    X = np.asarray(window, dtype=float).reshape(1, -1)
    return (X @ backbone["coef"] + backbone["intercept"]).reshape(-1)


class CyclicScheduler:
    """半马尔可夫相位调度：驻留结束后以 p_cyc 走向下一模式，否则随机。"""

    def __init__(self, n_modes: int, dwells: tuple, p_cyc: float, rng):
        assert len(dwells) == n_modes
        self.n_modes = n_modes
        self.dwells = tuple(int(d) for d in dwells)
        self.p_cyc = float(p_cyc)
        self.rng = rng
        self.mode = 0
        self.remaining = self.dwells[0]

    def step(self) -> int:
        m = self.mode
        self.remaining -= 1
        if self.remaining == 0:
            if self.rng.random() < self.p_cyc:
                nxt = (self.mode + 1) % self.n_modes
            else:
                nxt = int(self.rng.integers(self.n_modes))
            self.mode, self.remaining = nxt, self.dwells[nxt]
        return m

    def reset(self) -> None:
        self.mode = 0
        self.remaining = self.dwells[0]


class ModeBank:
    """K 个模式慢状态。update 只动当前模式（相位隔离）；update_global 供对照 A4。"""

    def __init__(self, n_modes: int, n_vars: int, rho: float, clip: float):
        self.n_modes = n_modes
        self.rho = float(rho)
        self.clip = float(clip)
        self.states = np.zeros((n_modes, n_vars))

    def reset(self) -> None:
        self.states[:] = 0.0

    def update(self, mode: int, residual: np.ndarray) -> None:
        r = np.clip(np.asarray(residual, dtype=float), -self.clip, self.clip)
        self.states[mode] = self.rho * self.states[mode] + (1.0 - self.rho) * r

    def update_global(self, residual: np.ndarray) -> None:
        r = np.clip(np.asarray(residual, dtype=float), -self.clip, self.clip)
        self.states = self.rho * self.states + (1.0 - self.rho) * r


def _fit_ridge(A, Y, ridge):
    A1 = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    params = np.linalg.solve(A1.T @ A1 + ridge * np.eye(A1.shape[1]), A1.T @ Y)
    return params[:-1], params[-1]


def fit_controller(states, targets, ridge):
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    z = (states - mean) / std
    coef, intercept = _fit_ridge(z, targets, ridge)
    return {"coef": coef, "intercept": intercept, "mean": mean, "std": std}


def apply_controller(ctl, state):
    z = (np.asarray(state, dtype=float) - ctl["mean"]) / ctl["std"]
    return z @ ctl["coef"] + ctl["intercept"]


def _compose_correction(controllers, states, weights=None) -> np.ndarray:
    """全模式连续修正：sum_k g_k(s_k)（或加权的 sum_k w_k g_k(s_k)）。"""
    parts = [apply_controller(c, s) for c, s in zip(controllers, states)]
    if weights is None:
        return sum(parts)
    return sum(w * p for w, p in zip(weights, parts))


def _zero_controller(n_vars):
    return {"coef": np.zeros((n_vars, n_vars)), "intercept": np.zeros(n_vars),
            "mean": np.zeros(n_vars), "std": np.ones(n_vars)}


def _metrics(errors):
    ae = np.abs(errors)
    se = errors**2
    q99 = np.quantile(se, 0.99)
    return float(np.mean(np.minimum(se, q99))), float(np.mean(ae))


def _fit_backbone(cfg: EnvConfig):
    train = make_series(cfg, cfg.seed, cfg.n_train + cfg.lookback)
    X, y = _windows(train, cfg.lookback)
    return fit_dlinear(X, y, cfg.ridge)


def _fit_controllers(x, backbone, cfg, isolated: bool):
    """校准段收集 (状态 -> 残差) 并逐模式拟合同一控制器。

    isolated=True：相位隔离更新（循环臂）；False：全局更新（全局/反应式臂）。
    两种情形下记录规则一致：每步记录此刻每个（isolated 时为活动）模式的状态，
    目标为该步残差。
    """
    sched = CyclicScheduler(cfg.n_modes, cfg.dwells, cfg.p_cyc_structured,
                            np.random.default_rng(cfg.seed + 555))
    bank = ModeBank(cfg.n_modes, cfg.n_vars, cfg.rho, cfg.state_clip)
    feats = [[] for _ in range(cfg.n_modes)]
    tgts = [[] for _ in range(cfg.n_modes)]
    for t in range(cfg.lookback, x.shape[0]):
        m = sched.step()
        base = predict_dlinear(backbone, x[t - cfg.lookback : t])
        r = x[t] - base
        if isolated:
            feats[m].append(bank.states[m].copy())
            tgts[m].append(r)
            bank.update(m, r)
        else:
            for k in range(cfg.n_modes):
                feats[k].append(bank.states[k].copy())
                tgts[k].append(r)
            bank.update_global(r)
    controllers = []
    for k in range(cfg.n_modes):
        if len(feats[k]) < cfg.n_vars + 2:
            controllers.append(_zero_controller(cfg.n_vars))
        else:
            controllers.append(
                fit_controller(np.stack(feats[k]), np.stack(tgts[k]), cfg.controller_ridge)
            )
    return controllers


def _is_pulse_step(cfg: EnvConfig, t_abs: int) -> bool:
    return cfg.env == "env3" and t_abs % cfg.pulse_period == cfg.pulse_phase


def _prequential_shared(x, backbone, controllers, cfg, p_cyc, seed, isolated):
    """循环/全局臂共用预测规则（全模式连续修正），唯一差异 = 更新规则。

    返回 (wmse, mae, n, pollution)；pollution = 脉冲步上非活动模式状态变化均值。
    """
    sched = CyclicScheduler(cfg.n_modes, cfg.dwells, p_cyc, np.random.default_rng(seed))
    bank = ModeBank(cfg.n_modes, cfg.n_vars, cfg.rho, cfg.state_clip)
    preds, targets = [], []
    pollution, n_pulse = 0.0, 0
    for t in range(cfg.lookback, x.shape[0]):
        m = sched.step()
        base = predict_dlinear(backbone, x[t - cfg.lookback : t])
        preds.append(base + _compose_correction(controllers, bank.states))
        targets.append(x[t])
        before = bank.states.copy()
        r = x[t] - base
        if isolated:
            bank.update(m, r)
        else:
            bank.update_global(r)
        if _is_pulse_step(cfg, t):
            n_pulse += 1
            others = [k for k in range(cfg.n_modes) if k != m]
            pollution += float(np.abs(bank.states[others] - before[others]).sum())
    wmse, mae = _metrics(np.array(preds) - np.array(targets))
    return wmse, mae, len(preds), (pollution / max(n_pulse, 1))


def _gate_features(window):
    w = np.asarray(window, dtype=float)
    return np.concatenate([w.mean(axis=0), w.std(axis=0),
                           w[-1], np.abs(w[-1] - w.mean(axis=0))])


def _prequential_reactive(x, backbone, controllers, cfg, seed):
    """反应式门控臂（对照 A1）：全局更新 + 窗口特征 softmax 加权各模式修正。

    与循环臂控制器对称（同一 fit_controller）；门控权重由校准段候选误差
    责任拟合（温度 1 的 softmax 目标，ridge 回归）。
    """
    # 校准：全局更新轨迹上拟合门控
    bank = ModeBank(cfg.n_modes, cfg.n_vars, cfg.rho, cfg.state_clip)
    feats, cands, tgts = [], [], []
    for t in range(cfg.lookback, x.shape[0]):
        base = predict_dlinear(backbone, x[t - cfg.lookback : t])
        cand = [base + apply_controller(controllers[k], bank.states[k])
                for k in range(cfg.n_modes)]
        feats.append(_gate_features(x[t - cfg.lookback : t]))
        cands.append(cand)
        tgts.append(x[t])
        bank.update_global(x[t] - base)
    feats = np.stack(feats)
    tgts = np.stack(tgts)
    cands = np.stack([np.stack(c) for c in cands])
    err = ((cands - tgts[:, None, :]) ** 2).sum(axis=2)
    resp = np.exp(-err / 1.0)
    resp /= resp.sum(axis=1, keepdims=True)
    W, b = _fit_ridge(feats, resp, cfg.controller_ridge)

    bank.reset()
    preds = []
    for t in range(cfg.lookback, x.shape[0]):
        base = predict_dlinear(backbone, x[t - cfg.lookback : t])
        w = _gate_features(x[t - cfg.lookback : t])
        logits = np.clip(w @ W + b, -30, 30)
        pi = np.exp(logits - logits.max())
        pi /= pi.sum()
        preds.append(base + _compose_correction(controllers, bank.states, weights=pi))
        bank.update_global(x[t] - base)
    wmse, mae = _metrics(np.array(preds) - tgts)
    return wmse, mae, len(preds)


def _prequential_base(x, backbone, cfg):
    preds = [predict_dlinear(backbone, x[t - cfg.lookback : t])
             for t in range(cfg.lookback, x.shape[0])]
    tgts = x[cfg.lookback:]
    return _metrics(np.array(preds) - np.array(tgts)) + (len(preds),)


def _load(cfg, backbone):
    if backbone is None:
        backbone = _fit_backbone(cfg)
    else:
        backbone = {k: v.copy() for k, v in backbone.items()}
    calib = make_series(cfg, cfg.seed + 10_000, cfg.n_calibrate + cfg.lookback)
    test = make_series(cfg, cfg.seed + 20_000, cfg.n_test + cfg.lookback)
    return backbone, calib, test


def run_a1(cfg: EnvConfig, backbone: Optional[dict]) -> A1Result:
    backbone, calib, test = _load(cfg, backbone)
    ctl_cyc = _fit_controllers(calib, backbone, cfg, isolated=True)
    ctl_glb = _fit_controllers(calib, backbone, cfg, isolated=False)
    base_wmse, base_mae, n = _prequential_base(test, backbone, cfg)
    cyclic_wmse, cyclic_mae, _, _ = _prequential_shared(
        test, backbone, ctl_cyc, cfg, cfg.p_cyc_structured, cfg.seed + 31_000, True)
    reactive_wmse, reactive_mae, _ = _prequential_reactive(
        test, backbone, ctl_glb, cfg, cfg.seed + 41_000)
    return A1Result(base_wmse, cyclic_wmse, reactive_wmse,
                    base_mae, cyclic_mae, reactive_mae, n)


def run_a2(cfg: EnvConfig, backbone: Optional[dict]) -> A2Result:
    backbone, calib, test = _load(cfg, backbone)
    ctl = _fit_controllers(calib, backbone, cfg, isolated=True)
    s_wmse, s_mae, n, _ = _prequential_shared(
        test, backbone, ctl, cfg, cfg.p_cyc_structured, cfg.seed + 31_000, True)
    u_wmse, u_mae, _, _ = _prequential_shared(
        test, backbone, ctl, cfg, 1.0 / cfg.n_modes, cfg.seed + 31_000, True)
    return A2Result(s_wmse, u_wmse, s_mae, u_mae, n)


def run_a4(cfg: EnvConfig, backbone: Optional[dict]) -> A4Result:
    backbone, calib, test = _load(cfg, backbone)
    ctl_iso = _fit_controllers(calib, backbone, cfg, isolated=True)
    ctl_glb = _fit_controllers(calib, backbone, cfg, isolated=False)
    i_wmse, i_mae, n, i_pol = _prequential_shared(
        test, backbone, ctl_iso, cfg, cfg.p_cyc_structured, cfg.seed + 31_000, True)
    g_wmse, g_mae, _, g_pol = _prequential_shared(
        test, backbone, ctl_glb, cfg, cfg.p_cyc_structured, cfg.seed + 31_000, False)
    return A4Result(i_wmse, g_wmse, i_mae, g_mae, i_pol, g_pol, n)

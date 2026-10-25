"""残差状态表示学习：对照 C1 最小实现（证伪规格 v0.1 §2）。

学习对象：收缩型状态单元（alpha 门控 tanh 递归），校准段 BPTT 训练，
测试阶段冻结。手工反向传播（无自动微分依赖），梯度由有限差分测试校验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ResidualConfig:
    seed: int
    n_vars: int = 3
    n_train: int = 300
    n_calibrate: int = 400
    n_test: int = 500
    lookback: int = 24
    m: int = 2                      # 状态扩张因子，d = n_vars * m
    periods: tuple = (24, 168)
    season_amp: tuple = (1.0, 0.8)
    level_decay: float = 0.997
    drift_prob: float = 0.002
    drift_jump: float = 1.5
    noise_scale: float = 0.25
    env: str = "env1"               # env1 | env3 | env5 | env6
    pulse_period: int = 48
    pulse_phase: int = 7
    pulse_scale: float = 4.0
    ridge: float = 1e-4
    controller_ridge: float = 1.0
    spec_penalty: float = 1e-3      # ||W_s||_F^2 谱惩罚
    train_epochs: int = 25
    restarts: int = 3
    adam_lr: float = 0.02


@dataclass(frozen=True)
class C1Result:
    base_wmse: float
    learned_wmse: float
    e1_wmse: float
    e2_wmse: float
    base_mae: float
    learned_mae: float
    e1_mae: float
    e2_mae: float
    n_predictions: int


# ---------------------------------------------------------------- 环境

def make_series(cfg: ResidualConfig, seed: int, n_obs: int) -> np.ndarray:
    """env1/3 与前两条路线保持一致以便跨路线比较；env5 方差体制切换；env6 领先-滞后。"""
    rng = np.random.default_rng(seed)
    var_scale = 0.5 + rng.random(cfg.n_vars)
    phases = rng.uniform(0, 2 * np.pi, size=len(cfg.periods))
    out = np.empty((n_obs, cfg.n_vars))

    if cfg.env == "env6":
        # 公共漂移创新 u_t：变量0当期加载，变量1延迟 2 步加载
        u = np.zeros((n_obs + 2, cfg.n_vars))
        innov = (rng.random((n_obs, cfg.n_vars)) < cfg.drift_prob) * rng.choice(
            np.array([-1.0, 1.0]), size=(n_obs, cfg.n_vars)) * cfg.drift_jump
        for t in range(n_obs):
            u[t + 2] = cfg.level_decay * u[t + 1] + innov[t]
        for t in range(n_obs):
            season = sum(
                amp * np.sin(2 * np.pi * t / p + phases[j]) * var_scale
                for j, (amp, p) in enumerate(zip(cfg.season_amp, cfg.periods))
            )
            out[t, 0] = u[t + 2, 0] + season[0] + cfg.noise_scale * rng.normal()
            lag = u[t, 0] if cfg.n_vars > 1 else 0.0
            rest = (u[t + 2, 1:] if cfg.n_vars > 1 else np.zeros(0))
            out[t, 1:] = 0.6 * lag + rest + season[1:] + cfg.noise_scale * rng.normal(size=cfg.n_vars - 1) if cfg.n_vars > 1 else out[t, :1]
        return out

    regime = np.ones(n_obs)
    if cfg.env == "env5":
        state, r = 1.0, 1.0
        for t in range(n_obs):
            if rng.random() < 0.005:
                r = 0.5 if r > 1 else 1.8
            regime[t] = r
    level = np.zeros(cfg.n_vars)
    for t in range(n_obs):
        jump = (rng.random(cfg.n_vars) < cfg.drift_prob) * rng.choice(
            np.array([-1.0, 1.0]), size=cfg.n_vars) * cfg.drift_jump * regime[t]
        level = cfg.level_decay * level + jump
        season = sum(
            amp * np.sin(2 * np.pi * t / p + phases[j]) * var_scale
            for j, (amp, p) in enumerate(zip(cfg.season_amp, cfg.periods))
        )
        noise = cfg.noise_scale * regime[t] * rng.normal(size=cfg.n_vars)
        pulse = 0.0
        if cfg.env == "env3" and t % cfg.pulse_period == cfg.pulse_phase:
            pulse = cfg.pulse_scale * rng.normal(size=cfg.n_vars)
        out[t] = level + season + noise + pulse
    return out


# ---------------------------------------------------------------- 骨干

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


def _fit_backbone(cfg):
    train = make_series(cfg, cfg.seed, cfg.n_train + cfg.lookback)
    X = np.stack([train[t - cfg.lookback : t] for t in range(cfg.lookback, train.shape[0])])
    X = X.reshape(X.shape[0], -1)
    return fit_dlinear(X, train[cfg.lookback:], cfg.ridge)


def _base_preds(backbone, x, lookback):
    return np.stack([predict_dlinear(backbone, x[t - lookback : t])
                     for t in range(lookback, x.shape[0])])


# ---------------------------------------------------------------- 学习状态单元

class StateCell:
    """s_t = a*s_{t-1} + (1-a)*tanh(W_r r_t + W_s s_{t-1} + b); y = P s + q。"""

    def __init__(self, n_vars: int, m: int, rng):
        d = n_vars * m
        self.n_vars, self.d = n_vars, d
        self.params = {
            "a": 2.0 * np.ones(d),                      # sigmoid(a)=alpha≈0.88
            "W_r": 0.1 * rng.standard_normal((d, n_vars)),
            "W_s": 0.1 * rng.standard_normal((d, d)),
            "b": np.zeros(d),
            "P": 0.1 * rng.standard_normal((n_vars, d)),
            "q": np.zeros(n_vars),
        }

    def step(self, s, r):
        p = self.params
        alpha = 1.0 / (1.0 + np.exp(-p["a"]))
        z = p["W_r"] @ r + p["W_s"] @ s + p["b"]
        return alpha * s + (1.0 - alpha) * np.tanh(z)

    def predict(self, s):
        return self.params["P"] @ s + self.params["q"]

    def loss_and_grad(self, rs, ys, spec_penalty=0.0):
        """严格 prequential 展开：预测用 s_{t-1}，再更新。返回 (loss, grads)。"""
        p = self.params
        alpha = 1.0 / (1.0 + np.exp(-p["a"]))
        T = rs.shape[0]
        s_prevs = np.empty((T, self.d))
        us = np.empty((T, self.d))
        ss = np.zeros((T + 1, self.d))
        loss = 0.0
        es = np.empty((T, self.n_vars))
        for t in range(T):
            sp = ss[t]
            s_prevs[t] = sp
            pred = p["P"] @ sp + p["q"]
            e = pred - ys[t]
            es[t] = e
            loss += 0.5 * float(e @ e)
            z = p["W_r"] @ rs[t] + p["W_s"] @ sp + p["b"]
            u = np.tanh(z)
            us[t] = u
            ss[t + 1] = alpha * sp + (1.0 - alpha) * u
        loss += spec_penalty * float((p["W_s"] ** 2).sum())

        grads = {k: np.zeros_like(v) for k, v in p.items()}
        grads["W_s"] += 2.0 * spec_penalty * p["W_s"]
        D = np.zeros(self.d)
        for t in range(T - 1, -1, -1):
            sp, u = s_prevs[t], us[t]
            beta = (1.0 - alpha) * (1.0 - u ** 2)
            gz = beta * D
            grads["W_r"] += np.outer(gz, rs[t])
            grads["W_s"] += np.outer(gz, sp)
            grads["b"] += gz
            grads["a"] += D * (sp - u) * alpha * (1.0 - alpha)
            grads["P"] += np.outer(es[t], sp)
            grads["q"] += es[t]
            # s_t 的总梯度来自：预测 t+1（P^T e_{t+1} 已在上一轮并入 D）、
            # 经更新式传回 s_{t-1} 的两条路径，以及当前预测 e_t 对 s_{t-1} 的作用
            D = alpha * D + p["W_s"].T @ gz + p["P"].T @ es[t]
        return loss, grads


class E2Cell:
    """对照 E2：逐维元学习衰减率，无非线性、无跨变量项。y = P s + q。"""

    def __init__(self, n_vars: int, rng):
        self.n_vars = n_vars
        self.params = {
            "a": 2.0 * np.ones(n_vars),
            "P": 0.1 * rng.standard_normal((n_vars, n_vars)),
            "q": np.zeros(n_vars),
        }

    def step(self, s, r):
        alpha = 1.0 / (1.0 + np.exp(-self.params["a"]))
        return alpha * s + (1.0 - alpha) * r

    def predict(self, s):
        return self.params["P"] @ s + self.params["q"]

    def loss_and_grad(self, rs, ys, spec_penalty=0.0):
        p = self.params
        alpha = 1.0 / (1.0 + np.exp(-p["a"]))
        T = rs.shape[0]
        ss = np.zeros((T + 1, self.n_vars))
        es = np.empty((T, self.n_vars))
        loss = 0.0
        for t in range(T):
            e = p["P"] @ ss[t] + p["q"] - ys[t]
            es[t] = e
            loss += 0.5 * float(e @ e)
            ss[t + 1] = alpha * ss[t] + (1.0 - alpha) * rs[t]
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        D = np.zeros(self.n_vars)
        for t in range(T - 1, -1, -1):
            grads["a"] += D * (ss[t] - rs[t]) * alpha * (1.0 - alpha)
            grads["P"] += np.outer(es[t], ss[t])
            grads["q"] += es[t]
            D = alpha * D + p["P"].T @ es[t]
        return loss, grads


# ---------------------------------------------------------------- 训练器（手工 Adam）

def _adam_train(cell, rs, ys, cfg: ResidualConfig, rng):
    params = {k: v.copy() for k, v in cell.params.items()}
    ms = {k: np.zeros_like(v) for k, v in params.items()}
    vs = {k: np.zeros_like(v) for k, v in params.items()}
    best_loss, best = np.inf, None
    b1, b2, eps = 0.9, 0.999, 1e-8
    for epoch in range(1, cfg.train_epochs + 1):
        cell.params = params
        loss, grads = cell.loss_and_grad(rs, ys, cfg.spec_penalty)
        if loss < best_loss:
            best_loss = loss
            best = {k: v.copy() for k, v in params.items()}
        for k in params:
            ms[k] = b1 * ms[k] + (1 - b1) * grads[k]
            vs[k] = b2 * vs[k] + (1 - b2) * grads[k] ** 2
            mh = ms[k] / (1 - b1 ** epoch)
            vh = vs[k] / (1 - b2 ** epoch)
            params[k] -= cfg.adam_lr * mh / (np.sqrt(vh) + eps)
    cell.params = best if best is not None else params
    return best_loss


def train_cell(n_vars, m, rs, ys, cfg: ResidualConfig, seed):
    best_loss, best_cell = np.inf, None
    for i in range(cfg.restarts):
        rng = np.random.default_rng(seed + 1000 * i)
        cell = StateCell(n_vars, m, rng)
        loss = _adam_train(cell, rs, ys, cfg, rng)
        if loss < best_loss:
            best_loss, best_cell = loss, cell
    return best_cell


def train_e2(n_vars, rs, ys, cfg: ResidualConfig, seed):
    best_loss, best_cell = np.inf, None
    for i in range(max(cfg.restarts - 1, 1)):
        rng = np.random.default_rng(seed + 2000 * i)
        cell = E2Cell(n_vars, rng)
        loss = _adam_train(cell, rs, ys, cfg, rng)
        if loss < best_loss:
            best_loss, best_cell = loss, cell
    return best_cell


# ---------------------------------------------------------------- E1 基线

def _fit_ridge(A, Y, ridge):
    A1 = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    params = np.linalg.solve(A1.T @ A1 + ridge * np.eye(A1.shape[1]), A1.T @ Y)
    return params[:-1], params[-1]


def _ema_states(rs, rho):
    T, n = rs.shape
    ss = np.zeros((T + 1, n))
    for t in range(T):
        ss[t + 1] = rho * ss[t] + (1 - rho) * rs[t]
    return ss[:T]  # 预测用 s_{t-1}


def _eval_ema(backbone, test, rho, W, b, lookback):
    base = _base_preds(backbone, test, lookback)
    rs_full = test[lookback:] - base
    ss = _ema_states(rs_full, rho)
    preds = base + ss @ W.T + b
    return preds - test[lookback:]


# ---------------------------------------------------------------- 指标与总控

def _metrics(errors):
    ae = np.abs(errors)
    se = errors**2
    q99 = np.quantile(se, 0.99)
    return float(np.mean(np.minimum(se, q99))), float(np.mean(ae))


def _prequential_learned(backbone, test, cell, lookback):
    base = _base_preds(backbone, test, lookback)
    rs_full = test[lookback:] - base
    s = np.zeros(cell.d if isinstance(cell, StateCell) else cell.n_vars)
    preds = np.empty_like(rs_full)
    for t in range(len(rs_full)):
        preds[t] = cell.predict(s)
        s = cell.step(s, rs_full[t])
    return _metrics(preds + base - test[lookback:]) + (len(rs_full),)


def run_c1(cfg: ResidualConfig, backbone: Optional[dict]) -> C1Result:
    if backbone is None:
        backbone = _fit_backbone(cfg)
    else:
        backbone = {k: v.copy() for k, v in backbone.items()}
    calib = make_series(cfg, cfg.seed + 10_000, cfg.n_calibrate + cfg.lookback)
    test = make_series(cfg, cfg.seed + 20_000, cfg.n_test + cfg.lookback)
    base_c = _base_preds(backbone, calib, cfg.lookback)
    rs_c = calib[cfg.lookback:] - base_c
    ys_c = rs_c  # 训练目标：残差

    # 学习臂
    cell = train_cell(cfg.n_vars, cfg.m, rs_c, ys_c, cfg, cfg.seed + 31_000)
    learned_wmse, learned_mae, n = _prequential_learned(backbone, test, cell, cfg.lookback)

    # E2 臂
    e2 = train_e2(cfg.n_vars, rs_c, ys_c, cfg, cfg.seed + 41_000)
    e2_wmse, e2_mae, _ = _prequential_learned(backbone, test, e2, cfg.lookback)

    # E1 臂：网格选优
    base_t = _base_preds(backbone, test, cfg.lookback)
    errs_t = test[cfg.lookback:] - base_t
    base_wmse, base_mae = _metrics(errs_t)
    best = (np.inf, None)
    for rho in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99):
        ss_c = _ema_states(rs_c, rho)
        W, b = _fit_ridge(ss_c, ys_c, cfg.controller_ridge)
        # 校准损失用逐步预测
        loss = float(np.mean((ss_c @ W.T + b - ys_c) ** 2))
        if loss < best[0]:
            best = (loss, (rho, W, b))
    rho, W, b = best[1]
    e1_errs = _eval_ema(backbone, test, rho, W, b, cfg.lookback)
    e1_wmse, e1_mae = _metrics(e1_errs)

    return C1Result(base_wmse, learned_wmse, e1_wmse, e2_wmse,
                    base_mae, learned_mae, e1_mae, e2_mae, n)

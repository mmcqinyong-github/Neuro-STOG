"""W1 kill-test：共享胶质增益（P1）vs 逐突触自由增益 vs 分组归一化。

预注册来源：`一区论文创作idea-v1.md` §5 的 W1 行，与 §2 P1 的否决条件 N1/N3。

设计史（两次失败，都已定位，记录在此以免重犯）
============================================
v1 失败：骨干用 9*24=216 维联合 ridge + 300 训练样本，train wmse=0.024 / test=3.74
（60 倍过拟合）。残差被骨干方差淹没，四臂差异不可测。→ 改逐通道 ridge（test 0.80）。
v2 失败：给共享坐标做「域内跨输出标准化」，而域均值坐标在域内各输出上完全相等，
跨输出标准化把它整块归零（实测非零元素比例 0.0），S/L/D 三臂退化为只预测截距，
数值完全相同、看似"持平"。→ **共享场与跨输出归一化天然互斥**，L 臂必须换形式。
v3 失败：把共同成分做成 ω 幅度可调的加性项后，发现**每个通道在自己的历史里都能直接
观测到它**，逐突触臂无需跨通道信息即可估计。ω 再大也不产生共享优势（ω=2 时
F 0.4665 vs S 0.4716，F 反而更好）。→ 共享的价值不来自"共性存在"，
而来自"估计共性的样本不够"。

v4（本版）的机制陈述
==================
共享是**降自由度**，不是"获知更多信息"。因此它的收益只在校准样本相对参数稀缺时出现：
- F  own 坐标(15) × 逐输出权重           params = nv*(15+1)          = 144
- FP 池化坐标(d_c) × 逐输出权重           params = nv*(d_c+1)         =  81
- S  池化坐标(d_c) × **域共享**权重 + 带符号增益 g_i
                                          params = n_dom*d_c+2*nv     =  42
     约束 Σ_{i∈e} g_i = |e|（域内通量守恒，同时固定 g 与 w_e 的尺度）
- L  同 S 但 g_i ≡ 1（S 的嵌套子模型）      params = n_dom*d_c+nv       =  33
- D  S + 通量守恒扩散 ġ = -κL_e g + u      params = 42+4              =  46

三个对照各自封死一条替代解释：
  L  ← "赢只是用了池化坐标 + 更强正则"（若 S≈L，则胶质增益无独立贡献 → N3 触发）
  FP ← "赢只是池化输入降噪，与参数共享无关"（若 S≈FP，则共享权重无贡献）
  F  ← "赢只是正则化"（F 参数最多，长校准下应最强）

预注册预测（可被推翻）
====================
(P-a) S 的收益随 n_calib 增大而单调收缩：n_calib=80 时 S 最佳，n_calib=400 时 F/FP 反超。
      存在交叉点 n*。若不存在交叉（S 全程不占优，或全程占优），(P-a) 被推翻。
(P-b) 增益不得退化：||g − 1||_∞ > 0.1。若所有 g_i ≈ 1，S 与 L 数值等价，生物结构为空。
(P-c) 异质性 hetero 上升时 S 相对 F 的优势收缩（方差收益被偏差吃掉）。

裁决（N3）
=========
若 S 在整张 (ω, n_calib, λ) 网格上都不显著优于 L 与 FP → 分组归一化/池化输入已足够，
生物共享结构无独立贡献，**P1 归零，转 P3**。

时序纪律（不可违反）
==================
预测只用 [t-L, t) 窗口；残差在观测到 x_t 之后才允许写入；测试阶段无梯度；骨干冻结。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from cyclic_mode import EnvConfig, _metrics, make_series

N_FEAT = 15
RIDGE_BACKBONE = 10.0

# 池化坐标只取"慢"特征：24 步均值 / 末值 / 7 步差分均值 / 24 步标准差。
# 这是胶质慢通路的最小可用坐标集——共享场必须是低维的，否则参数节约的来源就没了。
POOL_IDX = (3, 7, 9, 6)
D_COORD = 2 * len(POOL_IDX)      # 域池化 4 + 全局池化 4 = 8


# --------------------------------------------------------------------------
# 骨干：逐通道 ridge
# --------------------------------------------------------------------------
def fit_backbone_ci(x: np.ndarray, lookback: int, lam: float = RIDGE_BACKBONE) -> dict:
    nv = x.shape[1]
    W = np.empty((nv, lookback))
    b = np.empty(nv)
    for j in range(nv):
        X = np.stack([x[t - lookback : t, j] for t in range(lookback, x.shape[0])])
        y = x[lookback:, j]
        A = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        p = np.linalg.solve(A.T @ A + lam * np.eye(lookback + 1), A.T @ y)
        W[j], b[j] = p[:-1], p[-1]
    return {"W": W, "b": b, "lookback": lookback}


def predict_backbone_ci(bb: dict, window: np.ndarray) -> np.ndarray:
    w = np.asarray(window, dtype=float).reshape(bb["lookback"], -1)
    return np.sum(w.T * bb["W"], axis=1) + bb["b"]


# --------------------------------------------------------------------------
# 环境
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class W1Env:
    seed: int
    env: str = "env1"
    n_vars: int = 9
    n_train: int = 300
    n_calibrate: int = 400
    n_test: int = 500
    lookback: int = 24
    noise_scale: float = 0.25
    shift_weight: float = 0.0     # ω
    shift_rho: float = 0.97
    hetero: float = 0.0
    m_e: int = 3
    diffusion_kappa: float = 0.6
    diffusion_step: float = 0.35
    gain_lr: float = 0.02
    gain_clip: float = 4.0
    gain_floor: float = -4.0

    def to_env_config(self, seed: int) -> EnvConfig:
        base = "env3" if self.env.startswith("env3") else "env1"
        return EnvConfig(
            seed=seed, n_vars=self.n_vars, n_train=self.n_train,
            n_calibrate=self.n_calibrate, n_test=self.n_test,
            lookback=self.lookback, noise_scale=self.noise_scale,
            ridge=RIDGE_BACKBONE, env=base,
        )


def domain_index(n_vars: int, m_e: int) -> list[tuple[int, ...]]:
    assert n_vars % m_e == 0, "n_vars 必须能被 m_e 整除"
    return [tuple(range(i, i + m_e)) for i in range(0, n_vars, m_e)]


def shared_drive(n_obs: int, seed: int, rho: float) -> np.ndarray:
    rng = np.random.default_rng(seed + 777_777)
    z = np.empty(n_obs)
    v = 0.0
    for t in range(n_obs):
        v = rho * v + np.sqrt(1.0 - rho * rho) * rng.normal()
        z[t] = v
    return z


def channel_signs(n_vars: int, seed: int, hetero: float, domains) -> np.ndarray:
    rng = np.random.default_rng(seed + 888_888)
    sign = np.ones(n_vars)
    for idx in domains:
        sel = list(idx)
        k = int(round(hetero * len(sel)))
        if k > 0:
            k = min(k, len(sel) - 1)
            sign[rng.choice(sel, size=k, replace=False)] = -1.0
    return sign


def make_series_w1(env: W1Env, seed: int, n_obs: int, inject_shift: bool) -> np.ndarray:
    """训练段不含共享成分（骨干无法预先吸收），仅校准/测试段注入。"""
    x = make_series(env.to_env_config(seed), seed, n_obs)
    if not inject_shift or env.shift_weight == 0.0:
        return x
    doms = domain_index(env.n_vars, env.m_e)
    z = shared_drive(n_obs, seed, env.shift_rho)
    sign = channel_signs(env.n_vars, seed, env.hetero, doms)
    return x + env.shift_weight * np.outer(z, sign)


# --------------------------------------------------------------------------
# 特征
# --------------------------------------------------------------------------
def per_var_features(w: np.ndarray) -> np.ndarray:
    L, nv = w.shape
    f = np.empty((nv, N_FEAT))
    f[:, 0] = w[-3:].mean(axis=0)
    f[:, 1] = w[-6:].mean(axis=0)
    f[:, 2] = w[-12:].mean(axis=0)
    f[:, 3] = w[-24:].mean(axis=0)
    f[:, 4] = w[-6:].std(axis=0)
    f[:, 5] = w[-12:].std(axis=0)
    f[:, 6] = w[-24:].std(axis=0)
    f[:, 7] = w[-1]
    f[:, 8] = w[-1] - w[-2]
    f[:, 9] = np.diff(w[-7:], axis=0).mean(axis=0)
    f[:, 10] = np.diff(w[-13:], axis=0).mean(axis=0)
    f[:, 11] = w[-12:].max(axis=0)
    f[:, 12] = w[-12:].min(axis=0)
    f[:, 13] = w[-24:].max(axis=0) - w[-24:].min(axis=0)
    f[:, 14] = (w[-1] - w[0]) / float(L)
    return f


def bank_at(w: np.ndarray, domains) -> tuple[np.ndarray, np.ndarray]:
    """(own (nv,15), slots (nv,D_COORD))。

    slots 仅含 POOL_IDX 选中的慢特征，且先域池化再全局池化——低维是共享场降自由度的前提。
    """
    pv = per_var_features(w)
    nv = pv.shape[0]
    sub = pv[:, list(POOL_IDX)]                       # (nv, 4)
    dom = np.empty((nv, len(POOL_IDX)))
    for idx in domains:
        pooled = sub[list(idx)].mean(axis=0)
        for j in idx:
            dom[j] = pooled
    glob = np.broadcast_to(sub.mean(axis=0), (nv, len(POOL_IDX)))
    return pv, np.concatenate([dom, glob], axis=1)


def bank_trajectory(x, lookback, domains):
    n = x.shape[0]
    own = np.empty((n - lookback, x.shape[1], N_FEAT))
    slt = np.empty((n - lookback, x.shape[1], D_COORD))
    for i, t in enumerate(range(lookback, n)):
        o, s = bank_at(x[t - lookback : t], domains)
        own[i], slt[i] = o, s
    return own, slt


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def _ridge(A, Y, lam):
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)


def _scaler(fit_block, blocks):
    flat = fit_block.reshape(-1, fit_block.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return [(b - mean) / std for b in blocks]


def _project_flux(g, domains):
    out = g.copy()
    for idx in domains:
        sel = list(idx)
        out[sel] += 1.0 - out[sel].mean()
    return out


def _laplacian(m):
    A = np.zeros((m, m))
    for i in range(m - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return np.diag(A.sum(axis=1)) - A


# --------------------------------------------------------------------------
# 臂
# --------------------------------------------------------------------------
@dataclass
class Arm:
    name: str
    predict: Callable
    n_params: int
    gains: np.ndarray | None = None


def fit_free(own, slots, Y, lam, domains):
    n, nv, d = own.shape
    A = np.concatenate([own, np.ones((n, nv, 1))], axis=2)
    W = np.stack([_ridge(A[:, j, :], Y[:, j : j + 1], lam)[:, 0] for j in range(nv)])

    def predict(o, s, y, _A=A, _W=W):
        A2 = np.concatenate([o, np.ones((o.shape[0], o.shape[1], 1))], axis=2)
        return np.einsum("nvd,vd->nv", A2, _W)

    return Arm("F", predict, nv * (d + 1))


def _fit_field(B, Y, domains, g, b, lam):
    nv = B.shape[1]
    W = np.zeros((nv, B.shape[2]))
    for idx in domains:
        rows = np.concatenate([g[j] * B[:, j, :] for j in idx], axis=0)
        col = np.concatenate([(Y[:, j] - b[j]).reshape(-1, 1) for j in idx], axis=0)
        w_e = _ridge(rows, col, lam)[:, 0]
        for j in idx:
            W[j] = w_e
    return W


def fit_per_syn_pooled(own, slots, Y, lam, domains):
    """FP：池化坐标 + 逐输出独立权重。封死"只是池化降噪"这条解释。"""
    n, nv, d = slots.shape
    A = np.concatenate([slots, np.ones((n, nv, 1))], axis=2)
    W = np.stack([_ridge(A[:, j, :], Y[:, j : j + 1], lam)[:, 0] for j in range(nv)])

    def predict(o, s, y, _W=W):
        A2 = np.concatenate([s, np.ones((s.shape[0], s.shape[1], 1))], axis=2)
        return np.einsum("nvd,vd->nv", A2, _W)

    return Arm("FP", predict, nv * (d + 1))


def fit_shared(own, slots, Y, lam, domains, n_iter=12):
    """S：池化坐标 + 域共享场 w_e + 带符号增益 g_i（Σ_{i∈e} g_i = |e|）。"""
    n, nv, d = slots.shape
    g = np.ones(nv)
    b = np.zeros(nv)
    W = np.zeros((nv, d))
    for _ in range(n_iter):
        W = _fit_field(slots, Y, domains, g, b, lam)
        q = np.einsum("nvd,vd->nv", slots, W)
        for j in range(nv):
            g[j] = float(q[:, j] @ Y[:, j]) / (float(q[:, j] @ q[:, j]) + 1e-10)
        g = _project_flux(g, domains)
        for j in range(nv):
            b[j] = float(np.mean(Y[:, j] - g[j] * q[:, j]))

    def predict(o, s, y, _W=W, _g=g, _b=b):
        return _b[None, :] + _g[None, :] * np.einsum("nvd,vd->nv", s, _W)

    return Arm("S", predict, len(domains) * d + 2 * nv, gains=g.copy())


def fit_grouped_ln(own, slots, Y, lam, domains):
    """L：S 的嵌套子模型，g_i ≡ 1。N3 的直接裁决对象。"""
    n, nv, d = slots.shape
    W = _fit_field(slots, Y, domains, np.ones(nv), np.zeros(nv), lam)
    b = np.array([float(np.mean(Y[:, j] - slots[:, j, :] @ W[j])) for j in range(nv)])

    def predict(o, s, y, _W=W, _b=b):
        return _b[None, :] + np.einsum("nvd,vd->nv", s, _W)

    return Arm("L", predict, len(domains) * d + nv, gains=np.ones(nv))


def fit_shared_diffusion(env, own, slots, Y, lam, domains):
    """D：S + 通量守恒扩散 ġ = -κL_e g + u，u = η·(x_t − 预测_t)。严格因果。"""
    n, nv, d = slots.shape
    Laps = [_laplacian(len(idx)) for idx in domains]

    def _step(g, u):
        gn = g.copy()
        for di, idx in enumerate(domains):
            sel = list(idx)
            gn[sel] = g[sel] + env.diffusion_step * (
                -env.diffusion_kappa * (Laps[di] @ g[sel]) + u[sel]
            )
        gn = _project_flux(gn, domains)
        return np.clip(gn, env.gain_floor, env.gain_clip)

    W = _fit_field(slots, Y, domains, np.ones(nv), np.zeros(nv), lam)
    g_tr = np.empty((n, nv))
    g = np.ones(nv)
    corr = np.empty((n, nv))
    for t in range(n):
        g_tr[t] = g
        corr[t] = g * np.einsum("vd,vd->v", slots[t], W)
        g = _step(g, env.gain_lr * (Y[t] - corr[t]))
    b = np.array([float(np.mean(Y[:, j] - corr[:, j])) for j in range(nv)])
    W = _fit_field(slots, Y, domains, g_tr.mean(axis=0), b, lam)

    def predict(o, s, y, _W=W, _b=b):
        m = s.shape[0]
        out = np.empty((m, nv))
        gg = np.ones(nv)
        for t in range(m):
            out[t] = gg * np.einsum("vd,vd->v", s[t], _W)
            gg = _step(gg, env.gain_lr * (y[t] - out[t]))
        return _b[None, :] + out

    return Arm("D", predict, len(domains) * d + 2 * nv + 4,
               gains=g_tr[-1].copy() if n else np.ones(nv))


# --------------------------------------------------------------------------
# 单点评估
# --------------------------------------------------------------------------
def prepare(env: W1Env, n_calib: int):
    train = make_series_w1(env, env.seed, env.n_train + env.lookback, inject_shift=False)
    backbone = fit_backbone_ci(train, env.lookback)
    calib = make_series_w1(env, env.seed + 10_000, n_calib + env.lookback, inject_shift=True)
    test = make_series_w1(env, env.seed + 20_000, env.n_test + env.lookback, inject_shift=True)
    return backbone, calib, test


def _rows(x, backbone, env, domains, upto, offset=0):
    """offset 允许在长序列上滑动取段（真实数据用）。"""
    idx = list(range(env.lookback + offset, env.lookback + offset + upto))
    base = np.stack([predict_backbone_ci(backbone, x[t - env.lookback : t]) for t in idx])
    y = x[idx]
    own, slt = bank_trajectory(x[: env.lookback + offset + upto], env.lookback, domains)
    if offset > 0:
        own, slt = own[offset:], slt[offset:]
    return base, y, y - base, own, slt


ARMS = ("F", "FP", "S", "L", "D")


def run_point_real(env: W1Env, train: np.ndarray, calib: np.ndarray, test: np.ndarray,
                   lam: float, n_calib: int, arms=ARMS) -> dict:
    """真实序列版本：与 run_point 完全相同的五臂与指标，只是数据来自外部。

    train / calib / test 均为 (n_obs + lookback, n_vars)，已含 lookback 前缀。
    因果关系不变：骨干只在 train 拟合；calib 只用于拟合机制；test 只评估。
    """
    domains = domain_index(env.n_vars, env.m_e)
    n_calib = int(min(n_calib, calib.shape[0] - env.lookback - 1))
    backbone = fit_backbone_ci(train, env.lookback)

    b_ca, y_ca, r_ca, own_ca, slt_ca = _rows(calib, backbone, env, domains, n_calib)
    b_te, y_te, _, own_te, slt_te = _rows(test, backbone, env, domains, env.n_test)

    own_ca, own_te = _scaler(own_ca, [own_ca, own_te])
    slt_ca, slt_te = _scaler(slt_ca, [slt_ca, slt_te])

    base_wmse, base_mae = _metrics(y_te - b_te)
    res = {"base": (base_wmse, base_mae, 1), "n_test": y_te.shape[0]}

    fitters = {
        "F": lambda: fit_free(own_ca, slt_ca, r_ca, lam, domains),
        "FP": lambda: fit_per_syn_pooled(own_ca, slt_ca, r_ca, lam, domains),
        "S": lambda: fit_shared(own_ca, slt_ca, r_ca, lam, domains),
        "L": lambda: fit_grouped_ln(own_ca, slt_ca, r_ca, lam, domains),
        "D": lambda: fit_shared_diffusion(env, own_ca, slt_ca, r_ca, lam, domains),
    }
    for k in arms:
        arm = fitters[k]()
        err = y_te - (b_te + arm.predict(own_te, slt_te, y_te))
        w, mae = _metrics(err)
        res[k] = (w, mae, arm.n_params)
        if arm.gains is not None:
            res[f"g_{k}"] = arm.gains
    return res


def delta_res_real(env: W1Env, train: np.ndarray, calib: np.ndarray, test: np.ndarray,
                   n_calib: int, lags: int = 6, pooled: bool = True,
                   superset: bool = True) -> float:
    """真实序列上的 Δ_res：全信息 oracle 相对无修正基线的最大可达优势（%）。

    ⚠️ 设计约束（P3 规格 v1.2，由实测错误反推得出）
    ============================================
    **oracle 必须是所有被试机制载体的超集**，否则它就不是上界。
    实测错误：v1.1 的 oracle 只有 3×6(滞后) + 3×5(聚合) = 33 维，
    而 F 臂的载体 own 有 3×15 = 45 维，F 合法地超过了 oracle——
    这不是"oracle 不完备"的细节问题，是**上界定义错误**。

    v1.2 的 oracle 特征集（一次写定）：
      1. 全通道 × `lags` 阶滞后                （3×6 = 18）
      2. 全通道 × 15 维多尺度特征（即 per_var_features 的全量）（3×15 = 45）
      3. 全通道 × 域池化坐标 8 维              （3×8 = 24）
    合计 87 维 > 任何单臂的 45 维 ⇒ 超集成立。

    高维 + 400 样本 ⇒ 必须靠 ridge 收缩。lam 网格含大值（到 1e5）以保证不因过拟合而
    低估上界——否则 oracle 反而比机制差，同样破坏上界性质。
    """
    backbone = fit_backbone_ci(train, env.lookback)
    n_calib = int(min(n_calib, calib.shape[0] - env.lookback - 1))
    domains = domain_index(env.n_vars, env.m_e)

    def resid(x, upto):
        idx = list(range(env.lookback, env.lookback + upto))
        base = np.stack([predict_backbone_ci(backbone, x[t - env.lookback : t]) for t in idx])
        return x[idx] - base

    def feats(x, upto):
        rows = []
        for t in range(env.lookback, env.lookback + upto):
            parts = [np.concatenate([x[t - l] for l in range(1, lags + 1)])]
            if pooled:
                w = x[t - env.lookback : t]
                parts.append(per_var_features(w).ravel())          # 3×15 = 45
                _, slt = bank_at(w, domains)                       # (nv, D_COORD)
                if superset:
                    parts.append(slt.ravel())                      # 3×8 = 24
            rows.append(np.concatenate(parts))
        return np.stack(rows)

    r_te, r_ca = resid(test, env.n_test), resid(calib, n_calib)
    Xte, Xca = feats(test, env.n_test), feats(calib, n_calib)
    none, _ = _metrics(r_te)
    best = np.inf
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5):
        A = np.concatenate([Xca, np.ones((len(Xca), 1))], axis=1)
        P = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ r_ca)
        pred = np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1) @ P
        w, _ = _metrics(r_te - pred)
        best = min(best, w)
    return float((none - best) / none * 100.0)

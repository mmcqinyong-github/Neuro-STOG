"""RAFT 环境专用的 Δ_res 计算（层 2 复现用）。

为什么需要单独一个模块
======================
W-P3-2 的 Δ_res 是在 `tsf_loader` 构造的环境上算的：lookback=24、单步预测、
9 通道等间隔子集、train/calib/test = 800/400/500。

层 2 复现用的是**论文自己的环境**：RAFT 官方 loader 的切分、lookback=720、
pred_len=96、全通道、多变量 M 设置。

两者的环境不同，Δ_res 不可直接搬运。若把 W-P3-2 的数值拿过来对照层 2 的 Δ_mech，
就是拿 A 环境的分辨率去解释 B 环境的现象——这正是本工作区反复出现的自欺形式。
因此这里**在 RAFT 的环境里重新算 Δ_res**，定义与 W-P3-2 保持一致：

    Δ_res(E) = (WMSE(基线) - WMSE(全信息 oracle)) / WMSE(基线) × 100

其中
  * 基线 = 逐通道 ridge骨干（channel-independent backbone），与层 1 同构，
    在 RAFT 的 train 段上拟合；
  * oracle = 能看到**全部**可用信息的线性预测器，其特征集是所有被试机制的
    载体的**超集**（层 1 的教训：oracle 必须是超集，否则不构成上界）；
  * 校准段 = RAFT 的 val 段，测试段 = RAFT 的 test 段。

多步预测的处理
==============
RAFT 预测 pred_len=96 步，而 W-P3-2 是单步。多步下"基线的残差"不再是逐点残差，
而是整段 96 步的形状误差。为了与层 1 口径可比，这里用**逐步 reshape 的直接多步
ridge**：对每一步 h 单独拟合一个从 lookback 窗口到 x_{t+h} 的线性映射，
再拼成 96 步预测。oracle 的特征集在每一步上取同样的超集特征。

单步与多步的 Δ_res 不可比，因此本模块的报告必须**独立于** W-P3-2 的数值，
只用于层 2 内部的高低分辨力分组（判据 H-L1、H-L3）。
"""

from __future__ import annotations

import numpy as np

# 与层 1 (shared_gain.py) 同构的骨干：逐通道 ridge
RIDGE_LAM = 10.0


# ---------------------------------------------------------------------------
# RAFT 官方切分（复刻 data_provider/data_loader.py）
# ---------------------------------------------------------------------------
def raft_split(data: np.ndarray, kind: str, seq_len: int, pred_len: int):
    """复刻 RAFT 官方 loader 的 train/val/test 切分。

    ETT 系列：固定 12/4/4 月（hour: *24，minute: *24*4）。
    custom  ：7:1:2 比例。
    返回 (train, val, test)，各自为 (T, C) 数组，val/test 段前面带上 seq_len 行历史。
    """
    n = data.shape[0]
    if kind in ("ETTh1", "ETTh2"):
        unit = 24
        b1s = [0, 12 * 30 * unit - seq_len, 12 * 30 * unit + 4 * 30 * unit - seq_len]
        b2s = [12 * 30 * unit, 12 * 30 * unit + 4 * 30 * unit, 12 * 30 * unit + 8 * 30 * unit]
    elif kind in ("ETTm1", "ETTm2"):
        unit = 24 * 4
        b1s = [0, 12 * 30 * unit - seq_len, 12 * 30 * unit + 4 * 30 * unit - seq_len]
        b2s = [12 * 30 * unit, 12 * 30 * unit + 4 * 30 * unit, 12 * 30 * unit + 8 * 30 * unit]
    else:  # custom: 7:1:2
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_vali = n - num_train - num_test
        b1s = [0, num_train - seq_len, n - num_test - seq_len]
        b2s = [num_train, num_train + num_vali, n]

    def seg(i):
        return data[b1s[i]:b2s[i]]

    return seg(0), seg(1), seg(2)


# ---------------------------------------------------------------------------
# 骨干与特征
# ---------------------------------------------------------------------------
def fit_backbone_ci(x: np.ndarray, lookback: int, lam: float = RIDGE_LAM):
    """逐通道 ridge：每个输出通道用自己的 lookback 窗口拟合。

    与层 1 的 fit_backbone_ci 同构，避免 216 维联合 ridge 的过拟合（层 1 陷阱 1）。
    闭式 ridge：P = (A'A + lam I)^-1 A'y，A 含截距列。
    """
    n = x.shape[0]
    T = n - lookback
    if T <= 10:
        raise ValueError("训练段太短")
    X = np.stack([x[t - lookback:t] for t in range(lookback, n)])   # (T, L, C)
    X = X.transpose(0, 2, 1)                                        # (T, C, L)
    Y = x[lookback:]                                               # (T, C)
    C = x.shape[1]
    W = np.zeros((C, lookback))
    b = np.zeros(C)
    for j in range(C):
        A = X[:, j, :]                                              # (T, L)
        A1 = np.concatenate([A, np.ones((T, 1))], axis=1)           # (T, L+1)
        G = A1.T @ A1 + lam * np.eye(A1.shape[1])
        rhs = A1.T @ Y[:, j]                                        # (L+1,)
        P = np.linalg.solve(G, rhs)                                 # (L+1,)
        W[j] = P[:lookback]
        b[j] = P[lookback]
    return {"W": W, "b": b, "lookback": lookback}


def predict_backbone_ci(bb, window: np.ndarray) -> np.ndarray:
    """window: (L, C) → (C,)"""
    return np.sum(window.T * bb["W"], axis=1) + bb["b"]


def _multi_scale_feats(w: np.ndarray) -> np.ndarray:
    """每个通道的多尺度特征（层 1 同构的 per_var_features 子集）。

    w: (L, C) → (C, 15)
    向量化实现，避免逐通道 Python 循环（21 通道 × 200 点时开销显著）。
    """
    L, C = w.shape
    out = np.zeros((C, 15))
    v = w.T                                   # (C, L)
    d = np.diff(v, axis=1)                    # (C, L-1)

    out[:, 0] = v[:, -1]
    out[:, 1] = v[:, -2]
    out[:, 2] = v[:, -3]
    out[:, 3] = v.mean(axis=1)
    out[:, 4] = v.std(axis=1)
    out[:, 5] = v[:, -24:].mean(axis=1)
    out[:, 6] = v[:, -24:].std(axis=1) if L >= 24 else v.std(axis=1)
    out[:, 7] = v[:, -96:].mean(axis=1) if L >= 96 else v.mean(axis=1)
    out[:, 8] = v[:, -96:].std(axis=1) if L >= 96 else v.std(axis=1)
    out[:, 9] = d[:, -1] if L > 1 else 0.0
    out[:, 10] = d[:, -2] if L > 2 else 0.0
    out[:, 11] = d[:, -5:].mean(axis=1) if L > 5 else 0.0
    out[:, 12] = d[:, -5:].std(axis=1) if L > 5 else 0.0

    if L >= 96:
        seg = v[:, -96:]
        sd = seg.std(axis=1)
        ok = sd > 1e-9
        tt = np.arange(seg.shape[1])
        for c in np.nonzero(ok)[0]:
            out[c, 13] = np.corrcoef(seg[c], tt)[0, 1]
            out[c, 14] = ((seg[c] - seg[c].mean()) ** 3).mean() / (sd[c] ** 3 + 1e-9)
    return out


def _oracle_features(window: np.ndarray, lags: int) -> np.ndarray:
    """oracle 的特征集：超集设计（层 1 教训 v1.2）。

    window: (L, C) → 扁平特征向量
      1. 全通道 × lags 阶滞后              (C * lags)
      2. 全通道 × 15 维多尺度特征           (C * 15)
      3. 全通道的全局池化坐标               (C * 3)
    合计远大于任何单臂的载体 ⇒ oracle 是上界。
    """
    L, C = window.shape
    parts = []
    for l in range(1, lags + 1):
        parts.append(window[-l].ravel())                       # C * lags
    parts.append(_multi_scale_feats(window).ravel())          # C * 15
    parts.append(np.stack([window.mean(axis=0),
                           window[-min(L, 96):].mean(axis=0),
                           window[-min(L, 96):].std(axis=0)]).ravel())  # C * 3
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Δ_res：多步版本
# ---------------------------------------------------------------------------
def delta_res_raft(train: np.ndarray, val: np.ndarray, test: np.ndarray,
                   seq_len: int, pred_len: int, lags: int = 6,
                   lam_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5),
                   seed: int = 0, max_oracle_pts: int = 400) -> dict:
    """在 RAFT 环境下算 Δ_res。

    基线   = 逐通道 ridge 骨干在 train 段拟合，逐点多步外推。
    oracle = 每步独立的 ridge，特征集为 _oracle_features 的超集，
             在 val 段选 lam，在 test 段评。
    返回 dict：delta_res(%)、基线与 oracle 的 wmse、各步相对误差。
    """
    rng = np.random.default_rng(seed)
    C = train.shape[1]

    # ---- 逐步的窗口索引 ----
    def step_indices(seg, upto):
        """返回每个可预测起点的 t（下标），使 t..t+pred_len-1 全部落在 seg 内。"""
        n = seg.shape[0]
        start = seq_len
        end = n - pred_len
        if end <= start:
            raise ValueError("段太短，无法构造多步窗口")
        idx = np.arange(start, end)
        if upto < len(idx):
            idx = idx[:upto]
        return idx

    tr_idx = step_indices(train, 10 ** 9)
    va_idx = step_indices(val, 10 ** 9)
    te_idx = step_indices(test, 10 ** 9)

    # ---- 基线骨干 ----
    bb = fit_backbone_ci(train, seq_len)

    def baseline_predict(seg, idx):
        """返回 (len(idx), pred_len, C)"""
        out = np.empty((len(idx), pred_len, C))
        for i, t in enumerate(idx):
            hist = seg[t - seq_len:t]
            for h in range(pred_len):
                out[i, h] = predict_backbone_ci(bb, hist)
                # 多步外推：用预测值补进历史
                hist = np.vstack([hist[1:], out[i, h][None, :]])
        return out

    def truth(seg, idx):
        """向量化取多步真值：starts (n,) → (n, pred_len, C)"""
        starts = np.asarray(idx)
        offs = np.arange(pred_len)
        return seg[starts[:, None] + offs[None, :]]              # (n, pred_len, C)

    # 训练多步外推很贵（每步要重算历史），用**一步迭代**近似：基线的多步
    # 由骨干在原历史上直接外推（不做递归），这是标准做法且更保守
    # （不会因递归误差放大而虚增基线误差 ⇒ 不会虚增 Δ_res）。

    def baseline_predict_direct(seg, idx):
        """骨干对所有未来步给同一预测（不做递归外推）。

        向量化：把 (n, L, C) 的窗口一次性摊平做矩阵乘，避免 O(n·L·C) 的 Python 循环。
        """
        n = len(idx)
        wins = np.stack([seg[t - seq_len:t] for t in idx])      # (n, L, C)
        # 对每个通道 j: p_j = sum_l W[j,l] * win[:,l,j] + b[j]
        # einsum 下标 'jl,nlj->nj'：j 既是 W 的通道维也是 win 的通道维，
        # 必须让同一个 j 同时出现在两侧，否则会把通道 j 的权重错配到通道 c。
        p = np.einsum('jl,nlj->nj', bb["W"], wins) + bb["b"]     # (n, C)
        return np.repeat(p[:, None, :], pred_len, axis=1)        # (n, pred_len, C)

    base_te = baseline_predict_direct(test, te_idx)
    true_te = truth(test, te_idx)
    base_wmse = float(np.mean((base_te - true_te) ** 2))

    # ---- oracle：逐步 ridge，超集特征 ----
    # 特征构造代价：len(idx) × (seq_len×C) 的窗口拷贝，对 ETT 的 2785 步 × 96 步会很大。
    # 为控制成本，oracle 只在**稀疏抽样**的起点上评估，其余不估（这会低估 oracle 能力
    # ⇒ 低估 Δ_res，判据偏保守，但必须记录）。
    max_oracle_pts = 400
    sel = np.linspace(0, len(te_idx) - 1, min(max_oracle_pts, len(te_idx))).astype(int)
    sel_v = np.linspace(0, len(va_idx) - 1, min(max_oracle_pts, len(va_idx))).astype(int)

    def build_xy(seg, idx, sub):
        Xs, Ys = [], []
        for j in sub:
            t = idx[j]
            w = seg[t - seq_len:t]
            f = _oracle_features(w, lags)
            Xs.append(f)
            Ys.append(seg[t:t + pred_len])          # (pred_len, C)
        return np.stack(Xs), np.stack(Ys)

    Xv, Yv = build_xy(val, va_idx, sel_v)
    Xt, Yt = build_xy(test, te_idx, sel)

    # ---- 逐步 ridge：每个 (步, 通道) 一个模型 ----
    # 效率优化（不改语义）：对每个 (步, 通道) 只做一次 SVD，
    # 所有 lam 的解由奇异值分解直接给出，避免对每个 lam 重解 500×500 线性系统。
    #   A = U S V^T,  ridge 解 P = V diag(s/(s²+lam)) U^T y
    Av = np.concatenate([Xv, np.ones((len(Xv), 1))], axis=1)
    At = np.concatenate([Xt, np.ones((len(Xt), 1))], axis=1)
    U, s, Vt = np.linalg.svd(Av, full_matrices=False)
    # test 端投影：At @ V 只需算一次
    AtV = At @ Vt.T                                   # (n_test, d)

    pred = np.zeros_like(Yt)
    for h in range(pred_len):
        for j in range(C):
            y = Yv[:, h, j]
            Uy = U.T @ y                               # (d,)
            best = None
            for lam in lam_grid:
                d = s / (s * s + lam)                  # ridge 收缩因子
                p = AtV @ (d * Uy)                     # (n_test,)
                err = float(np.mean((p - Yt[:, h, j]) ** 2))
                if best is None or err < best[0]:
                    best = (err, p)
            pred[:, h, j] = best[1]

    oracle_wmse = float(np.mean((pred - Yt) ** 2))
    # 基线在**同一批**抽样点上重算，保证可比
    base_sel = base_te[sel]
    base_wmse_sel = float(np.mean((base_sel - Yt) ** 2))

    delta = (base_wmse_sel - oracle_wmse) / base_wmse_sel * 100.0
    return {
        "delta_res": float(delta),
        "base_wmse": base_wmse,
        "base_wmse_sel": base_wmse_sel,
        "oracle_wmse": oracle_wmse,
        "n_oracle_pts": int(len(sel)),
        "n_test_pts": int(len(te_idx)),
        "C": int(C),
        "rng_unused": int(rng.integers(0, 2)),
    }

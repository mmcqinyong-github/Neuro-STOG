"""W-P1-R：潜变量可识别性假说（LIH）的模型、闭式解与蒙特卡洛。

预注册：`W-P1-R可识别性相变预注册规格-v1.0.md`
纪律（预注册 §7）：**闭式解先于蒙特卡洛写出并存档**，防止用数值结果反推公式。

模型（预注册 §2.1）
------------------
    γ_i = ḡ + δ_i,      δ_i ~ N(0, τ²)          i = 1..m   真实增益
    c_t ~ N(0, 1)                                           域级共同驱动
    z_{i,t} = c_t + ξ_{i,t},  ξ ~ N(0, σ_ξ²)                单元对 c 的观测
    y_{i,t} = γ_i·c_t + ε_{i,t},  ε ~ N(0, σ²)              输出

估计器只看到 (z, y)，看不到 c。

两个口径
--------
G_total  自然部署：F 用 z_{i,*} 预测，S 用池化 z̄_* 预测。
         （S 在测试时也用池化输入，这是域模型的真实部署方式）
G_param  机制分解：测试时**双方都给定真 c**，只剩参数估计的差异。
         用于分离"参数共享"与"测试输入池化"两个效应。

主判据用 G_total（预注册闭式解即此口径），G_param 作补充诊断。
"""

from __future__ import annotations

import numpy as np


# ===========================================================================
# 一、闭式解（先写，先存档）
# ===========================================================================
def iota(sigma_xi2: float) -> float:
    """单单元可识别度 ι = 1/(1+σ_ξ²)。"""
    return 1.0 / (1.0 + sigma_xi2)


def A_(sigma_xi2: float) -> float:
    """衰减量 A = ι·σ_ξ² = σ_ξ²/(1+σ_ξ²)，即单单元预测的渐近误差因子。"""
    return sigma_xi2 / (1.0 + sigma_xi2)


def A_m(sigma_xi2: float, m: int) -> float:
    """池化后的衰减量 A_m = σ_ξ²/(m+σ_ξ²)。"""
    return sigma_xi2 / (m + sigma_xi2)


def tau_hat2(tau2: float, m: int) -> float:
    """域内**经验**异质性的期望：E[(1/m)Σ(γ_i − ḡ)²] = τ²(1 − 1/m)。"""
    return tau2 * (1.0 - 1.0 / m) if m > 1 else 0.0


def closed_form_total(m: int, sigma_xi2: float, n: int, tau2: float,
                      gbar: float = 1.0, sigma2: float = 1.0):
    """自然部署口径的闭式 MSE（预注册附录 A，含 v1.3 修正）。

    【修订 v1.3 / N-R3 第一次修正】初版漏掉了**不可约的输出噪声** σ²：
    目标 y = γc + ε 含 ε，任何预测器的 MSE 都至少是 σ²。
    首次 mc 校验（540 格）显示 σ_ξ²→0 时相对误差达 1e6%（MC≈1.00008 vs CF≈0.00008），
    差值恰为 σ²。**G = MSE_F − MSE_S 不受影响**（σ² 在相减时抵消），
    这与 selftest 中 G 的误差仅 ~1% 一致。

    返回 (MSE_F, MSE_S, G_total)。G > 0 表示共享优于独立。
    """
    return closed_form_total_that(m, sigma_xi2, n, tau_hat2(tau2, m),
                                  gbar, sigma2)


def closed_form_total_that(m: int, sigma_xi2: float, n: int,
                           tau_hat2_val: float, gbar: float = 1.0,
                           sigma2: float = 1.0):
    """直接以**域内经验异质性 τ̂² 的实现值**代入的闭式解。

    为什么需要这个变体：闭式解中 MSE 只通过 τ̂² 依赖异质性，但 τ̂² 在有限 m 下
    本身是随机变量（τ̂² = τ²·χ²_{m−1}/m）。用 E[τ̂²] = τ²(1−1/m) 代入会引入
    与 τ̂² 抽样波动同量级的误差（m=6、τ²=0.5 时 SD(τ̂²)≈0.26 而均值≈0.417）。
    判据 R1 的正确检验方式因此是**配对**：每次重复用其实际实现的 τ̂² 代入本函数，
    再与该重复的 MC 结果比对。这样可干净分离"闭式解是否正确"与"τ̂² 抽样波动"。

    【修订 v1.4 / N-R3 第二次修正】初版方差项只保留 σ²ι/n，遗漏了 γ² 相关项。
    诊断证据：n=800、σ_ξ²=4、m=6 下，聚合相对误差随 τ² 单调增大
    （τ²=0.01→0.29%，0.1→1.42%，0.5→4.62%，1.0→7.28%，2.0→10.56%）。
    完整推导（保留 Var(Σc²) 与 Cov(N,D) 项，两处 γ² 项恰好部分抵消）：

        Var(γ̂_i) = [γ_i²σ_ξ² + σ²(1+σ_ξ²)] / [n(1+σ_ξ²)²]
        ⇒ MSE 方差贡献 = Var(γ̂)·(1+σ_ξ²) = σ²/n + γ_i²σ_ξ²/[n(1+σ_ξ²)]

    自洽性检查：σ_ξ²=0 ⇒ Var(γ̂)=σ²/n（z=c 时的 OLS 方差），无 γ² 残留 ✓
    池化臂同理（观测噪声 σ_ξ²/m、输出噪声 σ²/m）。
    """
    Aa, Am = A_(sigma_xi2), A_m(sigma_xi2, m)
    S = 1.0 + sigma_xi2
    g2 = gbar**2
    # F：域平均 γ² → ḡ² + τ̂²
    var_f = sigma2 / n + (g2 + tau_hat2_val) * sigma_xi2 / (n * S)
    # S：池化后观测噪声 σ_ξ²/m、输出噪声 σ²/m，回归系数为 ḡ
    var_s = sigma2 / (m * n) + g2 * sigma_xi2 / (n * (m + sigma_xi2))
    mse_f = sigma2 + (g2 + tau_hat2_val) * Aa + var_f
    mse_s = sigma2 + tau_hat2_val + g2 * Am + var_s
    return mse_f, mse_s, mse_f - mse_s


def closed_form_param(m: int, sigma_xi2: float, n: int, tau2: float,
                      gbar: float = 1.0, sigma2: float = 1.0):
    """测试时给定真 c 的口径（机制分解用）。

    渐近（n→∞）主导项：
        MSE_F^param = Var(γ̂_i)      + (ḡ²+τ̂²)(1−ι)²
        MSE_S^param = Var(ĝ)         + τ̂² + ḡ²(1−ι_m)²
    有限 n 的方差项按 O(1/n) 保留主项（σ²ι/n 与 σ²ι_m/(mn)）。
    """
    th2 = tau_hat2(tau2, m)
    io, io_m = iota(sigma_xi2), 1.0 / (1.0 + sigma_xi2 / m)
    var_f = sigma2 * io / n
    var_s = sigma2 * io_m / (m * n)
    # 同样含不可约噪声 σ²（v1.3 修正）
    mse_f = sigma2 + var_f + (gbar**2 + th2) * (1.0 - io) ** 2
    mse_s = sigma2 + var_s + th2 + gbar**2 * (1.0 - io_m) ** 2
    return mse_f, mse_s, mse_f - mse_s


def closed_form_total_inf_n(m: int, sigma_xi2: float, tau2: float,
                            gbar: float = 1.0) -> float:
    """n→∞ 极限下的 G_total。预注册 C2 的判决量。

    G_∞ = ḡ²(A − A_m) − τ̂²(1 − A)
        = ḡ²σ_ξ²(m−1)/[(1+σ_ξ²)(m+σ_ξ²)] − τ̂²/(1+σ_ξ²)
    """
    th2 = tau_hat2(tau2, m)
    Aa, Am = A_(sigma_xi2), A_m(sigma_xi2, m)
    return gbar**2 * (Aa - Am) - th2 * (1.0 - Aa)


def tau_star2(m: int, sigma_xi2: float, n: int,
              gbar: float = 1.0, sigma2: float = 1.0) -> float:
    """使 G_total = 0 的先验异质性 τ²（m ≥ 2 有定义）。

    由 G = (m−1)/m·σ²/n + ḡ²(A−A_m) − τ̂²(1−A) = 0 解出 τ̂²*，
    再用 τ̂² = τ²(1−1/m) 换回 τ²：

        τ²* = σ²(1+σ_ξ²)/n + m·ḡ²·σ_ξ²/(m+σ_ξ²)

    【修订 v1.2】预注册 §3 的 τ*² 表达式多了一个 (m−1)/m 因子、
    且第二项缺因子 m，与本函数不符。**以本函数为准**（见 §附录 B）。
    """
    if m < 2:
        return float("nan")
    return sigma2 * (1.0 + sigma_xi2) / n + m * gbar**2 * sigma_xi2 / (m + sigma_xi2)


# ===========================================================================
# 二、蒙特卡洛（忠实实现模型，不调用闭式解）
# ===========================================================================
def mc_draw(m: int, n: int, sigma_xi2: float, tau2: float,
            gbar: float, sigma2: float, rng: np.random.Generator,
            n_test: int = 20000, n_rep: int = 1):
    """一次蒙特卡洛：返回 (mse_f, mse_s, mse_f_param, mse_s_param)。

    形状约定：所有数组 (n_rep, m, n)。
    """
    sx, se = np.sqrt(sigma_xi2), np.sqrt(sigma2)

    # 训练段
    c = rng.standard_normal((n_rep, 1, n))
    xi = sx * rng.standard_normal((n_rep, m, n))
    eps = se * rng.standard_normal((n_rep, m, n))
    delta = np.sqrt(tau2) * rng.standard_normal((n_rep, m, 1))
    gamma = gbar + delta                       # (n_rep, m, 1)

    z = c + xi                                 # (n_rep, m, n)
    y = gamma * c + eps                        # (n_rep, m, n)

    # ---- F：逐单元回归 y_i on z_i（无截距） ----
    num_f = np.sum(z * y, axis=2)              # (n_rep, m)
    den_f = np.sum(z * z, axis=2)
    ghat_f = num_f / den_f                     # (n_rep, m)

    # ---- S：池化后回归 ȳ on z̄ ----
    z_bar = z.mean(axis=1)                     # (n_rep, n)
    y_bar = y.mean(axis=1)
    ghat_s = np.sum(z_bar * y_bar, axis=1) / np.sum(z_bar * z_bar, axis=1)  # (n_rep,)

    # ---- 测试段 ----
    c_t = rng.standard_normal((n_rep, 1, n_test))
    xi_t = sx * rng.standard_normal((n_rep, m, n_test))
    eps_t = se * rng.standard_normal((n_rep, m, n_test))
    z_t = c_t + xi_t
    y_t = gamma * c_t + eps_t                  # 目标

    # G_total：F 用 z_{i,*}，S 用池化 z̄_*
    pred_f = ghat_f[:, :, None] * z_t
    z_bar_t = z_t.mean(axis=1, keepdims=True)  # (n_rep, 1, n_test)
    pred_s = ghat_s[:, None, None] * z_bar_t

    mse_f = np.mean((y_t - pred_f) ** 2, axis=(1, 2))
    mse_s = np.mean((y_t - pred_s) ** 2, axis=(1, 2))

    # G_param：双方都用真 c 作输入
    pred_f_p = ghat_f[:, :, None] * c_t
    pred_s_p = ghat_s[:, None, None] * c_t
    mse_f_p = np.mean((y_t - pred_f_p) ** 2, axis=(1, 2))
    mse_s_p = np.mean((y_t - pred_s_p) ** 2, axis=(1, 2))

    # 本次重复实际实现的域内经验异质性 τ̂² = (1/m)Σ(γ_i − ḡ)²
    gam = gamma[:, :, 0]                       # (n_rep, m)
    th2_actual = np.var(gam, axis=1, ddof=0)   # 总体方差 = (1/m)Σ(γ−ḡ)²

    return mse_f, mse_s, mse_f_p, mse_s_p, th2_actual


def mc_replicate(m: int, n: int, sigma_xi2: float, tau2: float,
                 gbar: float, sigma2: float, n_rep: int, seed: int,
                 n_test: int = 20000, batch: int = 20):
    """重复 n_rep 次，分批以控制内存。返回四个数组，各长 n_rep。"""
    rng = np.random.default_rng(seed)
    outs: list[list[np.ndarray]] = [[] for _ in range(5)]
    done = 0
    while done < n_rep:
        b = min(batch, n_rep - done)
        r = mc_draw(m, n, sigma_xi2, tau2, gbar, sigma2, rng, n_test=n_test, n_rep=b)
        for k in range(5):
            outs[k].append(r[k])
        done += b
    return [np.concatenate(o) for o in outs]

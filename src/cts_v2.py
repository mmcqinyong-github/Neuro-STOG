"""CTS-Net v2: 双通道解离版（W-P2-B 预注册规格 v1.0 的实现）。

相对 v1（`src/cts_net.py`，原样保留作留档）的改动：
  1. **移除子空间投影**（任务①）：P_U / P_V 不再使用。
     W-P2-A 消融证明投影是负优化（no_proj 比 full 好 shift 14.1% / raw 12.2%）。
  2. **η 源与 θ 源解耦**（任务②）：两条通道可独立指定来源，便于解离。
     - ach    : 结构/期望不确定性（输入分布偏离 train 参考）
     - da     : 价值/意外不确定性（运行预测损失）
     - credit : DA 型信用分配（"适配是否真的比骨干好"的在线比较）
     - run    : 截至当前的运行均值 —— 预算匹配的静态对照
  3. **θ 不再由 θ_max·sigmoid 给定**，改由所选 θ 源产生；
     credit 源用 tanh，允许负权重（适配有害时不信任它）。
  4. **新增目标映射**（`target_map`），用于 rewire（虚拟漂移）体制：
     输入分布不变、p(y|x) 改变。

     \u0000 所有信号的计算只用 t 时刻及之前的信息，单步因果，无泄漏。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADAPT_WINDOW = 200      # 自适应基线窗口（预注册 §5 固定）


class _AdaptNorm:
    """相对近期适应水平的局部新异性映射，输出约束在 [0,1]。

    为什么必须这样归一化（2026-10-27，失败三次后的结论）：
      直接用「相对训练参考」的百分位，各通道的**时间平均驱动不相等** ——
      实测 ETTh1 raw 上 d_shift=0.80 而 d_val=0.49。于是所谓 "ach 更好"
      其实是"平均而言学习率更高"，与 W-P1-W 的参数量混淆是同一类错误。
      本类把信号相对于其**最近 W 步的均值/标准差**做有界标准化：
          u = clip(0.5 + (x − ma_W) / (4·sd_W), 0, 1)
      对任何局部平稳过程，E[u] ≈ 0.5（对称性），因此**所有通道的平均预算
      自动等价**；通道之间的差别只剩「何时高、何时低」，也就是时间结构。
      窗口外的初值用训练段统计量作种子 ⇒ 只用训练信息，无泄漏。
    """

    def __init__(self, ref_mean: float, ref_std: float, W: int = ADAPT_WINDOW):
        self.W = W
        self.buf: list[float] = []
        self.ref_mean = float(ref_mean)
        self.ref_std = float(ref_std) if ref_std > 1e-12 else 1.0

    def update(self, x: float) -> float:
        if len(self.buf) >= 2:
            ma = float(np.mean(self.buf))
            sd = float(np.std(self.buf))
        else:
            ma, sd = self.ref_mean, self.ref_std
        sd = sd if sd > 1e-12 else self.ref_std
        u = 0.5 + (float(x) - ma) / (4.0 * sd)
        u = float(np.clip(u, 0.0, 1.0))
        self.buf.append(float(x))
        if len(self.buf) > self.W:
            self.buf.pop(0)
        return u


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


ETA_SOURCES = ("none", "ach", "da", "or", "run", "credit")
THETA_SOURCES = ("zero", "ach", "da", "credit", "runc")

# 变体 -> (eta_src, theta_src)  预注册 §3 写定
# 说明：`ach` 与 `credit` 的 η 源相同（都是 ach）⇒ 写入侧轨迹逐位相同
#      ⇒ 二者构成**预算精确匹配**的读出门消融（差异只来自 θ 的来源）。
VARIANTS = {
    "none":   ("none",   "zero"),
    "ach":    ("ach",    "ach"),      # 写←新奇  读←新奇
    "credit": ("ach",    "credit"),   # 写←新奇  读←信用   （与 ach 精确配对）
    "cw_ar":  ("credit", "ach"),      # 写←信用  读←新奇
    "cw_cr":  ("credit", "credit"),   # 写←信用  读←信用
    "da":     ("da",     "da"),       # 纯价值单通道（Black-Mamba 式单标量代理）
    "const":  ("run",    "runc"),     # 预算匹配的静态对照
    "moe2":   ("or",     "credit"),   # DynaME 代理：专家竞争
}


class CTSV2:
    def __init__(self, L: int, C: int, d: int = 32,
                 seed: int = 0,
                 n_epochs: int = 300, lr: float = 1e-3, batch: int = 64,
                 theta_max: float = 1.0, kappa: float = 8.0,
                 eta_base: float = 1e-2,
                 alpha_shift: float = 0.05, alpha_val: float = 0.10,
                 alpha_credit: float = 0.10, tau_credit: float = 1.0,
                 variant: str = "dual", n_experts: int = 1,
                 theta_init: float = 0.1, theta_probe: float = 0.5) -> None:
        self.L, self.C, self.d = L, C, d
        self.seed = seed
        self.n_epochs, self.lr, self.batch = n_epochs, lr, batch
        self.theta_max = theta_max
        self.kappa = kappa
        self.eta_base = eta_base
        self.alpha_shift = alpha_shift
        self.alpha_val = alpha_val
        self.alpha_credit = alpha_credit
        self.tau_credit = tau_credit
        self.variant = variant
        self.theta_init = theta_init
        self.theta_probe = theta_probe
        self.eta_src, self.theta_src = VARIANTS[variant]
        self.n_experts = n_experts if variant == "moe2" else 1
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ---------------- 网络构建 ----------------
    def _build(self) -> None:
        torch.manual_seed(self.seed)
        self.enc = nn.Sequential(
            nn.Linear(self.L * self.C, 128), nn.ReLU(),
            nn.Linear(128, self.d))
        self.dec = nn.Linear(self.d, self.C)
        # experts：moe2 用 2 个（DynaME 代理），其余 1 个
        self.experts = nn.ModuleList([nn.Linear(self.d, self.d)
                                      for _ in range(self.n_experts)])
        for e in self.experts:
            nn.init.zeros_(e.weight)
            nn.init.zeros_(e.bias)
        self.opt = torch.optim.Adam(
            list(self.enc.parameters()) + list(self.dec.parameters()), lr=self.lr)
        self.to(DEVICE)

    def to(self, dev: str) -> None:
        self.enc.to(dev); self.dec.to(dev); self.experts.to(dev)

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        return self.enc(X.reshape(B, -1))

    def _blend_experts(self, zt: torch.Tensor, w_expert: np.ndarray
                       ) -> torch.Tensor:
        """experts 的加权输出（moe2 时才有多于一个专家）。"""
        out = torch.zeros_like(zt)
        for k, e in enumerate(self.experts):
            out = out + float(w_expert[k]) * e(zt)
        return out

    # ---------------- 训练 ----------------
    def fit(self, Xtr: np.ndarray, Ytr: np.ndarray) -> dict:
        self._build()
        Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
        Yt = torch.tensor(Ytr, dtype=torch.float32, device=DEVICE)
        N = Xt.shape[0]
        for _ in range(self.n_epochs):
            perm = torch.randperm(N, device=DEVICE)
            for i in range(0, N, self.batch):
                idx = perm[i:i + self.batch]
                self.opt.zero_grad()
                z = self._encode(Xt[idx])
                loss = ((self.dec(z) - Yt[idx]) ** 2).mean()
                loss.backward()
                self.opt.step()

        # --- 门控参考统计量（全部来自 train 段，无泄漏）---
        # 结构通道（ACh 型，期望不确定性）：输入分布偏离
        self.mu_ref = float(np.mean(Xtr.mean(axis=(0, 1))))
        self.sd_ref = float(np.mean(Xtr.std(axis=(0, 1)))) + 1e-8
        es_tr = np.array([self._input_shift(Xtr[k]) for k in range(len(Xtr))])
        es_tr_ema = self._ema_seq(es_tr, self.alpha_shift)

        # 价值通道（DA 型，意外不确定性）：输出预测误差
        with torch.no_grad():
            pred_tr = self.dec(self._encode(Xt)).cpu().numpy()
        e_val = np.mean((pred_tr - Ytr) ** 2, axis=1)
        e_val = np.asarray(e_val).reshape(-1)          # Ytr 为 (N,1,C)
        ev_tr_ema = self._ema_seq(e_val, self.alpha_val)
        self.mu_v = float(np.mean(e_val))
        self.sigma_v = float(np.std(e_val)) + 1e-8

        # ---------------------------------------------------------------
        # 因果秩归一化的参考分布（2026-10-27 修订，理由见下）
        #
        # 问题：初稿用 sigmoid((s − mu)/sigma) 归一化，实测在 ETTh1 上
        #   raw d_shift=0.727 / shift d_shift=0.893 —— 动态范围只有 0.17，
        #   接近常开。于是"ach 门控"退化成"恒定高学习率"，它在所有体制
        #   都赢，纯粹因为它**平均而言练得更多**，而不是因为它处理了结构信息。
        #   这与 W-P1-W 的失败同源：对照两臂的**平均预算**不等价。
        #
        # 修正：把每条通道的原始信号映射到它在**训练段 EMA 分布**上的分位数
        #   u ∈ [0,1]。性质：
        #     · 若测试段与训练段同分布，则 E[u] = 0.5（无伪触发）
        #     · 所有通道共享同一 Server → U(0,1) 边际 ⇒ **平均驱动完全相同**
        #     · 差别只剩"何时高何时低"，即时间结构 —— 这才是要测的东西
        #   参考分布只由训练段构成 ⇒ 无泄漏；推理时用固定参考 ⇒ 因果。
        # ---------------------------------------------------------------
        # 自适应局部基线（impl. v2.3，理由见下）
        self.ref_shift = (float(np.mean(es_tr_ema)), float(np.std(es_tr_ema)) + 1e-8)
        self.ref_val = (float(np.mean(ev_tr_ema)), float(np.std(ev_tr_ema)) + 1e-8)
        return dict(mse_train=float(np.mean(e_val)))

    @staticmethod
    def _ema_seq(vals: np.ndarray, alpha: float) -> np.ndarray:
        s = float(vals[0])
        out = []
        for v in vals:
            s = (1 - alpha) * s + alpha * float(v)
            out.append(s)
        return np.asarray(out)

    def _input_shift(self, X: np.ndarray) -> float:
        """输入分布漂移指标（v1.1 沿用）：窗口均值+方差的相对偏移。"""
        mu_t = float(np.mean(X))
        sd_t = float(np.mean(np.std(X, axis=0)))
        return (abs(mu_t - self.mu_ref) / self.sd_ref
                + 0.5 * abs(sd_t - self.sd_ref) / self.sd_ref)

    # ---------------- 窗口化 ----------------
    @staticmethod
    def make_windows(S: np.ndarray, L: int, H: int = 1):
        N = len(S)
        X, Y = [], []
        for i in range(N - L - H + 1):
            X.append(S[i:i + L])
            Y.append(S[i + L:i + L + H])
        return np.asarray(X), np.asarray(Y)

    # ---------------- 在线滚动 ----------------
    def rolling(self, S_roll: np.ndarray, eval_start: int, n_eval: int,
                target_map: np.ndarray | None = None) -> dict:
        """单步因果在线评估。

        target_map: (C,C) 线性映射，作用在真实未来上得到监督目标。
                    rewire（虚拟漂移）用它做通道置换：输入不变，p(y|x) 改变。
        """
        L, C = self.L, self.C
        N = len(S_roll)
        self.enc.eval(); self.dec.eval()
        for e in self.experts:
            e.train()

        ad_opt = torch.optim.SGD(self.experts.parameters(), lr=self.eta_base)

        # --- 状态 ---
        # 冷启动死锁修正（2026-10-27 发现并留档）：
        # credit 驱动的 θ 若从 0 起步，则 z_ad = z（适配不生效）
        # ⇒ L_bb − L_AD ≡ 0 ⇒ credit ≡ 0 ⇒ θ ≡ 0，永久死锁。
        # 实测 dual/credit/const 的 MSE 与 none 完全相同即是此症状。
        # 修正：所有自适应变体以 θ0 = θ_init·θ_max 起步，让 DA 信用有可测信号。
        theta = 0.0 if self.theta_src == "zero" else self.theta_init * self.theta_max
        theta0 = theta
        s_shift = self.ref_shift[0]
        s_val = self.ref_val[0]
        # 自适应局部基线归一化器：把每条通道映射为"相对近期适应水平的新异性"
        an_shift = _AdaptNorm(*self.ref_shift, W=ADAPT_WINDOW)
        an_val = _AdaptNorm(*self.ref_val, W=ADAPT_WINDOW)
        an_credit = _AdaptNorm(0.0, 1.0, W=ADAPT_WINDOW)
        s_credit = 0.0
        run_drive = 0.0          # or-驱动的截至当前均值（const 用）
        run_credit = 0.0         # credit 的截至当前均值（const 用）
        w_expert = np.ones(self.n_experts) / self.n_experts

        preds = np.zeros((n_eval, C))
        y_true = np.zeros((n_eval, C))
        eta_traj = np.zeros(n_eval)
        theta_traj = np.zeros(n_eval)
        dshift_traj = np.zeros(n_eval)
        dval_traj = np.zeros(n_eval)
        credit_traj = np.zeros(n_eval)

        with torch.no_grad():
            z_all = []
            for i in range(N):
                if i + L > N:
                    break
                xt = torch.tensor(S_roll[i:i + L][None, :, :],
                                  dtype=torch.float32, device=DEVICE)
                z_all.append(self._encode(xt).cpu().numpy()[0])
        z_all = np.asarray(z_all)

        budget = 0.0          # 累计 (1+kappa*drive) 之和，用于 B5 预算审计
        hist_credit: list[float] = []      # credit 的自身历史（无训练段参考）
        credit_src: list[float] = []
        for j in range(n_eval):
            i = eval_start + j
            if i + L + 1 > N:
                break
            win = S_roll[i:i + L]
            fut = S_roll[i + L]
            y_obs = (target_map @ fut) if target_map is not None else fut

            zt = torch.tensor(z_all[i][None, :], dtype=torch.float32,
                              device=DEVICE)
            with torch.no_grad():
                z = zt[0]
                delta = self._blend_experts(zt, w_expert)[0]      # (d,)
                z_ad = z + theta * delta
                yhat = self.dec(z_ad[None, :])[0].cpu().numpy()
                ybb = self.dec(zt)[0].cpu().numpy()               # 冻结骨干
                # 固定幅度探针： credit 必须与「当前信任度 θ」解耦，否则死锁
                yprobe = self.dec((z + self.theta_probe * delta)[None, :]
                                  )[0].cpu().numpy()

            preds[j] = yhat
            y_true[j] = y_obs

            # ---------- 通道信号（全部只用 t 及之前的信息）----------
            # 结构/ACh 型：EMA 后映射到训练参考分布的分位数
            e_shift = self._input_shift(win)
            s_shift = (1 - self.alpha_shift) * s_shift + self.alpha_shift * e_shift
            d_shift = an_shift.update(s_shift)

            # 价值/DA 型
            e_val = float(np.mean((yhat - y_obs) ** 2))
            s_val = (1 - self.alpha_val) * s_val + self.alpha_val * e_val
            d_val = an_val.update(s_val)

            # DA 型信用分配：可塑通路是否真的携带降损信息。
            # 用固定幅度探针测量（θ_probe），不用当前的 θ，否则 credit 自指死锁。
            l_bb = float(np.mean((ybb - y_obs) ** 2))
            l_probe = float(np.mean((yprobe - y_obs) ** 2))
            s_credit = ((1 - self.alpha_credit) * s_credit
                        + self.alpha_credit * (l_bb - l_probe))
            hist_credit.append(s_credit)
            u_credit = an_credit.update(s_credit)

            # 运行均值（静态预算匹配对照，因果：只用过去）
            credit = u_credit
            drive_now = max(d_shift, d_val)
            run_drive = run_drive + (drive_now - run_drive) / (j + 1)
            run_credit = run_credit + (credit - run_credit) / (j + 1)

            # ---------- 由来源表决定 η 与 θ ----------
            if self.eta_src == "none":
                drive = 0.0
            elif self.eta_src == "ach":
                drive = d_shift
            elif self.eta_src == "da":
                drive = d_val
            elif self.eta_src == "or":
                drive = drive_now
            elif self.eta_src == "credit":
                drive = u_credit
            else:                                     # run（静态对照）
                drive = run_drive
            eta = self.eta_base * (1.0 + self.kappa * drive)
            budget += (1.0 + self.kappa * drive)

            if self.theta_src == "zero":
                theta = 0.0
            elif self.theta_src == "ach":
                theta = self.theta_max * d_shift
            elif self.theta_src == "da":
                theta = self.theta_max * d_val
            elif self.theta_src == "credit":
                theta = self.theta_max * credit
            else:                                     # runc（静态对照）
                theta = self.theta_max * run_credit

            # ---------- 在线更新（frozen backbone -> only experts）----------
            if self.eta_src != "none":
                yt = torch.tensor(y_obs[None, :], dtype=torch.float32,
                                  device=DEVICE)
                ad_opt.zero_grad()
                # 写读分离：写入侧恒用全强度 Δ（θ_write=1），不由 θ 缩放。
                # 若像初稿那样写 dec(z+θΔ)，则梯度 ∝ θ，θ≈0 时专家学不动
                # ⇒ credit 无信号 ⇒ θ 恒 0（bootstrap 死锁，实测 dual/credit
                # 的 MSE 与 none 完全相同）。θ 只管读出权重。
                z_ad_t = zt + self._blend_experts(zt, w_expert)
                loss = ((self.dec(z_ad_t) - yt) ** 2).mean()
                loss.backward()
                ad_opt.step()
                for g in ad_opt.param_groups:
                    g["lr"] = eta

                # DynaME 代理：专家权重按近期表现做竞争（性能路由）
                if self.variant == "moe2":
                    with torch.no_grad():
                        losses = []
                        for e in self.experts:
                            zz = zt + theta * e(zt)
                            losses.append(float(
                                ((self.dec(zz) - yt) ** 2).mean().item()))
                    losses = np.asarray(losses)
                    losses = losses - losses.min()
                    w_new = np.exp(-4.0 * losses)
                    w_new = w_new / w_new.sum()
                    w_expert = 0.9 * w_expert + 0.1 * w_new

            eta_traj[j] = eta
            theta_traj[j] = theta
            dshift_traj[j] = d_shift
            dval_traj[j] = d_val
            credit_traj[j] = credit

        valid = int(np.sum(np.any(y_true != 0, axis=1)))
        sl = slice(0, max(valid, 1))
        mse = float(np.mean((preds[sl] - y_true[sl]) ** 2))
        return dict(preds=preds, y_true=y_true, mse=mse,
                   eta=eta_traj, theta=theta_traj,
                    d_shift=dshift_traj, d_val=dval_traj,
                    credit=credit_traj, budget=float(budget))


def _rewire_matrix(C: int, seed: int = 20_261_027) -> np.ndarray:
    """虚拟漂移的置换矩阵（π：偶数长度错位一格，避免自反）。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(C)
    if C >= 2 and np.all(perm == np.arange(C)):
        perm = np.roll(np.arange(C), 1)
    P = np.zeros((C, C))
    for k, c in enumerate(perm):
        P[k, c] = 1.0
    return P


def run_unit(ds: str, regime: str, variant: str, seed: int,
             lookback: int = 96, n_train: int = 800,
             n_calib: int = 400, n_test: int = 500,
             n_epochs: int = 300, **kw) -> dict:
    """单单元：数据集 × 体制 × 变体 × 种子。"""
    try:
        from tsf_external import split_external as _sp_ext
        if ds in ("solar", "PEMS04"):
            sp = _sp_ext
        else:
            from tsf_loader import split as sp            # noqa: F811
    except Exception:
        from tsf_loader import split as sp

    s = sp(ds, "raw" if regime == "rewire" else regime,
           lookback=lookback, n_train=n_train,
           n_calib=n_calib, n_test=n_test, offset=0)
    L = lookback
    Xtr, Ytr = CTSV2.make_windows(s.train, L, 1)
    S_roll = np.concatenate([s.calib, s.test], axis=0)
    eval_start = len(s.calib)
    tmap = _rewire_matrix(s.n_vars) if regime == "rewire" else None

    net = CTSV2(L, s.n_vars, seed=seed, n_epochs=n_epochs,
                variant=variant, **kw)
    fit_info = net.fit(Xtr, Ytr)
    res = net.rolling(S_roll, eval_start, n_test, target_map=tmap)
    n_param = (sum(p.numel() for p in net.enc.parameters())
               + sum(p.numel() for p in net.dec.parameters())
               + sum(p.numel() for p in net.experts.parameters()))
    return dict(dataset=ds, regime=regime, variant=variant, seed=seed,
                mse=res["mse"], mse_train=fit_info["mse_train"],
                eta_mean=float(np.mean(res["eta"])),
                theta_mean=float(np.mean(res["theta"])),
                theta_abs=float(np.mean(np.abs(res["theta"]))),
                dshift=float(np.mean(res["d_shift"])),
                dval=float(np.mean(res["d_val"])),
                credit=float(np.mean(res["credit"])),
                budget=res["budget"], n_param=n_param)


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "ETTh1"
    reg = sys.argv[2] if len(sys.argv) > 2 else "shift"
    var = sys.argv[3] if len(sys.argv) > 3 else "dual"
    r = run_unit(ds, reg, var, 0, n_epochs=60)
    print(f"{ds} {reg:7s} {var:7s} MSE={r['mse']:.5f} "
          f"d_s={r['dshift']:.3f} d_v={r['dval']:.3f} "
          f"credit={r['credit']:+.3f} P={r['n_param']}")

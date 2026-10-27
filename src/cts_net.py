"""CTS-Net: Complementary Teaching-Signal Subspace Network.

原理（对应 W-P2-A预注册规格 §1）：
  - 稳定子空间（皮层样，主骨干冻结）：z_stable = P_V @ z_t
  - 可塑子空间（海马样，在线适配头）：z_plastic = P_U @ Adapter(z_t)
  - 子空间旋转角 theta(t) 由漂移信号 d_shift 驱动（Buzsaki Nature 2026）
  - 双通道：结构/漂移通道（ACh 型，状态转移预测误差）+ 价值通道（DA 型，输出误差）
  - lam=0 => theta 恒 0 且适配头冻结（纯骨干）；lam=1 => 全开

实现说明（预注册 §3）：
  - U,V 为 train 表征的 PCA 奇异向量（U=top-r, V=next-r，正交），P_U=UU^T, P_V=VV^T
  - 可塑修正被显式投影到可塑子空间 => 几何上正交于稳定（M4 实现自检）
  - 门控参数 mu_s/sigma_s 从 train 段估计（无泄漏）
  - H=1（单步滚动），在线逐点评估
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


class CTSNet:
    def __init__(self, L: int, C: int, d: int = 32, r_sub: int = 8,
                 lam: float = 1.0, seed: int = 0,
                 n_epochs: int = 300, lr: float = 1e-3, batch: int = 64,
                 theta_max: float = 1.0, kappa_shift: float = 2.0,
                 eta_base: float = 1e-3, alpha_shift: float = 0.05,
                 alpha_val: float = 0.10, variant: str = "full") -> None:
        self.L, self.C, self.d = L, C, d
        self.r_sub = min(r_sub, d // 2)
        self.lam = lam
        self.seed = seed
        self.n_epochs, self.lr, self.batch = n_epochs, lr, batch
        self.theta_max = theta_max
        self.kappa_shift = kappa_shift
        self.eta_base = eta_base
        self.alpha_shift, self.alpha_val = alpha_shift, alpha_val
        self.variant = variant          # full / value_only / shift_only / no_subspace
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ---------------- 网络构建 ----------------
    def _build(self) -> None:
        torch.manual_seed(self.seed)
        self.enc = nn.Sequential(
            nn.Linear(self.L * self.C, 128), nn.ReLU(),
            nn.Linear(128, self.d))
        self.dec = nn.Linear(self.d, self.C)
        self.adapter = nn.Linear(self.d, self.d)
        # 适配头初始为 0（lam=0 时输出 = 骨干，无注入）
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)
        self.opt = torch.optim.Adam(
            list(self.enc.parameters()) + list(self.dec.parameters()), lr=self.lr)
        self.to(DEVICE)

    def to(self, dev: str) -> None:
        self.enc.to(dev); self.dec.to(dev); self.adapter.to(dev)

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        return self.enc(X.reshape(B, -1))

    # ---------------- 训练 ----------------
    def fit(self, Xtr: np.ndarray, Ytr: np.ndarray) -> dict:
        """Xtr:(N,L,C) Ytr:(N,C) —— 训练主骨干 + 估计 U,V,W_pred + 门控统计量。"""
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

        # 子空间基：train 表征 PCA
        with torch.no_grad():
            Z = self._encode(Xt).cpu().numpy()          # (N,d)
        mu_z = Z.mean(0, keepdims=True)
        Zc = Z - mu_z
        _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        V = Vt.T                                        # (d,d) 右奇异向量
        self.U = V[:, :self.r_sub]                      # 可塑基
        self.V = V[:, self.r_sub:2 * self.r_sub]        # 稳定基（正交于 U）
        self.P_U = self.U @ self.U.T
        self.P_V = self.V @ self.V.T

        # 状态转移模型 W_pred：z_{t-1} -> z_t（最小二乘）
        Zp, Zf = Z[:-1], Z[1:]
        Wp = np.linalg.lstsq(Zp, Zf, rcond=None)[0]     # (d,d)
        self.W_pred = Wp
        e_shift = np.linalg.norm(Zp @ Wp - Zf, axis=1) / np.sqrt(self.d)
        self.mu_s = float(np.mean(e_shift))
        self.sigma_s = float(np.std(e_shift)) + 1e-8
        # 价值通道门控统计量（train 输出误差）
        with torch.no_grad():
            pred_tr = self.dec(self._encode(Xt)).cpu().numpy()
        e_val = np.mean((pred_tr - Ytr) ** 2, axis=1)
        self.mu_v = float(np.mean(e_val))
        self.sigma_v = float(np.std(e_val)) + 1e-8

        # --- v1.1 修订：结构/漂移通道改在输入空间检测分布漂移 ---
        # ACh 型"结构学习"= 检测环境布局（输入分布）变化，与输出价值无关。
        # 参考统计量从 train 段窗口估计（无泄漏）。
        # e_shift 在"原始输入空间"（z-score 后但含 shift/noise 注入）计算：
        #   均值漂移分量 |mu_t - mu_ref|/sig_ref  + 方差漂移分量 |sd_t - sig_ref|/sig_ref
        # 这样 shift（常数偏移→均值跳变）与 noise（方差增大）都能被结构通道捕获，
        # 而不依赖表征空间（表征会吸收常数偏移，见 W-P1-W 的 shift 不变性发现）。
        mu_ref = float(np.mean(Xtr.mean(axis=(0, 1))))
        sd_ref = float(np.mean(Xtr.std(axis=(0, 1)))) + 1e-8
        self.mu_ref, self.sd_ref = mu_ref, sd_ref
        es_tr = np.array([self._input_shift(Xtr[k]) for k in range(len(Xtr))])
        self.mu_s = float(np.mean(es_tr))
        self.sigma_s = float(np.std(es_tr)) + 1e-8
        return dict(mse_train=float(np.mean(e_val)))

    def _input_shift(self, X: np.ndarray) -> float:
        """输入分布漂移指标：窗口均值+方差的相对偏移（跨通道平均）。"""
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
    def rolling(self, S_roll: np.ndarray, eval_start: int, n_eval: int) -> dict:
        """在 S_roll 上从位置 0 滚动，评估位置 [eval_start, eval_start+n_eval)。

        返回 dict(preds, y_true, theta_traj, d_shift_traj, d_val_traj,
                   angle_traj, mse)
        """
        L, C = self.L, self.C
        N = len(S_roll)
        # 冻结主骨干（稳定记忆）；适配头可更新（可塑）
        self.enc.eval(); self.dec.eval()
        self.adapter.train()
        ad_opt = torch.optim.SGD(self.adapter.parameters(), lr=self.eta_base)

        theta = 0.0
        s_shift = self.mu_s
        s_val = self.mu_v
        d_shift = 0.0
        d_val = 0.0
        preds = np.zeros((n_eval, C))
        y_true = np.zeros((n_eval, C))
        theta_traj = np.zeros(n_eval)
        dshift_traj = np.zeros(n_eval)
        dval_traj = np.zeros(n_eval)
        angle_traj = np.zeros(n_eval)
        prev_z = None

        with torch.no_grad():
            z_all = []
            for i in range(N):
                if i + L > N:
                    break
                win = S_roll[i:i + L]
                xt = torch.tensor(win[None, :, :], dtype=torch.float32, device=DEVICE)
                z_all.append(self._encode(xt).cpu().numpy()[0])
        z_all = np.asarray(z_all)   # (N-L+1, d)

        for j in range(n_eval):
            i = eval_start + j
            if i + L + 1 > N:
                break
            win = S_roll[i:i + L]
            y_obs = S_roll[i + L]
            xt = torch.tensor(win[None, :, :], dtype=torch.float32, device=DEVICE)
            zt = torch.tensor(z_all[i][None, :], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                z = self._encode(xt)[0]                       # (d,)
                z_adj = z + theta * (self.adapter(z[None, :])[0])
                yhat = self.dec(z_adj[None, :])[0].cpu().numpy()

            preds[j] = yhat
            y_true[j] = y_obs

            # --- 结构/漂移通道（ACh 型）：输入分布漂移（v1.1）---
            e_shift = self._input_shift(win)              # win = S_roll[i:i+L]
            s_shift = (1 - self.alpha_shift) * s_shift + self.alpha_shift * e_shift
            d_shift = _sigmoid((s_shift - self.mu_s) / self.sigma_s)

            # --- 价值通道（DA 型）：输出误差 ---
            e_val = float(np.mean((yhat - y_obs) ** 2))
            s_val = (1 - self.alpha_val) * s_val + self.alpha_val * e_val
            d_val = _sigmoid((s_val - self.mu_v) / self.sigma_v)

            # --- 门控（variant 消融：full / no_theta / no_gate / no_proj）---
            drive_theta = self.variant in ("full", "no_proj")
            drive_eta = self.variant in ("full", "no_proj", "no_theta")
            do_update = self.variant in ("full", "no_theta", "no_gate", "no_proj")
            do_proj = self.variant in ("full", "no_theta", "no_gate")
            if self.lam > 0:
                theta = (self.lam * self.theta_max * d_shift) if drive_theta else 0.0
                eta = (self.eta_base * (1 + self.lam * self.kappa_shift * d_shift)
                       if drive_eta else self.eta_base)
                if do_update:
                    for g in ad_opt.param_groups:
                        g["lr"] = eta
                    zt = torch.tensor(z_all[i][None, :], dtype=torch.float32, device=DEVICE)
                    yt = torch.tensor(y_obs[None, :], dtype=torch.float32, device=DEVICE)
                    ad_opt.zero_grad()
                    z_adj_t = zt + theta * self.adapter(zt)
                    loss = ((self.dec(z_adj_t) - yt) ** 2).mean()
                    loss.backward()
                    # 梯度投影到可塑子空间（正交于稳定 => 不破坏稳定结构）
                    with torch.no_grad():
                        gvec = self.adapter.weight.grad.cpu().numpy()
                        G = gvec.reshape(self.d, self.d)
                        Gproj = (self.P_U @ G @ self.P_U) if do_proj else G
                        self.adapter.weight.grad = torch.tensor(
                            Gproj.reshape(-1), dtype=torch.float32,
                            device=DEVICE).reshape(self.adapter.weight.shape)
                    ad_opt.step()
            else:
                theta = 0.0

            # 子空间夹角（实现自检：可塑修正 vs 稳定方向）
            z_np = z.cpu().numpy()
            with torch.no_grad():
                z_adj_np = (z + theta * self.adapter(zt)[0]).cpu().numpy()
            z_stable = self.P_V @ z_np
            z_plastic = self.P_U @ (z_adj_np - z_np)
            denom = (np.linalg.norm(z_stable) * np.linalg.norm(z_plastic))
            ang = 90.0
            if denom > 1e-9:
                cos = abs(float(np.dot(z_stable, z_plastic) / denom))
                ang = float(np.degrees(np.arccos(min(1.0, cos))))
            angle_traj[j] = ang
            theta_traj[j] = theta
            dshift_traj[j] = d_shift
            dval_traj[j] = d_val
            prev_z = z.cpu().numpy()

        mse = float(np.mean((preds - y_true) ** 2))
        return dict(preds=preds, y_true=y_true, mse=mse,
                    theta=theta_traj, d_shift=dshift_traj, d_val=dval_traj,
                    angle=angle_traj)


def run_unit(ds: str, regime: str, lam: float, seed: int,
             lookback: int = 96, n_train: int = 800,
             n_calib: int = 400, n_test: int = 500,
             n_epochs: int = 300, **kw) -> dict:
    """单单元：数据集×体制×λ×种子。返回指标 dict。"""
    from tsf_loader import split
    s = split(ds, regime, lookback=lookback, n_train=n_train,
              n_calib=n_calib, n_test=n_test, offset=0)
    L = lookback
    # train 监督对（用 split.train 前段，避免用到 calib 体制切换）
    Xtr, Ytr = CTSNet.make_windows(s.train, L, 1)
    # 滚动序列 = calib + test（体制切换在 calib 1/3 处可见）
    S_roll = np.concatenate([s.calib, s.test], axis=0)
    eval_start = len(s.calib)
    n_eval = n_test
    net = CTSNet(L, s.n_vars, lam=lam, seed=seed, n_epochs=n_epochs, **kw)
    fit_info = net.fit(Xtr, Ytr)
    res = net.rolling(S_roll, eval_start, n_eval)
    return dict(dataset=ds, regime=regime, lam=lam, seed=seed,
                mse=res["mse"],
                mse_train=fit_info["mse_train"],
                theta_mean=float(np.mean(res["theta"])),
                theta_max=float(np.max(res["theta"])),
                dshift_mean=float(np.mean(res["d_shift"])),
                dval_mean=float(np.mean(res["d_val"])),
                angle_mean=float(np.mean(res["angle"])),
                n_param=sum(p.numel() for p in net.enc.parameters())
                        + sum(p.numel() for p in net.dec.parameters())
                        + sum(p.numel() for p in net.adapter.parameters()))


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "ETTh1"
    reg = sys.argv[2] if len(sys.argv) > 2 else "raw"
    lam = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    r = run_unit(ds, reg, lam, 0, n_epochs=60)
    print(f"{ds} {reg} lam={lam}: MSE={r['mse']:.6f} train={r['mse_train']:.6f} "
          f"theta_max={r['theta_max']:.3f} dshift={r['dshift_mean']:.3f} "
          f"angle={r['angle_mean']:.1f} nparam={r['n_param']}")

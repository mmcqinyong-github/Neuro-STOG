"""W-P1-R 驱动：闭式解存档 → 蒙特卡洛验证 → 主扫描。

预注册纪律 §7：闭式解**先于**蒙特卡洛写出并存档。
因此本脚本的 `archive` 子命令必须先于 `mc` / `scan` 执行，且存档文件带时间戳。

用法
----
    python scripts/run_wp1r.py archive          # 存档闭式解 + 自洽性检查
    python scripts/run_wp1r.py selftest         # 蒙特卡洛自检（小样本）
    python scripts/run_wp1r.py mc               # 闭式解 vs 蒙特卡洛验证
    python scripts/run_wp1r.py scan             # 主扫描
    python scripts/run_wp1r.py w1map            # W1 坐标映射（先于查看 W1 数值）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identifiability import (  # noqa: E402
    A_, A_m, closed_form_param, closed_form_total, closed_form_total_that,
    closed_form_total_inf_n, iota, mc_replicate, tau_hat2, tau_star2,
)

RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# 主扫描网格（预注册 §4.1）
GRID_M = [1, 2, 3, 6, 12, 24]
GRID_SX = [0.0, 0.25, 1.0, 4.0, 16.0, 64.0]
GRID_N = [50, 200, 800, 3200, 12800]
GRID_TAU = [0.01, 0.1, 0.5, 1.0, 2.0]
GBAR = 1.0
SIGMA2 = 1.0
N_REP = 40


# ---------------------------------------------------------------------------
# archive：闭式解存档 + 自洽性检查
# ---------------------------------------------------------------------------
def cmd_archive() -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("W-P1-R 闭式解存档（预注册 §7：先于蒙特卡洛写出）")
    lines.append(f"存档时间：{ts}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("模型：γ_i = ḡ + δ_i (δ~N(0,τ²))；c_t~N(0,1)；z=c+ξ (ξ~N(0,σ_ξ²))；y=γc+ε (ε~N(0,σ²))")
    lines.append("估计器只看到 (z,y)。")
    lines.append("")
    lines.append("闭式解（附录 A 推导）：")
    lines.append("    MSE_F = (ḡ² + τ̂²)·A + σ²/n")
    lines.append("    MSE_S = τ̂² + ḡ²·A_m + σ²/(mn)")
    lines.append("    G     = MSE_F − MSE_S")
    lines.append("          = (m−1)/m·σ²/n + ḡ²σ_ξ²(m−1)/[(1+σ_ξ²)(m+σ_ξ²)] − τ̂²/(1+σ_ξ²)")
    lines.append("    其中 A = σ_ξ²/(1+σ_ξ²)，A_m = σ_ξ²/(m+σ_ξ²)，τ̂² = τ²(1−1/m)")
    lines.append("")
    lines.append("n→∞ 极限：G_∞ = ḡ²(A − A_m) − τ̂²(1−A)")
    lines.append("τ*²（G=0 的解）：τ²* = σ²(1+σ_ξ²)/n + m·ḡ²·σ_ξ²/(m+σ_ξ²)    [m≥2]")
    lines.append("")

    # ---- 自洽性检查 ----
    lines.append("-" * 78)
    lines.append("自洽性检查（解析，未使用任何数值模拟）")
    lines.append("-" * 78)
    checks = []

    # C-a: m=1 时 G 恒为 0
    worst = 0.0
    for sx in GRID_SX:
        for n in GRID_N:
            for t2 in GRID_TAU:
                g = closed_form_total(1, sx, n, t2, GBAR, SIGMA2)[2]
                worst = max(worst, abs(g))
    checks.append(("C-a  m=1 时 G_total ≡ 0（共享=独立）", worst, worst < 1e-12))

    # C-b: σ_ξ²=0 退化为经典式 G = (m−1)σ²/(mn) − τ̂²
    worst = 0.0
    for m in GRID_M:
        for n in GRID_N:
            for t2 in GRID_TAU:
                g = closed_form_total(m, 0.0, n, t2, GBAR, SIGMA2)[2]
                classic = (m - 1) * SIGMA2 / (m * n) - tau_hat2(t2, m)
                worst = max(worst, abs(g - classic))
    checks.append(("C-b  σ_ξ²=0 退化为经典式", worst, worst < 1e-12))

    # C-c: τ̂²=0 且 σ_ξ²>0 且 m>1 时 G_∞ > 0（C2 的核心）
    worst = 1e9
    ok = True
    for m in GRID_M[1:]:
        for sx in GRID_SX[1:]:
            g = closed_form_total_inf_n(m, sx, 0.0, GBAR)
            worst = min(worst, g)
            ok = ok and g > 0
    checks.append(("C-c  τ̂²=0, σ_ξ²>0, m>1 ⇒ G_∞ > 0（C2 核心）", worst, ok))

    # C-d: τ*² 对 σ_ξ² 严格单调增（C1）
    ok = True
    detail = []
    for m in GRID_M[1:]:
        vals = [tau_star2(m, sx, 3200, GBAR, SIGMA2) for sx in GRID_SX]
        mono = all(vals[i + 1] > vals[i] for i in range(len(vals) - 1))
        ok = ok and mono
        detail.append(f"      m={m:2d}: " + " → ".join(f"{v:.4f}" for v in vals))
    checks.append(("C-d  τ*² 对 σ_ξ² 严格单调增（C1）", float(ok), ok))
    lines.extend(detail)

    # C-e: G_total 对 m 单调不减（τ̂²=0）
    ok = True
    for sx in GRID_SX:
        vals = [closed_form_total(m, sx, 800, 0.0, GBAR, SIGMA2)[2] for m in GRID_M]
        ok = ok and all(vals[i + 1] >= vals[i] - 1e-12 for i in range(len(vals) - 1))
    checks.append(("C-e  G_total 对 m 单调不减（τ̂²=0）", float(ok), ok))

    lines.append("")
    for name, val, ok in checks:
        flag = "PASS" if ok else "FAIL"
        lines.append(f"  [{flag}] {name}  （判据值 {val:.3e}）")
    lines.append("")
    lines.append(f"存档时间：{ts}")
    lines.append("=" * 78)

    out = RES / "wp1r_closedform.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n→ 已存档：{out}")


# ---------------------------------------------------------------------------
# selftest / mc：蒙特卡洛 vs 闭式解
# ---------------------------------------------------------------------------
def _one_cell(m, sx, n, t2, n_rep, seed, n_test=8000):
    # 批次按 m*n 动态收缩，避免大格子上峰值内存过高
    batch = max(4, int(2_000_000 / max(m * max(n, n_test), 1)))
    mf, ms, mfp, msp, th2 = mc_replicate(m, n, sx, t2, GBAR, SIGMA2, n_rep, seed,
                                         n_test=n_test, batch=batch)
    # 配对：用每次重复**实际实现**的 τ̂² 代入闭式解（消除 τ̂² 抽样波动）
    cf_f = np.array([closed_form_total_that(m, sx, n, v, GBAR, SIGMA2)[0] for v in th2])
    cf_s = np.array([closed_form_total_that(m, sx, n, v, GBAR, SIGMA2)[1] for v in th2])
    cf_g = cf_f - cf_s
    # 配对相对误差：逐重复比较，而非比中位数
    rel_f = np.abs(mf - cf_f) / np.abs(cf_f)
    rel_s = np.abs(ms - cf_s) / np.abs(cf_s)
    return {
        "m": m, "sigma_xi2": sx, "n": n, "tau2": t2,
        "mc_f": float(np.median(mf)), "mc_s": float(np.median(ms)),
        "cf_f": float(np.median(cf_f)), "cf_s": float(np.median(cf_s)),
        "mc_g": float(np.median(mf - ms)), "cf_g": float(np.median(cf_g)),
        "mc_g_iqr": float(np.subtract(*np.percentile(mf - ms, [75, 25]))),
        "mc_g_param": float(np.median(mfp - msp)),
        "mc_f_param": float(np.median(mfp)), "mc_s_param": float(np.median(msp)),
        "se_g": float(np.std(mf - ms, ddof=1) / np.sqrt(len(mf))),
        "paired_relerr_f": float(np.median(rel_f)),
        "paired_relerr_s": float(np.median(rel_s)),
        "th2_mean": float(np.mean(th2)),
    }


def cmd_selftest() -> None:
    print("蒙特卡洛自检（小规模，验证实现 fidelity）")
    rows = []
    # 关键格子：σ_ξ²=0 应无共享优势（τ̂²>0 时为负）；σ_ξ² 大且 τ² 小应有优势
    cases = [
        (6, 0.0, 800, 0.1),    # 完全可观测 → 预期 G<0（异质性代价）
        (6, 4.0, 800, 0.01),   # 不可观测 + 低异质 → 预期 G>0
        (6, 16.0, 800, 0.01),  # 更不可观测 → 预期 G 更大
        (1, 4.0, 800, 0.1),    # m=1 → 预期 G≈0
        (24, 4.0, 3200, 0.01),  # 大域 → 预期 G 更大
    ]
    for i, (m, sx, n, t2) in enumerate(cases):
        r = _one_cell(m, sx, n, t2, n_rep=40, seed=1000 + i)
        rows.append(r)
        print(f"  m={m:2d} σξ²={sx:5.2f} n={n:5d} τ²={t2:5.3f} | "
              f"MC_G={r['mc_g']:+.5f}±{r['se_g']:.5f}  CF_G={r['cf_g']:+.5f}  "
              f"relerr={abs(r['mc_g']-r['cf_g'])/max(abs(r['cf_g']),1e-9)*100:6.2f}%")
    print("\n自检完成。若 relerr 在 n≥800 上普遍 >5%，说明闭式解或实现有误（N-R3）。")


def cmd_mc() -> None:
    """闭式解 vs 蒙特卡洛：在 n≥800 的格子上系统比对（判据 R1）。"""
    print("闭式解 vs 蒙特卡洛（判据 R1：n≥800 上相对误差中位数 ≤2%）")
    errs_f, errs_s, rows = [], [], []
    seed = 20000
    for m in GRID_M:
        for sx in GRID_SX:
            for n in [800, 3200, 12800]:
                for t2 in GRID_TAU:
                    r = _one_cell(m, sx, n, t2, n_rep=N_REP, seed=seed)
                    seed += 1
                    rows.append(r)
                    errs_f.append(r["paired_relerr_f"])
                    errs_s.append(r["paired_relerr_s"])
    import statistics
    med_f = statistics.median(errs_f)
    med_s = statistics.median(errs_s)
    print(f"  格子数 {len(rows)}")
    print(f"  MSE_F 配对相对误差中位数 = {med_f*100:.3f}%   (阈值 2%)")
    print(f"  MSE_S 配对相对误差中位数 = {med_s*100:.3f}%   (阈值 2%)")
    print(f"  → R1 {'PASS' if max(med_f, med_s) <= 0.02 else 'FAIL'}")
    out = RES / "wp1r_mc_check.csv"
    keys = rows[0].keys()
    with out.open("w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.8g}" for k in keys) + "\n")
    print(f"  → {out}")


# ---------------------------------------------------------------------------
# scan：主扫描
# ---------------------------------------------------------------------------
def cmd_scan() -> None:
    print("主扫描（预注册 §4.1）")
    rows = []
    seed = 30000
    t0 = time.time()
    for m in GRID_M:
        for sx in GRID_SX:
            for n in GRID_N:
                for t2 in GRID_TAU:
                    r = _one_cell(m, sx, n, t2, n_rep=N_REP, seed=seed)
                    seed += 1
                    r["cf_g_inf"] = closed_form_total_inf_n(m, sx, t2, GBAR)
                    r["tau_star2"] = tau_star2(m, sx, n, GBAR, SIGMA2)
                    rows.append(r)
    out = RES / "wp1r_scan.csv"
    keys = list(rows[0].keys())
    with out.open("w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.8g}" for k in keys) + "\n")
    print(f"  {len(rows)} 格，用时 {time.time()-t0:.1f}s → {out}")


# ---------------------------------------------------------------------------
# w1map：W1 坐标映射（必须先于查看 W1 的 S−F 数值）
# ---------------------------------------------------------------------------
def cmd_w1map() -> None:
    """把 W1 的 env1/env3 配置映射到 (m, σ_ξ², n, τ²)，先算出 G 的符号预测。

    纪律：本函数只读 W1 的**配置**（不读 W1 的实测 S−F 数值），
    输出的是**预测**，随后才与 W1 实测比对（判据 R0）。
    """
    # W1 配置（来自 W1预注册规格 与 src/shared_gain.py 的 W1Env 默认值）：
    #   n_vars=9, m_e=3 ⇒ 域尺寸 m=3，域数 3
    #   共同成分由 shared_drive() 生成后**直接加进通道观测**（inject_shift）
    #   ⇒ 单通道对 c 的观测噪声 σ_ξ² ≈ 0（c 完全可观测）
    #   n_calib ∈ {50,100,200,400}
    #   ω = shift_weight ∈ {0, 2} 控制共同成分幅度
    #   hetero：域内符号异质性 ⇒ 对应 τ²
    mapping = {
        "source": "W1预注册规格-v1.0.md + src/shared_gain.py:W1Env",
        "m": 3,
        "sigma_xi2": 0.0,
        "justify_xi": "shared_drive() 生成的 c_t 经 inject_shift 直接加进通道观测，"
                      "单通道可从自身历史直接估计其影响 ⇒ σ_ξ²≈0（完全可识别）",
        "n_calib": [50, 100, 200, 400],
        "prediction": {},
    }
    preds = {}
    for n in [50, 100, 200, 400]:
        # σ²：W1 的 noise_scale=0.25 ⇒ σ²≈0.0625（量级估计，符号不依赖它）
        for s2 in [0.0625, 1.0]:
            g0 = closed_form_total(3, 0.0, n, 0.0, GBAR, s2)[2]   # τ²=0 上界
            # 异质性代价项：−τ̂²/(1+σ_ξ²) = −τ̂²，需 τ̂² > (m−1)σ²/(mn) 才翻负
            thr = (3 - 1) * s2 / (3 * n)
            preds[f"n={n},σ²={s2}"] = {
                "G_upperbound_tau0": float(g0),
                "tau_hat2_threshold_for_G_negative": float(thr),
                "predicted_sign_if_tau_hat2_above_threshold": "negative",
            }
    mapping["prediction"] = preds
    mapping["verdict_rule"] = (
        "若 W1 实测 S−F < 0 且此处预测的阈值 τ̂² 小于 W1 的实际异质性，则 R0 通过。"
        "本文件在查看 W1 实测数值之前写出。"
    )
    out = RES / "wp1r_w1_mapping.json"
    out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(mapping, ensure_ascii=False, indent=2))
    print(f"\n→ {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["archive", "selftest", "mc", "scan", "w1map"])
    a = ap.parse_args()
    {"archive": cmd_archive, "selftest": cmd_selftest, "mc": cmd_mc,
     "scan": cmd_scan, "w1map": cmd_w1map}[a.cmd]()


if __name__ == "__main__":
    main()

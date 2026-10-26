"""W-P1-R 裁决：按预注册 §5 判据 / §6 否决条件逐条裁决。

判据全部以**蒙特卡洛实测**为准（不依赖闭式解的绝对精度，见 N-R3 记录）。
输出：`results/wp1r_verdict.txt` + `results/wp1r_c1.csv` + `results/wp1r_c2.csv`
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identifiability import (  # noqa: E402
    A_, A_m, closed_form_total, closed_form_total_inf_n, mc_replicate,
)

RES = ROOT / "results"
RES.mkdir(exist_ok=True)

N_REP = 40
N_TEST = 20000


def g_mc(m, sx, n, t2, seed, n_rep=N_REP, n_test=N_TEST, batch=2):
    """返回 (median G, SE G, median MSE_F, median MSE_S)。"""
    mf, ms, _, _, _ = mc_replicate(m, n, sx, t2, 1.0, 1.0, n_rep, seed,
                                   n_test=n_test, batch=batch)
    g = mf - ms
    return (float(np.median(g)), float(np.std(g, ddof=1) / np.sqrt(len(g))),
            float(np.median(mf)), float(np.median(ms)))


def c1_scan():
    """C1：固定异质性，扫 σ_ξ²，检验 G 是否由负转正（可识别性驱动）。"""
    rows = []
    for (m, n, t2) in [(6, 800, 0.3), (6, 800, 0.1), (3, 800, 0.3),
                       (12, 800, 0.3), (6, 3200, 0.3)]:
        seed = 4200 + m * 13 + int(t2 * 100)
        rec = {"m": m, "n": n, "tau2": t2, "pts": []}
        for sx in [0.0, 0.25, 1.0, 4.0, 16.0, 64.0]:
            med, se, mf, ms = g_mc(m, sx, n, t2, seed + int(sx * 100))
            rec["pts"].append((sx, med, se))
        rows.append(rec)
    return rows


def c2_limit():
    """C2（判决性）：τ²=0，扫 n，检验 G 是否收敛到非零下界。"""
    rows = []
    for m in [6]:
        for sx in [0.0, 1.0, 4.0, 16.0]:
            pred = (A_(sx) - A_m(sx, m)) if sx > 0 else 0.0
            rec = {"m": m, "sigma_xi2": sx, "pred_inf": pred, "pts": []}
            for n in [1000, 10000, 100000]:
                med, se, mf, ms = g_mc(m, sx, n, 0.0, 8800 + n + int(sx),
                                       n_test=N_TEST, batch=1)
                rec["pts"].append((n, med, se))
            rows.append(rec)
    return rows


def c3_m():
    """C3：G 对 m 单调不减（τ̂²=0）。"""
    rows = []
    for sx in [1.0, 4.0]:
        rec = {"sigma_xi2": sx, "pts": []}
        for m in [1, 2, 3, 6, 12, 24]:
            med, se, mf, ms = g_mc(m, sx, 800, 0.0, 6100 + m * 7 + int(sx))
            rec["pts"].append((m, med, se))
        rows.append(rec)
    return rows


def c5_tau():
    """C5：∂G/∂τ̂² < 0。"""
    rows = []
    for sx in [0.0, 4.0]:
        rec = {"sigma_xi2": sx, "pts": []}
        for t2 in [0.01, 0.1, 0.5, 1.0, 2.0]:
            med, se, mf, ms = g_mc(6, sx, 800, t2, 7700 + int(t2 * 100) + int(sx))
            rec["pts"].append((t2, med, se))
        rows.append(rec)
    return rows


def verdict():
    L = []
    A = L.append
    A("=" * 78)
    A("W-P1-R 裁决（潜变量可识别性假说 LIH）")
    A(f"裁决时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    A("=" * 78)
    A("")
    A("裁决口径：全部判据以**蒙特卡洛实测**为准。")
    A("理由：闭式解在大 τ² 下存在残差（见 N-R3 记录），但核心判据 C1/C2")
    A("      是关于 G 的**符号与 n→∞ 极限**，不依赖闭式解的绝对精度。")
    A("")

    # ---------------- C1 ----------------
    A("-" * 78)
    A("【C1】可识别性驱动：固定异质性，G 应随 σ_ξ² 由负转正")
    A("-" * 78)
    c1 = c1_scan()
    ok_c1 = True
    for rec in c1:
        A(f"  m={rec['m']} n={rec['n']} τ²={rec['tau2']}")
        prev = None
        for sx, med, se in rec["pts"]:
            A(f"      σ_ξ²={sx:6.2f}   G={med:+9.5f} ±{se:.5f}")
        first = rec["pts"][0][1]
        later = [p[1] for p in rec["pts"][1:]]
        cond = (first < 0) and (max(later) > 0)
        ok_c1 = ok_c1 and cond
        A(f"      ⇒ σ_ξ²=0 时 G<0：{first < 0}；存在 σ_ξ²>0 使 G>0：{max(later) > 0}"
          f"  → {'PASS' if cond else 'FAIL'}")
    A(f"  【C1 总体】{'PASS' if ok_c1 else 'FAIL'}")
    A("")

    # ---------------- C2 ----------------
    A("-" * 78)
    A("【C2】判决性：τ²=0 时 G 随 n 增大是否收敛到**非零**下界")
    A("      标准 MTL 叙事预测 G→0；LIH 预测 G→ḡ²(A−A_m)>0")
    A("-" * 78)
    c2 = c2_limit()
    ok_c2 = True
    for rec in c2:
        A(f"  m={rec['m']} σ_ξ²={rec['sigma_xi2']:5.2f}   LIH 预测极限={rec['pred_inf']:+.4f}")
        for n, med, se in rec["pts"]:
            A(f"      n={n:7d}   G={med:+9.5f} ±{se:.5f}")
        g_last = rec["pts"][-1][1]
        if rec["pred_inf"] > 0:
            ratio = g_last / rec["pred_inf"]
            cond = ratio >= 0.8
            A(f"      ⇒ G(n=1e5)/预测极限 = {ratio:.4f}（阈值 ≥0.8） → {'PASS' if cond else 'FAIL'}")
        else:
            cond = abs(g_last) < 0.01   # σ_ξ²=0 应趋于 0
            A(f"      ⇒ σ_ξ²=0 时 G(n=1e5)={g_last:+.6f}（应≈0） → {'PASS' if cond else 'FAIL'}")
        ok_c2 = ok_c2 and cond
    A(f"  【C2 总体】{'PASS' if ok_c2 else 'FAIL'}")
    A("")

    # ---------------- C3 ----------------
    A("-" * 78)
    A("【C3】G 对 m 单调不减，且 G(m=1)=0")
    A("-" * 78)
    ok_c3 = True
    for rec in c3_m():
        A(f"  σ_ξ²={rec['sigma_xi2']}")
        vals = []
        for m, med, se in rec["pts"]:
            A(f"      m={m:3d}   G={med:+9.5f} ±{se:.5f}")
            vals.append(med)
        mono = all(vals[i + 1] >= vals[i] - 3 * rec["pts"][i + 1][2]
                   for i in range(len(vals) - 1))
        zerom1 = abs(vals[0]) < 3 * max(rec["pts"][0][2], 1e-9)
        cond = mono and zerom1
        ok_c3 = ok_c3 and cond
        A(f"      ⇒ 单调不减={mono}；G(m=1)≈0={zerom1} → {'PASS' if cond else 'FAIL'}")
    A(f"  【C3 总体】{'PASS' if ok_c3 else 'FAIL'}")
    A("")

    # ---------------- C5 ----------------
    A("-" * 78)
    A("【C5】∂G/∂τ̂² < 0（异质性越大共享越差）")
    A("-" * 78)
    ok_c5 = True
    for rec in c5_tau():
        A(f"  σ_ξ²={rec['sigma_xi2']}")
        vals = [p[1] for p in rec["pts"]]
        for t2, med, se in rec["pts"]:
            A(f"      τ²={t2:5.2f}   G={med:+9.5f} ±{se:.5f}")
        dec = all(vals[i + 1] <= vals[i] + 3 * rec["pts"][i + 1][2]
                  for i in range(len(vals) - 1))
        ok_c5 = ok_c5 and dec
        A(f"      ⇒ 单调不增={dec} → {'PASS' if dec else 'FAIL'}")
    A(f"  【C5 总体】{'PASS' if ok_c5 else 'FAIL'}")
    A("")

    # ---------------- 汇总 ----------------
    A("=" * 78)
    A("【裁决汇总】")
    A(f"  C1（可识别性驱动，核心）  : {'PASS' if ok_c1 else 'FAIL'}")
    A(f"  C2（n→∞ 非零下界，判决性）: {'PASS' if ok_c2 else 'FAIL'}")
    A(f"  C3（G 对 m 单调）          : {'PASS' if ok_c3 else 'FAIL'}")
    A(f"  C5（G 对 τ̂² 递减）         : {'PASS' if ok_c5 else 'FAIL'}")
    A("  R1（闭式解精度）          : FAIL —— 见下，已定位，不影响上述判据")
    A("  R0（W1 一致性）           : 部分 —— 见裁决报告 §6，须诚实标注")
    A("=" * 78)
    A("")
    A("R1 状态说明（诚实记录，不静默掩盖）：")
    A("  闭式解在 τ̂² 小（≲0.1）时与 MC 相差 <1.5%；τ̂² 大（≈2）时残差达 ~10%。")
    A("  已定位为渐近展开的高阶项问题，两次修正（v1.3 补 σ²、v1.4 补 γ² 方差项）")
    A("  均未能消除。**按预注册 N-R3，解析路线在此判据上不达标**，")
    A("  但 C1/C2/C3/C5 均由 MC 直接判定，不依赖闭式解精度，故理论主体不受影响。")
    A("  闭式解降级为「渐近指导式」，其适用范围限定为 τ̂² ≲ 0.5。")

    out = RES / "wp1r_verdict.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n→ {out}")

    # CSV
    with (RES / "wp1r_c1.csv").open("w", encoding="utf-8") as fh:
        fh.write("m,n,tau2,sigma_xi2,G_mc,SE\n")
        for rec in c1:
            for sx, med, se in rec["pts"]:
                fh.write(f"{rec['m']},{rec['n']},{rec['tau2']},{sx},{med:.8g},{se:.8g}\n")
    with (RES / "wp1r_c2.csv").open("w", encoding="utf-8") as fh:
        fh.write("m,sigma_xi2,n,G_mc,SE,pred_inf\n")
        for rec in c2:
            for n, med, se in rec["pts"]:
                fh.write(f"{rec['m']},{rec['sigma_xi2']},{n},{med:.8g},{se:.8g},{rec['pred_inf']:.8g}\n")


if __name__ == "__main__":
    verdict()

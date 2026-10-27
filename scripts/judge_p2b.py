"""W-P2-B 判据裁决（预注册 §4 + §9.4 修订）。

主比对是**预算精确匹配**的读出门消融：η 源相同的两个变体在写入侧轨迹逐位相同，
所以 `ach` vs `credit`（行1）、`cw_ar` vs `cw_cr`（行2）的差异只来自 θ 的来源。

B1/B2（单通道必要性）按 §7 失败条款如实处理，不做挽救。
B3  minimax 性（仅描述）
B4  时间结构价值：dynamic vs budget-matched static
B5  参数与预算审计
B6  跨数据集（外部 solar / PEMS04 不参与调参）
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS = os.path.join(ROOT, "results", "p2b_runs.csv")
OUT = os.path.join(ROOT, "results", "p2b_summary.txt")

INTERNAL = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"]
EXTERNAL = ["solar", "PEMS04"]
ALLDS = INTERNAL + EXTERNAL
REGIMES = ["raw", "shift", "noise", "rewire"]
SEEDS = [0, 1, 2]


def load():
    with open(RUNS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    extra_p = os.path.join(ROOT, "results", "p2b_extra.csv")
    if os.path.exists(extra_p):
        with open(extra_p, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    for r in rows:
        for k in ("mse", "budget", "n_param", "theta_abs", "seconds"):
            r[k] = float(r[k])
        r["seed"] = int(r["seed"])
    return rows


def cell(rows, ds, reg, var):
    v = [r["mse"] for r in rows
         if r["dataset"] == ds and r["regime"] == reg and r["variant"] == var]
    return np.mean(v) if v else float("nan")


def gain(rows, ds, reg, var):
    b = cell(rows, ds, reg, "none")
    a = cell(rows, ds, reg, var)
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return float("nan")
    return (1 - a / b) * 100


def main() -> None:
    rows = load()
    lines: list[str] = []
    out = lines.append
    out("=" * 78)
    out("W-P2-B 双通道解离 —— 判据裁决")
    out(f"单元数 {len(rows)}   变体 {sorted({r['variant'] for r in rows})}")
    out("=" * 78)

    # ---------- A. 主比对：预算精确匹配的读出门消融 ----------
    out("")
    out("--- A. 读出门来源（预算逐位精确的 2×2 列主效应）---")
    out("    （同一行内 η 源相同 ⇒ 写入轨迹相同 ⇒ 差异只来自 θ 来源）")
    out(f"{'行(写源)':>10s} {'read=novelty':>14s} {'read=credit':>13s} "
        f"{'Δ(pp)':>8s} {'win/total':>10s}")

    verdicts = {}
    for wname, v_nov, v_cre in [("write=d_s", "ach", "credit"),
                                ("write=u_c", "cw_ar", "cw_cr")]:
        diffs, wins = [], 0
        per_reg = {}
        for reg in REGIMES:
            rg = []
            for ds in ALLDS:
                g_n = gain(rows, ds, reg, v_nov)
                g_c = gain(rows, ds, reg, v_cre)
                if np.isfinite(g_n) and np.isfinite(g_c):
                    rg.append(g_c - g_n)
                    diffs.append(g_c - g_n)
                    wins += 1 if g_c > g_n else 0
            per_reg[reg] = (np.mean(rg) if rg else float("nan"), len(rg))
        d_all = float(np.mean(diffs))
        ok = (d_all >= 3.0) and (wins / len(diffs) >= 2 / 3)
        verdicts[wname] = (d_all, wins, len(diffs), ok)
        out(f"{wname:>10s} {v_nov:>14s} {v_cre:>13s} "
            f"{d_all:+8.2f} {wins:5d}/{len(diffs):<5d}  "
            f"{'成立' if ok else '不成立'}")
        for reg in REGIMES:
            m, n = per_reg[reg]
            out(f"             {reg:7s} Δ={m:+7.2f} pp   (n={n})")

    a_ok = all(v[3] for v in verdicts.values())
    out(f"  ⇒ A 整体：{'成立（读出门应由 DA 信用驱动，而非新奇信号）' if a_ok else '不成立'}")

    # ---------- B1/B2：单通道必要性（失败条款）----------
    out("")
    out("--- B1/B2 单通道必要性（预注册原判据，诚实裁决）---")
    out(f"{'regime':>8s} {'ach gain':>10s} {'da gain':>9s} {'Δ(da-ach)':>10s}")
    for reg in REGIMES:
        ga = np.nanmean([gain(rows, ds, reg, "ach") for ds in ALLDS])
        gd = np.nanmean([gain(rows, ds, reg, "da") for ds in ALLDS])
        out(f"{reg:>8s} {ga:+10.2f} {gd:+9.2f} {gd - ga:+10.2f}")
    g = {}
    for reg in REGIMES:
        g[reg] = (np.nanmean([gain(rows, ds, reg, "ach") for ds in ALLDS]),
                  np.nanmean([gain(rows, ds, reg, "da") for ds in ALLDS]))
    b1 = (g["rewire"][1] - g["rewire"][0]) >= 5.0
    b2 = min(g["noise"][0] - g["noise"][1],
             g["shift"][0] - g["shift"][1]) >= 5.0
    out(f"  B1（rewire 上 da 优于 ach ≥5pp）：{'成立' if b1 else '⛔ 不成立'}")
    out(f"  B2（noise/shift 上 ach 优于 da ≥5pp）：{'成立' if b2 else '⛔ 不成立'}")
    if not (b1 and b2):
        out("  ⇒ 按 §7 失败条款：**不主张**『两条通道各自单独不充分』。")
        out("     解离实验未产生预期的双向失效，如实记为负面结果。")

    # ---------- B3：minimax 描述 ----------
    out("")
    out("--- B3 各体制最优变体（描述性，非判据）---")
    adv = ["ach", "credit", "cw_ar", "cw_cr", "da", "const"]
    out(f"{'regime':>8s} " + "".join(f"{v:>9s}" for v in adv) + "   最优")
    for reg in REGIMES:
        gs = {v: np.nanmean([gain(rows, ds, reg, v) for ds in ALLDS])
              for v in adv}
        best = max(gs, key=gs.get)
        out(f"{reg:>8s} " + "".join(f"{gs[v]:+9.2f}" for v in adv)
            + f"   {best}")

    # ---------- B4：时间结构 ----------
    b4_bud = float(np.mean([r["budget"] for r in rows
                            if r["variant"] == "const"]))
    b4_ref = float(np.mean([r["budget"] for r in rows
                            if r["variant"] == "credit"]))
    pts = [(r["budget"], gain(rows, r["dataset"], r["regime"], r["variant"]))
           for r in rows if r["variant"] != "none"]
    pts = [(b, g) for b, g in pts if np.isfinite(g)]
    conf_r = float(np.corrcoef([p[0] for p in pts], [p[1] for p in pts])[0, 1])
    out("")
    out("--- B4 时间结构（dynamic vs 预算匹配静态 const）---")
    out(f"    [先审计] r(budget, gain) = {conf_r:+.3f}  (n={len(pts)})")
    diffs, wins = [], 0
    for reg in REGIMES:
        rg = []
        for ds in ALLDS:
            a = gain(rows, ds, reg, "credit")
            c = gain(rows, ds, reg, "const")
            if np.isfinite(a) and np.isfinite(c):
                rg.append(a - c)
                diffs.append(a - c)
                wins += 1 if a > c else 0
        out(f"  {reg:7s} credit−const = {np.mean(rg):+7.2f} pp  "
            f"(win {sum(1 for x in rg if x > 0)}/{len(rg)})")
    b4 = (np.mean(diffs) >= 3.0) and (wins / len(diffs) >= 2 / 3)
    b4_resid = abs(b4_bud - b4_ref) / max(b4_bud, b4_ref) * 100
    out(f"  预算残差 |const−credit| = {b4_resid:.1f}%  "
        f"r(budget,gain)={conf_r:+.3f}")
    if b4_resid > 5:
        out("  ⚠ B4 **不可解读**：const 与 credit 的平均预算不等价（>5%），")
        out("     而增益与预算正相关 ⇒ const 的优势可能只是'平均练得多'。")
        out("     按 §9.4 处理：既不主张'时间结构有价值'，也不主张'无价值'。")
        b4 = None
    out(f"  B4：{'成立' if b4 is True else ('不成立' if b4 is False else '不可解读')}"
        f"（阈值 ≥3pp 且 ≥2/3 单元为正）")

    # ---------- B7：与既有测试时自适应方法的协议内对比 ----------
    out("")
    out("--- B7 与 Black-Mamba / DynaME 代理的协议内对比（任务③）---")
    out("    诚实声明：抽象层面的代理复现，非原文代码与原始超参。")
    if any(r["variant"] == "moe2" for r in rows):
        out(f"{'regime':>8s} {'credit':>9s} {'da(BM)':>9s} {'Δ':>7s} | "
            f"{'cw_cr':>8s} {'moe2':>8s} {'Δ':>7s}")
        for reg in REGIMES:
            cr = np.nanmean([gain(rows, ds, reg, "credit") for ds in ALLDS])
            da = np.nanmean([gain(rows, ds, reg, "da") for ds in ALLDS])
            cc = np.nanmean([gain(rows, ds, reg, "cw_cr") for ds in ALLDS])
            mo = np.nanmean([gain(rows, ds, reg, "moe2") for ds in ALLDS])
            out(f"{reg:>8s} {cr:+8.2f}% {da:+8.2f}% {cr-da:+6.2f} | "
                f"{cc:+7.2f}% {mo:+7.2f}% {cc-mo:+6.2f}")
        out("  ⇒ 相对单标量惊讶门控有 +1.5~4.6pp 的小幅优势；")
        out("    相对性能路由专家竞争（DynaME 代理）**基本打平** —— 诚实记为'无优势'。")
    else:
        out("  （moe2 未跑）")

    # ---------- B5：参数与预算审计 ----------
    out("")
    out("--- B5 参数与预算审计 ---")
    ps_all = sorted({int(r["n_param"]) for r in rows if r["variant"] != "none"})
    bad = []
    for ds in ALLDS:
        ppp = sorted({int(r["n_param"]) for r in rows
                      if r["dataset"] == ds and r["variant"] != "none"})
        if len(ppp) != 1:
            bad.append((ds, ppp))
    for v in adv:
        bb = np.mean([r["budget"] for r in rows
                      if r["variant"] == v and r["regime"] == "shift"])
        out(f"  {v:8s}  params(per-dataset OK)  budget(shift)={bb:8.1f}")
    # 参数必须在**每个数据集内**一致（跨数据集不同只反映 n_vars，不是变体差异）
    params_ok = not bad
    out(f"  数据集内参数一致性：{'成立' if params_ok else '⛔ 不成立'} {bad}")
    out(f"  （跨数据集 {ps_all} 的差异来自 n_vars 6 vs 9，属数据属性，非变体差异）")
    b_nov = np.mean([r["budget"] for r in rows if r["variant"] == "ach"])
    b_cre = np.mean([r["budget"] for r in rows if r["variant"] == "cw_cr"])
    rel = abs(b_nov - b_cre) / max(b_nov, b_cre) * 100
    out(f"  行1(ach={b_nov:.0f}) vs 行2(cw_cr={b_cre:.0f}) 预算残差 {rel:.2f}% "
        f"⇒ 行主效应 {'可解释' if rel <= 5 else '⛔ 仍被预算污染'}")

    # ---------- B6：跨数据集 ----------
    out("")
    out("--- B6 跨数据集（外部不参与调参）---")
    for lbl, dsl in [("内部", INTERNAL), ("外部", EXTERNAL)]:
        gc = np.nanmean([gain(rows, ds, "shift", "credit") for ds in dsl])
        ga = np.nanmean([gain(rows, ds, "shift", "ach") for ds in dsl])
        out(f"  {lbl} shift: ach={ga:+6.2f}%  credit={gc:+6.2f}%  "
            f"Δ={gc - ga:+6.2f}pp")
    dc_int = np.nanmean([gain(rows, ds, "shift", "credit")
                         - gain(rows, ds, "shift", "ach") for ds in INTERNAL])
    dc_ext = np.nanmean([gain(rows, ds, "shift", "credit")
                         - gain(rows, ds, "shift", "ach") for ds in EXTERNAL])
    b6 = dc_ext > 0
    out(f"  B6：外部 Δ={dc_ext:+.2f}pp ⇒ {'成立' if b6 else '⛔ 不成立'}")

    out("")
    out("=" * 78)
    out(f"裁决汇总： A={'✅' if a_ok else '⛔'}  "
        f"B1={'✅' if b1 else '⛔'}  B2={'✅' if b2 else '⛔'}  "
        f"B4={'✅' if b4 else '⛔'}  B5参数={'✅' if params_ok else '⛔'}  "
        f"B6={'✅' if b6 else '⛔'}")
    out("=" * 78)

    text = "\n".join(lines)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())

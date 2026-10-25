"""W-P3-3 层 2 裁决：按预注册规格计算 H-L1 / H-L2 / H-L3 / H-L4。

输入：
  results/p33_raw/*.json          —— 官方 RAFT 复现的逐次 MSE
  results/p33_dres.csv            —— RAFT 环境上的 Δ_res
  results/p32_points.csv          —— 层 1 的 Δ_res（用于对照，非层 2 判据）

输出：
  results/p33_summary.txt
  results/p33_points.csv
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Paper-Neuro-STOG")
RAW = ROOT / "results" / "p33_raw"

# 论文 Table 8 (RAFT paper, ablation on retrieval module, multivariate, pred_len=96)
# RAFT 完整 vs Without Retrieval —— 这是层 2 的**外部对照**
PAPER_TABLE8 = {
    "ETTh1": dict(raft=0.367, without_retrieval=0.379),
    "ETTh2": dict(raft=0.276, without_retrieval=0.282),
    "ETTm1": dict(raft=0.302, without_retrieval=0.306),
    "ETTm2": dict(raft=0.164, without_retrieval=0.167),
    "weather": dict(raft=0.165, without_retrieval=0.182),
    "exchange_rate": dict(raft=0.091, without_retrieval=0.089),
}
# Table 16 主结果（= Table 8 的 RAFT 行）
PAPER_MAIN = {k: v["raft"] for k, v in PAPER_TABLE8.items()}

LOW_RES_THRESHOLD = 5.0   # 预注册 §3：|Δ_res| < 5% 为低分辨力区（层 2 未用到，
                          # 层 2 改用预注册 v1.2 H-L1′ 的中位数切分）


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_ci(x, y, n_boot=10000, seed=0, alpha=0.05):
    """逐点 bootstrap 的 Spearman ρ 的百分位 CI。"""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = spearman(x[idx], y[idx])
        if not math.isnan(r):
            vals.append(r)
    vals = np.array(vals)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def iqr(vals):
    v = np.asarray(vals, dtype=float)
    q1, q3 = np.percentile(v, [25, 75])
    return float(q3 - q1)


def main() -> None:
    # ---------- 读取复现结果 ----------
    runs = []
    for p in sorted(RAW.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "mse" in r:
            runs.append(r)

    if not runs:
        print("没有可用复现结果")
        return

    # ---------- 读取 RAFT 环境 Δ_res ----------
    dres = {}
    dres_path = ROOT / "results" / "p33_dres.csv"
    if dres_path.exists():
        for row in csv.DictReader(dres_path.open(encoding="utf-8")):
            dres[row["dataset"]] = float(row["delta_res"])

    # ---------- 逐数据集汇总 ----------
    per_ds = {}
    for r in runs:
        per_ds.setdefault(r["dataset"], {}).setdefault(r["arm"], {})[r["seed"]] = r

    lines = []
    def out(s=""):
        lines.append(s)
        print(s, flush=True)

    out("=" * 78)
    out("W-P3-3 层 2 复现裁决（RAFT 官方代码，ICML 2025）")
    out("=" * 78)

    # ---------- 矩阵完整性检查（预注册 v1.3 C.4：未完成的部分须如实标注） ----------
    expected = []
    for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "exchange_rate"]:
        for arm in ("on", "B"):
            for s in (0, 1, 2):
                expected.append((ds, arm, s))
    for ds in ("ETTh1", "exchange_rate"):
        for s in (0, 1):
            expected.append((ds, "A", s))
    have_keys = {(r["dataset"], r["arm"], r["seed"]) for r in runs}
    missing = [e for e in expected if e not in have_keys]
    out()
    out(f"【矩阵完整性】预注册 v1.3 矩阵 {len(expected)} 次，已完成 {len(have_keys & set(expected))} 次，"
        f"缺失 {len(missing)} 次")
    if missing:
        by_ds = {}
        for ds, arm, s in missing:
            by_ds.setdefault((ds, arm), []).append(s)
        for (ds, arm), seeds in sorted(by_ds.items()):
            out(f"    缺失: {ds:14s} arm={arm:2s} seeds={sorted(seeds)}")
        out("    ⇒ 裁决中缺失部分一律标注「未执行」，不得用已完成部分的结论暗示整体通过。")
    complete = not missing
    if complete:
        out("    ⇒ 矩阵完整，以下裁决为最终裁决。")

    out()
    out("【逐数据集逐档位实测 MSE】")
    out(f"{'dataset':14s} {'arm':4s} {'seeds':6s} {'median':>9s} {'IQR':>8s} "
        f"{'min':>8s} {'max':>8s}")
    summary_rows = []
    for ds in sorted(per_ds):
        for arm in ("on", "A", "B"):
            if arm not in per_ds[ds]:
                continue
            seeds = per_ds[ds][arm]
            mses = [seeds[s]["mse"] for s in sorted(seeds)]
            med = float(np.median(mses))
            iq = iqr(mses)
            out(f"{ds:14s} {arm:4s} {len(mses):<6d} {med:9.4f} {iq:8.4f} "
                f"{min(mses):8.4f} {max(mses):8.4f}")
            summary_rows.append(dict(dataset=ds, arm=arm, n_seeds=len(mses),
                                     median_mse=round(med, 6), iqr=round(iq, 6),
                                     min_mse=round(min(mses), 6),
                                     max_mse=round(max(mses), 6),
                                     seed_mses=";".join(f"{m:.6f}" for m in mses)))

    # ---------- Δ_mech：on 相对 A / B 的改善 ----------
    out()
    out("【Δ_mech：官方代码下机制带来的相对改善 %】")
    out("  Δ_mech(on vs B) = (MSE_B - MSE_on)/MSE_B * 100   （B = 只保留线性预测器，"
        "等价论文 Without Retrieval）")
    out("  Δ_mech(on vs A) = (MSE_A - MSE_on)/MSE_A * 100   （A = 结构不变但检索前向期置零）")
    out()
    out(f"{'dataset':14s} {'Δres%':>9s} {'res区':>8s} "
        f"{'dmechB':>9s} {'IQR_B':>8s} {'SNR_B':>7s} {'符号一致B':>9s} "
        f"{'dmechA':>9s} {'IQR_A':>8s} {'SNR_A':>7s} {'符号一致A':>9s} "
        f"{'best':>4s}")
    dmech_rows = []
    for ds in sorted(per_ds):
        if "on" not in per_ds[ds]:
            continue
        on_seeds = per_ds[ds]["on"]
        seeds_common = sorted(set(on_seeds))
        dr = dres.get(ds)
        # res_zone 稍后由 v1.2 的中位数切分统一决定，此处先留空
        row = dict(dataset=ds, delta_res=("" if dr is None else round(dr, 4)),
                   res_zone="")
        for arm, tag in (("B", "B"), ("A", "A")):
            if arm not in per_ds[ds]:
                row[f"dmech_{tag}"] = ""
                row[f"iqr_{tag}"] = ""
                row[f"snr_{tag}"] = ""
                row[f"sign_consistent_{tag}"] = ""
                continue
            arm_seeds = per_ds[ds][arm]
            common = [s for s in seeds_common if s in arm_seeds]
            if not common:
                continue
            dm = []
            for s in common:
                m_on, m_arm = on_seeds[s]["mse"], arm_seeds[s]["mse"]
                dm.append((m_arm - m_on) / m_arm * 100.0)
            med = float(np.median(dm))
            iq = iqr(dm)
            snr = abs(med) / iq if iq > 1e-12 else float("inf") if abs(med) > 0 else 0.0
            consistent = all(d > 0 for d in dm) or all(d < 0 for d in dm)
            row[f"dmech_{tag}"] = round(med, 4)
            row[f"iqr_{tag}"] = round(iq, 4)
            row[f"snr_{tag}"] = (round(snr, 3) if snr != float("inf") else "inf")
            row[f"sign_consistent_{tag}"] = int(consistent)
            row[f"dmech_{tag}_seedvals"] = ";".join(f"{d:.4f}" for d in dm)
        # best arm
        meds = {}
        for arm in ("on", "A", "B"):
            if arm in per_ds[ds]:
                meds[arm] = float(np.median([per_ds[ds][arm][s]["mse"]
                                             for s in per_ds[ds][arm]]))
        row["best_arm"] = min(meds, key=meds.get) if meds else ""
        dmech_rows.append(row)

    # ---------- 分组：按层 2 的 |Δ_res| 排序中位数切分（预注册 v1.2 H-L1′） ----------
    out()
    out("【分组：按 |Δ_res| 排序中位数切分（预注册 v1.2 H-L1′）】")
    out("  注：层 1（W-P3-2, lookback=24/单步/9通道）下 ETTh2/ETTm2/weather 为无分辨力；")
    out("      层 2（RAFT 官方环境, lookback=720/96步/全通道）下三者均为高分辨力。")
    out("      ⇒ 分辨力是环境的性质，不是数据集的固有属性。")
    have = [r for r in dmech_rows
            if r.get("dmech_B") not in ("", None)
            and r.get("delta_res") not in ("", None)]
    ordered = sorted(have, key=lambda r: abs(float(r["delta_res"])))
    mid = len(ordered) // 2
    hi_set = {r["dataset"] for r in ordered[mid:]}
    lo_set = {r["dataset"] for r in ordered[:mid]}
    for r in dmech_rows:
        if r.get("delta_res") in ("", None):
            continue
        if r["dataset"] in hi_set:
            r["res_zone"] = "高"
        elif r["dataset"] in lo_set:
            r["res_zone"] = "低"
        else:
            # 有 Δ_res 但缺 Δ_mech（未执行）：不归入任何组，避免被误判为"低"
            r["res_zone"] = ""
    out("  |Δ_res| 排序: " + " < ".join(
        f"{r['dataset']}({abs(float(r['delta_res'])):.1f})" for r in ordered))
    out(f"  高组 = {sorted(hi_set)}")
    out(f"  低组 = {sorted(lo_set)}")

    # ---------- 逐数据集 Δ_mech 汇总表（分组后打印，res_zone 已确定） ----------
    out()
    out("【逐数据集 Δ_mech 明细】")
    out(f"{'dataset':14s} {'Δres%':>9s} {'res区':>8s} "
        f"{'dmechB':>9s} {'IQR_B':>8s} {'SNR_B':>7s} {'符号一致B':>9s} "
        f"{'dmechA':>9s} {'IQR_A':>8s} {'SNR_A':>7s} {'符号一致A':>9s} "
        f"{'best':>4s}")
    for row in dmech_rows:
        def g(k, w=9, p=4):
            v = row.get(k, "")
            return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{str(v):>{w}}"
        out(f"{row['dataset']:14s} {g('delta_res',9,2):>9s} {row.get('res_zone',''):>8s} "
            f"{g('dmech_B'):>9s} {g('iqr_B',8)} {g('snr_B',7,3):>7s} "
            f"{g('sign_consistent_B',9,0):>9s} "
            f"{g('dmech_A'):>9s} {g('iqr_A',8)} {g('snr_A',7,3):>7s} "
            f"{g('sign_consistent_A',9,0):>9s} "
            f"{str(row.get('best_arm','')):>4s}")

    # ---------- H-L4：复现质量 ----------
    out()
    out("【H-L4 复现质量：官方代码实测 vs 论文报告】")
    out(f"{'dataset':14s} {'论文RAFT':>9s} {'实测on':>9s} {'相对差%':>9s} "
        f"{'论文WoRetr':>11s} {'实测B':>9s} {'相对差%':>9s} {'判定':>8s}")
    hl4_rows = []
    for ds in sorted(per_ds):
        if ds not in PAPER_TABLE8:
            continue
        p_raft = PAPER_TABLE8[ds]["raft"]
        p_wo = PAPER_TABLE8[ds]["without_retrieval"]
        med_on = float(np.median([per_ds[ds]["on"][s]["mse"]
                                  for s in per_ds[ds]["on"]])) if "on" in per_ds[ds] else None
        med_b = float(np.median([per_ds[ds]["B"][s]["mse"]
                                 for s in per_ds[ds]["B"]])) if "B" in per_ds[ds] else None
        d_on = (med_on - p_raft) / p_raft * 100 if med_on else float("nan")
        d_b = (med_b - p_wo) / p_wo * 100 if med_b else float("nan")
        ok = (not math.isnan(d_on)) and abs(d_on) <= 5.0
        out(f"{ds:14s} {p_raft:9.3f} "
            f"{(f'{med_on:9.4f}' if med_on else '     —   ')} "
            f"{(f'{d_on:+9.2f}' if med_on else '      —   ')} "
            f"{p_wo:11.3f} "
            f"{(f'{med_b:9.4f}' if med_b else '     —   ')} "
            f"{(f'{d_b:+9.2f}' if med_b else '      —   ')} "
            f"{('PASS' if ok else 'FAIL'):>8s}")
        hl4_rows.append(dict(dataset=ds, paper_raft=p_raft, measured_on=med_on,
                             rel_diff_on=round(d_on, 3) if med_on else "",
                             paper_without=p_wo, measured_b=med_b,
                             rel_diff_b=round(d_b, 3) if med_b else "",
                             verdict="PASS" if ok else "FAIL"))

    # ---------- H-L1：跨环境稳健性 ----------
    out()
    out("【H-L1′ 判决性判据：机制增益在高/低分辨力环境间的稳健性差异】")
    out("  口径：稳健 = 符号在全部种子上完全一致 且 SNR = |median|/IQR > 1")
    hl1_pass = hl1_pass_A = False
    hl1_gap = hl1_gap_A = float("nan")
    hl1_detail = None
    for tag in ("B", "A"):
        hi_ok = hi_n = lo_ok = lo_n = 0
        for row in dmech_rows:
            if row.get("res_zone") not in ("高", "低"):
                continue
            d = row.get(f"dmech_{tag}")
            snr = row.get(f"snr_{tag}")
            cons = row.get(f"sign_consistent_{tag}")
            if d == "" or snr == "":
                continue
            good = bool(cons) and (snr == "inf" or float(snr) > 1.0)
            if row["res_zone"] == "高":
                hi_n += 1
                hi_ok += int(good)
            else:
                lo_n += 1
                lo_ok += int(good)
        gap = (hi_ok / hi_n - lo_ok / lo_n) * 100 if hi_n and lo_n else float("nan")
        passed = (hi_n >= 2 and lo_n >= 2 and not math.isnan(gap) and gap >= 40.0)
        out(f"  档 {tag}：高分辨力 {hi_ok}/{hi_n} 稳健，低分辨力 {lo_ok}/{lo_n} 稳健，"
            f"差 = {gap:+.1f} pp  →  {'通过' if passed else '未通过'}"
            f"（需各≥2 数据集且差≥40pp）")
        if tag == "B":
            hl1_gap, hl1_pass, hl1_detail = gap, passed, (hi_ok, hi_n, lo_ok, lo_n)
        else:
            hl1_gap_A, hl1_pass_A = gap, passed

    # ---------- 事后诊断（非预注册判据，不替代 H-L1′） ----------
    # 动机：H-L1′ 的"稳健"口径只问"符号是否一致且 SNR>1"，**不区分方向**。
    # 若某环境的机制增益"稳定地为负"（机制稳定有害），它会被计为"稳健"，
    # 从而低估高/低分辨力两组之间的差别。下面这一栏按**带符号方向**重算，
    # 用于诊断"增益的方向是否随 Δ_res 的方向/大小变化"。
    # **它不是预注册判据，不参与 H-L1′ 的通过/未通过判定。**
    out()
    out("【事后诊断（非判据）：带方向的稳健性】")
    out("  口径：正增益 = Δ_mech 的中位 > 0 且符号在全部种子上为正且 SNR>1；"
        "负增益 = 中位 < 0 且符号全为负且 SNR>1")
    for tag in ("B", "A"):
        hi_pos = hi_neg = hi_n2 = 0
        lo_pos = lo_neg = lo_n2 = 0
        for row in dmech_rows:
            if row.get("res_zone") not in ("高", "低"):
                continue
            d = row.get(f"dmech_{tag}")
            snr = row.get(f"snr_{tag}")
            cons = row.get(f"sign_consistent_{tag}")
            if d == "" or snr == "":
                continue
            s = float("inf") if snr == "inf" else float(snr)
            if not (bool(cons) and s > 1.0):
                continue
            if row["res_zone"] == "高":
                hi_n2 += 1
                if float(d) > 0:
                    hi_pos += 1
                else:
                    hi_neg += 1
            else:
                lo_n2 += 1
                if float(d) > 0:
                    lo_pos += 1
                else:
                    lo_neg += 1
        if hi_n2 or lo_n2:
            out(f"  档 {tag}：高组 正 {hi_pos} / 负 {hi_neg}（共 {hi_n2}）；"
                f"低组 正 {lo_pos} / 负 {lo_neg}（共 {lo_n2}）")
            # 方向一致性：高组正 & 低组负 → 增益方向随分组翻转
            flip = (hi_pos > 0 and lo_neg > 0 and hi_neg == 0 and lo_pos == 0)
            out(f"    方向翻转（高组全正、低组全负）：{'是' if flip else '否'}")

    # ---------- H-L2：ρ(Δ_res, Δ_mech)，带符号（预注册 v1.2） ----------
    out()
    out("【H-L2：Spearman ρ(Δ_res, Δ_mech) —— 用带符号 Δ_res】")
    hl2 = {}
    for tag in ("B", "A"):
        xs, ys = [], []
        for row in dmech_rows:
            if row.get(f"dmech_{tag}") == "" or row.get("delta_res") == "":
                continue
            xs.append(float(row["delta_res"]))
            ys.append(float(row[f"dmech_{tag}"]))
        if len(xs) < 3:
            out(f"  档 {tag}：样本不足（n={len(xs)}），无法计算")
            continue
        rho = spearman(xs, ys)
        lo, hi = boot_ci(xs, ys)
        passed = (rho > 0) and (lo > 0)
        out(f"  档 {tag}：n={len(xs)}  ρ = {rho:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]"
            f"  →  {'通过' if passed else '未通过'}")
        hl2[tag] = dict(rho=rho, ci_lo=lo, ci_hi=hi, n=len(xs), verdict="PASS" if passed else "FAIL")

    # 补充（非判据）：|Δ_res| 口径的 ρ。预注册 v1.2 明确 H-L2 必须用带符号 Δ_res，
    # 下面这行**不参与裁决**，仅用于诊断"符号是否驱动了相关"。
    out()
    out("  [补充诊断，非判据] 若改用 |Δ_res|：")
    for tag in ("B", "A"):
        xs, ys = [], []
        for row in dmech_rows:
            if row.get(f"dmech_{tag}") == "" or row.get("delta_res") == "":
                continue
            xs.append(abs(float(row["delta_res"])))
            ys.append(float(row[f"dmech_{tag}"]))
        if len(xs) < 3:
            continue
        rho_abs = spearman(xs, ys)
        lo_abs, hi_abs = boot_ci(xs, ys)
        out(f"    档 {tag}：ρ(|Δ_res|, Δ_mech) = {rho_abs:+.4f}  "
            f"95%CI [{lo_abs:+.4f}, {hi_abs:+.4f}]")

    # ---------- H-L3：层 1 的「无分辨力」判定是否被推翻 ----------
    out()
    out("【H-L3：层 1 判定的低分辨力数据集，官方代码的机制增益是否不可复现】")
    out("  层 1（W-P3-2，raw 体制数据集级中位）判定为无分辨力的：ETTh2 / ETTm2 / weather")
    LAYER1_LOW = {"ETTh2", "ETTm2", "weather"}
    hl3_rows = []
    hl3_good = 0
    hl3_pos = 0
    hl3_seen = 0
    for row in dmech_rows:
        if row["dataset"] not in LAYER1_LOW:
            continue
        d = row.get("dmech_B")
        snr = row.get("snr_B")
        cons = row.get("sign_consistent_B")
        if d == "" or snr == "":
            out(f"  {row['dataset']:14s} 未执行（缺 on/B 配对）")
            continue
        s = float(snr) if snr != "inf" else float("inf")
        stable = bool(cons) and s > 1.0
        positive = float(d) > 0
        hl3_seen += 1
        hl3_good += int(stable)
        hl3_pos += int(positive)
        out(f"  {row['dataset']:14s} Δmech(B) = {d:+7.3f}%  IQR = {row.get('iqr_B',0):6.3f}  "
            f"SNR = {str(snr):>7s}  符号一致 = {cons}  正增益 = {positive}  → "
            f"{'稳定且显著为正' if (stable and positive) else '未达稳定正增益'}")
        hl3_rows.append(dict(dataset=row["dataset"], dmech_B=d, snr_B=str(snr),
                              sign_consistent=cons, stable=stable,
                              positive=positive))
    # v1.2：通过条件 = 三者中至少 2 个"符号一致且 SNR>1"，且 Δ_mech 显著为正
    hl3_pass = (hl3_seen >= 2 and hl3_good >= 2 and hl3_pos >= 2)
    if hl3_seen < 2:
        out(f"  统计：仅 {hl3_seen}/3 个数据集已执行，样本不足 → "
            f"**未执行**（不构成失败；预注册 v1.3 C.4）")
    else:
        out(f"  统计：稳定 {hl3_good}/{hl3_seen}，正增益 {hl3_pos}/{hl3_seen}"
            f"  →  {'通过（三者在层 2 高分辨力环境下增益稳定且为正，与层 1 不矛盾）' if hl3_pass else '失败（须修改 P3 结论）'}")

    # ---------- 汇总判定 ----------
    out()
    out("=" * 78)
    out("【裁决汇总】")
    if not complete:
        out("  ⚠ 矩阵未跑完（缺失 %d 次）。以下判定只覆盖已执行部分，" % len(missing))
        out("    未执行部分不得被读作通过，也不得被读作失败。")
    hl1_label = "通过" if hl1_pass else "未通过"
    if hl1_detail is None or hl1_detail[1] == 0 or hl1_detail[3] == 0:
        hl1_label = "未执行（样本不足）"
    if hl1_detail:
        out(f"  H-L1（判决性，档 B）：{hl1_label}  "
            f"差 {hl1_gap:+.1f} pp  高 {hl1_detail[0]}/{hl1_detail[1]} / 低 {hl1_detail[2]}/{hl1_detail[3]}")
    else:
        out(f"  H-L1（判决性，档 B）：{hl1_label}")
    hl1A_label = "通过" if hl1_pass_A else "未通过"
    if hl1_gap_A != hl1_gap_A:  # NaN
        hl1A_label = "未执行（样本不足）"
    out(f"  H-L1（档 A）        ：{hl1A_label}  差 {hl1_gap_A:+.1f} pp")
    for tag in ("B", "A"):
        if tag in hl2:
            h = hl2[tag]
            out(f"  H-L2（档 {tag}）        ：{h['verdict']}  ρ={h['rho']:+.4f} "
                f"CI[{h['ci_lo']:+.4f}, {h['ci_hi']:+.4f}] n={h['n']}")
        else:
            out(f"  H-L2（档 {tag}）        ：未执行（配对样本不足）")
    out(f"  H-L3               ："
        f"{'通过（层 1 未被推翻）' if hl3_pass else ('失败（须修改 P3）' if hl3_seen >= 2 else '未执行（样本不足）')}")
    if hl4_rows:
        n_pass = sum(1 for r in hl4_rows if r["verdict"] == "PASS")
        out(f"  H-L4（复现质量）    ：{n_pass}/{len(hl4_rows)} 数据集与论文报告值相差 ≤5%")
    out("=" * 78)

    # ---------- 落盘 ----------
    (ROOT / "results" / "p33_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    # 预注册 §7 指定的文件名（别名，内容同上，保证产出清单可核）
    (ROOT / "results" / "p33_raft_summary.txt").write_text("\n".join(lines),
                                                          encoding="utf-8")
    if summary_rows:
        p = ROOT / "results" / "p33_points.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        # 预注册 §7 指定的文件名
        p2 = ROOT / "results" / "p33_raft_points.csv"
        with p2.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    if dmech_rows:
        keys = sorted({k for r in dmech_rows for k in r})
        p = ROOT / "results" / "p33_dmech.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(dmech_rows)
    print(f"\n写入 {ROOT / 'results' / 'p33_summary.txt'}")


if __name__ == "__main__":
    main()

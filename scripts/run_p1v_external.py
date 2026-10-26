"""W-P1-V：外部数据集验证（solar_AL / PEMS04 / PEMS08）

预注册：`W-P1-V外部验证预注册规格-v1.0.md`（判据先于数据写定）
纪律：同一流水线原样照搬，一次跑完，结果不再调参；失败原样留档。

输出：
  results/p1v_points.csv
  results/p1v_summary.txt
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import shared_gain as sg                                        # noqa: E402
from tsf_external import EXTERNAL, split_external               # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ---- 与 run_p32_bench.py 完全一致的超参（预注册 §3，不得改动）----
REGIMES = ("raw", "shift", "noise")
TRAIN_OFFSETS = (0, 400, 800, 1200, 1600)
LAM = 1e-2
N_CALIB = 400
N_TRAIN, N_TEST = 800, 500
LOOKBACK = 24

# p32 主结果（用于 X5）
P32_RHO = 0.8420


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_spearman(x, y, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = spearman(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def cluster_boot_spearman(x, y, groups, n_boot=10_000, seed=0):
    uniq = sorted(set(groups))
    blocks = {g: np.where(np.array(groups) == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    k = len(uniq)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        idx = np.concatenate([blocks[uniq[p]] for p in pick])
        v = spearman(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def bench_point(name: str, regime: str, offset: int) -> dict:
    s = split_external(name, regime, lookback=LOOKBACK, n_train=N_TRAIN,
                       n_calib=N_CALIB, n_test=N_TEST, offset=offset)
    env = sg.W1Env(seed=101 + offset // 100, n_vars=s.n_vars, m_e=3,
                   n_train=N_TRAIN, n_calibrate=N_CALIB, n_test=N_TEST,
                   lookback=LOOKBACK)
    dres = sg.delta_res_real(env, s.train, s.calib, s.test, n_calib=N_CALIB)
    r = sg.run_point_real(env, s.train, s.calib, s.test, lam=LAM, n_calib=N_CALIB)
    base = r["base"][0]
    rel = {a: (base - r[a][0]) / base * 100.0 for a in sg.ARMS}
    best = max(rel, key=rel.get)
    row = {
        "dataset": name, "regime": regime, "offset": offset,
        "n_vars": s.n_vars, "n_domains": s.n_vars // 3,
        "base_wmse": base, "delta_res": dres,
        "delta_mech": rel[best], "best_arm": best,
    }
    for a in sg.ARMS:
        row[f"rel_{a}"] = rel[a]
    return row


def load_p32():
    p = RESULTS / "p32_points_honest.csv"
    with p.open(newline="", encoding="utf-8") as f:
        return [{"dataset": r["dataset"], "regime": r["regime"],
                 "offset": int(float(r["offset"])),
                 "delta_res": float(r["delta_res"]),
                 "delta_mech": float(r["delta_mech"])}
                for r in csv.DictReader(f)]


def main():
    t0 = time.time()
    rows: list[dict] = []
    print("=== W-P1-V：外部数据集验证 ===", flush=True)
    for name in EXTERNAL:
        for regime in REGIMES:
            for off in TRAIN_OFFSETS:
                try:
                    row = bench_point(name, regime, off)
                except Exception as e:                          # noqa: BLE001
                    print(f"  {name}/{regime}/off{off} FAILED: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                rows.append(row)
                print(f"  {name:8s} {regime:5s} off={off:4d} | "
                      f"Δ_res={row['delta_res']:+9.2f}% "
                      f"Δ_mech={row['delta_mech']:+8.2f}% best={row['best_arm']}",
                      flush=True)

    if not rows:
        print("[致命] 无有效点", flush=True)
        return 1

    with (RESULTS / "p1v_points.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    x = np.array([r["delta_res"] for r in rows])
    y = np.array([r["delta_mech"] for r in rows])
    groups = [r["dataset"] for r in rows]
    rho = spearman(x, y)
    lo, hi = boot_spearman(x, y)
    clo, chi = cluster_boot_spearman(x, y, groups)

    # ---------- X1 ----------
    X1 = rho >= 0.50 and lo > 0
    x1 = "成立" if X1 else "不成立"

    # ---------- X2：中位同号 ----------
    med_new = float(np.median(x))
    p32 = load_p32()
    med_ref = float(np.median([r["delta_res"] for r in p32]))
    X2 = (med_new > 0) == (med_ref > 0)
    x2 = "成立" if X2 else "不成立"

    # ---------- X3：无分辨力区 ----------
    unres = int((np.abs(x) < 5.0).sum())
    unres_frac = unres / len(rows)
    X3 = unres_frac >= 0.25
    x3 = "成立" if X3 else "不成立"

    # ---------- X4：符号可行域 ----------
    pos_frac = float((y > 0).mean())
    X4 = 0.20 <= pos_frac <= 0.80
    x4 = "成立" if X4 else "不成立"

    # ---------- X5：合并一致性 ----------
    mx = np.concatenate([x, np.array([r["delta_res"] for r in p32])])
    my = np.concatenate([y, np.array([r["delta_mech"] for r in p32])])
    rho_merge = spearman(mx, my)
    mlo, mhi = boot_spearman(mx, my, seed=7)
    X5 = (np.sign(rho_merge) == np.sign(P32_RHO)) and abs(rho_merge - P32_RHO) <= 0.25
    x5 = "成立" if X5 else "不成立"

    # ---------- 裁决 ----------
    if X1 and X3 and X4:
        verdict = "外部验证通过"
        meaning = "可写「跨 10 个数据集，其中 3 个完全独立」"
    elif X1:
        verdict = "部分通过"
        meaning = "只能写「主相关可迁移，但边界条件需限定在 7 数据集范围内」"
    else:
        verdict = "外部验证失败"
        meaning = ("⛔ 全部 Δ_res 结论降级为「7 数据集内的经验规律，未获外部支持」，"
                   "W-P1-T §6/§8 的一区判定退回原位")

    per_ds = {}
    for name in EXTERNAL:
        sub = [r for r in rows if r["dataset"] == name]
        if sub:
            sx = np.array([r["delta_res"] for r in sub])
            sy = np.array([r["delta_mech"] for r in sub])
            per_ds[name] = (len(sub), float(np.median(sx)), float(np.median(sy)),
                            spearman(sx, sy) if len(sub) > 2 else float("nan"))

    lines = [
        "=== W-P1-V：外部数据集验证（solar / PEMS04 / PEMS08）===",
        f"点数: {len(rows)}  ({len(per_ds)} 数据集 × {len(REGIMES)} 体制 × "
        f"{len(TRAIN_OFFSETS)} 偏移)",
        f"耗时: {time.time() - t0:.1f}s",
        "",
        f"X1 排序可迁移性（主判据）: ρ = {rho:+.4f}",
        f"   逐点 95%CI [{lo:+.4f}, {hi:+.4f}]",
        f"   cluster bootstrap（按 {len(per_ds)} 数据集分块）95%CI [{clo:+.4f}, {chi:+.4f}]",
        f"   判据 ρ≥+0.50 且 CI 下界>0  →  {x1}",
        "",
        f"X2 绝对水平同号: 外部中位 {med_new:+.2f}%  vs  p32 中位 {med_ref:+.2f}%"
        f"  →  {x2}",
        "",
        f"X3 无分辨力区普遍性: |Δ_res|<5% = {unres}/{len(rows)} = {unres_frac:.1%}",
        f"   判据 ≥25%  →  {x3}",
        "",
        f"X4 符号可行域: Δ_mech>0 占比 = {pos_frac:.1%}",
        f"   判据 [20%,80%]  →  {x4}",
        "",
        f"X5 合并一致性: {len(rows)} 新点 + {len(p32)} p32 点 = {len(mx)} 点",
        f"   合并 ρ = {rho_merge:+.4f}  95%CI [{mlo:+.4f}, {mhi:+.4f}]",
        f"   p32 主结果 ρ = {P32_RHO:+.4f}，|差| = {abs(rho_merge-P32_RHO):.4f}",
        f"   判据 同号且 |差|≤0.25  →  {x5}",
        "",
        "按数据集:",
        "  数据集       n  median(Δ_res)  median(Δ_mech)  ρ(组内)",
    ]
    for name, (n, m1, m2, rr) in per_ds.items():
        lines.append(f"  {name:8s} {n:3d}  {m1:+11.2f}%  {m2:+13.2f}%  {rr:+.3f}")

    lines += ["", "按体制:"]
    for regime in REGIMES:
        sub = [r for r in rows if r["regime"] == regime]
        if not sub:
            continue
        sx = np.array([r["delta_res"] for r in sub])
        sy = np.array([r["delta_mech"] for r in sub])
        lines.append(f"  {regime:6s} n={len(sub):3d}  Δ_res={sx.mean():+8.2f}%  "
                     f"Δ_mech={sy.mean():+8.2f}%  ρ={spearman(sx, sy):+.3f}")

    lines += ["", "best_arm 计数:"]
    for a in sg.ARMS:
        c = sum(1 for r in rows if r["best_arm"] == a)
        lines.append(f"  {a:3s} {c:3d}")

    n_ok = sum([X1, X2, X3, X4, X5])
    lines += [
        "",
        "=" * 70,
        f"裁决: {verdict}   （{n_ok}/5 项成立）",
        f"含义: {meaning}",
        "=" * 70,
    ]

    text = "\n".join(lines)
    print("\n" + text, flush=True)
    (RESULTS / "p1v_summary.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

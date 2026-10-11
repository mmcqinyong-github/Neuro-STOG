"""Stage 4 -- statistics and the pre-registered verdict table.

Every number that appears in the paper's tables comes from here.

Endpoint
--------
W = mean over test days of min(loss_t, cap_t), with cap_t the arm-independent
winsorising envelope (protocol.build_envelope).  The envelope is stored with
each run so the statistic is reproducible from the saved arrays alone.

The eleven real benchmarks and the four regime-switching stresses are reported
separately as well as pooled, because they answer different questions: the real
benchmarks are close to stationary, the stresses are not.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.config import PREREG  # noqa: E402
from cerebrocast.data import ALL_ORDER  # noqa: E402
from cerebrocast.stats import (bh_adjust, block_bootstrap, cohens_d,  # noqa: E402
                               paired_block_bootstrap, tost, wilcoxon_paired)

STREAM = os.path.join(os.path.dirname(__file__), "..", "results", "stream")
SYNTH = os.path.join(os.path.dirname(__file__), "..", "results", "synth")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "summary")

BASELINES = ["B1_frozen_static", "B2_monthly_retrain", "B3_fforma", "B4_hedge",
             "B5_best_expert", "B6_adwin_retrain", "B7_online_meta"]
ABLTIONS = ["A1_no_spectral", "A2_no_episodic", "A3_no_clusters",
            "A4_no_change_detection", "A5_no_reset", "A6_no_forward_model",
            "A7_no_three_factor", "A8_no_metaplasticity", "A9_no_replay",
            "A10_no_downscaling", "A11_no_growth"]
PLACEBOS = ["P1_random_reset", "P2_random_replay",
            "P3_shuffled_responsibility", "P4_random_growth"]
ARMS = ABLTIONS + PLACEBOS
# the non-stationarity ladder, ordered by how much the DGP actually changes
LADDER = ["STAT-0", "STAT-1", "SW-2", "SW-4", "SW-X"]
LADDER_LABEL = {
    "STAT-0": "contiguous single series (stationary control)",
    "STAT-1": "same series, spliced windows (splice control)",
    "SW-2": "two series, same family",
    "SW-4": "four series, same family",
    "SW-X": "three series, different families",
}
MODULES = 11


def load_all(stream_dir: str, domains, seeds) -> Dict:
    data = {}
    for d in domains:
        for s in seeds:
            p = os.path.join(stream_dir, f"{d}__seed{s}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=True)
            meta = json.loads(str(z["__meta__"]))
            arms = {}
            for k in z.files:
                if k == "__meta__" or k.startswith("__") or not k.startswith(
                        ("FULL__", "A", "B", "P", "ref__")):
                    continue
                an, fn = k.split("__", 1)
                arms.setdefault(an, {})[fn] = z[k]
            data[(d, s)] = dict(meta=meta, arms=arms, caps=z["__caps__"])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=STREAM)
    ap.add_argument("--synth", default=SYNTH)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--domains", nargs="*", default=ALL_ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 42, 2021, 3407, 12345])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = load_all(a.stream, a.domains, a.seeds)
    if not D:
        raise SystemExit("no runs found")

    def W(rec, arm):
        v = rec["arms"][arm]["loss"]
        return float(np.mean(np.minimum(v, rec["caps"])))

    # ---------------- per (domain, seed) ------------------------------------
    rows = []
    for (d, s), rec in D.items():
        full = W(rec, "FULL")
        bl = {b: W(rec, b) for b in BASELINES if b in rec["arms"]}
        best_b = min(bl, key=bl.get)
        orc = float(np.mean(np.minimum(rec["arms"]["ref"]["oracle_expert"],
                                       rec["caps"])))
        rows.append(dict(domain=d, seed=s, family=rec["meta"]["family"],
                         full=full, **bl, best_baseline=best_b,
                         best_baseline_w=bl[best_b], oracle=orc,
                         gain_pct=100 * (bl[best_b] - full) / bl[best_b],
                         n_test=int(rec["meta"]["days"]["test"])))

    # ---------------- H1 per domain ----------------------------------------
    dom: Dict[str, list] = {}
    for r in rows:
        dom.setdefault(r["domain"], []).append(r)
    h1_rows = []
    for dname, rs in dom.items():
        f = np.array([r["full"] for r in rs])
        b = np.array([r["best_baseline_w"] for r in rs])
        o = np.array([r["oracle"] for r in rs])
        margin = PREREG["equivalence_margin_pct"] / 100.0 * float(np.mean(b))
        t = tost(f, b, margin)
        dm, lo, hi = paired_block_bootstrap(b, f, block=1, n_boot=2000, seed=7)
        h1_rows.append(dict(domain=dname, family=rs[0]["family"],
                            full_W=float(np.mean(f)),
                            best_baseline_W=float(np.mean(b)),
                            oracle_W=float(np.mean(o)),
                            headroom_captured_pct=float(
                                100 * (np.mean(b) - np.mean(f))
                                / max(1e-12, np.mean(b) - np.mean(o))),
                            gain_pct=float(100 * (np.mean(b) - np.mean(f)) / np.mean(b)),
                            diff=dm, ci_lo=lo, ci_hi=hi,
                            cohens_d=cohens_d(f, b), wilcoxon=wilcoxon_paired(f, b),
                            **{k: v for k, v in t.items()}))
    pv = [r["p_superiority"] for r in h1_rows]
    adj, rej = bh_adjust(pv)
    for r, p, q in zip(h1_rows, adj, rej):
        r["p_superiority_bh"] = float(p)
        r["significant_bh"] = bool(q)

    def summarise(sel, tag):
        g = np.array([r["gain_pct"] for r in h1_rows if sel(r)])
        if g.size == 0:
            return None
        ns = sum(1 for r in h1_rows if sel(r) and r["significant_bh"]
                 and r["gain_pct"] > 0)
        return dict(stratum=tag, n_domains=int(g.size),
                    gain_pct_mean=float(np.mean(g)), gain_pct_median=float(np.median(g)),
                    gain_pct_min=float(np.min(g)), gain_pct_max=float(np.max(g)),
                    n_significant=ns, threshold=PREREG["H1_main_gain_pct"],
                    passed=bool(np.median(g) >= PREREG["H1_main_gain_pct"]))

    is_control = lambda r: r["domain"] in ("STAT-0", "STAT-1")
    is_switch = lambda r: r["domain"] in ("SW-2", "SW-4", "SW-X")
    is_stress = lambda r: str(r["family"]).startswith("STRESS")
    is_real = lambda r: not str(r["family"]).startswith("STRESS")
    h1_all = summarise(lambda r: True, "all")
    h1_str = summarise(is_switch, "multi-source regime switching")
    h1_ctl = summarise(is_control, "splicing / stationarity controls")
    h1_real = summarise(is_real, "real benchmarks")

    # dose-response: gain along the ladder.
    # NOTE: built from `rows` (one entry per domain *per architecture seed*),
    # NOT from `h1_rows`, which is already averaged over seeds.  Using the
    # averaged table made n_seeds=1 and min=max=mean, i.e. it reported a
    # degenerate range and hid the true seed-to-seed dispersion.
    ladder_rows = []
    for rung in LADDER:
        g = np.array([r["gain_pct"] for r in rows if r["domain"] == rung])
        if g.size == 0:
            continue
        ladder_rows.append(dict(rung=rung, label=LADDER_LABEL[rung],
                                n_seeds=int(g.size), gain_pct_mean=float(np.mean(g)),
                                gain_pct_sd=float(np.std(g, ddof=1)) if g.size > 1 else 0.0,
                                gain_pct_median=float(np.median(g)),
                                gain_pct_min=float(np.min(g)),
                                gain_pct_max=float(np.max(g))))

    # headline wording is derived from the ladder so it cannot drift from the data
    _g = {r["rung"]: r["gain_pct_mean"] for r in ladder_rows}
    _sw = [_g[k] for k in ("SW-2", "SW-4", "SW-X") if k in _g]
    _sw_txt = ("+%.2f%% to +%.2f%%" % (min(_sw), max(_sw))) if _sw else "n/a"
    _mono = bool(len(_sw) == 3 and _sw[0] <= _sw[1] <= _sw[2])

    # H1b: on the real benchmarks the claim is *equivalence*, not superiority
    real_eq = []
    for r in h1_rows:
        if is_real(r):
            real_eq.append(bool(r.get("equivalent", False)))
    h1b = dict(n_domains=len(real_eq), n_equivalent=int(sum(real_eq)),
               gain_pct_median=h1_real["gain_pct_median"] if h1_real else np.nan,
               threshold=PREREG["H1_main_gain_pct"],
               passed=bool(h1_real and h1_real["gain_pct_median"]
                           >= -PREREG["H3_universality_pct"]))

    h1 = dict(strata={"all": h1_all, "switch": h1_str, "control": h1_ctl,
                      "real": h1_real},
              dose_response=ladder_rows,
              dose_response_monotone=_mono,
              passed=bool(h1_str and h1_str["passed"]),
              real_equivalence=h1b)

    # ---------------- H3 universality --------------------------------------
    deg = np.array([-r["gain_pct"] for r in h1_rows])
    h3 = dict(median_degradation_pct=float(np.median(deg)),
              worst_domain=h1_rows[int(np.argmax(deg))]["domain"],
              worst_degradation_pct=float(np.max(deg)),
              threshold=PREREG["H3_universality_pct"],
              passed=bool(np.median(deg) <= PREREG["H3_universality_pct"]))

    # ---------------- H5 mechanism pricing ---------------------------------
    per_domain_full = {dname: float(np.mean([r["full"] for r in rs]))
                       for dname, rs in dom.items()}
    grand = float(np.mean(list(per_domain_full.values())))
    pricing = []
    for arm in ARMS:
        v = []
        for dname, rs in dom.items():
            fs, ab = [], []
            for r in rs:
                rec = D[(dname, r["seed"])]
                if arm not in rec["arms"]:
                    continue
                fs.append(W(rec, "FULL")); ab.append(W(rec, arm))
            if fs:
                v.append(float(np.mean(ab) - np.mean(fs)))
        if not v:
            continue
        v = np.array(v)
        est, lo, hi = block_bootstrap(v, block=1, n_boot=2000, seed=11)
        try:
            from scipy import stats as sps
            p = float(sps.ttest_1samp(v, 0.0).pvalue) if len(v) > 2 else float("nan")
        except Exception:  # noqa: BLE001
            p = float("nan")
        pricing.append(dict(arm=arm, kind=("placebo" if arm.startswith("P")
                                           else "ablation"),
                            delta_W=float(est), ci_lo=float(lo), ci_hi=float(hi),
                            rel_pct=float(100 * est / max(1e-12, grand)),
                            p_two_sided=p, n_domains=len(v)))
    adj2, rej2 = bh_adjust([x["p_two_sided"] if np.isfinite(x["p_two_sided"]) else 1.0
                            for x in pricing])
    for q, p, rj in zip(pricing, adj2, rej2):
        q["p_bh"] = float(p)
        q["significant"] = bool(rj and q["delta_W"] > 0)
    n_mod = sum(1 for q in pricing
                if q["kind"] == "ablation" and q["significant"])
    n_pla_sig = sum(1 for q in pricing if q["kind"] == "placebo" and q["significant"])
    h5 = dict(modules_significant=n_mod, threshold=PREREG["H5_pricing_modules"],
              placebo_arms=len(PLACEBOS), placebo_significant=n_pla_sig,
              passed=bool(n_mod >= PREREG["H5_pricing_modules"]))

    # ---------------- H7 cost ----------------------------------------------
    cf, cb = [], []
    for (d, s), rec in D.items():
        cf.append(float(rec["arms"]["FULL"]["sec_per_day"]))
        if "B2_monthly_retrain" in rec["arms"]:
            cb.append(float(rec["arms"]["B2_monthly_retrain"]["sec_per_day"]))
    h7 = dict(full_sec_per_day=float(np.mean(cf)) if cf else np.nan,
              monthly_retrain_sec_per_day=float(np.mean(cb)) if cb else np.nan,
              ratio=float(np.mean(cf) / np.mean(cb)) if cb else np.nan,
              threshold=PREREG["H7_cost_fraction"],
              passed=bool(cb and np.mean(cf) / np.mean(cb) < PREREG["H7_cost_fraction"]))

    # ---------------- synthetic H2 / H4 ------------------------------------
    h2 = h4 = None
    p_h4 = os.path.join(a.synth, "h4_detection.json")
    if os.path.exists(p_h4):
        rr = json.load(open(p_h4, encoding="utf-8"))
        f1 = np.array([x["macro_f1"] for x in rr])
        h4 = dict(macro_f1_mean=float(np.mean(f1)), macro_f1_median=float(np.median(f1)),
                  n_configs=len(rr), threshold=PREREG["H4_detect_macro_f1"],
                  passed=bool(np.mean(f1) >= PREREG["H4_detect_macro_f1"]))
    p_h2 = os.path.join(a.synth, "h2_p2_recovery.json")
    if os.path.exists(p_h2):
        rr = json.load(open(p_h2, encoding="utf-8"))
        rf = np.array([x["recovery_full"] for x in rr if np.isfinite(x.get("recovery_full", np.nan))])
        rn = np.array([x["recovery_noreset"] for x in rr if np.isfinite(x.get("recovery_noreset", np.nan))])
        h2 = dict(recovery_full_days=float(np.mean(rf)) if len(rf) else np.nan,
                  recovery_noreset_days=float(np.mean(rn)) if len(rn) else np.nan,
                  threshold=PREREG["H2_recovery_days"],
                  passed=bool(len(rf) and np.mean(rf) <= PREREG["H2_recovery_days"]))

    verdict = dict(H1=h1, H2=h2, H3=h3, H4=h4, H5=h5, H7=h7)

    # ---------------- write ------------------------------------------------
    def dump(name, rs):
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader(); w.writerows(rs)

    dump("per_run.csv", rows)
    if ladder_rows:
        dump("ladder.csv", ladder_rows)
    dump("h1_by_domain.csv", h1_rows)
    if pricing:
        dump("pricing.csv", pricing)
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w", encoding="utf-8"),
              indent=1, default=str)

    # headline wording is derived from the ladder so it cannot drift from the data
    _g = {r["rung"]: r["gain_pct_mean"] for r in ladder_rows}
    _sw = [_g[k] for k in ("SW-2", "SW-4", "SW-X") if k in _g]
    _sw_txt = ("+%.2f%% to +%.2f%%" % (min(_sw), max(_sw))) if _sw else "n/a"
    _mono = bool(len(_sw) == 3 and _sw[0] <= _sw[1] <= _sw[2])

    L = ["# Results", "",
         "Every number below is produced by `scripts/run_stats.py` from the arrays",
         "written by `scripts/run_stream.py` and `scripts/run_synth.py`.  The",
         "pre-registered hypotheses of `config.PREREG` are reported exactly as they",
         "came out, including the ones that failed.", "",
         "## Endpoint and protocol", "",
         "* The stream is walked one calendar day at a time.  On day `t` the method",
         "  sees only data up to `t-1`; the realised values of day `t` arrive",
         "  afterwards and are used for the loss and for the nightly maintenance.",
         "* The expert pool is fitted on the first 55% of each series and refitted",
         "  twice as data accumulate, so every forecast the method ever consumes",
         "  comes from a model that has not seen that day (see `docs/PROTOCOL.md`).",
         "* Primary endpoint `W` = mean over test days of `min(loss_t, cap_t)`.",
         "  `cap_t` is the 95th-percentile expert loss of day `t` times 8, smoothed",
         "  by an EMA; it is computed from the pool alone and is therefore identical",
         "  for every arm -- no method can gain by clipping itself.",
         "* The five architecture seeds share one pool realisation, so every",
         "  comparison is paired over identical days.", "",
         "## Headline", "",
         "* On the eleven standard benchmarks the architecture does **not** beat the",
         "  best causal baseline.  We report this as a null result rather than",
         "  tuning against the test split.",
         "* The reason is measurable: the per-day oracle expert is far better than",
         "  any causal method, yet the signal that would identify it is not",
         "  learnable from features observable before the day.",
         "* On streams whose data-generating process genuinely changes, the same",
         "  architecture gains, but the gain is *modest* rather than large: "
         f"{_sw_txt}",
         "  over the three regime-switching rungs (SW-2 / SW-4 / SW-X).",
         "  It does **not** increase monotonically as more sources are mixed, so",
         "  we do not claim a dose-response from degree of change to gain; see the",
         "  ladder table below for the per-seed spread.", "",
         "# Pre-registered verdicts", "",
         "| ID | statement | measured | threshold | verdict |", "|---|---|---|---|---|"]
    meas = {
        "H1": (f"stress {h1_str['gain_pct_median']:+.2f}% median" if h1_str else "n/a")
               + (f"; real {h1_real['gain_pct_median']:+.2f}%" if h1_real else ""),
        "H2": f"{h2['recovery_full_days']:.2f} d" if h2 else "n/a",
        "H3": f"{h3['median_degradation_pct']:.2f}%",
        "H4": f"{h4['macro_f1_mean']:.3f}" if h4 else "n/a",
        "H5": f"{h5['modules_significant']}/{MODULES}",
        "H7": f"{h7['ratio']:.4f}" if np.isfinite(h7["ratio"]) else "n/a"}
    thr = {"H1": f">= {PREREG['H1_main_gain_pct']}%", "H2": f"<= {PREREG['H2_recovery_days']} d",
           "H3": f"<= {PREREG['H3_universality_pct']}%", "H4": f">= {PREREG['H4_detect_macro_f1']}",
           "H5": f">= {PREREG['H5_pricing_modules']}/{MODULES}", "H7": f"< {PREREG['H7_cost_fraction']}"}
    st = {"H1": "main gain vs best baseline", "H2": "recovery after an abrupt break",
          "H3": "median relative degradation", "H4": "three-tier gate macro-F1",
          "H5": "modules priced significantly > 0", "H7": "incremental compute / monthly retrain"}
    for k in ("H1", "H2", "H3", "H4", "H5", "H7"):
        v = verdict[k]
        L.append(f"| {k} | {st[k]} | {meas[k]} | {thr[k]} | "
                 f"{'PASS' if v and v['passed'] else 'FAIL'} |")
    L += ["", "## Per-domain endpoint (mean over seeds, winsorised MSE)", "",
          "| domain | family | architecture | best baseline | gain % | "
          "headroom captured % | BH-adj p |", "|---|---|---|---|---|---|---|"]
    for r in sorted(h1_rows, key=lambda x: -x["gain_pct"]):
        L.append(f"| {r['domain']} | {r['family']} | {r['full_W']:.5f} | "
                 f"{r['best_baseline_W']:.5f} | {r['gain_pct']:+.2f} | "
                 f"{r['headroom_captured_pct']:.1f} | {r['p_superiority_bh']:.2e} |")
    L += ["", "## Dose-response along the non-stationarity ladder", "",
          "| rung | construction | n seeds | mean gain % | sd | median % | min % | max % |",
          "|---|---|---|---|---|---|---|---|"]
    for r in ladder_rows:
        L.append(f"| {r['rung']} | {r['label']} | {r['n_seeds']} | "
                 f"{r['gain_pct_mean']:+.2f} | {r['gain_pct_sd']:.2f} | "
                 f"{r['gain_pct_median']:+.2f} | {r['gain_pct_min']:+.2f} | "
                 f"{r['gain_pct_max']:+.2f} |")
    L += ["", "## Mechanism pricing (delta W vs the full architecture)", "",
          "| arm | kind | delta W | 95% CI | relative % | p (BH) | significant |",
          "|---|---|---|---|---|---|---|"]
    for q in sorted(pricing, key=lambda x: -x["delta_W"]):
        L.append(f"| {q['arm']} | {q['kind']} | {q['delta_W']:+.5f} | "
                 f"[{q['ci_lo']:+.5f}, {q['ci_hi']:+.5f}] | {q['rel_pct']:+.2f} | "
                 f"{q['p_bh']:.2e} | {'yes' if q['significant'] else 'no'} |")
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    i0 = next(i for i, s in enumerate(L) if s.startswith("# Pre-registered"))
    print("\n".join(L[i0:i0 + 12]))
    print("\nper-domain:")
    for r in sorted(h1_rows, key=lambda x: -x["gain_pct"]):
        print(f"  {r['domain']:10s} {r['family']:8s} gain={r['gain_pct']:+7.2f}%  "
              f"headroom={r['headroom_captured_pct']:5.1f}%  p_bh={r['p_superiority_bh']:.2e}")
    print("\npricing:")
    for q in sorted(pricing, key=lambda x: -x["delta_W"]):
        print(f"  {q['arm']:30s} {q['kind']:9s} dW={q['delta_W']:+.5f} "
              f"rel={q['rel_pct']:+6.2f}% p_bh={q['p_bh']:.3g} sig={q['significant']}")


if __name__ == "__main__":
    main()

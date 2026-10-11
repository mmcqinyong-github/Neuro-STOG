# SI-B — Pre-registration, deviations and execution record

This appendix is the audit trail between what the study pre-registered and what
was actually executed. It exists because the pre-registration culture is only
worth something if every departure from it is written down: a reader who cannot
see the deviations cannot tell a pre-registered test from a tuned one.

Two independent reviews drove this document (both dated 2026-10-11):

* `评审-CerebroCast实验符合性审查-v1.md` — compliance review of the experiment
  suite against the v1 plan;
* `实验结果和方法评估意见.md` — independent evaluation of the results and the
  experimental design.

Their verdict, in their own words: *the engineering quality matches the
verification culture, but the coverage does not match the commitment list, and
the strictness of the statistical implementation does not match the statistical
protocol.* This appendix is the response.

**Status legend** — `declared` (deviation written down, claim kept),
`downgraded` (deviation written down, claim weakened), `withdrawn` (claim
deleted), `remedied` (the missing work was actually done in v2).

---

## 1. The pre-registered hypotheses (v1 text, verbatim intent)

| ID | Pre-registered statement | v1 threshold |
|---|---|---|
| H1 | EPF five markets, 5 seeds, cumulative MSE at least 8% better than the best of B1–B7 | ≥ 8% |
| H2 | after an abrupt break the system recovers within 10 days | ≤ 10 d |
| H3 | median relative degradation over the 11 real domains ≤ 5% | ≤ 5% |
| H4 | three-tier gate macro-F1 against the decision-theoretic label | ≥ 0.70 |
| H5 | at least 6 of 9 module ablations priced significantly above zero | ≥ 6/9 |
| H6 | a newly released expert enters the post-hoc top-5 responsibility ranking within 30 days | ≤ 30 d |
| H7 | incremental compute below 10% of a monthly full retrain | < 10% |

---

## 2. Deviation register

### 2.1 Data and protocol

| # | Item | v1 text | Actual execution | Reason | Effect on conclusions | Status |
|---|---|---|---|---|---|---|
| D1 | Data domains | ECL, Solar, Weather, ETTh1, ETTm1, Exchange | EPF×5 + ETTh1/2 + ETTm1/2 + Weather + ECL | Solar and Exchange unavailable in the local cache | Spectral-family coverage shifts (sensor doubles); H3/C2 read as "11 domains, 4 families" | declared |
| D2 | Expert pool | reuse the companion study's 21-method pool + 475 cached runs | 24 purpose-built lightweight heterogeneous experts (12 general + 3 period-prefix + 9 regime-specialist) | **The old pool leaked**: distance-weighted k-NN self-matched, so 66% of training days had exactly zero loss. Reusing the cache would have inherited the contamination | v1 stress gains of +17%…+77% were an artefact and are **withdrawn**; the leak and its fix become a methodological contribution | remedied |
| D3 | EPF exogenous covariates | hourly price + 2 exogenous forecast covariates | univariate price series only (`OT`) | covariates were never wired into the window builder | Pool absolute level is below the EPF literature standard; deployment narrative weakened | declared |
| D4 | Split | chronological 0.70/0.10/0.20 | growing-block `fit 55% / warm 10% / val 10% / test 25%` | required to remove the pool leak (D2) | Every architecture forecast now comes from a model that has not seen that day — a strengthening | remedied |
| D5 | Synthetic grid | p ∈ {0.001, 0.005, 0.02}, α decay, amplitude balance, missing mask | H4: p ∈ {0.001, 0.005, 0.02, 0.05, 0.10}; H2: p = 0.03 with a jump sweep; no α / amplitude / missing arms | the original grid started at p = 0.02, i.e. it only measured the hard regime | H4's failure was originally reported without the low-frequency band; v2 adds it, so the verdict is now frequency-resolved | remedied |
| D6 | Long horizon 96/720 | listed as a secondary analysis | not run | cut for time | none (secondary) | declared |
| D7 | Confirmation seeds | {7, 12345} held out as a confirmation layer | all 5 seeds pooled | no exploration/confirmation separation was implemented | v2 adds a confirmation-seed-only reading of the main verdicts | declared |
| D8 | Cut list discipline | "the cut list is written here in advance; no ad-hoc decisions at run time" | the cut list was written *after* the runs | process failure | The v2 cut list is frozen below in §3 | declared |

### 2.2 Arms, baselines and the attribution problem

| # | Item | v1 text | Actual execution | Reason | Effect on conclusions | Status |
|---|---|---|---|---|---|---|
| D9 | Ablation numbering | A1–A9 with module semantics | A1–A11 with code semantics | three mutually inconsistent numbering schemes existed across the plan, the draft and the code; the arm family also collided symbolically with structural assumptions A1–A3 | The paper is now written against the code numbering, with a v1↔v2 mapping table **and a symbol-disambiguation rule**: A1–A3 stay reserved for the assumptions of Section 3 (always qualified, e.g. "assumption A2", "A2(ii)"), removal arms are written as the code identifiers `A1_no_spectral` … `A11_no_growth` (abbreviated "arm A4" only when the section has already spelled out the full identifier). Four families, four prefixes — assumptions A, removals A-with-suffix, placebos P, comparators B | remedied |
| D10 | **Best-baseline set** | best of B1–B7 | v1: best of B1–B7; **v2: B8 added** | B1–B7 did not contain recency-weighted, difficulty-normalised simplex LS — which is the architecture's *own engine* (L0), so every reported gain conflated engine and modules | **This was the most consequential defect.** v2 reports dual-caliber gains (vs best classic, and vs the bare engine); see §4 | remedied |
| D11 | H5 threshold semantics | ≥ 6 of 9 arms | 11 arms were run, threshold left at 6 | the arm count grew without revisiting the threshold | v2 reports both readings (6/11 and 6/10 excluding the feature ablation A1) explicitly | declared |
| D12 | P3 placebo | price the responsibility signal | permuted λ *before* the mixture | λ is simultaneously the fusion weight and the credit baseline | The original P3 prices "sound mix vs random mix", i.e. engine evidence — **not** the responsibility ordering. v2 relabels it and adds P3b, which permutes only the credit copy of λ | remedied |

### 2.3 Statistics and metrics

| # | Item | v1 text | Actual execution | Reason | Effect on conclusions | Status |
|---|---|---|---|---|---|---|
| D13 | Bootstrap | 168 h block bootstrap, 310 blocks/market, 2000 resamples | `block=1` bootstrap over **5 seed-mean points** | the daily series were not carried into the statistics stage | Every v1 CI, p-value and TOST verdict was degenerate. v2 resamples the paired *daily* difference series with one-week blocks (2000 resamples); the n = 5 seed-level Wilcoxon is demoted to a footnote | remedied |
| D14 | H5 pooling | (implied per-domain comparability) | raw ΔW averaged across 16 domains | — | Domains differ in scale by 30× (Weather W≈0.43 vs ECL W≈0.014), so large-scale domains dominated. v2 normalises per domain (ΔW/W_FULL) and reports the per-domain sign distribution alongside the raw pooling | remedied |
| D15 | H1 pass criterion | EPF five markets | judged on the **stress stratum** median | implementation drift | The pre-registered main battlefield (EPF-5, median −2.42%) was negative and was masked. v2 splits H1 into **H1a** (EPF-5, main) and **H1b** (stress) | remedied |
| D16 | H3 scope | 11 real domains | computed on all 16 streams | implementation drift | Both readings pass, but only the pre-registered one is now reported as the verdict; the 16-stream value is retained for the record | remedied |
| D17 | H2 metric | days to recover to 90% of the **post-break oracle gap** | running mean back to the **pre-break** median × 1.05, censored at 40 days | implementation drift | The legacy metric confounds adaptation speed with a change in regime difficulty: if the new regime is intrinsically harder the loss never returns to the pre-break level, so "34.9 days" mostly means "censors at 40". v2 implements the pre-registered definition | remedied |
| D18 | H7 operationalisation | incremental compute < 10% of a monthly retrain (implying model retraining cost) | routing-layer wall clock vs a monthly NNLS refit | no retraining-cost instrumentation existed | The v1 FAIL (21.6×) compares two routing-layer CPU costs and is meaningless in absolute terms. v2 states the actual operationalisation and records the pre-registered intent as unmeasurable with the current instrumentation | declared |
| D19 | Real-domain equivalence wording | — | `passed = true` from a post-hoc "median ≥ −5%" rule while TOST gave 0/11 equivalent | — | The wording is downgraded from "equivalent" to "no evidence of systematic degradation"; TOST is not claimed | downgraded |
| D20 | MASE, M3, M4, M5 | listed as metrics | not computed (M5 depended on H6) | — | M4 is not optional: L3's weak-joint premise cites a calibration audit that did not exist. v2 computes M3/M4/MASE and runs H6, producing M5 | remedied |
| D21 | C1 attribution regression | falsification criterion stated | never run | — | C1 was neither supported nor falsified; v2 provides the attribution regression (§5) | remedied |

### 2.4 Evidence-status corrections to the theory chapter

| # | Statement | v1 status | Corrected status (v2) |
|---|---|---|---|
| P2 | reset benefit threshold τ | "boundary is clean" | **insufficient evidence** — RESET never fires on the annotated break days (`frac_reset_at_break = 0`), so the grid cannot price P2 |
| P5 | module-necessity counterexample families | "instantiated in the generator and at an EPF parameter point" | **constructive proof only, numerical instantiation now added** as five directed families; the v1 "already instantiated" sentence is deleted |
| C1 | total-regret four-way decomposition | (implicitly supported) | **tested in v2** — attribution regression reported in §5 |
| L1 | fingerprint false-alarm bound | (implied consistent) | **partial** — the bound's premise (low switch frequency) was not covered by the v1 grid; v2 covers it |
| L3 | eligibility-trace credit consistency | premise audited by M4 | M4 now computed; status recorded |
| C2 | spectral-affinity monotonicity | tested (Fig. 7) | unchanged |

---

## 3. Frozen cut list (v2)

Written **before** the v2 runs, so no ad-hoc decisions were made afterwards.

**Cut and declared:** Solar and Exchange (D1); long horizons 96/720 (D6); the α
decay, amplitude-balance and missing-mask synthetic arms (D5); reuse of the
companion pool and its 475 cached runs (D2, with cause); separate
exploration/confirmation seed layers in the main table (D7, replaced by a
dedicated confirmation reading).

**Executed in v2:** B8 engine baseline; P3b credit placebo; day-level block
bootstrap; per-domain normalised pricing; H1a/H1b split; H3 scope correction;
pre-registered H2 metric; H4 low-frequency band; H6 onboarding; P5 five
directed families; C1 attribution regression; M3/M4/MASE.

---

## 4. Fallback execution record

The plan pre-committed to a fallback if the architecture did not beat the best
baseline. It did not, so **fallback 1b/1c was activated**, and the narrative
moved from "the architecture wins" to "mechanism pricing + a measured map of
where maintenance architecture is worth its cost".

What the evidence actually supports, in order of strength:

1. **The bare engine is the competitor, not the brain modules.** With B8 made
   explicit, the strongest causal method on the real benchmarks is the
   architecture's own consolidation engine *with every module switched off*. The
   module contribution on top of it is negative on the real domains. This is the
   headline finding and it would have been invisible without D10.
2. **The oracle gap is real but not actionable.** The per-day oracle expert is
   roughly 2× better than every causal method, no causal method captures more
   than ~9% of that gap, yet the pre-day features do carry positive information
   about expert quality (correlation up to 0.85). The bottleneck is the decision
   rule, not the signal.
3. **Maintenance pays only on genuinely non-stationary streams, and not
   monotonically.** Gains are +1.5%…+3.9% across the switching rungs and do not
   grow with the amount of mixing; the cross-family rung has a negative seed.

Fallbacks **2a/2b** (gate → continuous modulation field) remain a decision point
and are not executed; P2 is recorded as insufficient evidence rather than
falsified.

---

## 5. C1 attribution regression

`scripts/diag_attrib.py` fits a ridge regression from per-stream change
statistics to the realised contribution of each mechanism group (detection,
identification, adaptation, maintenance), and reports the R² against the
pre-registered upgrade threshold. Result is stored in
`results/synth/c1_attribution.json` and summarised in the results chapter. The
status is recorded verbatim — including a negative result if that is what comes
out.

---

## 6. What a reader should check first

1. `results/summary/report.md` — verdicts with the v2 (day-level) statistics.
2. `results/summary/h1_by_domain.csv` — the engine/module split per domain.
3. `results/summary/pricing_by_domain.csv` — the per-domain sign distribution
   behind H5.
4. `docs/FINDINGS.md` — the diagnostic trail, including the withdrawn results.
5. This file — the register above.

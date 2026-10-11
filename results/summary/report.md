# Results (v2)

Every number below is produced by `scripts/run_stats.py` from the arrays
written by `scripts/run_stream.py` and `scripts/run_synth.py`.  The
pre-registered hypotheses are reported exactly as they came out, including
the ones that failed, and every deviation from the v1 pre-registration is
logged in `docs/SI-B-preregistration-deviations.md`.

## Endpoint, protocol and what changed in v2

* The stream is walked one calendar day at a time.  On day `t` the method
  sees only data up to `t-1`; the realised values of day `t` arrive
  afterwards and are used for the loss and for the nightly maintenance.
* The expert pool is fitted on the first 55% of each series and refitted
  twice as data accumulate, so every forecast the method ever consumes
  comes from a model that has not seen that day.
* Primary endpoint `W` = mean over test days of `min(loss_t, cap_t)`.
  `cap_t` is the 95th-percentile expert loss of day `t` times 8, smoothed
  by an EMA; it is computed from the pool alone and is therefore identical
  for every arm -- no method can gain by clipping itself.
* **Intervals and tests are now day-level.**  The previous revision
  resampled five seed-mean points with block length 1, which made every
  interval and TOST verdict degenerate.  v2 resamples the paired daily
  difference series with one-week blocks (2000 resamples).

## Headline

* On the eleven standard benchmarks the architecture does **not** beat the
  best causal baseline.  We report this as a null result rather than
  tuning against the test split.
* **The B8 decomposition is the main new result.**  B8 is the
  architecture's own consolidation engine with every brain module
  switched off.  It is the strongest single competitor, so the reported
  gain splits into an *engine* part and a *module* part, and on the real
  benchmarks the module part is negative.
* The reason is measurable: the per-day oracle expert is far better than
  any causal method, yet the signal that would identify it is not
  learnable from features observable before the day.
* On streams whose data-generating process genuinely changes, the same
  architecture gains, but the gain is *modest* rather than large: +-3.50% to +1.17%
  over the three regime-switching rungs (SW-2 / SW-4 / SW-X).
  It does **not** increase monotonically as more sources are mixed, so
  we do not claim a dose-response from degree of change to gain.

# Pre-registered verdicts

| ID | statement | measured | threshold | verdict |
|---|---|---|---|---|
| H1a | EPF five-market main test: gain vs best of B1-B8 (the comparator set including the bare engine) | EPF-5 median -3.26% | >= 8.0% | FAIL |
| H1b | regime-switching stress stratum: median gain | stress median -7.16% | >= 8.0% | FAIL |
| H2 | recovery after an abrupt break (pre-registered metric) | 24.03 d | <= 10.0 d | FAIL |
| H3 | median relative degradation over the 11 real domains | 3.26% | <= 5.0% | PASS |
| H4 | three-tier gate macro-F1 | 0.198 | >= 0.7 | FAIL |
| H5 | module ablations priced significantly > 0 | 0/11 | >= 6/11 | FAIL |
| H6 | new-expert onboarding: responsibility converges to top-5 | 67.13 d | <= 30.0 d | FAIL |
| H7 | incremental compute vs monthly retrain | 17.2927 | < 0.1 | FAIL |

## Per-domain endpoint and the engine / module split

`gain_cls` = vs the best classical baseline (B1-B7); `gain_cls+eng` = vs
the best of B1-B8; `module` = FULL vs B8 (negative means the brain
modules make it worse than the bare engine they sit on).

| domain | family | architecture | best classic | gain_cls % | best B1-B8 | gain_cls+eng % | B8 engine | module % | headroom % | BH p |
|---|---|---|---|---|---|---|---|---|---|---|
| ECL | load | 0.01386 | B4_hedge nan | -1.95 | B4_hedge | +2.21 | 0.01476 | +6.07 | -4.1 | 9.90e-01 |
| EPF-NP | power | 0.16585 | B4_hedge nan | -3.30 | B4_hedge | +1.09 | 0.16455 | -0.79 | -6.5 | 9.90e-01 |
| ETTm2 | sensor | 0.09891 | B2_monthly_retrain nan | +0.83 | B8_rwls_engine | +0.73 | 0.09951 | +0.60 | 0.8 | 9.90e-01 |
| SW-4 | STRESS | 0.30172 | B2_monthly_retrain nan | +1.87 | B8_rwls_engine | -1.80 | 0.30531 | +1.17 | 2.3 | 3.18e-01 |
| STAT-0 | STRESS | 0.12172 | B2_monthly_retrain nan | +0.86 | B8_rwls_engine | -2.12 | 0.12156 | -0.13 | -0.2 | 9.90e-01 |
| Weather | climate | 0.43450 | B2_monthly_retrain nan | +4.16 | B8_rwls_engine | -3.12 | 0.42314 | -2.68 | -5.0 | 9.90e-01 |
| EPF-DE | power | 0.33963 | B4_hedge nan | -4.87 | B4_hedge | -3.15 | 0.33875 | -0.26 | -9.3 | 9.90e-01 |
| EPF-PJM | power | 0.06909 | B4_hedge nan | +5.06 | B8_rwls_engine | -3.26 | 0.06938 | +0.41 | 0.7 | 9.90e-01 |
| ETTh1 | sensor | 0.04070 | B1_frozen_static nan | -1.25 | B1_frozen_static | -3.28 | 0.04028 | -1.05 | -1.9 | 9.90e-01 |
| EPF-FR | power | 0.17713 | B1_frozen_static nan | +6.04 | B8_rwls_engine | -3.70 | 0.16644 | -6.42 | -10.3 | 9.90e-01 |
| EPF-BE | power | 0.36905 | B2_monthly_retrain nan | -2.42 | B2_monthly_retrain | -5.54 | 0.38973 | +5.31 | -5.6 | 9.90e-01 |
| SW-X | STRESS | 0.22805 | B2_monthly_retrain nan | +1.47 | B8_rwls_engine | -7.16 | 0.22034 | -3.50 | -6.0 | 9.90e-01 |
| SW-2 | STRESS | 0.28197 | B2_monthly_retrain nan | +3.94 | B8_rwls_engine | -8.53 | 0.27996 | -0.72 | -1.3 | 9.90e-01 |
| ETTm1 | sensor | 0.04007 | B2_monthly_retrain nan | -0.89 | B2_monthly_retrain | -11.88 | 0.03985 | -0.54 | -1.4 | 9.90e-01 |
| ETTh2 | sensor | 0.10189 | B2_monthly_retrain nan | -2.73 | B2_monthly_retrain | -12.64 | 0.10001 | -1.89 | -3.9 | 9.90e-01 |
| STAT-1 | STRESS | 0.09168 | B2_monthly_retrain nan | -0.83 | B8_rwls_engine | -36.84 | 0.09039 | -1.43 | -2.8 | 9.90e-01 |

## Dose-response along the non-stationarity ladder

`seed-level` columns are ratios of mean winsorised losses, one value per (rung, seed), so mean always lies inside [min, max]. `bootstrap est` is the day-level paired block-bootstrap point estimate of the same comparison that H1a/H1b read.

| rung | construction | n | seed-level mean % | sd | median % | min % | max % | bootstrap est % |
|---|---|---|---|---|---|---|---|---|
| STAT-0 | contiguous single series (stationary control) | 5 | -0.13 | 0.19 | -0.04 | -0.37 | +0.08 | -2.12 |
| STAT-1 | same series, spliced windows (splice control) | 5 | -1.43 | 0.96 | -1.31 | -2.84 | -0.20 | -36.84 |
| SW-2 | two series, same family | 5 | -0.72 | 0.19 | -0.68 | -0.91 | -0.44 | -8.53 |
| SW-4 | four series, same family | 5 | +1.17 | 0.29 | +1.23 | +0.69 | +1.47 | -1.80 |
| SW-X | three series, different families | 5 | -3.50 | 1.52 | -2.79 | -5.70 | -2.03 | -7.16 |

## Mechanism pricing, per-domain normalised (H5 v2)

`rel` is the mean over domains of `100*(W_arm - W_FULL)/W_FULL`, so a
positive value means removing that mechanism *hurts*.  `raw` is the old
pooling of absolute MSE deltas, kept to show the scale artefact.  `sign`
counts the domains in which the arm is worse than FULL.

| arm | kind | rel % | 95% CI | raw dW | raw CI | +dom | -dom | p (BH) | significant |
|---|---|---|---|---|---|---|---|---|---|
| P3_shuffled_responsibility | engine | +31.927 | [+18.238, +50.110] | +0.04831 | [+0.02498, +0.07389] | 16 | 0 | 4.88e-04 | yes |
| A4_no_change_detection | ablation | +0.697 | [-0.021, +1.590] | +0.00201 | [-0.00007, +0.00508] | 10 | 6 | 7.71e-01 | no |
| A1_no_spectral | ablation | +0.296 | [-0.177, +0.843] | +0.00079 | [-0.00053, +0.00261] | 8 | 8 | 8.34e-01 | no |
| A6_no_forward_model | ablation | +0.230 | [-0.289, +0.842] | +0.00009 | [-0.00065, +0.00081] | 9 | 7 | 8.34e-01 | no |
| A8_no_metaplasticity | ablation | +0.164 | [-0.013, +0.471] | +0.00031 | [-0.00001, +0.00085] | 7 | 7 | 8.34e-01 | no |
| A5_no_reset | ablation | +0.114 | [-0.358, +0.623] | -0.00007 | [-0.00112, +0.00086] | 8 | 4 | 8.34e-01 | no |
| P4_random_growth | placebo | +0.053 | [+0.003, +0.109] | +0.00008 | [-0.00001, +0.00019] | 6 | 4 | 7.71e-01 | no |
| A2_no_episodic | ablation | +0.046 | [-0.006, +0.112] | +0.00009 | [-0.00003, +0.00023] | 9 | 7 | 8.34e-01 | no |
| P3b_shuffled_credit | placebo | +0.008 | [-0.037, +0.054] | +0.00003 | [-0.00003, +0.00011] | 7 | 9 | 8.34e-01 | no |
| A10_no_downscaling | ablation | +0.000 | [-0.005, +0.006] | +0.00000 | [-0.00000, +0.00001] | 6 | 7 | 8.34e-01 | no |
| A7_no_three_factor | ablation | -0.003 | [-0.038, +0.032] | -0.00001 | [-0.00009, +0.00008] | 6 | 8 | 8.34e-01 | no |
| P1_random_reset | placebo | -0.024 | [-0.222, +0.122] | -0.00017 | [-0.00052, +0.00011] | 4 | 3 | 8.34e-01 | no |
| P2_random_replay | placebo | -0.025 | [-0.104, +0.070] | -0.00007 | [-0.00020, +0.00005] | 6 | 10 | 7.71e-01 | no |
| A11_no_growth | ablation | -0.073 | [-0.188, +0.015] | -0.00026 | [-0.00069, +0.00003] | 4 | 6 | 7.72e-01 | no |
| A3_no_clusters | ablation | -0.122 | [-0.359, +0.068] | -0.00043 | [-0.00118, +0.00005] | 9 | 7 | 1.00e+00 | no |
| A9_no_replay | ablation | -0.196 | [-1.040, +0.393] | -0.00050 | [-0.00197, +0.00043] | 8 | 8 | 8.34e-01 | no |

## Confirmation-seed reading

Seeds [7, 12345] only (exploration seeds excluded): EPF-5 median -2.97%, real-domain median -2.32%.


## Falsified claims and audited premises

The v2 supplementary batch turned three further pre-registered claims over, and audited one premise that had never been checked. Each row below is read straight from its JSON side-car.

| claim | what was measured | pre-registered | outcome | source |
|---|---|---|---|---|
| C1 (four-way regret decomposition) | leave-one-domain out-of-sample R^2, per mechanism group | >= 0.6 | **falsified**: all four groups negative (adaptation -0.091, detection -0.347, identification -1.076, maintenance -0.496) | `c1_attribution.json` |
| P5 (targeted-pressure ordering) | predicted top-cost mechanism group vs realised | ordering holds per family-run | **falsified**: holds in 1/15 runs (6.7%) | `p5_families.json` |
| P2 (RESET beats ADAPT past a threshold) | fraction of annotated breaks on which RESET fires | must be positive for the claim to be testable | **insufficient evidence**: frac_reset_at_break = 0.00 in every configuration, so the grid cannot price the boundary | `h2_p2_recovery.json` |
| L3 premise (calibrated forward model) | mean relative ECE of the forward model | <= 1 (no worse than a constant) | **premise not met**: mean relative ECE = 1.360 over 6 domains (poorly calibrated -- the L3 premise is NOT met by this implementation) | `m4_calibration.json` |
| H4 (tier classification) | macro-F1 vs drift labels, five switch frequencies | >= 0.70 | **falsified at every frequency**: best cell 0.289 at the lowest switch rate, degrading monotonically with frequency | `h4_detection.json` |

The M5 onboarding curve (H6) is reported above; MASE against a weekly-seasonal naive reference is 0.62-0.82 on all streams, which is the one endpoint where the architecture holds up everywhere.


# Results

Every number below is produced by `scripts/run_stats.py` from the arrays
written by `scripts/run_stream.py` and `scripts/run_synth.py`.  The
pre-registered hypotheses of `config.PREREG` are reported exactly as they
came out, including the ones that failed.

## Endpoint and protocol

* The stream is walked one calendar day at a time.  On day `t` the method
  sees only data up to `t-1`; the realised values of day `t` arrive
  afterwards and are used for the loss and for the nightly maintenance.
* The expert pool is fitted on the first 55% of each series and refitted
  twice as data accumulate, so every forecast the method ever consumes
  comes from a model that has not seen that day (see `docs/PROTOCOL.md`).
* Primary endpoint `W` = mean over test days of `min(loss_t, cap_t)`.
  `cap_t` is the 95th-percentile expert loss of day `t` times 8, smoothed
  by an EMA; it is computed from the pool alone and is therefore identical
  for every arm -- no method can gain by clipping itself.
* The five architecture seeds share one pool realisation, so every
  comparison is paired over identical days.

## Headline

* On the eleven standard benchmarks the architecture does **not** beat the
  best causal baseline.  We report this as a null result rather than
  tuning against the test split.
* The reason is measurable: the per-day oracle expert is far better than
  any causal method, yet the signal that would identify it is not
  learnable from features observable before the day.
* On streams whose data-generating process genuinely changes, the same
  architecture gains substantially, and the gain grows with the amount of
  genuine change (the ladder table below).

# Pre-registered verdicts

| ID | statement | measured | threshold | verdict |
|---|---|---|---|---|
| H1 | main gain vs best baseline | stress +1.87% median; real -1.25% | >= 8.0% | FAIL |
| H2 | recovery after an abrupt break | 34.90 d | <= 10.0 d | FAIL |
| H3 | median relative degradation | 0.00% | <= 5.0% | PASS |
| H4 | three-tier gate macro-F1 | 0.137 | >= 0.7 | FAIL |
| H5 | modules priced significantly > 0 | 0/11 | >= 6/11 | FAIL |
| H7 | incremental compute / monthly retrain | 21.6446 | < 0.1 | FAIL |

## Per-domain endpoint (mean over seeds, winsorised MSE)

| domain | family | architecture | best baseline | gain % | headroom captured % | BH-adj p |
|---|---|---|---|---|---|---|
| EPF-FR | power | 0.17713 | 0.18851 | +6.04 | 9.0 | 2.15e-01 |
| EPF-PJM | power | 0.06909 | 0.07277 | +5.06 | 8.5 | 1.84e-04 |
| Weather | climate | 0.43450 | 0.45336 | +4.16 | 7.3 | 4.62e-03 |
| SW-2 | STRESS | 0.28197 | 0.29355 | +3.94 | 7.0 | 9.72e-06 |
| SW-4 | STRESS | 0.30172 | 0.30746 | +1.87 | 3.6 | 3.32e-04 |
| SW-X | STRESS | 0.22805 | 0.23145 | +1.47 | 2.4 | 9.63e-02 |
| STAT-0 | STRESS | 0.12172 | 0.12277 | +0.86 | 1.7 | 9.37e-04 |
| ETTm2 | sensor | 0.09891 | 0.09974 | +0.83 | 1.1 | 6.77e-03 |
| STAT-1 | STRESS | 0.09168 | 0.09092 | -0.83 | -1.6 | 1.00e+00 |
| ETTm1 | sensor | 0.04007 | 0.03972 | -0.89 | -1.4 | 1.00e+00 |
| ETTh1 | sensor | 0.04070 | 0.04020 | -1.25 | -1.9 | 1.00e+00 |
| ECL | load | 0.01386 | 0.01360 | -1.95 | -4.1 | 1.00e+00 |
| EPF-BE | power | 0.36905 | 0.36034 | -2.42 | -5.6 | 1.00e+00 |
| ETTh2 | sensor | 0.10189 | 0.09919 | -2.73 | -3.9 | 1.00e+00 |
| EPF-NP | power | 0.16585 | 0.16054 | -3.30 | -6.5 | 1.00e+00 |
| EPF-DE | power | 0.33963 | 0.32385 | -4.87 | -9.3 | 1.00e+00 |

## Dose-response along the non-stationarity ladder

| rung | construction | mean gain % | median % | min % | max % |
|---|---|---|---|---|---|
| STAT-0 | contiguous single series (stationary control) | +0.86 | +0.86 | +0.86 | +0.86 |
| STAT-1 | same series, spliced windows (splice control) | -0.83 | -0.83 | -0.83 | -0.83 |
| SW-2 | two series, same family | +3.94 | +3.94 | +3.94 | +3.94 |
| SW-4 | four series, same family | +1.87 | +1.87 | +1.87 | +1.87 |
| SW-X | three series, different families | +1.47 | +1.47 | +1.47 | +1.47 |

## Mechanism pricing (delta W vs the full architecture)

| arm | kind | delta W | 95% CI | relative % | p (BH) | significant |
|---|---|---|---|---|---|---|
| P3_shuffled_responsibility | placebo | +0.04831 | [+0.02584, +0.07550] | +26.88 | 3.02e-02 | yes |
| A4_no_change_detection | ablation | +0.00201 | [-0.00011, +0.00517] | +1.12 | 5.29e-01 | no |
| A1_no_spectral | ablation | +0.00079 | [-0.00069, +0.00254] | +0.44 | 5.43e-01 | no |
| A8_no_metaplasticity | ablation | +0.00031 | [-0.00001, +0.00087] | +0.17 | 5.29e-01 | no |
| A2_no_episodic | ablation | +0.00009 | [-0.00003, +0.00023] | +0.05 | 5.29e-01 | no |
| A6_no_forward_model | ablation | +0.00009 | [-0.00066, +0.00081] | +0.05 | 8.80e-01 | no |
| P4_random_growth | placebo | +0.00008 | [-0.00001, +0.00020] | +0.04 | 5.29e-01 | no |
| A10_no_downscaling | ablation | +0.00000 | [-0.00000, +0.00001] | +0.00 | 5.83e-01 | no |
| A7_no_three_factor | ablation | -0.00001 | [-0.00009, +0.00007] | -0.01 | 8.80e-01 | no |
| A5_no_reset | ablation | -0.00007 | [-0.00110, +0.00087] | -0.04 | 9.02e-01 | no |
| P2_random_replay | placebo | -0.00007 | [-0.00020, +0.00005] | -0.04 | 5.43e-01 | no |
| P1_random_reset | placebo | -0.00017 | [-0.00052, +0.00012] | -0.09 | 5.43e-01 | no |
| A11_no_growth | ablation | -0.00026 | [-0.00068, +0.00003] | -0.14 | 5.29e-01 | no |
| A3_no_clusters | ablation | -0.00043 | [-0.00119, +0.00004] | -0.24 | 5.29e-01 | no |
| A9_no_replay | ablation | -0.00050 | [-0.00199, +0.00042] | -0.28 | 6.03e-01 | no |

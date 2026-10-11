# Protocol

This document describes exactly how the streaming experiment is built and scored,
so that every reported number can be traced to the code that produced it.

## 1. Data

Eleven real domains in four spectral families, plus five constructed stress
ladder rungs.

| family | domains | source |
|---|---|---|
| power (electricity price) | EPF-NP, EPF-PJM, EPF-BE, EPF-FR, EPF-DE | hourly price, **univariate** (`OT` column only) |
| sensor | ETTh1, ETTh2, ETTm1, ETTm2 | 15-minute, aggregated to hourly |
| climate | Weather | 10-minute, aggregated to hourly |
| load | ECL | row mean |

Two domains named in the original plan (Solar, Exchange) are **absent** and two
extra sensor domains were added; this is declared as deviation D1 in
`docs/SI-B-preregistration-deviations.md`. The EPF exogenous forecast covariates
are **not** used (deviation D3).

### Data layout

`src/cerebrocast/data.py` expects the source CSVs under a dataset root
(`EPF_DIR` / `DATA_DIR` at the top of that file). Point those constants at your
local copy; the files themselves are the public ETT / Weather / ECL / EPF
benchmark datasets and are not redistributed here.

## 2. Pool construction (`scripts/build_pool.py`)

The expert pool is built with a **growing-block, out-of-sample** protocol:

```
fit 55% | warm 10% | val 10% | test 25%      (chronological)
```

The pool is refitted twice as data accumulate (fit → warm, fit+warm → val,
fit+warm+val → test), and `fit_predict()` only ever predicts days *after* the
block it was fitted on. Consequently every forecast the architecture consumes on
day `t` comes from a model that has not seen day `t`.

This replaced an earlier "fit once on the training block, then predict the whole
series" protocol, under which the distance-weighted k-NN expert **self-matched**:
66% of training days had exactly zero loss, which inflated the stress-domain
gains to +17%…+77%. Those results are withdrawn; see `docs/FINDINGS.md`.

The pool is a single realisation shared by all five architecture seeds, so every
architecture comparison is paired over identical days.

## 3. Streaming replay (`scripts/run_stream.py`)

For each (domain, seed):

1. the fingerprint encoder is fitted on the **fit block only**, with no future
   statistics;
2. the architecture warms up on the fit block;
3. the free scalars are calibrated by coordinate descent on the **validation
   block**, then frozen;
4. the system walks the validation block and then the test block, recording only
   test-day losses.

Nothing after step 3 touches the test split.

### Endpoint

Every arm is recorded as a raw per-day mean squared error. Two arm-independent
transforms are then applied identically to every arm:

* `W = mean_t min(loss_t, cap_t)` — the primary endpoint. `cap_t` is the
  95th-percentile expert loss of day `t` times 8, EMA-smoothed, computed from the
  pool alone, so no arm can gain by clipping itself.
* `N = mean_t min(loss_t, cap_t) / median_expert_loss_t` — secondary.

Calibration optimises `W` on the validation block.

### Arms

* `FULL` — the complete architecture.
* `A1…A11` — one mechanism removed each.
* `P1…P4`, `P3b` — placebos that remove the *structure* of a mechanism while
  preserving its rate/capacity.
* `B1…B7` — the classical control conditions.
* **`B8_rwls_engine`** — the architecture's own consolidation engine
  (difficulty-normalised, recency-weighted simplex LS) with every brain module
  switched off. This is the competitor that makes the engine/module split
  measurable.

## 4. Statistics (`scripts/run_stats.py`)

* **Day-level paired block bootstrap**: the paired daily difference series is
  resampled in one-week blocks, 2000 times. The unit of replication is the
  calendar day.
* **TOST** equivalence on the same daily differences, margin = 1% of the
  reference MSE.
* **BH** multiplicity control per comparison family.
* **Mechanism pricing** is computed after per-domain normalisation
  (`100*(W_arm - W_FULL)/W_FULL`) and reported with the per-domain sign
  distribution; pooling raw MSE deltas across domains is reported only to show
  the scale artefact.
* The `n = 5` seed-level Wilcoxon is reported as a footnote, not as the test.

## 5. Synthetic stages

* `scripts/run_synth.py` — H4 (three-tier gate against decision-theoretic oracle
  labels, including the low-switch-frequency band) and H2 (recovery to 90% of the
  post-break oracle gap, the pre-registered definition).
* `scripts/run_p5.py` — the five directed stress families of Section 5.4.
* `scripts/run_onboard.py` — H6: three good experts are reserved and released
  inside the test split; the availability mask keeps them out of both the mixture
  and the sufficient statistics, so onboarding is not free.
* `scripts/diag_attrib.py` — the C1 attribution regression.
* `scripts/metrics_addendum.py` — M3 (selection entropy), M4 (forward-model
  calibration) and MASE.

## 6. Leakage gates

Five gates are enforced: (i) fingerprints use fit-block statistics only;
(ii) the pool is out-of-sample by construction; (iii) calibration uses the
validation block only; (iv) the test block is never re-fitted; (v) the sleep
phase may only use data dated ≤ the current day.

# Control experiments and diagnostics

This file records the checks that decide how the main results should be read.
Every number quoted here is reproducible with the script named next to it.

## 1. The expert pool was leaking, and it changed the headline

`scripts/diag_learn.py`, `scripts/build_pool.py`

The first version of the pool fitted each expert once on the training split and
then used that fit to predict every day, including the training days themselves.
For the distance-weighted k-NN expert this is fatal: the query point is in its
own neighbour set, the distance is zero, and `weights="distance"` copies the
training target back.  Measured on EPF-NP:

| split | days where some expert has loss exactly 0 |
|---|---|
| train | 1003 / 1523 |
| validation | 0 / 217 |
| test | 0 / 437 |

The test split was clean, but the architecture *warms up* on the training split,
so it learned weights from a table in which one expert looked perfect.  Under the
leaked pool the stress benchmarks showed gains of +17% to +77%.  After the pool
was rebuilt with the growing-block protocol (`build_pool.py`, section 1 of the
module docstring) the same benchmarks show the numbers in
`results/summary/ladder.csv`.  **The large stress gains were an artefact of the
leak and are withdrawn.**

## 2. Is the choice of expert learnable at all?

`scripts/diag_learn.py`

For every benchmark we fit a ridge map from features observable *before* day `t`
(spectral fingerprint, yesterday's per-expert log loss, two EMAs of it) to the
day-normalised loss vector of day `t`.  Fitted on warm+val, scored on test.

* The correlation between predicted and realised expert quality is **positive**
  everywhere it is defined, reaching 0.85 on the cross-family ladder rung.  So
  expert quality is not random noise.
* Nevertheless, choosing `argmin` of the prediction is *worse* than always
  using the single best fixed expert, on every benchmark.  The extreme of a
  noisy prediction is dominated by prediction error.

That is the crux: the signal exists, but a hard decision on it loses money.

## 3. Soft predictive gates do not rescue it either

`scripts/diag_gate.py`

Replacing the hard pick with a tempered gate, and comparing against the whole
classical causal family on the same clean pool (seed 7, winsorised endpoint,
temperature tuned on validation only):

| router | median gain over the family champion |
|---|---|
| best fixed expert | −15.2% |
| static simplex fitted on validation | −2.3% |
| **recency-weighted simplex LS (`rwls`)** | **0.0% (is the champion)** |
| Hedge on realised losses | −9.8% |
| pure predictive gate `w ∝ exp(−η ẑ_pred)` | −11.6% |
| recency + predictive mixture | −25% … −53% |

The champion is a recency-weighted, difficulty-normalised simplex least-squares
combination.  Every addition — including the forward-model gate that the
architecture is built around — makes it worse.  Approximately **half** of the
available oracle headroom is left on the table by *all* of these routers.

## 4. Which of the nine modules actually carry the effect?

`scripts/diag_layers.py` (seed 7, adding mechanisms back one at a time)

| layer | EPF-NP | ETTh1 |
|---|---|---|
| L0 pure recency-LS engine | +3.66% | +4.69% |
| L1 + cluster field | **+4.66%** | +4.72% |
| L2 + change detection | +4.61% | +4.77% |
| L3 + LC-NE reset | +4.61% | +4.77% |
| L4 + forward-error gate | +4.61% | **+0.71%** |
| L5 + three-factor plasticity | +4.61% | +1.01% |
| L6 + metaplasticity | +4.61% | +0.96% |
| L7 FULL | +4.41% | +1.55% |

Two facts stand out.

1. The **cluster field** is the only module that reliably adds anything (+0.03 to
   +1.0 points), and it is the module that makes the architecture a *structured*
   rather than flat aggregator.
2. The **forward-error gate** is actively harmful where it engages: on ETTh1 it
   costs 4.1 points.  Validation-based calibration selected `β = 0.5` there, i.e.
   the validation split did not reject it.  We report this rather than removing
   the module on the strength of test-set evidence.

The three-tier gate behaves as designed even though it is rare: on the ladder
rungs it fires RESET on exactly the days that follow a true splice and stays
silent on the stationary rungs.  Its rarity (a handful of days out of a
five-hundred-day test stream) is why it has almost no measurable effect.

## 5. What the synthetic ground truth says

`scripts/run_synth.py`, `results/synth/*.json`

* The discrete three-tier classification does not match a decision-theoretic
  ground truth (macro-F1 far below the pre-registered 0.70).  Reported as a
  failed hypothesis.
* The boundary at which RESET beats ADAPT is clean: RESET only pays when the
  jump is comparable to or larger than the local scale of the series.

## 6. Reading of the whole picture

The headroom is real (the per-day oracle is roughly 2× better than any causal
method on these benchmarks) but it is not reachable by (i) the seven classical
routers, (ii) the architecture, or (iii) predictive gating built on the same
observable features.  That is a negative result about *this benchmark suite*,
and we state it as such instead of tuning against the test split.

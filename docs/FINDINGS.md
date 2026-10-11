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

What remains is small, and its *sign depends on which comparator set is used* —
which is the whole point of §1b:

| rung | vs best of B1–B7 (classic) | seed-level, vs best of B1–B8 |
|---|---|---|
| SW-2 | +3.94% | −0.72% (range −0.91 … −0.44) |
| SW-4 | +1.87% | **+1.17%** (range +0.69 … +1.47) |
| SW-X | +1.47% | −3.50% (range −5.70 … −2.03) |

So the earlier statement "the architecture gains +1.5%…+3.9% on the switching
rungs" is true **only against the classical family**, and it evaporates once the
architecture's own engine is admitted to the comparator set: two of the three
rungs turn negative, and only SW-4 survives.  The gain is also **not monotone**
in the number of sources mixed (SW-2 > SW-4 > SW-X does not hold in either
caliber), so no dose-response from degree of change to gain should be read off
this table.  (`run_stats.py` builds the ladder from the per-seed rows; an earlier
revision aggregated the already-seed-averaged table, which forced `n_seeds=1` and
`min=max=mean` and hid this spread entirely.)

## 1b. The missing baseline: the engine *is* the competitor

`scripts/run_stream.py` (arm `B8_rwls_engine`), `results/summary/h1_by_domain.csv`

§3 already showed that the strongest classical causal combiner on this pool is
recency-weighted, difficulty-normalised simplex least squares (`rwls`) — which is
exactly the architecture's own consolidation engine (L0) with the brain modules
switched off.  That combiner was nevertheless **absent from the B1–B7 control
set**, so every reported "architecture vs best baseline" gain silently mixed two
very different things: the value of the engine, and the value of the nine
mechanisms layered on it.

Baseline **B8** is that engine, calibrated on the validation split under the same
protocol as every other arm.  Once it is present, the decomposition is direct:

* **engine contribution** = B8 vs the best of B1–B7;
* **module contribution** = FULL vs B8.

The result is unambiguous and unwelcome: **B8 is the best of the eight
comparators on every one of the five ladder rungs and on nine of the sixteen
domains overall, and FULL is worse than B8.**  For example on SW-2, seeds 7
(the table is the seed-5 mean unless noted, from `results/summary/h1_by_domain.csv`
and the per-run table):

| arm | W |
|---|---|
| **B8 (bare engine)** | **0.27996** |
| FULL (engine + nine modules) | 0.28197 |
| B2 month-retrain (previous best baseline) | 0.29355 (seed 7) |
| B1 frozen | 0.30132 (seed 7) |

So the modules *subtract* about 0.7% on top of the engine, while the engine
itself beats the previously-best classical baseline by about 4.6%.  Across all
sixteen domains the module contribution is negative in eleven of them, ranging
from **+6.07%** (ECL, where the fusion geometry evidently helps) to **−6.42%**
(EPF-FR).  Any claim of the form "the brain-inspired architecture improves on the
best causal method" has to be re-read as "its consolidation engine does, and its
mechanisms cost a little".  This is why the v2 result tables report **two** gain
columns and why the results chapter leads with the engine/module split rather
than with a single gain figure.

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
  ground truth.  The v2 grid adds the low-switch-frequency band that the first
  grid omitted, and the verdict is now **frequency-resolved** rather than a single
  number — 60 configurations, 5 switch rates × 2 spike levels × 2 gradual flags ×
  3 seeds:

  | `p_switch` | macro-F1 (mean) | min |
  |---|---|---|
  | 0.001 | 0.284 | 0.229 |
  | 0.005 | 0.289 | 0.263 |
  | 0.02  | 0.223 | 0.178 |
  | 0.05  | 0.116 | 0.095 |
  | 0.10  | 0.077 | 0.056 |

  Degradation is monotone in switch frequency, so the detector is not uniformly
  useless — it is *usable only when the regime is calm*, and even there 0.29 is
  far below the pre-registered 0.70.  H4 therefore fails at **every** frequency
  measured, but the failure is much less severe in the low-frequency band.  Note
  also that the labels are oracle labels built by `argmin` over a noisy quantity
  and are strongly class-imbalanced, so the reading mixes detector error with
  label error.
* **RETRACTED — "the boundary at which RESET beats ADAPT is clean".**  That
  conclusion is withdrawn.  `frac_reset_at_break = 0.0` in every configuration of
  `results/synth/h2_p2_recovery.json`: RESET never fires on the annotated break
  days, so this grid cannot price the RESET/ADAPT boundary at all.  The
  proposition P2 is recorded as **insufficient evidence**, not as supported.
* The legacy recovery metric (running mean back to the pre-break median, censored
  at 40 days) is also withdrawn: it conflates adaptation speed with a change in
  regime difficulty, so the "≈35 days" it reported largely means "censored".  v2
  implements the pre-registered definition (recovery to 90% of the post-break
  *oracle gap*) and re-runs the jump sweep:

  | `jump_scale` | legacy `recovery_full` | pre-registered `oracle90_full` | `oracle90_noreset` | post-break loss FULL / noreset |
  |---|---|---|---|---|
  | 0.25 | 34.8 | 22.7 | 23.6 | 0.4414 / 0.4490 |
  | 1.00 | 34.7 | 23.6 | 23.6 | 0.4226 / 0.4384 |
  | 4.00 | 35.1 | 23.4 | 23.6 | 0.4115 / 0.4328 |

  Two readings follow.  First, the legacy metric was indeed a censoring artefact —
  it barely moves with `jump_scale` (34.4–35.1) whereas the pre-registered one
  does not move much either but is well inside the 40-day window (22.7–23.9), so
  it is actually measured rather than clipped.  Second, **the reset module does
  not shorten recovery**: FULL and the no-reset counterfactual recover in the same
  number of days at every jump scale.  What it does do is lower the post-break
  loss *level* (e.g. −4.9% relative at `jump_scale = 4`).  That is a level effect,
  not a speed effect, and it is the honest way to state what §4's L3 row shows.

## 5b. P5: the mechanism-cost ordering does not reproduce

`scripts/run_p5.py`, `src/cerebrocast/synth_p5.py`, `results/synth/p5_families.json`

P5 predicted that a *targeted* pressure family would make the corresponding
mechanism group the most expensive one.  Five families were instantiated on the
real streaming pipeline (15 family-seed runs, each running FULL plus eleven
ablations and aggregating the cost into four groups):

| family | predicted top group | realised top group | holds | group costs (det/id/adapt/maint, %) |
|---|---|---|---|---|
| `periodic_jump` | detection | adaptation | ✗ | −8.44 / +0.43 / +11.14 / +0.05 |
| `rare_high_loss` | detection | detection | ✓ (1 of 3 seeds) | +0.03 / 0.00 / +0.02 / +0.01 |
| `heavy_tail` | maintenance | detection | ✗ | +0.18 / −0.00 / −0.09 / −0.00 |
| `regime_cycle` | identification | adaptation | ✗ | −5.67 / +0.28 / +6.73 / +0.02 |
| `linear_drift` | adaptation | detection | ✗ | +0.15 / −0.00 / −0.04 / +0.00 |

**The ordering prediction holds on 1 of 15 runs (6.7%).**  The realised pattern is
nearly orthogonal to the prediction: the adaptation group dominates under both
`periodic_jump` and `regime_cycle`, the detection group dominates under
`heavy_tail` and `linear_drift`, and `rare_high_loss` barely stresses any group at
all (all four costs within ±0.03%).  P5 is therefore reported as a **negative
result about the group decomposition**, not as a calibrated map from pressure
type to mechanism cost.  Combined with §4 (where the cluster field is the only
consistently positive layer and the forward gate is negative), the honest reading
is that the ablation cost is not cleanly attributable to the mechanism the
pressure family was designed to stress.


## 5c. H6: new experts are not found within the pre-registered budget

`scripts/run_onboard.py`, `scripts/merge_onboard.py`,
`results/synth/h6_onboarding.json` (240 events: 16 domains × 5 seeds × 3 experts)

The strongest missing experiment in the first revision was onboarding.  Three
experts whose *post-hoc* test-period rank is inside the top 5 are reserved and
then released into the pool at 1/3 and 2/3 of the test split.  The availability
mask blocks them from the mixture **and** from the second-moment accumulator, so
they contribute nothing before release and the system must notice them after the
fact.  The counterfactual arm keeps the same three experts permanently absent.

| reading | value |
|---|---|
| onboarding events | 240 |
| converged to top-5 responsibility rank within 30 days | **≈ 21%** |
| mean days to top-5, over events that converge | ≈ 65 |
| pre-registered verdict | **FAIL** (threshold 30 days) |

The failures are not marginal: the per-shard converged fractions are 16.7%,
25.0%, 25.0% and 16.7%, and the non-converging experts simply never enter the
top-5 within the window (their recorded `days_to_top5_rank` is `nan` because the
run censors at the end of the test split).  So the honest statement is that the
responsibility signal does **not** reliably discover a good new expert inside one
month.  This is the first direct evidence for failure mode **F10** (pool-growth
onboarding failure), which was previously an assertion in the catalogue with no
experiment behind it; **F9** (novel regime *combinations*) remains without
direct evidence and must be flagged as such in the text.

## 5d. The forward model is not calibrated, so Lemma L3's weak joint fails

`scripts/metrics_addendum.py`, `results/synth/m4_calibration.json`

L3's weak joint cites a calibration audit of the forward error model.  That
audit did not exist in the first revision, so it could not be cited either way.
It has now been run: predicted per-expert loss is binned into deciles and
compared with the realised mean in each bin, over six domains.

| domain | ECE | relative ECE |
|---|---|---|
| ETTh1 | 0.040 | 1.102 |
| ECL | 0.023 | 1.759 |
| EPF-BE | 0.539 | 1.748 |
| EPF-NP | 0.157 | 1.138 |
| SW-2 | 0.447 | 1.635 |
| SW-X | 0.204 | 0.775 |
| **mean** | — | **1.360** |

A relative ECE above 1 means the model's confidence ordering is worth *less* than
a constant prediction.  The conclusion is therefore that **the L3 premise is not
met by this implementation**, and L3 must be carried as a conditional statement
with an explicit unmet premise rather than as a supported lemma.  Two companion
readings from the same script: the selection-frequency entropy is normalised to
0.85 of uniform on the stationary rung with no degenerate days, and MASE against
a weekly-seasonal naive reference is 0.62–0.82 on **all sixteen** streams — the
one endpoint where the architecture is unambiguously better than a simple
reference.

## 5e. C1 is falsified by its own pre-registered test

`scripts/diag_attrib.py`, `results/synth/c1_attribution.json`

C1 conjectured that the regret decomposes into detection, identification,
adaptation and maintenance terms that can be *predicted from observable
environment statistics*.  The pre-registered falsification criterion was a
leave-one-domain out-of-sample R² of at least 0.6, using eight features that are
all computable from the stream itself (no arm information): mean drift, the
fractions of ADAPT/HOLD/RESET days, fingerprint drift, difficulty IQR, the noise
floor, and the pool median loss.

| mechanism group | out-of-sample R² | mean cost % |
|---|---|---|
| detection | **−0.347** | +0.697 |
| identification | **−1.076** | −0.038 |
| adaptation | **−0.091** | +0.172 |
| maintenance | **−0.496** | −0.022 |
| pooled | **−0.034** | — |

Every group is negative, i.e. the regression predicts the mechanism cost *worse
than the group mean*.  C1 is recorded as **falsified**, not as untested.


## 6. Reading of the whole picture

The headroom is real (the per-day oracle is roughly 2× better than any causal
method on these benchmarks) but it is not reachable by (i) the classical routers,
(ii) the architecture, or (iii) predictive gating built on the same observable
features.  That is a negative result about *this benchmark suite*, and we state
it as such instead of tuning against the test split.

With B8 present (§1b) the picture sharpens into a statement about *when*
maintenance architecture is worth its cost, which is the fallback the plan
pre-committed to:

1. On near-stationary real data the heavy architecture is a **net cost** on top of
   its own engine, and the engine alone already beats the classical baselines.
2. Once the engine is in the comparator set, even the change-driven streams do not
   reliably favour the full architecture: only the four-source same-family rung
   SW-4 stays positive (seed-level +0.69% … +1.47%), while SW-2 and SW-X turn
   negative.  Against the *classical* family alone the same rungs read
   +1.47% … +3.94%, and it is important not to quote that number without saying
   which comparator set produced it.
3. The gap to the oracle is dominated by the **decision rule**, not by missing
   information: the pre-day features do carry positive information about expert
   quality, but neither hard selection nor a soft gate can convert it into money.
4. The mechanism-level evidence points the same way.  Zero of eleven removal arms
   is significant after per-domain normalisation; the only significant placebo
   (P3, +31.9%) permutes the fusion weights and is therefore *engine* evidence;
   and the placebo that isolates the responsibility ordering (P3b, +0.008%) is
   indistinguishable from zero.  Onboarding (H6, ≈21% convergence in 30 days),
   the pressure-ordering prediction (P5, 1/15), the four-way decomposition (C1,
   negative R²) and the forward-model calibration premise (M4, relative ECE 1.36)
   all fail their pre-registered tests.  The one endpoint with no caveat is MASE
   against a weekly-seasonal naive reference: 0.62–0.82 on all sixteen streams.

## 7. What changed between the two revisions

The first revision of this experiment suite reported a monotone dose-response, a
+12%…+77% stress gain, and a single "architecture vs best baseline" gain column.
All three were artefacts of defects that were only found by auditing the suite
against its own pre-registration:

| defect | effect | where it is fixed |
|---|---|---|
| k-NN self-matching leak in the pool | inflated stress gains to +17%…+77% | `build_pool.py` growing-block protocol; §1 |
| B8 (the bare engine) missing from the baselines | conflated engine and module contributions | `run_stream.py`; §1b |
| `block=1` bootstrap over 5 seed-mean points | degenerate CIs, TOST with no power | `run_stats.py` v2, day-level blocks |
| raw ΔW pooled across 30×-different scales | H5 and the single "significant" arm were scale artefacts | `run_stats.py` v2, per-domain normalisation |
| P3 placebo permuted the fusion weights | priced "sound mix vs random mix", not the credit ordering | `system.py` P3b; `config.py` |
| H1 judged on the stress stratum | hid that the pre-registered EPF-5 battlefield is negative | `run_stats.py` v2, H1a/H1b |
| H2 metric vs the *pre-break* level | "34.9 days" mostly meant "censored at 40" | `run_synth.py`, pre-registered definition |
| H4 grid starting at p = 0.02 | the low-frequency band was never measured, so the verdict was not frequency-resolved | `run_synth.py`, p ∈ {0.001…0.10}; §5 |
| missing H6 / P5 / C1 / M3 / M4 / MASE | claims without evidence | `run_onboard.py` + `merge_onboard.py`; `run_p5.py` + `synth_p5.py`; `diag_attrib.py`; `metrics_addendum.py` |
| three D-numbering schemes (plan, SI-B, draft) and an A-symbol collision between arms and assumptions | the plan and the deviation register disagreed on what "D5" meant; arms A1–A11 shadowed assumptions A1–A3 | plan §6 header + D9; Section 4/6 draft headers |

The last three rows changed conclusions rather than presentation: **H6 fails
(≈21% convergence in 30 days), P5's ordering holds in 1/15 runs, C1's
out-of-sample R² is negative for every mechanism group, and the L3 calibration
premise is not met (relative ECE 1.36)**.  Those are reported as falsifications
and as an unmet premise, not as gaps to be filled by a later revision.

Every one of these is logged in
`docs/SI-B-preregistration-deviations.md`, which is the authoritative record.

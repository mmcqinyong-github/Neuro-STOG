# Neuro-STOG

Code, results and figures for the CerebroCast experiments: a calendar-day
streaming architecture that maintains a heterogeneous pool of forecasting
experts, and a controlled study of *when* such maintenance actually pays.

## What is in here

| path | contents |
|---|---|
| `src/cerebrocast/` | the architecture, the eight control conditions, the synthetic generators, the statistics |
| `scripts/`        | the pipeline stages, plus the diagnostic scripts used during development |
| `results/summary/`| every number that appears in the paper (CSV + `verdict.json` + `report.md`) |
| `results/synth/`  | controlled-drift results (H2, H4, H6, P5, C1, M3/M4/MASE) |
| `figures/`        | Fig. 1-9 as vector PDF/SVG and 300 dpi PNG |
| `docs/`           | protocol, findings, and the pre-registration deviation register |

## Headline results

Read `results/summary/report.md` -- it reports the pre-registered hypotheses
exactly as they came out, including the ones that **failed**, and
`docs/SI-B-preregistration-deviations.md` records every departure from the
original pre-registration.

* The architecture does **not** beat the best causal baseline on the eleven
  standard benchmarks.  We report this as a null result and explain why: the
  per-day oracle expert is far better than any causal method, but the signal that
  would identify it does not survive the decision rule.
* **The bare engine is the real competitor.**  Baseline `B8` is the
  architecture's own consolidation engine with every brain module switched off.
  It is the strongest single method on the real benchmarks, so the reported gain
  splits into an *engine* part and a *module* part -- and the module part is
  negative on the real domains.  A first revision that omitted `B8` conflated the
  two; the dual-caliber tables in `report.md` fix that.
* On streams whose process genuinely changes, the answer depends on the
  comparator set, and it is quoted both ways.  Against the classical family
  (B1-B7) the switching rungs gain +1.47% to +3.94%; once the architecture's own
  engine (B8) is admitted, the same rungs read -3.50% to +1.17% at the seed level
  and two of the three turn negative.  Neither caliber is monotone in the amount
  of mixing, so we claim no dose-response.  An earlier revision reported
  +12% to +77% with a monotone ladder; that came from a leaked expert pool and is
  withdrawn.
* **Mechanism pricing prices the engine, not the mechanisms.**  After per-domain
  normalisation, **0 of 11** removal arms are significant.  The only significant
  placebo, `P3` (+31.9%), permutes the *fusion weights* and therefore prices
  "sound mixture versus random mixture".  The placebo that actually tests the
  responsibility ordering, `P3b`, is +0.02% with a confidence interval
  containing zero.
* **Three further pre-registered claims are falsified, and we say so.**  The
  drift-detector's tier classification fails at every switch frequency measured
  (macro-F1 0.08-0.29 against a threshold of 0.70, degrading monotonically with
  frequency); the targeted-pressure ordering of Proposition P5 holds in 1 of 15
  family-seed runs; and the four-way regret decomposition of Conjecture C1 has
  **negative** out-of-sample R^2 for every mechanism group, so mechanism cost is
  *not* predictable from observable environment statistics.
* **One premise is audited and found wanting.**  The forward model's calibration
  (metric M4) has a mean relative ECE of 1.36, i.e. worse than a constant
  prediction -- so Lemma L3's weak joint does not hold for this implementation,
  and we report it as a conditional statement rather than a verified one.
* Where the architecture does hold up: MASE against a weekly-seasonal naive
  reference is 0.62-0.82 on all sixteen streams, and the real-domain median
  relative degradation stays inside the pre-registered 5% band.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/build_pool.py
python scripts/build_stress.py 7 42 2021 3407 12345
python scripts/run_stream.py --rounds 2
python scripts/run_synth.py
python scripts/run_onboard.py          # H6; may be sharded with --out results/obN
python scripts/merge_onboard.py        # concatenate the shards
python scripts/run_p5.py               # five targeted pressure families
python scripts/metrics_addendum.py     # M3, M4, MASE
python scripts/diag_attrib.py          # C1 attribution regression
python scripts/run_stats.py
python scripts/make_figures.py
```

The sharded H6 run is the one actually used; `run_onboard.py` is deterministic
given `--domains` and `--seeds`, and `merge_onboard.py` de-duplicates on
`(domain, seed, expert)` so a repeated shard cannot double-count.

The raw forecasts are cached under `results/pool` and `results/stream` and are
not distributed; the commands above rebuild them from the public source datasets
(see `docs/PROTOCOL.md` for the expected layout).

## Figures

Every figure is produced by `scripts/make_figures.py` with matplotlib, directly
from the saved result arrays.  No image-generation model is involved anywhere in
the pipeline, so any figure can be traced back to the numbers that produced it.

## License

MIT -- see `LICENSE`.

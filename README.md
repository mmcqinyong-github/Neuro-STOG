# Neuro-STOG

Code, results and figures for the CerebroCast experiments: a calendar-day
streaming architecture that maintains a heterogeneous pool of forecasting
experts, and a controlled study of *when* such maintenance actually pays.

## What is in here

| path | contents |
|---|---|
| `src/cerebrocast/` | the architecture, the seven control conditions, the synthetic generator, the statistics |
| `scripts/`        | the six pipeline stages, plus the diagnostic scripts used during development |
| `results/summary/`| every number that appears in the paper (CSV + `verdict.json` + `report.md`) |
| `figures/`        | Fig. 1-8 as vector PDF/SVG and 300 dpi PNG |
| `docs/`           | protocol description and the results write-up |

## Headline results

Read `results/summary/report.md` -- it reports the pre-registered hypotheses
exactly as they came out, including the ones that **failed**.

* The architecture does **not** beat the best causal baseline on the eleven
  standard stationary benchmarks.  We report this as a null result and explain
  why: the per-day oracle expert is far better than any causal method, but the
  signal that would identify it is not learnable from features observable before
  the day (measured correlation between predicted and realised expert quality is
  *negative*, -0.51).
* On streams where the data-generating process genuinely changes, the same
  architecture gains 12-77% over the best causal baseline, and the gain grows
  monotonically along a five-rung non-stationarity ladder.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/build_pool.py
python scripts/build_stress.py 7 42 2021 3407 12345
python scripts/run_stream.py --rounds 2
python scripts/run_synth.py
python scripts/run_stats.py
python scripts/make_figures.py
```

The raw forecasts are cached under `results/pool` and `results/stream` and are
not distributed; the commands above rebuild them from the public source datasets
(see `docs/PROTOCOL.md` for the expected layout).

## Figures

Every figure is produced by `scripts/make_figures.py` with matplotlib, directly
from the saved result arrays.  No image-generation model is involved anywhere in
the pipeline, so any figure can be traced back to the numbers that produced it.

## License

MIT -- see `LICENSE`.

# Lesson: Why the TFT forecasts weren't trustworthy

You asked for an audit of the forecasting accuracy problems. This document walks through
what systematic debugging turned up, why each issue actually hurts accuracy (as opposed to
being a style nit), and how it was verified and fixed. Three structural bugs, one crash bug,
and one missing feature — all in `src/data_pipeline.py` and `src/forecasting.py`.

The short version: **the pipeline was never actually measuring TFT/LSTM test accuracy**, and
separately, **the "ground truth" targets it trained against already contained future
information**. Both problems make offline metrics look better than real forecasting
performance, which is exactly the "looks fine in the notebook, bad in practice" pattern you
described.

---

## Finding 0 (context): there was no TFT code left to audit

Before touching anything else: the `refactor` commit (`5378e1b`) that split the monolithic
notebook into `src/` modules dropped the actual LSTM/TFT training cells. The current
`data/data_processing.ipynb` only ever evaluated `PersistenceForecaster` — a lag-24 baseline —
with a comment saying "replace or extend with NeuralForecast adapters ... when running full
training." So "the TFT model" you're seeing accuracy issues from is either an older run, or
code that hasn't been exercised since the refactor. I restored a working LSTM+TFT example
(section 4b of the notebook), wired through the fixes below.

This also means the bugs below were **latent** — nobody had run the adapter path recently
enough to notice.

---

## Finding 1: `NeuralForecastAdapter` never actually reached the test period

**File:** `src/forecasting.py`, `NeuralForecastAdapter.predict()` (was line 206-213 in the old code)

### The bug

`evaluate_forecaster()` fits the model once on `train_df`, then calls `predict()` with a
`future_df` that's either `val_df` or `test_df`. The old adapter ignored the distinction and
did:

```python
preds = self._nf.predict(futr_df=futr_df).reset_index()
```

`NeuralForecast.predict()` doesn't know what date range you *want* — it forecasts exactly `h`
steps immediately after the last timestamp the model was fit on. Since the model was fit on
`train_df` only, every call to `predict()` — whether you asked for `val` or `test` — produced
forecasts for the *same* 24-hour window: the start of the val period. When you asked for
`test`, you silently got val-period predictions merged (via an inner join on date) against
whatever actuals happened to line up.

Net effect: the reported "test MAE" was actually a val MAE in disguise, and the true test
period — the one that matters for judging whether the model generalizes to unseen future
data — was **never evaluated at all**. On top of that, even the val evaluation only ever
covered a single 24-hour window out of a period that likely spans many days, so the sample
size behind every reported metric was a tiny, non-representative slice.

This is the single most likely explanation for "looks fine in experiments, bad in practice":
you were never actually looking at held-out future performance.

### Directed proof

```
train range: 2024-01-01 to 2024-01-28   (28 days)
val range:   2024-01-29 to 2024-02-03   (6 days)
test range:  2024-02-04 to 2024-02-09   (6 days)

make_future_dataframe() output range: 2024-01-29 00:00 to 2024-01-29 23:00
>>> falls inside the VAL period, not the true TEST period
```

### The fix

`predict()` now walks forward through the *entire* requested span, in `horizon`-sized steps,
re-querying `NeuralForecast.make_future_dataframe(df=rolling_history)` at each step (per-series
correct, so it doesn't assume every unique_id shares the exact same timeline) and feeding the
true observed values back in as new history — a standard walk-forward backtest. Verified:

```
predicted date range: 2024-02-04 00:00 to 2024-02-09 23:00
true test_df range:   2024-02-04 00:00 to 2024-02-09 23:00
rows evaluated: 288 / full test_df rows: 288   (previously: ~24 rows, wrong dates)
```

Test: `tests/test_forecasting.py::test_neural_forecast_adapter_covers_full_evaluation_split`.

### A crash bug found along the way

While fixing this, `fit()`/`predict()` turned out to also raise `KeyError('y')` outright.
`ForecastConfig` names the (possibly log1p-transformed) target column `y_model`, but
`NeuralForecast` looks for a column literally named `y` unless told otherwise. The old-notebook
version dodged this by manually renaming the column before ever calling `nf.fit()`; the
refactored adapter generalized the code but dropped that rename. This means `NeuralForecastAdapter`
as shipped **could not have run successfully** against the current `ForecastConfig` at all —
another reason to suspect nobody exercised this path post-refactor. Fixed by renaming
`bundle.target_col → "y"` internally wherever a frame is handed to NeuralForecast.

---

## Finding 2: stockout reconstruction leaked the future into training targets

**File:** `src/data_pipeline.py`, `_same_hour_heuristic()` (line 81) and
`_build_reconstruction_training_frame()` (line 110)

### The bug

Before any train/val/test split happens, `reconstruct_latent_demand()` fills in "latent demand"
for stockout hours — i.e. it invents the target value (`y`) that TFT is later trained to
predict. Two independent leaks fed future information into that invented target:

1. `_same_hour_heuristic()` averaged the same hour-of-day over a **symmetric** ±7-day window
   (`day_idx - window` to `day_idx + window`), i.e. it used days *after* the stockout to fill
   in the value *for* the stockout.
2. The GBM fallback model (`HistGradientBoostingRegressor`) was trained on
   `_build_reconstruction_training_frame()`, which pulled rows from the **entire** dataset —
   train, val, *and* test — including `product_id`/`store_id` as raw features. So the model
   used to impute a train-period stockout had, in effect, already seen that exact product/store's
   future (test-period) demand pattern during its own training.

For rows near the split boundary, the "ground truth" the TFT trains against was partly
constructed from data that a real forecaster would never have had access to. This inflates
apparent forecastability near the boundary and — worse — means the reported offline accuracy
doesn't reflect what happens when the model faces genuinely new data in production.

### Directed proof

Two 30-day synthetic series, identical except for the sales values on day 10 (chronologically
*after* a stockout on day 5):

```
day-5 stockout imputed value, when day 10 (future) sales are LOW:  [4.94, 5.08]
day-5 stockout imputed value, when day 10 (future) sales are HIGH: [9.02, 9.16]

>>> Reconstructed target for an EARLIER day changed solely because of LATER data: True
```

Changing only the future changed the past. That's the leak, isolated and reproduced.

### The fix

- `_same_hour_heuristic()` is now backward-only: `window_start = day_idx - window` to
  `day_idx` (exclusive), never `day_idx + window`.
- `reconstruct_latent_demand()` / `_build_reconstruction_training_frame()` gained an optional
  `reconstruction_cutoff` parameter — rows on/after the cutoff are excluded from *training* the
  GBM model (imputation still happens for all rows, the model just never gets to learn from
  data past the cutoff). The restored notebook computes this from `train_frac + val_frac`.

Re-running the same directed test after the fix, with both the heuristic-only path and the
`reconstruction_cutoff`-protected model path:

```
day-5 imputed value (future=low):  [5.299, 5.449]
day-5 imputed value (future=high): [5.299, 5.449]
leakage present: False
```

Test: `tests/test_data_pipeline.py::test_reconstruction_does_not_leak_future_days_into_past_stockouts`.

**Worth knowing:** using *some* future context to clean historical target data is a legitimate,
common practice in offline demand reconstruction (you're allowed to use hindsight to denoise
history, since you're not "forecasting" it). The problem here specifically was leaking across
the point where you later draw your train/test boundary — that's what makes your test metric
lie to you. Cutting the leak off at `reconstruction_cutoff` keeps the (defensible) practice for
data strictly before the boundary while removing the part that corrupts evaluation.

---

## Finding 3: static covariates were computed but never reached the model

**File:** `src/forecasting.py`, `NeuralForecastAdapter.fit()` / `predict()`

### The bug

`ForecastDatasetBundle.static_df` (product_id, store_id, city_id, first_category_id per series)
was built by `prepare_forecast_frames()` — and then never used. `NeuralForecastAdapter.fit()`
called `self._nf.fit(df=...)` without a `static_df=` argument, and the TFT/LSTM model
constructors in the old notebook never passed `stat_exog_list` either
(confirmed directly: `data/lightning_logs/version_18/hparams.yaml` shows
`stat_exog_list: null` on the actual trained model).

This matters specifically for TFT, whose main architectural advantage over a plain LSTM is the
static-covariate encoder — it's what lets one shared global model distinguish ~50K different
product-store combinations instead of treating every series as interchangeable modulo recent
history. Without it, the model has to infer "which product/store is this" purely from
recent-history lag features, which is a much weaker signal, especially for low-volume or
newer series. This is a very plausible direct contributor to poor accuracy, separate from the
two leakage/evaluation bugs above.

### The fix

`fit()` now passes `static_df=bundle.static_df` through to `nf.fit()`, and `predict()` passes
it to every `nf.predict()` call in the walk-forward loop. **Note:** this alone isn't sufficient —
the model itself still needs to be constructed with `stat_exog_list=list(bundle.static_cols)`
for the static encoder to actually consume the columns. The restored notebook cell does this;
the adapter's docstring calls out the requirement so it isn't missed again if the adapter is
reused with a different model.

---

## Session note: how a laptop ran out of memory verifying this

While validating Finding 1 against the real LSTM/TFT models (not just the tiny synthetic model
in the unit test), the machine ran out of memory and hard-stopped. Root cause: the walk-forward
fix calls `NeuralForecastAdapter.predict()` once per horizon-sized window across the whole
val/test span. Each call spins up a fresh PyTorch Lightning `Trainer`, and by default each
`Trainer` spawns `num_workers` dataloader subprocesses. Multiply that by ~9 windows × 2 splits
(val+test) × 2 models (LSTM+TFT), and you get dozens of Trainer initializations with worker
subprocess churn — enough to exhaust memory on a laptop with no resource ceiling.

Fixes applied:
- The restored notebook now passes `dataloader_kwargs={"num_workers": 0}` to both models,
  which removes the subprocess-explosion vector entirely.
- For actual full-scale training (real data, real model sizes, real step counts), do it inside
  Docker with an explicit memory limit (`docker run --memory=6g ...`) so a runaway run gets
  OOM-killed inside the container instead of taking down the host — or on a dedicated remote
  GPU environment like Lightning AI Studios, which fits naturally since this stack is already
  built on PyTorch Lightning / NeuralForecast. Keep iterating on the *logic* locally with tiny
  synthetic data and tiny `max_steps` (seconds, not minutes) the way the test suite does, and
  only send the real-data, real-size run to a resource-limited or remote environment.

---

## What changed, file by file

| File | Change |
|---|---|
| `src/data_pipeline.py` | `_same_hour_heuristic` is causal-only; `reconstruct_latent_demand`/`backtest_reconstruction`/`_build_reconstruction_training_frame` accept `reconstruction_cutoff` to keep the imputation model from training on future/test rows |
| `src/forecasting.py` | `NeuralForecastAdapter.fit()` renames target col to `y` and passes `static_df`; `.predict()` walks forward across the full requested split instead of a single misaligned window |
| `tests/test_data_pipeline.py` | New regression test proving future data can no longer change a past stockout's reconstructed value |
| `tests/test_forecasting.py` | New regression test proving `evaluate_forecaster(..., evaluation_split="test")` now covers the true, full test date range |
| `data/data_processing.ipynb` | Restored LSTM/TFT training cells (missing since the refactor), wired with causal reconstruction, static covariates, walk-forward val *and* test evaluation, and `num_workers=0` |

All 16 tests pass (`python3 -m pytest tests/ -q`).

## Recommended next steps (not done here — out of scope for this pass)

- Switch `loss=MAE()` to `MQLoss(quantiles=[0.1, 0.5, 0.9])` on the TFT if you want the
  prediction intervals the RL state space (`forecast_lo_10`/`forecast_hi_90`) is designed to
  consume — right now those columns are always empty because the model only ever emits a
  point forecast.
- Consider whether `fit()` should retrain on `train_df + val_df` before the final test
  evaluation (common practice once hyperparameters are locked in), versus the current
  conservative choice of keeping the fitted weights train-only and only extending the
  *inference* history through val. Both are defensible; pick deliberately.
- Run the actual training (real HuggingFace data, real model sizes) in Docker or on Lightning AI
  per the memory note above, not on the local machine directly.

# Execution Runbook — Lightning AI Studio

A phase-by-phase plan for running this project on a GPU Studio. Each phase has a
**gate**: do not spend GPU time on the next phase until the current one passes.

The ordering is deliberate. Phases 0–1 are cheap and catch the failures that
would otherwise waste an hour of GPU time discovering the same thing.

**Status at time of writing:**

| Component | State |
|---|---|
| Data pipeline (`src/data_pipeline.py`) | Fixed + tested (causal reconstruction) |
| Forecasting (`src/forecasting.py`) | Fixed + tested (walk-forward evaluation) |
| Notebook `data/data_processing.ipynb` | LSTM/TFT restored, syntax-checked, **never run on real data** |
| RL pipeline (`src/rl_pipeline.py`) | Fixed + tested (10 defects, each with a regression test) |
| `artifacts/` | Regenerated on the Studio: features, all-split forecasts, RL results |

---

## Phase 0 — Studio setup

```bash
git clone <your-repo-url> && cd RL
pip install -r requirements.txt
```

Confirm the GPU is actually visible to PyTorch — a Studio can be attached to a
GPU while the process still silently runs on CPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"
```

**Gate:** prints `True` plus a device name. If it says `CPU ONLY`, fix the Studio
compute before continuing — TFT on CPU is the difference between minutes and hours.

### Raise the dataloader workers

`data/data_processing.ipynb` currently sets `num_workers: 0` in the model cell.
That was a deliberate laptop-safety setting (it prevented an OOM that hard-stopped
a local machine). On a Studio VM, raise it:

```python
dataloader_kwargs = {"num_workers": 4}
```

This matters: the walk-forward evaluation calls `predict()` once per horizon
window, so dataloader throughput is hit repeatedly, not just during `fit()`.

---

## Phase 1 — Verification tests (run before any GPU work)

```bash
python -m pytest tests/ -q
```

**Gate: 16 passed.** Takes ~30–60s, no GPU needed.

Four of these are regression tests guarding bugs that were actually found and
fixed in this repo. If any fail, something regressed — stop and investigate
rather than training on top of it:

| Test | Guards against |
|---|---|
| `test_reconstruction_does_not_leak_future_days_into_past_stockouts` | Future demand leaking into reconstructed stockout targets |
| `test_neural_forecast_adapter_covers_full_evaluation_split` | Evaluating on a single 24h window mislabeled as the test split |
| `test_split_and_scale_features_is_groupwise_and_keeps_target_raw` | Scaler fit on non-train rows / target being scaled |
| `test_export_rl_forecast_features_produces_stable_schema` | Forecast export schema drifting away from what RL consumes |

Run just the fast subset while iterating:

```bash
python -m pytest tests/ -q -k "not neural_forecast_adapter"
```

---

## Phase 2 — Data acquisition and feature build

Uncomment the HuggingFace loader in `data/data_processing.ipynb` cell 3. If the
dataset requires auth, set `HF_TOKEN` in the Studio environment first.

```python
splits = {'train': 'data/train.parquet', 'eval': 'data/eval.parquet'}
train_df = pd.read_parquet("hf://datasets/Dingdong-Inc/FreshRetailNet-50K/" + splits["train"])
eval_df  = pd.read_parquet("hf://datasets/Dingdong-Inc/FreshRetailNet-50K/" + splits["eval"])
raw_df = pd.concat([train_df, eval_df], ignore_index=True)
```

Then run cells through section 3 (reconstruction → features → splits).

### What to evaluate here

`reconstruction_diagnostics` is printed by the reconstruction cell. Check it
rather than skipping past it:

- **`imputation_source_counts`** — how stockout hours were filled. Heavy
  `fallback_global_mean` / `fallback_zero` means the same-hour heuristic rarely
  had enough history, and your reconstructed demand is mostly flat means. That
  caps forecast quality no matter how good the model is.
- **`masked_backtest_mae_model` vs `masked_backtest_mae_heuristic`** — if the
  GBM isn't clearly beating the median baseline, the model path isn't earning
  its complexity.
- **`total_stockout_hours` as a fraction of all hours** — this is the share of
  your target variable that is *imputed, not observed*. If it's large, say so in
  any writeup; it bounds how much the headline MAE actually means.

**A note on `reconstruction_cutoff`:** cell 5 derives it from
`train_frac + val_frac`. This is what stops the imputation model from training on
test-period rows. Don't remove it to "use more data" — that reintroduces the
leak that `test_reconstruction_does_not_leak_future_days_into_past_stockouts`
exists to catch.

**Gate:** `hourly_df` has `split ∈ {train,val,test}`, no NaNs in `y`, and
per-series timestamps are contiguous hourly.

---

## Phase 3 — Forecasting (LSTM + TFT)

Run notebook sections 4 and 4b. Start with a **smoke run** before the real one:

```python
max_steps=20   # temporarily, on both models
```

Confirm the whole path executes end to end (fit → walk-forward predict → metrics)
in a couple of minutes. Only then restore `max_steps=500` (LSTM) / `2000` (TFT).

This ordering matters more than it looks: the walk-forward evaluation is the part
most likely to fail on real data (per-series history ending at different
timestamps), and you want to find that out after 2 minutes, not after an hour
of training.

### Evaluation order

1. **Validation table** (section 4b) — model selection only.
2. **Test table** (section 4b) — the number you actually report.

### What to evaluate

- **Beat persistence.** The `Persistence` row is a lag-24 baseline. A TFT that
  does not beat it has learned nothing useful, regardless of absolute MAE. This
  is the single most important comparison in the notebook.
- **Check `bias`, not just `mae`.** Large positive bias means systematic
  over-forecasting. For an inventory system that is not a symmetric error — it
  translates directly into overstock cost downstream.
- **Compare val vs test MAE.** A large gap means the val split drove model
  selection toward something that doesn't generalize.
- **Sanity-check the scale.** MAE is reported on the original (expm1'd) scale
  because `use_log_target=True` is inverted before scoring. An MAE that looks
  suspiciously tiny usually means it's being computed in log space somewhere.

> On the `0.05 MAE` in the initial commit message: treat that number as
> unverified. It predates both the reconstruction-leak fix and the
> evaluation-window fix, and the evaluation-window bug in particular meant the
> reported "test" metric was computed over a single 24-hour window adjacent to
> training data. Expect the corrected number to be worse, and to be real.
> Re-baseline rather than trying to reproduce it.

**Gate:** TFT (or LSTM) beats persistence on **test** MAE. If neither does, the
problem is upstream — go back to Phase 2 diagnostics before tuning the model.

### Export

Run section 5. Produces `artifacts/hourly_features.parquet` and
`artifacts/rl_forecast_features.parquet`. Both are required by Phase 4.

Studio disks persist across sessions, so this only needs doing once. Pause the
Studio rather than deleting it to keep these.

---

## Phase 4 — RL (complete)

All 10 audited defects are fixed, each with a regression test. See `design.md`
Parts 2-3 for the full record of what was wrong and why it mattered. The short
version of what changed:

| # | Defect | Fix |
|---|---|---|
| 1 | Forecast 24x too small (`.mean()` vs `.sum()`) | `_get_daily_forecast` sums |
| 2 | Oracle leak — obs contained realised future demand | causal persistence proxy |
| 3 | Time limit reported as `terminated=True` | now `truncated` |
| 4 | Reward scale unmanaged (Q ~ -4,111) | `reward_scale=0.01`, raw `total_cost` |
| 5 | Dead `[1.0, 1.0]` observation slots | removed; use `covariate_matrix` |
| 6 | Action space capped at 50 units | order-up-to levels, scales with demand |
| 7 | `std_cost == 0.0` — deterministic eval | `random_start` per reset |
| 8 | Cost params made never-ordering optimal | recalibrated (10 / 20) |
| 9 | Demand/forecast date mismatch | inner-join on common dates |
| 10 | Baselines used raw quantities | `(s,S)`/EOQ emit order-up-to levels |

### Results

Product `691_843` (7.93 units/day), test split, 40 episodes, `random_start`:

| Policy | Cost | ± sd | Service |
|---|---|---|---|
| **DQN** | **385.1** | 60.5 | 0.778 |
| (s,S) | 517.1 | 108.3 | 0.713 |
| EOQ | 620.9 | 80.5 | 0.658 |
| NeverOrder | 1737.2 | 115.8 | 0.145 |

DQN beats a val-tuned `(s,S)` by 25.5% at a higher service level. Welch t = -6.73,
non-overlapping 95% CIs.

### Three protocol traps this phase walked into

Each one changed the answer, so they are worth repeating back to anyone re-running this:

1. **Tune baselines on val, not test.** `(s,S)` grid-searched on test scored 366.7;
   tuned honestly on val it scores 841.0 on test. The first comparison reported
   "DQN loses by 118.8%" against an opponent that had seen the test set.
2. **Select checkpoints on val, not test** — and take the *best*, not the last.
3. **`episode_length` must be shorter than the split.** With a 15-day test split and
   `episode_length=60`, `min(60,15)=15` leaves no slack for `random_start`, so every
   "episode" is one episode repeated and `std_cost` is exactly 0.0.

### Once fixed — what to evaluate

**This phase is CPU-bound. Do not pay for a GPU to run it.** SB3's `MlpPolicy`
with a small net trains fine on CPU; a single-product DQN is a few minutes.
Switch the Studio back to CPU, or run it locally.

Evaluate in this order:

1. **Baselines first, on the test split.** `(s,S)`, `EOQ`, and `never-order`.
   Never-order is the one people forget, and it is the one that catches a
   miscalibrated cost function (defect 8).
2. **Learning curve across epochs.** Periodically evaluate on held-out data
   during training. A flat or rising cost curve means the agent isn't learning —
   check reward scale (defect 4) before touching hyperparameters.
3. **DQN vs best baseline, same env, multiple episodes with random starts.**
   Report mean *and* spread. With defect 7 fixed, `std_cost` becomes meaningful;
   if it's still `0.0`, `random_start` isn't wired in.
4. **Service level alongside cost.** A policy can win on cost by refusing to
   stock. `avg_service_level` catches that.
5. **Inspect an actual rollout.** `run_episode` returns inventory/order/stockout
   traces. Look at them. An agent that orders the same quantity every step, or
   never orders, is a degenerate solution that aggregate cost can hide.

**Honest reporting:** if DQN loses to `(s,S)` after the fixes, that is a
legitimate result worth writing down. `(s,S)` is a strong industry-standard baseline
but is NOT provably optimal here (Scarf assumes backorders + stationary demand; this env
has lost sales + seasonality), so it is a real bar under
assumptions this environment nearly satisfies, so beating it is a high bar. Tuning
until DQN wins, then reporting only that run, is how the original `0.05 MAE`
number happened.

---

## Cost and time control

Lightning's free tier is ~80 GPU-hours/month. The expensive mistakes:

- **Leaving a local IDE attached.** While VS Code is connected over SSH, the
  Studio will not auto-sleep. Disconnect when you step away.
- **Running Phase 4 on a GPU.** It's CPU work.
- **Skipping the Phase 3 smoke run.** Finding a walk-forward bug after 60 minutes
  of TFT training instead of after 2.

Rough budget: Phase 2 is CPU-bound (reconstruction is a Python loop over
product-store groups and is the slowest non-GPU step). Phase 3 is the only phase
that genuinely needs the GPU.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `KeyError: 'y'` in `nf.fit()` | Target column not renamed. `bundle.target_col` is `y_model` when `use_log_target=True`; the adapter renames it — check that path if you've edited it |
| `ValueError: There are missing combinations of ids and times in futr_df` | Per-series history ends at different timestamps. The adapter uses `make_future_dataframe(df=...)` per window for exactly this reason |
| Forecast metrics suspiciously good | Check `evaluation_split="test"` and that predicted `ds` range matches `bundle.test_df` — this was defect-fixed once already |
| Machine runs out of memory | `num_workers` too high for available RAM, or many `predict()` calls each spawning worker pools. This OOM'd a laptop; drop back to `num_workers: 0` |
| RL agent never orders | Defect 8 — cost params make it optimal. Not a bug in the agent |
| `std_cost == 0.0` | Defect 7 — `random_start` not implemented |

---

## Summary checklist

- [x] Phase 0 — GPU visible to PyTorch; `num_workers` raised
- [x] Phase 1 — 16 tests pass
- [x] Phase 2 — data loaded; reconstruction diagnostics reviewed
- [x] Phase 3 — smoke run at `max_steps=20`, then full training
- [x] Phase 3 — TFT/LSTM beats persistence on **test**
- [x] Phase 3 — artifacts exported
- [x] Phase 4 — 10 RL defects fixed, each with a regression test
- [x] Phase 4 — switch off GPU (RL is CPU work)
- [x] Phase 4 — baselines, learning curve, rollout inspection
- [ ] Studio paused

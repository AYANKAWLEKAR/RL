# Design Decisions & Challenges

A record of what was decided, what went wrong, and why — for StockSmart's forecasting
and RL pipeline. Written so a reader who wasn't there can pick it up cold.

The through-line: **almost every defect here was invisible to a passing test suite and
only appeared when the code met real data at real scale.** That's the main lesson.

---

## Part 1 — The headline finding

The project started from "big issues with accuracy for the TFT model." The reported
figure was `0.05 MAE`.

That number was never a measure of skill.

Verified on a random 50k-row sample of FreshRetailNet-50K:

| Metric | Value |
|---|---|
| Mean hourly demand | **0.042 units** |
| Median hourly demand | **0.000** |
| Hours that are exactly zero | **77.8%** |
| Mean daily demand | 1.00 unit/day |
| Series averaging ≥5 units/day | **1.5%** |

A model predicting `0` everywhere scores MAE ≈ mean(|y|) ≈ **0.042**. The headline
number was indistinguishable from predicting nothing.

Worse, it was structurally encouraged: **MAE is minimized by the conditional median**,
and the median of a 78%-zero distribution *is* zero. Training with `loss=MAE()` on this
data actively rewards the degenerate solution.

**Decision: add a `ZeroForecaster` control** and report it beside every model, forever.
`rel_zero < 1.0` is now the real gate — a model can beat the naive lag-24 baseline
while still being worse than predicting nothing. On the test split, that is exactly
what happens: Zero (0.5905) beats Persistence (0.6755).

> **Nuance worth keeping.** I predicted MAE loss would collapse TFT toward zero. On the
> *filtered* series that turned out to be **wrong** — selection raised the zero-share
> from 14.9% and TFT-MAE narrowly beat TFT-MQLoss. The concern was right about the full
> dataset and wrong about the subset actually trained on. Recorded because the wrong
> prediction is as instructive as the right ones.

---

## Part 2 — Design decisions

### D1. Restrict to non-degenerate series (502 of 50,000)

Two independent reasons, either sufficient:

- **Scale.** 4.85M daily rows → **116.4M hourly rows** after `expand_to_hourly`, which
  materializes a Python list of dicts via `.iterrows()`. Will not fit in 15GB RAM.
- **Degeneracy.** At a median of 0.00 units/hour, MAE measures sparsity, not skill.

Filtering to ≥5 units/day yields 502 series (mean 9.2, median 6.4 units/day) and a
zero-share of 14.9% instead of 77.8%. This is a **scoping decision, not a fix** — it
makes the problem well-posed rather than making a model look good.

### D2. Real data over a synthetic generator

The original approved design specified a seeded synthetic generator because no data
existed locally. Once the real dataset was cached on the Studio that premise expired.
Synthetic data would also have hidden every bug in Part 3 — all five surfaced only
under real scale, real multi-series panels, and the Studio's newer library versions.

### D3. Order-up-to action space

Actions were a fixed quantity grid `0..50`. At `lead_time=2`, a 28 units/day series
needs ~84 units of cover, so **26% of real series structurally could not avoid
stockouts** — the agent was being graded on an impossible task.

Actions are now **target inventory positions**; the env orders the shortfall. The same
grid works at any demand scale, and `(s,S)` remains directly expressible, so baselines
stay comparable.

### D4. Cost recalibration

With `fixed_order_cost=25` and `stockout_penalty=5`, never ordering is optimal below
~6.2 units/day — true for **21% of real series**. An agent that learned to order
nothing was *correct*, and would have looked broken.

Now 10 and 20. Verified by grid search across the real range:

| Daily demand | Never-order | Best (s,S) | Verdict |
|---|---|---|---|
| 5 | 19,615 | 2,240 | ordering wins 89% |
| 7 | 27,610 | 2,950 | ordering wins 89% |
| 15 | 59,602 | 4,955 | ordering wins 92% |
| 28 | 111,600 | 8,942 | ordering wins 92% |

### D5. Causal-only reconstruction

`_same_hour_heuristic` averaged a window **centered** on the stockout day, pulling in
later days; the GBM also trained on rows it would later impute. Proven with a directed
test: changing a *later* day's sales changed an *earlier* day's reconstructed value.

Now backward-only, plus a `reconstruction_cutoff` so the imputation model never trains
on test-period rows.

### D6. Walk-forward evaluation

A single `nf.predict()` only forecasts `h` steps after the fitted data. The original
code called it once and labelled the result "test" — so the reported metric covered a
**single 24-hour window adjacent to training data**, not the held-out span.

Now walks forward across the whole split, feeding back observed values as history.
Window boundaries come from `make_future_dataframe(df=...)` because per-series history
ends at different timestamps after lag-feature NaN drops.

### D7. Reward scaling and truncation

- Raw episode cost ≈ 1e4, so Q(s₀) ≈ **−4,111**. A freshly-initialised DQN outputs
  Q≈0 and must regress onto that. `reward_scale=0.01`; `total_cost` stays raw for
  reporting.
- Time limits reported `terminated=True`, so SB3 stopped bootstrapping at the boundary
  (`target = r` instead of `r + γ·maxQ'`), biasing values down. Now `truncated`.

### D8. `random_start`

A fixed start makes a single-product env deterministic: `std_cost == 0.0` always, and
"average over N episodes" measures **one episode N times**. Each reset now samples a
window, so evaluation has real variance.

---

## Part 3 — Challenges overcome

Five bugs, none of which any local test could have caught.

### C1. The adapter scored the **row index** as the forecast

The worst one. Symptom: `TFT-MAE mae=inf`, 94% of predictions past `expm1`'s overflow
threshold. Diagnostic output:

```
=== RAW predictions in LOG space (col=index) ===
  q0 = 0.0    q0.5 = 6023.5    q1.0 = 12047.0     (n = 12,048)
```

Values running 0→12,047 for 12,048 rows are **row numbers**.

NeuralForecast changed its output shape: ≤3.1 returns `unique_id` as the DataFrame
index, ≥3.2 as a regular column with a RangeIndex. The adapter called `.reset_index()`
unconditionally, which on ≥3.2 injects an `index` column that sorts first — and
`model_cols[0]` picked it.

Invisible locally because the pinned version there is **3.1.5**, and the existing
adapter test asserted forecast **dates**, never **values**. The version gap noted as a
Phase-0 footnote turned out to be the thing that mattered.

It also produced a false result: TFT-MQLoss appeared to beat TFT-MAE only because
MQLoss emits a `-median` column that column-selection caught, while MAE fell through to
the index. Not a loss-function finding at all.

### C2. `val_size` — per-series timesteps vs row count

NeuralForecast's `val_size` counts timesteps **per series**; the adapter passed
`len(val_df)`, a total row count. With 502 series of ~1,629 train steps each:
`1629 − 175198 = −173,569`, and every TFT run failed while only baselines produced
numbers. Works by accident at 1–2 series, which is all the tests covered.

### C3. CUDA OOM

TFT's `windows_batch_size` defaults to **1024**; memory scales with
`windows_batch × input_size × hidden_size`. At 168×128 that exceeds a 14.5GB T4. Set
explicitly → 14GB down to **1.5GB**.

### C4. Reconstruction was quadratic in practice

`model.predict()` on a **one-row DataFrame per stockout hour** (~292k calls), plus
`backtest_reconstruction` rebuilding the training frame and fitting a **second
identical GBM** for diagnostics. Stage 1 exceeded 20 minutes and never finished.

Batched into one `predict()` call and the fitted model threaded through:
**>20 min → 156s**. Imputations are provably independent — the heuristic reads
`sales`/`status`, which are never mutated, and `latent` is written but never read.

### C5. The oracle leak

With `forecast_series=None`, `_get_obs` fell back to `self.demand`, so the observation
contained the exact realised future:

```
actual demand[0:4] = [ 7. 99.  3. 42.]
obs forecast block = [ 7. 99.  3. 42.]
```

Any "RL beats baseline" result under that setup measured an oracle. Now a causal
persistence proxy built only from demand strictly before `t`.

---

## Part 4 — Results

Test split, 175,700 rows, 502 series:

| Model | MAE | rel_zero | rel_persist |
|---|---|---|---|
| Zero | 0.5905 | 1.000 | 0.874 |
| Persistence (lag-24) | 0.6755 | 1.144 | 1.000 |
| **TFT-MAE** | **0.2005** | **0.339** | **0.297** |
| TFT-MQLoss | 0.2065 | 0.350 | 0.306 |

TFT is **66% better than predicting zero** and **70% better than persistence**, with
small bias (−0.046) and reasonable val→test generalization (0.172 → 0.200).

This is the first number in the project that means anything: every earlier TFT figure
was either the all-zeros baseline (`0.05 MAE`), a crash (`n_obs=0`), or row indices
(`mae=inf`).

---

## Part 5 — Standing caveat

**32.6% of the target is imputed, and 98.2% of that came from the GBM**, not the
transparent same-hour heuristic — `short_stockout_max_hours=3` routes any day with >3
stockout hours to the model, and at a ~25% stockout rate that is most days.

So TFT is substantially learning to predict **another model's output**. Three
consequences:

1. **It flatters the metric.** A model's output is smoother than reality, so the
   imputed third is *easier* to predict than real demand.
2. **It cannot be validated.** There is no ground truth for stockout hours. The
   reported backtest (GBM 0.115 vs heuristic 0.209) scores the GBM on **in-stock**
   hours only — it says the model fits observed data, not that it imputes unobserved
   demand correctly, which is the only thing it's used for.
3. **It propagates.** The RL agent will optimize against demand that is one-third
   synthetic.

No code fix removes this. It is a property of the dataset plus the reconstruction
choice. **Any accuracy figure from this project should carry it.** The cheapest
improvement is to score only on hours where demand was actually observed — the number
gets worse and more honest.

---

## Part 6 — Open items

- **DQN not yet trained.** The env is now correct and calibrated; the agent hasn't run.
- **Forecast covers the test split only.** `make_env(split="train", forecast_df=...)`
  has no forecasts for train. `on_missing_forecast="naive"` unblocks it, but training
  on naive forecasts and testing on model forecasts is a **distribution shift** —
  a real decision, not a detail.
- **Only `forecast_median` exported.** No prediction intervals, because a point loss
  has no quantiles. MQLoss was within 3% and would supply them.
- **`predict()` re-processes full history each window** — O(n²), ~14 min per
  evaluation. Correct but wasteful.
- **`inverse_transform_predictions` has no clipping.** C1 was the source of the huge
  values so this no longer fires, but a single diverged prediction could still destroy
  a metric.

### On the eventual DQN result

`(s,S)` is provably optimal for this problem class under fairly general conditions, so
beating it is a **high bar, not a formality**. If DQN loses after a correct comparison,
that is a legitimate result worth writing down. Tuning until it wins and reporting only
that run is the same failure mode that produced the original `0.05 MAE`.

---

## Part 7 — RL results (single-product DQN)

Product `691_843`, 7.93 units/day (mid-range by construction, so the result is not an
artifact of a distribution edge). Test split, 40 episodes, `random_start`.

| Policy | Cost | ± sd | Service level |
|---|---|---|---|
| **DQN** | **385.1** | 60.5 | 0.778 |
| (s,S) | 517.1 | 108.3 | 0.713 |
| EOQ | 620.9 | 80.5 | 0.658 |
| NeverOrder | 1737.2 | 115.8 | 0.145 |

**DQN beats a val-tuned (s,S) by 25.5%** — at a *higher* service level (0.778 vs
0.713), so it is not buying cost savings by refusing to stock. Welch's t = −6.73
(df ≈ 61); the 95% CIs do not overlap ([366, 404] vs [484, 551]), so the gap is not
noise. The rollout uses 6 distinct order sizes with 50% zeros — not degenerate.

Learning curve (val) shows genuine learning then overfitting: 258 → 190 → **186 at
15k** → drifting up to 378 by 60k. The best-on-val checkpoint is what gets scored.

### Three methodology errors caught while producing this

Recorded because each one *changed the answer*, and the first two would have shipped a
wrong conclusion in opposite directions.

1. **Baseline overfit to test.** `(s,S)` was originally grid-searched on the test
   split, scoring 366.7. Tuned honestly on val it scores **841.0** on test. The first
   run therefore reported "DQN loses by 118.8%" against an opponent that had seen the
   test set. Both sides are now tuned/selected on val and scored once on test.
2. **Test leakage into checkpoint selection.** The learning curve originally ran on
   *test*, and the reported model was the *final* checkpoint rather than the best.
   Now the curve runs on val, and the best-on-val checkpoint is restored before the
   single test evaluation.
3. **Zero evaluation variance.** `std_cost` was exactly 0.0 because the launcher
   passed `--episode-length 60` against a 15-day test split: `min(60,15)=15` leaves no
   slack for `random_start`, so all 40 "episodes" were one episode repeated. With
   `episode_length=10` there are 6 distinct starts and the numbers carry real error
   bars. Every figure before this fix was a single-episode point estimate.

### Caveats on this result

- **One product, one seed.** No claim of generality. The category-level agent is
  untrained, and no seed-variance study was run.
- **Short horizons.** 61 train days / 16 val / 15 test, episodes of 10 days. The
  `initial_inventory=20` warmup is a meaningful share of a 10-day episode.
- **Unstable training.** Val cost degrades after ~15k steps. Best-on-val checkpointing
  handles it, but a genuinely converged agent would not need rescuing.
- **The imputation caveat from Part 5 still applies** — a third of the demand the
  agent optimizes against is GBM-generated.

---

## Part 8 — Training instability: investigated, and mostly *not* fixable

The val curve improves to ~15-25k steps then degrades (185 → 271 by 60k). Investigated
with instrumentation rather than hyperparameter guessing, because two candidate causes
need opposite remedies.

### The decisive measurement

Evaluating on **train and val at every checkpoint** separates them:

| | best (25k) | final (60k) | Δ |
|---|---|---|---|
| train cost | 188.5 | 200.6 | **+12.1** |
| val cost | 186.1 | 271.3 | **+85.1** |

Train barely moves while val degrades 7× more → **overfitting**, not optimization
blowup. Corroborated by `|Q|` growing only 2.3× (2.67 → 6.14, bounded) and TD loss
staying in range; a diverging Q-function would show neither.

### A real bug found on the way — that did NOT fix it

Every checkpoint reported `eps=0.010`, including the first at 5k steps. Cause: the
training loop called `agent.learn(chunk, reset_num_timesteps=False)` repeatedly, and
SB3 recomputes `_total_timesteps` per call, so the ε-schedule collapses to its floor
inside the first chunk. **The agent explored for ~2k of 60k steps.**

Hypothesis: restoring the schedule would cure the degradation. A/B, same seed and
budget, single variable:

| Variant | ε range | best val | final val | degradation |
|---|---|---|---|---|
| CHUNKED (was) | 0.010 … 0.010 | 182.4 @25k | 270.5 | +88.1 |
| SINGLE (fixed) | 0.010 … 0.794 | 179.4 @45k | 266.2 | **+86.9** |

The schedule is demonstrably repaired, and the degradation is **unchanged**. The
hypothesis was wrong. Recorded because a plausible-sounding fix that changes nothing is
worth knowing about — it would otherwise get "fixed" again by the next person.

The ε bug is still fixed, on correctness grounds: an agent that stops exploring after
3% of its budget is not doing what the config says.

### What actually causes it

**Not enough training data.** 61 train days at `episode_length=10` is ~51 distinct
windows; a 60k-step budget is ~6,000 episodes, so each window is replayed ~118 times.
Both A/B variants overfit identically because both see the same tiny window set.

This is a data-quantity limit, not a bug, and it has no hyperparameter fix. The
remedies are:

1. **Best-on-val checkpointing** — already in place. For overfitting this is the
   *correct* answer, not a workaround.
2. **More data** — train across many products rather than one product's 61 days.
   That is exactly what `CategoryInventoryEnv` is for, and is the strongest reason to
   build the category agent beyond the product claim itself.
3. **Smaller budget.** 60k steps over 51 windows is far past the point of return;
   the seed study uses 30k.

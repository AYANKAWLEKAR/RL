# ML Technical Decisions

Every substantive ML decision in this project, the problem that forced it, and how it
was resolved. Written for a reader who wasn't there.

**How this differs from the other docs:** `design.md` is the chronological narrative of
what happened. This file is the decision index — one entry per decision, each answering
*what problem forced this?* and *how do we know it worked?* If you only read one file to
judge the engineering, read this one.

**The pattern worth noticing:** most entries below were not found by reasoning about the
code. They were found by *measuring* it — and several overturned a confident prior. The
wrong predictions are kept in, marked, because a decision log that only records the
correct calls is a marketing document.

---

## Contents

| # | Decision | Forced by |
|---|---|---|
| [D1](#d1) | Add a zero-forecast baseline | Headline metric was measuring data sparsity |
| [D2](#d2) | Restrict to 502 non-degenerate series | 116M-row blowup + a degenerate metric |
| [D3](#d3) | Causal-only demand reconstruction | Future data leaked into past targets |
| [D4](#d4) | Walk-forward evaluation | "Test" metric covered one 24h window |
| [D5](#d5) | Select the median column explicitly | Row index was scored as the forecast |
| [D6](#d6) | `val_size` in per-series timesteps | Every TFT run failed at 502 series |
| [D7](#d7) | Bound `windows_batch_size` | CUDA OOM on a 14.5GB T4 |
| [D8](#d8) | Batch the imputation predictions | Reconstruction never finished |
| [D9](#d9) | Order-up-to action space | 26% of series couldn't avoid stockouts |
| [D10](#d10) | Recalibrate the cost function | Never-ordering was optimal for 21% of series |
| [D11](#d11) | Scale reward, truncate not terminate | Q-targets near −4,000; biased bootstrapping |
| [D12](#d12) | `random_start` for evaluation | `std_cost` was always exactly 0.0 |
| [D13](#d13) | Days-of-cover normalization | One action meant different things per product |
| [D14](#d14) | Bound the forecast to its causal prefix | 6 of 7 forecast slots leaked the future |
| [D15](#d15) | Seed and pair evaluation | Numbers weren't reproducible; baseline handicapped |
| [D16](#d16) | Best-on-val checkpointing | Training overfits a 61-day split |
| [D17](#d17) | Forecast ablation | Central claim was asserted, never tested |

---

<a name="d1"></a>
## D1 — Add a zero-forecast baseline

**Problem.** The project began from "big issues with TFT accuracy," with a reported
`0.05 MAE`. Measuring the data showed why that number meant nothing:

| Metric | Value |
|---|---|
| Mean hourly demand | 0.042 units |
| Median hourly demand | 0.000 |
| Hours exactly zero | 77.8% |

A model predicting `0` everywhere scores MAE ≈ mean(|y|) ≈ **0.042**. The headline was
indistinguishable from predicting nothing. Worse, it was *structurally encouraged*: MAE
is minimized by the conditional median, and the median of a 78%-zero distribution is
zero — so `loss=MAE()` actively rewards the degenerate solution.

**Decision.** Add `ZeroForecaster` and report `rel_zero` beside every model. A model that
cannot beat predicting zero has learned nothing, regardless of its MAE.

**Verification.** On the test split, Zero (0.5905) **beats** lag-24 Persistence (0.6755)
— so "beat the naive baseline" was too weak a gate. `rel_zero < 1.0` is the real one.

**Where I was wrong.** I predicted MAE loss would collapse TFT toward zero. On the
*filtered* series (D2) it did not — selection dropped the zero-share to 14.9%, the
conditional median stopped being zero, and TFT-MAE narrowly beat TFT-MQLoss. The concern
was right about the full dataset and wrong about the subset actually trained on.

---

<a name="d2"></a>
## D2 — Restrict to 502 non-degenerate series

**Problem.** Two independent blockers, either sufficient:

1. **Scale.** 4.85M daily rows → **116.4M hourly rows** after `expand_to_hourly`, which
   materializes a list of dicts via `.iterrows()`. Does not fit in 15GB RAM.
2. **Degeneracy.** At a median of 0.00 units/hour, MAE measures sparsity, not skill (D1).

**Decision.** Filter to series averaging ≥5 units/day → 502 series (mean 9.2, median 6.4),
zero-share 14.9% instead of 77.8%.

**Framing that matters.** This is a **scoping decision, not a fix**. It makes the problem
well-posed; it does not make a model look good. The 116M-row ceiling is a real engineering
limit of the current `.iterrows()` implementation and is documented as such rather than
hidden.

---

<a name="d3"></a>
## D3 — Causal-only demand reconstruction

**Problem.** The dataset records *sales*, not *demand* — during a stockout, sales are zero
but demand was not. Reconstruction fills those hours. But `_same_hour_heuristic` averaged a
window **centered** on the stockout day, pulling in *later* days, and the GBM trained on
rows it would later impute.

**How it was caught.** A directed test: hold everything fixed, change only a *later* day's
sales, and check whether an *earlier* day's reconstructed value moves.

```
day-5 stockout imputed, future day LOW : [4.94, 5.08]
day-5 stockout imputed, future day HIGH: [9.02, 9.16]   <- leak
```

**Decision.** Backward-only window, plus `reconstruction_cutoff` so the imputation model
never trains on test-period rows. Same test now asserts the values are identical.

**Standing caveat this does not remove.** **32.6% of the target is imputed**, and 98.2% of
that comes from the GBM rather than the transparent heuristic (`short_stockout_max_hours=3`
routes any day with >3 stockout hours to the model, and at a 25% stockout rate that's most
days). So the forecaster is substantially learning to predict *another model's output*.
No code change fixes this — it is a property of the dataset plus the reconstruction choice,
and every accuracy figure here should carry it.

---

<a name="d4"></a>
## D4 — Walk-forward evaluation

**Problem.** A single `nf.predict()` only forecasts `h` steps past the fitted data. The
original code called it once and labelled the result "test" — so the reported metric
covered a **single 24-hour window adjacent to training data**, not the held-out span.

**Decision.** Walk forward across the whole split, feeding back observed values as history.
Window boundaries come from `make_future_dataframe(df=...)` rather than a global date union,
because per-series history ends at different timestamps after lag-feature NaN drops.

**Cost discovered later.** The naive version carried the full accumulated history into
every call — O(n²), ~14 min per split. The model only conditions on `input_size` steps, so
trimming is result-neutral: **68 windows went from ~14 min to 39s.**

---

<a name="d5"></a>
## D5 — Select the prediction column explicitly

**Problem.** `TFT-MAE mae=inf`, with 94% of predictions past `expm1`'s overflow point.
Direct inspection of the raw output:

```
=== RAW predictions in LOG space (col=index) ===
  q0 = 0.0   q0.5 = 6023.5   q1.0 = 12047.0     (n = 12,048)
```

Values running 0 → 12,047 for 12,048 rows are **row numbers**. NeuralForecast changed its
output shape between versions: ≤3.1 returns `unique_id` as the index, ≥3.2 as a regular
column with a RangeIndex. An unconditional `.reset_index()` therefore injects an `index`
column that sorts first — and `model_cols[0]` picked it.

**Decision.** Normalize the frame shape explicitly and exclude reserved column names,
preferring an explicit `-median` column under quantile loss.

**Why it hid.** The pinned local version was **3.1.5** (old layout), and the existing
adapter test asserted forecast **dates**, never **values**. A version gap noted as a
footnote turned out to be the thing that mattered.

**It also produced a false result.** TFT-MQLoss appeared to beat TFT-MAE purely because
MQLoss emits a `-median` column that column-selection caught, while MAE fell through to the
index. Not a loss-function finding at all.

---

<a name="d6"></a>
## D6 — `val_size` counts per-series timesteps

**Problem.** Every TFT run failed; only baselines produced numbers. NeuralForecast's
`val_size` is a count of timesteps **per series**, but the adapter passed `len(val_df)` —
a total row count. With 502 series of ~1,629 train steps: `1629 − 175198 = −173,569`.

**Decision.** Convert to per-series and clamp so each series retains at least
`input_size + h` steps.

**Why it hid.** Works by accident at 1–2 series, which is all the tests covered. The
regression test now uses 60 series so the old behavior fails it.

---

<a name="d7"></a>
## D7 — Bound `windows_batch_size`

**Problem.** CUDA OOM at 14GB on a 14.5GB T4.

**Diagnosis.** TFT's `windows_batch_size` defaults to **1024**, and memory scales with
`windows_batch × input_size × hidden_size`. At 168 × 128 that exceeds the card.

**Decision.** Set it explicitly (128). Memory went 14GB → **1.5GB**, and training ran at
94% GPU utilization.

---

<a name="d8"></a>
## D8 — Batch the stockout imputation

**Problem.** Stage 1 exceeded 20 minutes and never finished on a 502-series slice.

**Diagnosis.** Two hot-path defects: `model.predict()` was called on a **one-row DataFrame
per stockout hour** (~292k calls, dominated by per-call overhead), and
`backtest_reconstruction` rebuilt the training frame and fit a **second identical GBM**
purely for diagnostics.

**Decision.** Collect all model-imputed hours and predict in one batched call; thread the
already-fitted model into the diagnostics. Safe because imputations are independent — the
heuristic reads `sales`/`status`, which are never mutated, and `latent` is written but
never read.

**Verification.** **>20 min (unfinished) → 156s.** The test asserting exact imputation
source counts passes unchanged, which is what makes it a refactor rather than a rewrite.

---

<a name="d9"></a>
## D9 — Order-up-to action space

**Problem.** Actions were a fixed quantity grid `0..50`. At `lead_time=2`, a 28 units/day
series needs ~84 units of cover, so **26% of real series structurally could not avoid
stockouts** — the agent was being graded on an impossible task.

**Decision.** Actions select a **target inventory position**; the env orders the shortfall.
The same grid works at any demand scale, and `(s,S)` stays directly expressible so baselines
remain comparable.

---

<a name="d10"></a>
## D10 — Recalibrate the cost function

**Problem.** With `fixed_order_cost=25` and `stockout_penalty=5`, never ordering is optimal
below ~6.2 units/day — true for **21% of real series**. An agent that learned to order
nothing was *correct*, and would have looked broken.

**Decision.** `fixed_order_cost=10`, `stockout_penalty=20`. Verified by grid search across
the real demand range:

| Daily demand | Never-order | Best (s,S) | Verdict |
|---|---|---|---|
| 5 | 19,615 | 2,240 | ordering wins 89% |
| 15 | 59,602 | 4,955 | ordering wins 92% |
| 28 | 111,600 | 8,942 | ordering wins 92% |

**Why never-order is a permanent baseline.** It is the control that catches exactly this
class of miscalibration, and it is the one people omit.

---

<a name="d11"></a>
## D11 — Scale the reward; truncate rather than terminate

**Two problems.**

1. Raw episode cost ≈ 1e4, so Q(s₀) ≈ **−4,111**. A freshly-initialised DQN outputs Q≈0
   and must regress onto that — huge TD targets from step one.
2. Time limits reported `terminated=True`, so SB3 stopped bootstrapping at the boundary
   (`target = r` instead of `r + γ·max Q'`), systematically biasing values down.

**Decisions.** `reward_scale=0.01` (with `total_cost` kept in raw units for reporting), and
report the time limit as `truncated`. The replay-buffer seeding path was corrected to match
— the stored bootstrap flag stays `False` on truncation.

---

<a name="d12"></a>
## D12 — `random_start` for evaluation

**Problem.** A fixed start makes a single-product env deterministic: `std_cost == 0.0`
always, so "average over N episodes" measures **one episode N times**.

**Decision.** Sample an episode window each reset.

**Trap discovered while using it.** `episode_length` must be **shorter than the split**. With
a 15-day test split and `episode_length=60`, `min(60,15)=15` leaves zero slack and every
"episode" is identical again — silently. This bit once: a full result table was produced
with `std_cost=0.0` throughout before the cause was spotted.

---

<a name="d13"></a>
## D13 — Days-of-cover normalization for the shared policy

**Problem.** Training one policy across 400 series spanning **5.4–28.2 units/day (5.2×)**
was blocked by two representation defects:

1. Product identity was `_current_idx / n_products` — an **arbitrary ordinal position in a
   list**, carrying no information about the product's demand scale.
2. Actions were absolute units, so action 4 (=20 units) is **4 days of cover** for a 5/day
   series and **0.7 days** for a 28/day one. The same action meant different things.

Left alone, the agent could only learn an average policy wrong at both ends — and the
result would have read as "the shared policy doesn't work" when the state representation
was at fault.

**Decision.** `scale_normalize` expresses state *and* actions in **days of cover** using a
per-series demand scale from `compute_train_scales` (TRAIN split only, so an evaluation env
never normalizes with held-out demand). `sS_policy_scaled` gives the baseline the same
treatment — a fixed-unit `(s,S)` against a scale-aware agent would be a rigged comparison.

---

<a name="d14"></a>
## D14 — Bound the forecast to its causal prefix

**Problem — the most serious defect found, and it survived an earlier "fix".** D-era work
closed an obvious oracle leak (the env fell back to true future demand when no forecast was
supplied). But *with* a forecast, the block still leaked.

The forecast series is a **rolling 1-day-ahead** forecast: `stage3` walks forward with
`h=24h` and re-bases on true actuals every window, so `forecasts[d]` was conditioned on
realized demand through day `d-1`. The env read `forecasts[now : now+7]`:

```
slot 0: conditioned through day now-1  -> OK
slot 1: conditioned through day now    -> LEAK
...
slot 6: conditioned through day now+5  -> LEAK
```

**6 of 7 slots carried demand the agent had not observed**, and the leak grew with slot
index. `(s,S)` reads no forecast, so 100% of the exploitation was agent-side — which
plausibly explains the signature result of *higher service level and lower cost*, exactly
what future knowledge buys.

**How it hid.** The causality test covered only the `forecast_series=None` path. The branch
every headline result ran under had **no causality test at all**.

**Decision.** `forecast_causal_steps` (default 1) bounds the block to the causal prefix;
remaining slots persist the last causal value. A regression test now pins the
`forecast_series is not None` path with an identifiable series.

**Honest consequence.** Every RL number produced before this fix is inflated by an unknown
amount and had to be re-measured. Restoring a genuine 7-day lookahead requires training the
TFT with `h=168` (single-origin multi-step), which is future work — the current agent sees
one causal day.

---

<a name="d15"></a>
## D15 — Seed and pair the evaluation

**Problem.** `evaluate_policy` called `env.reset()` with no seed. Three consequences:

- The baseline and the agent were scored on **different random episodes** — unpaired, with
  needless variance, when a paired design was free.
- A 36-point `(s,S)` grid search argmin'd over **evaluation noise** rather than policy
  quality, systematically handicapping the baseline the agent was compared against.
- **No reported number was reproducible.** Re-running any stage produced different figures,
  which defeats the purpose of committing results at all.

**Decision.** `evaluate_policy(..., seed=0)` by default; episode *i* uses `seed + i`, so any
two policies scored with the same seed face identical windows and products.

**Known limit not fixed by seeding.** The test split is 15 days and `episode_length=10`, so
there are only **6 distinct start windows**, overlapping by 5–9 days. Reporting "40
episodes" as 40 independent samples overstates the evidence — the effective *n* is closer to
6, and confidence intervals computed from 40 are too tight. Stated rather than papered over.

---

<a name="d16"></a>
## D16 — Best-on-val checkpointing, and why the instability isn't fixable

**Problem.** Val cost improved to ~15–25k steps then degraded (185 → 271 by 60k).

**How it was diagnosed.** Two candidate causes need *opposite* remedies, so guessing was
not an option. Evaluating on **train and val at every checkpoint** separates them:

| | best (25k) | final (60k) | Δ |
|---|---|---|---|
| train | 188.5 | 200.6 | **+12.1** |
| val | 186.1 | 271.3 | **+85.1** |

Train barely moves while val degrades 7× more → **overfitting**, not optimization blowup
(corroborated by `|Q|` growing only 2.3× and staying bounded).

**A real bug found on the way that did NOT fix it.** Every checkpoint reported `eps=0.010`,
including the first: calling `learn(chunk, reset_num_timesteps=False)` in a loop makes SB3
recompute `_total_timesteps` per call, collapsing the ε-schedule inside the first chunk.
**The agent explored for ~2k of 60k steps.** An A/B with the schedule repaired:

| Variant | ε range | degradation |
|---|---|---|
| chunked | 0.010 … 0.010 | +88.1 |
| single `learn()` | 0.010 … 0.794 | **+86.9** |

**Unchanged.** The hypothesis was wrong. Recorded because a plausible fix that changes
nothing would otherwise be re-attempted. The ε bug was still fixed on correctness grounds.

**Actual cause.** Data quantity: 61 train days at `episode_length=10` is ~51 distinct
windows, replayed ~118 times over a 60k budget. No hyperparameter addresses that.

**Decisions.** (a) best-on-val checkpointing — the *correct* answer to overfitting, not a
workaround; (b) more data via the shared multi-category policy (D13).

**Confirmation.** Training the shared policy over 400 series drove val degradation from
**+86.9 → +0.0**: the best checkpoint was the final one, and the curve was still improving
when the budget ran out. Removing the limit removed the instability, which is the cleanest
possible test of the diagnosis.

---

## Evaluation protocol — the rules these decisions produced

Three protocol errors were made *while producing results*, each of which changed the
answer. They're listed as rules because they generalize:

1. **Tune baselines on validation, never on test.** `(s,S)` grid-searched on test scored
   366.7; tuned honestly on val it scored **841.0** on test. The first comparison reported
   "DQN loses by 118.8%" against an opponent that had seen the test set.
2. **Select checkpoints on validation, and take the best — not the last.** The first run
   reported the final checkpoint (802.3) when the best was 419.3.
3. **`episode_length` must be shorter than the split**, or `random_start` has no slack and
   every episode is identical (D12).

---

## Open items, stated rather than buried

- **The imputation caveat (D3).** 32.6% of the target is GBM output. Scoring forecast
  accuracy on *observed-only* hours would quantify how much this flatters the metric; it is
  a one-line mask and is the highest-credibility-per-hour item outstanding.
- ~~**No forecast ablation.**~~ **Done — see [D17](#d17).** The answer is that the TFT
  forecast provides **no measurable benefit** over a naive forecast or over zeros
  (n=20/arm, both comparisons insignificant). A forecast-driven base-stock baseline is
  still missing and remains the natural next comparison.
- **Effective sample size (D15).** 15-day test split, 6 distinct overlapping windows.
  Reported confidence intervals are too tight.
- **Single-origin multi-step forecasts (D14).** Restoring a genuine 7-day lookahead needs a
  TFT trained with `h=168`.
- **`expand_to_hourly` scaling ceiling (D2).** `.iterrows()` caps the pipeline well below
  the full 50k-series dataset.

---

<a name="d17"></a>
## D17 — Forecast ablation: the transformer does not measurably help

**The gap this closes.** Every earlier result compared a DQN that reads a TFT forecast
against `(s,S)` and EOQ, which read no forecast at all. So "DQN beats `(s,S)` by 26%"
conflated a better **algorithm** with more **information**. The project's central claim —
that transformer forecasting improves restocking — had been asserted, never tested.

**Design.** Three arms differing *only* in what fills the 7 forecast slots, dispatched at
the single point the forecast enters the observation (`forecast_mode`):

| Arm | Slots contain | Question |
|---|---|---|
| `supplied` | causal TFT forecast | the real system |
| `zeros` | `0.0` × 7 | does the feature carry **any** information? |
| `persistence` | last observed demand | does a **learned** forecast beat a **naive** one? |

**Validity check.** `(s,S)` and never-order never read the forecast, so their test costs
must be identical across arms. They were, bit-for-bit, on both scopes — so any difference
is the agent, not the harness.

![Forecast ablation](images/forecast_ablation.png)

### Result (single product, 20 seeds per arm, paired episodes)

| Arm | Test cost | ± sd | Service |
|---|---|---|---|
| **supplied (TFT)** | **416.9** | 65.2 | 0.763 |
| persistence | 427.2 | 82.6 | 0.740 |
| zeros | 434.4 | 65.9 | 0.752 |

| Comparison | diff | 95% CI | Welch t | Verdict |
|---|---|---|---|---|
| TFT vs zeros | +17.5 | ± 40.6 | −0.84 | not significant |
| TFT vs persistence | +10.2 | ± 46.1 | −0.44 | not significant |

**The TFT forecast produces no measurable benefit.** Not against a naive forecast, and
not even against blanked slots. The ordering is directionally right, but seed spread
(65–83) dwarfs the gaps between arms (10–17).

### The part worth reading: I reported a false positive first

At **n=5** the same experiment gave TFT vs zeros `t = −2.65` — significant — and I wrote
that the forecast "did show an effect, against my stated expectation."

That was wrong. At **n=20 it collapsed to −0.84.** The n=5 run happened to draw TFT's
seeds low (302–417) while persistence carried two bad ones (500.7, 595.2); 15 more seeds
regressed both to the mean.

This is the same small-sample failure this project had already flagged three times
(effective n≈6, unseeded evaluation, `episode_length > split`) — and I walked into it
anyway by running a t-test on five points and believing it. **Recorded because the
correction is the useful part, not the original claim.**

### Scope of the negative result

The agent sees **one causal day** of forecast: the D14 leak fix collapsed slots 1-6 to a
persisted value, so this compares a 1-day TFT forecast against 1-day persistence — close
to the hardest case for a transformer to distinguish itself, especially at `lead_time=2`.

The honest statement is therefore: *a 1-day-ahead TFT forecast provides no measurable
benefit over naive persistence in this environment* — **not** "forecasting doesn't help
inventory RL." Separating those needs a TFT trained at `h=168` for a genuine single-origin
7-day forecast, which is GPU-bound and has not been run.

### What this means for the architecture

As configured, the second stage is not earning its complexity. You could feed the agent a
naive persistence forecast — or zeros — and lose nothing measurable. That is a real,
reportable finding, and more useful than the 26.1% headline it qualifies.

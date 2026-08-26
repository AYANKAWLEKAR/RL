from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import ForecastConfig, export_rl_forecast_features, prepare_forecast_frames
from src.rl_pipeline import (
    CategoryInventoryEnv,
    DQNConfig,
    InventoryEnv,
    InventoryEnvConfig,
    _get_daily_demand,
    _get_daily_forecast,
    compute_train_scales,
    make_multi_product_env,
    build_dqn_agent,
    evaluate_policy,
    make_category_env,
    make_env,
    sS_policy,
)


def _forecast_df_from_hourly(split_hourly_df: pd.DataFrame) -> pd.DataFrame:
    bundle = prepare_forecast_frames(split_hourly_df, ForecastConfig())
    preds = bundle.full_df[["unique_id", "ds", bundle.raw_target_col]].rename(columns={bundle.raw_target_col: "model_prediction"})
    return export_rl_forecast_features(preds)


def test_inventory_env_reports_time_limit_as_truncation_not_termination():
    """Running out of horizon is a TRUNCATION, not a terminal state.

    Reporting terminated=True made SB3 stop bootstrapping at the episode boundary
    (target = r instead of r + gamma*maxQ'), which systematically biases values down.
    """
    env = InventoryEnv(
        demand_series=np.array([3, 4, 5], dtype=np.float32),
        forecast_series=np.array([3, 4, 5], dtype=np.float32),
        config=InventoryEnvConfig(
            lead_time=2, episode_length=3, forecast_horizon=2, stockout_history_len=3, random_start=False
        ),
    )
    obs, _ = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    obs, reward, terminated, truncated, info = env.step(2)
    assert reward < 0
    assert info["step_cost"] > 0
    assert (terminated, truncated) == (False, False)
    env.step(0)
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated is False, "the world does not genuinely end; this is a time limit"
    assert truncated is True


def test_observation_never_contains_future_demand():
    """With no external forecast the env must NOT hand the agent the realised future.

    Previously _get_obs fell back to self.demand, so the forecast block was literally
    the next H days of true demand. Any "RL beats baseline" result under that setup
    was measuring an oracle, not a policy.
    """
    demand = np.array([7.0, 99.0, 3.0, 42.0, 11.0, 8.0, 5.0, 2.0], dtype=np.float32)
    env = InventoryEnv(
        demand_series=demand,
        forecast_series=None,
        config=InventoryEnvConfig(
            forecast_horizon=4, episode_length=8, lead_time=2, stockout_history_len=2, random_start=False
        ),
    )
    obs, _ = env.reset()
    fc = obs[1 + 2 : 1 + 2 + 4]
    assert not np.allclose(fc, demand[:4]), "forecast block leaks true future demand"
    assert np.allclose(fc, 0.0), "no history at t=0, so the causal proxy is 0"

    obs, _, _, _, _ = env.step(0)
    fc = obs[1 + 2 : 1 + 2 + 4]
    assert np.allclose(fc, demand[0]), "after one step the proxy is the last OBSERVED demand"


def test_supplied_forecast_block_is_causal_not_a_rolling_lookahead():
    """The path ALL headline RL results ran under had no causality test.

    The forecast series handed to the env is a ROLLING 1-day-ahead forecast: entry d
    was produced from an origin at day d-1, re-based on true actuals. Reading the whole
    7-slot block therefore surfaced entries conditioned on demand the agent had not
    observed - up to 6 days of lookahead, growing with slot index. (s,S) reads no
    forecast, so every bit of that was agent-side advantage.

    Only the causal prefix may come from the series; the rest persists the last causal
    value.
    """
    # a forecast series that is trivially identifiable per index
    demand = np.arange(1, 21, dtype=np.float32)
    forecasts = np.arange(101, 121, dtype=np.float32)  # forecasts[i] == 101 + i
    cfg = InventoryEnvConfig(
        forecast_horizon=7, forecast_causal_steps=1, lead_time=2,
        stockout_history_len=2, episode_length=15, random_start=False,
    )
    env = InventoryEnv(demand_series=demand, forecast_series=forecasts, config=cfg)
    obs, _ = env.reset()
    fc = obs[1 + 2 : 1 + 2 + 7]

    assert fc[0] == pytest.approx(101.0), "slot 0 is the genuinely causal next-day forecast"
    # the leak would have produced 102..107 here
    assert not np.allclose(fc[1:], forecasts[1:7]), "block must not expose the rolling lookahead"
    assert np.allclose(fc[1:], 101.0), "non-causal slots persist the last causal value"

    # advancing a step advances the causal entry by exactly one
    obs, _, _, _, _ = env.step(0)
    fc = obs[1 + 2 : 1 + 2 + 7]
    assert fc[0] == pytest.approx(102.0)
    assert np.allclose(fc[1:], 102.0)


def test_forecast_mode_gives_three_clean_ablation_arms():
    """The three arms must differ ONLY in what fills the forecast slots.

    The DQN reads a forecast that (s,S) never sees, so "DQN beats (s,S)" conflates a
    better algorithm with more information. These arms isolate the forecast:
      supplied    -> TFT series (real system)
      zeros       -> does the feature carry any information at all?
      persistence -> does a LEARNED forecast beat a naive one?
    """
    demand = np.arange(1, 21, dtype=np.float32)
    forecasts = np.arange(101, 121, dtype=np.float32)
    base = dict(forecast_horizon=7, forecast_causal_steps=1, lead_time=2,
                stockout_history_len=2, episode_length=15, random_start=False)

    def block(mode, step_once):
        env = InventoryEnv(demand_series=demand, forecast_series=forecasts,
                           config=InventoryEnvConfig(forecast_mode=mode, **base))
        obs, _ = env.reset()
        if step_once:
            obs, _, _, _, _ = env.step(0)
        return obs[1 + 2 : 1 + 2 + 7], obs.shape[0]

    supplied, dim_s = block("supplied", True)
    zeros, dim_z = block("zeros", True)
    persist, dim_p = block("persistence", True)

    # observation SHAPE is identical across arms - only the contents change
    expected_dim = 1 + base["lead_time"] + base["forecast_horizon"] + base["stockout_history_len"]
    assert dim_s == dim_z == dim_p == expected_dim

    assert supplied[0] == pytest.approx(102.0), "supplied uses the causal TFT entry"
    assert np.allclose(zeros, 0.0), "zeros arm blanks the block"
    assert np.allclose(persist, demand[0]), "persistence arm is the last OBSERVED demand"
    # the arms are genuinely different signals
    assert not np.allclose(supplied, zeros)
    assert not np.allclose(supplied, persist)

    # zeros/persistence must IGNORE the supplied series entirely, so forecast quality
    # cannot leak into the control arms
    other = np.arange(901, 921, dtype=np.float32)
    for mode in ("zeros", "persistence"):
        e1 = InventoryEnv(demand_series=demand, forecast_series=forecasts,
                          config=InventoryEnvConfig(forecast_mode=mode, **base))
        e2 = InventoryEnv(demand_series=demand, forecast_series=other,
                          config=InventoryEnvConfig(forecast_mode=mode, **base))
        o1, _ = e1.reset(); o2, _ = e2.reset()
        assert np.allclose(o1, o2), f"{mode} arm must not depend on the forecast series"


def test_evaluate_policy_is_seeded_and_paired():
    """Two policies scored with the same seed must face identical episodes.

    Unseeded reset() meant the baseline and the agent were compared on different random
    windows, and an (s,S) grid search argmin'd over evaluation noise.
    """
    rng = np.random.default_rng(0)
    demand = rng.gamma(4, 3, 300).astype(np.float32)
    cfg = InventoryEnvConfig(episode_length=20, random_start=True)

    def costs(seed):
        env = InventoryEnv(demand_series=demand, config=cfg)
        return evaluate_policy(env, lambda o, e: 0, n_episodes=6, seed=seed)["avg_cost"]

    assert costs(0) == pytest.approx(costs(0)), "same seed must reproduce exactly"
    assert costs(0) != pytest.approx(costs(99)), "different seeds must draw different windows"

    # paired: the never-order and (s,S) policies see the SAME demand windows
    e1 = InventoryEnv(demand_series=demand, config=cfg)
    e2 = InventoryEnv(demand_series=demand, config=cfg)
    evaluate_policy(e1, lambda o, e: 0, n_episodes=4, seed=7)
    evaluate_policy(e2, lambda o, e: sS_policy(o, e, s=20, S=50), n_episodes=4, seed=7)
    assert e1.history["demand"] == e2.history["demand"], "same seed => same realised demand"


def test_env_aligns_demand_and_forecast_on_common_dates():
    """A model cannot forecast the first input_size steps of a series, so forecasts
    legitimately start later than demand. Both are inner-joined on date rather than
    padded with invented values."""
    ds = pd.date_range("2024-01-01", periods=24 * 10, freq="h")
    hourly = pd.DataFrame({
        "unique_id": "A", "ds": ds, "y": 1.0, "split": "train", "first_category_id": 10,
    })
    # forecast is missing the first 3 days, as a real warmup gap would be
    fc_ds = ds[ds >= pd.Timestamp("2024-01-04")]
    forecast = pd.DataFrame({"unique_id": "A", "ds": fc_ds, "forecast_median": 1.0})

    env = make_env("A", hourly, split="train", forecast_df=forecast,
                   config=InventoryEnvConfig(episode_length=5, random_start=False))

    assert len(env.demand) == 7, "10 demand days inner-joined with 7 forecast days"
    assert env.forecasts is not None and len(env.forecasts) == len(env.demand)
    assert np.allclose(env.forecasts, 24.0), "daily forecast is the SUM of hourly values"


def test_daily_forecast_sums_to_match_daily_demand():
    """_get_daily_demand sums hourly y; _get_daily_forecast must sum too.

    Using .mean() made a *perfect* hourly forecast read 24x smaller than the demand
    the agent actually faced.
    """
    ds = pd.date_range("2024-01-01", periods=72, freq="h")
    hourly = pd.DataFrame({"unique_id": "A", "ds": ds, "y": 2.0, "split": "train", "first_category_id": 10})
    perfect = pd.DataFrame({"unique_id": "A", "ds": ds, "forecast_median": 2.0})

    demand = _get_daily_demand(hourly, "A", "train")
    forecast = _get_daily_forecast(perfect, "A", set(ds.date))

    assert np.allclose(demand, 48.0)
    assert np.allclose(forecast, demand), "a perfect hourly forecast must equal daily demand"


def test_order_up_to_action_scales_beyond_a_fixed_cap():
    """Actions are target inventory POSITIONS, so reachable cover scales with demand.

    A fixed 0..50 quantity grid could not cover 26% of real series (up to 28 units/day
    at lead_time=2 needs ~84 units).
    """
    cfg = InventoryEnvConfig(n_order_levels=20, order_level_step=5.0, lead_time=2, random_start=False)
    env = InventoryEnv(demand_series=np.full(30, 28.0, dtype=np.float32), config=cfg)
    env.reset()

    assert env.action_space.n == 21
    assert cfg.max_order == 100.0, "grid must reach beyond the old 50-unit cap"
    # from empty, the top action orders the full target
    env.inventory, env.pipeline = 0.0, np.zeros(2, dtype=np.float32)
    assert env._order_from_action(20) == 100.0
    # already well stocked -> the same action orders only the shortfall
    env.inventory = 90.0
    assert env._order_from_action(20) == 10.0
    env.inventory = 120.0
    assert env._order_from_action(20) == 0.0, "never order below the target position"


def test_random_start_gives_evaluation_real_variance():
    """A fixed start makes a single-product env deterministic, so std_cost is always
    exactly 0.0 and averaging over N episodes measures one episode N times."""
    rng = np.random.default_rng(0)
    demand = rng.gamma(4, 3, 400).astype(np.float32)

    fixed = InventoryEnv(demand_series=demand, config=InventoryEnvConfig(episode_length=100, random_start=False))
    varied = InventoryEnv(demand_series=demand, config=InventoryEnvConfig(episode_length=100, random_start=True))
    policy = lambda obs, e: sS_policy(obs, e, s=20, S=50)  # noqa: E731

    assert evaluate_policy(fixed, policy, n_episodes=4)["std_cost"] == 0.0
    assert evaluate_policy(varied, policy, n_episodes=8)["std_cost"] > 0.0


def test_reward_is_scaled_but_total_cost_stays_raw():
    """Q-values must be learnable while reported cost stays interpretable."""
    cfg = InventoryEnvConfig(episode_length=5, reward_scale=0.01, random_start=False)
    env = InventoryEnv(demand_series=np.full(10, 10.0, dtype=np.float32), config=cfg)
    env.reset()
    _, reward, _, _, info = env.step(4)
    assert reward == pytest.approx(-info["step_cost"] * cfg.reward_scale)
    assert env.total_cost == pytest.approx(info["step_cost"]), "total_cost stays in raw units"


def test_scale_normalize_makes_one_action_mean_the_same_thing_across_scales():
    """A shared policy needs actions in DAYS OF COVER, not absolute units.

    Real series span 5.4-28.2 units/day. Unnormalized, action 4 (=20 units) is 4 days
    of cover for a 5/day series and 0.7 days for a 28/day one, so the same action means
    different things and no single policy can be right for both.
    """
    small = np.full(60, 5.0, dtype=np.float32)
    large = np.full(60, 28.0, dtype=np.float32)

    plain = InventoryEnvConfig(scale_normalize=False, order_level_step=5.0, random_start=False)
    env = CategoryInventoryEnv([small, large], config=plain, scales=[5.0, 28.0])
    env.reset(seed=0)
    env.inventory, env.pipeline = 0.0, np.zeros(plain.lead_time, dtype=np.float32)
    env.scale = 5.0
    small_units = env._order_from_action(4)
    env.scale = 28.0
    large_units = env._order_from_action(4)
    assert small_units == large_units == 20.0, "unnormalized: identical units, different cover"
    assert small_units / 5.0 != pytest.approx(large_units / 28.0), "cover differs wildly"

    scaled = InventoryEnvConfig(scale_normalize=True, order_level_days=0.5, random_start=False)
    env2 = CategoryInventoryEnv([small, large], config=scaled, scales=[5.0, 28.0])
    env2.reset(seed=0)
    env2.inventory, env2.pipeline = 0.0, np.zeros(scaled.lead_time, dtype=np.float32)
    env2.scale = 5.0
    s_units = env2._order_from_action(4)
    env2.scale = 28.0
    l_units = env2._order_from_action(4)
    # same DAYS of cover on both, even though the unit counts differ
    assert s_units / 5.0 == pytest.approx(l_units / 28.0) == pytest.approx(2.0)


def test_scale_normalized_observation_is_comparable_across_products():
    """Two products differing only in scale should look identical in days-of-cover."""
    cfg = InventoryEnvConfig(scale_normalize=True, forecast_horizon=3, lead_time=2,
                             stockout_history_len=2, episode_length=20, random_start=False)
    small = np.full(40, 5.0, dtype=np.float32)
    large = np.full(40, 25.0, dtype=np.float32)
    env = CategoryInventoryEnv([small, large], forecast_series_list=[small, large],
                               config=cfg, scales=[5.0, 25.0])

    env.reset(seed=0)
    env._current_idx, env.demand, env.scale = 0, small, 5.0
    env.forecasts = small
    env.inventory = 10.0  # 2 days of cover
    obs_small = env._get_obs()

    env._current_idx, env.demand, env.scale = 1, large, 25.0
    env.forecasts = large
    env.inventory = 50.0  # also 2 days of cover
    obs_large = env._get_obs()

    # index 1 is inventory in days of cover; forecast block likewise
    assert obs_small[1] == pytest.approx(obs_large[1]) == pytest.approx(2.0)
    assert np.allclose(obs_small[2:], obs_large[2:]), "only the scale feature should differ"
    assert obs_small[0] != obs_large[0], "log1p(scale) still identifies absolute size"


def test_make_multi_product_env_spans_categories_and_uses_train_scales():
    ds = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    frames = []
    for uid, cat, level in [("a", 10, 1.0), ("b", 10, 4.0), ("c", 99, 2.0)]:
        f = pd.DataFrame({"unique_id": uid, "ds": ds, "y": level, "first_category_id": cat})
        f["split"] = "train"
        f.loc[f.index[-24 * 10:], "split"] = "test"
        frames.append(f)
    hourly = pd.concat(frames, ignore_index=True)

    uids = ["a", "b", "c"]
    scales = compute_train_scales(hourly, uids)
    assert scales["a"] == pytest.approx(24.0)
    assert scales["b"] == pytest.approx(96.0)

    env = make_multi_product_env(hourly, uids, split="test", config=InventoryEnvConfig(episode_length=5),
                                 min_series_days=5, scales=scales)
    assert env.n_products == 3, "spans both categories, unlike make_category_env"
    assert env.scales == [pytest.approx(24.0), pytest.approx(96.0), pytest.approx(48.0)]


def test_category_env_samples_products():
    env = CategoryInventoryEnv(
        demand_series_list=[np.array([1, 2, 3], dtype=np.float32), np.array([2, 3, 4], dtype=np.float32)],
        uid_list=["a", "b"],
        config=InventoryEnvConfig(episode_length=3, forecast_horizon=2, stockout_history_len=2),
    )
    obs, _ = env.reset(seed=7)
    assert obs.shape[0] == env.observation_space.shape[0]
    assert env.current_uid() in {"a", "b"}


def test_make_env_and_category_env_validate_lengths(split_hourly_df: pd.DataFrame):
    forecast_df = _forecast_df_from_hourly(split_hourly_df)
    uid = split_hourly_df["unique_id"].iloc[0]
    env = make_env(uid, split_hourly_df, split="train", forecast_df=forecast_df, config=InventoryEnvConfig(episode_length=10))
    assert isinstance(env, InventoryEnv)

    category_id = int(split_hourly_df["first_category_id"].mode().iloc[0])
    cat_env = make_category_env(
        category_id,
        split_hourly_df,
        split="train",
        forecast_df=forecast_df,
        config=InventoryEnvConfig(episode_length=10),
        min_series_days=5,
    )
    assert isinstance(cat_env, CategoryInventoryEnv)

    bad_forecast = forecast_df[forecast_df["unique_id"] != uid].copy()
    with pytest.raises(ValueError):
        make_env(uid, split_hourly_df, split="train", forecast_df=bad_forecast)


def test_build_dqn_agent_and_policy_evaluation(split_hourly_df: pd.DataFrame):
    forecast_df = _forecast_df_from_hourly(split_hourly_df)
    uid = split_hourly_df["unique_id"].iloc[0]
    env = make_env(uid, split_hourly_df, split="train", forecast_df=forecast_df, config=InventoryEnvConfig(episode_length=10))
    model = build_dqn_agent(env, DQNConfig(buffer_size=100, learning_starts=1, batch_size=8, net_arch=(32, 32)))
    assert model.observation_space.shape == env.observation_space.shape

    metrics = evaluate_policy(env, lambda obs, e: sS_policy(obs, e, s=5, S=12), n_episodes=2)
    assert set(metrics) == {"avg_cost", "std_cost", "avg_service_level"}
    assert 0.0 <= metrics["avg_service_level"] <= 1.0


def test_end_to_end_smoke_training(split_hourly_df: pd.DataFrame):
    forecast_df = _forecast_df_from_hourly(split_hourly_df)
    category_id = int(split_hourly_df["first_category_id"].mode().iloc[0])
    env = make_category_env(
        category_id,
        split_hourly_df,
        split="train",
        forecast_df=forecast_df,
        config=InventoryEnvConfig(episode_length=8, forecast_horizon=3, stockout_history_len=4),
        min_series_days=5,
    )
    model = build_dqn_agent(env, DQNConfig(buffer_size=100, learning_starts=1, batch_size=8, net_arch=(32, 32)))
    model.learn(total_timesteps=10, progress_bar=False)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert isinstance(int(action), int)

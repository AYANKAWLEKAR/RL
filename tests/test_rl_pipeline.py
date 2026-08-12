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

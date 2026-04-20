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


def test_inventory_env_step_and_termination():
    env = InventoryEnv(
        demand_series=np.array([3, 4, 5], dtype=np.float32),
        forecast_series=np.array([3, 4, 5], dtype=np.float32),
        config=InventoryEnvConfig(lead_time=2, episode_length=3, forecast_horizon=2, stockout_history_len=3),
    )
    obs, _ = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    obs, reward, terminated, _, info = env.step(2)
    assert reward < 0
    assert info["step_cost"] > 0
    assert terminated is False
    env.step(0)
    _, _, terminated, _, _ = env.step(0)
    assert terminated is True


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

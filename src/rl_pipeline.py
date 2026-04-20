from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import DQN


@dataclass
class InventoryEnvConfig:
    max_order: int = 50
    lead_time: int = 2
    initial_inventory: int = 20
    holding_cost: float = 0.5
    stockout_penalty: float = 5.0
    fixed_order_cost: float = 25.0
    variable_order_cost: float = 1.0
    episode_length: int = 365
    forecast_horizon: int = 7
    stockout_history_len: int = 14


@dataclass
class DQNConfig:
    learning_rate: float = 1e-3
    buffer_size: int = 50_000
    learning_starts: int = 500
    batch_size: int = 64
    gamma: float = 0.99
    target_update_interval: int = 250
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01
    net_arch: tuple[int, ...] = (256, 256)
    seed: int = 42
    verbose: int = 0


def _coerce_forecasts(demand: np.ndarray, forecasts: np.ndarray | None) -> np.ndarray | None:
    if forecasts is None:
        return None
    if len(forecasts) != len(demand):
        raise ValueError("forecast_series must have the same length as demand_series")
    return forecasts.astype(np.float32)


class InventoryEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        demand_series: np.ndarray,
        forecast_series: np.ndarray | None = None,
        covariate_matrix: np.ndarray | None = None,
        config: InventoryEnvConfig | None = None,
    ):
        super().__init__()
        self.config = config or InventoryEnvConfig()
        self.demand = np.asarray(demand_series, dtype=np.float32)
        if len(self.demand) == 0:
            raise ValueError("demand_series cannot be empty")
        self.forecasts = _coerce_forecasts(self.demand, np.asarray(forecast_series) if forecast_series is not None else None)
        self.covariates = covariate_matrix.astype(np.float32) if covariate_matrix is not None else None
        if self.covariates is not None and len(self.covariates) != len(self.demand):
            raise ValueError("covariate_matrix must align with demand_series length")
        self.episode_length = min(self.config.episode_length, len(self.demand))
        self.n_cov = self.covariates.shape[1] if self.covariates is not None else 0

        state_dim = 1 + self.config.lead_time + self.config.forecast_horizon + 2 + self.n_cov + self.config.stockout_history_len
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.config.max_order + 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.inventory = float(self.config.initial_inventory)
        self.pipeline = np.zeros(self.config.lead_time, dtype=np.float32)
        self.stockout_history = np.zeros(self.config.stockout_history_len, dtype=np.float32)
        self.total_cost = 0.0
        self.history = {"inventory": [], "demand": [], "order": [], "cost": [], "stockout": []}
        return self._get_obs(), {}

    def _get_obs(self):
        fc = np.zeros(self.config.forecast_horizon, dtype=np.float32)
        src = self.forecasts if self.forecasts is not None else self.demand
        end = min(self.t + self.config.forecast_horizon, len(src))
        fc[: end - self.t] = src[self.t:end]

        cov = np.zeros(self.n_cov, dtype=np.float32)
        if self.covariates is not None and self.t < len(self.covariates):
            cov = self.covariates[self.t]

        return np.concatenate(
            [
                [self.inventory],
                self.pipeline,
                fc,
                [1.0, 1.0],
                cov,
                self.stockout_history,
            ]
        ).astype(np.float32)

    def step(self, action):
        order_qty = int(action)
        received = self.pipeline[0]
        self.pipeline = np.roll(self.pipeline, -1)
        self.pipeline[-1] = 0
        self.inventory += float(received)

        if order_qty > 0:
            self.pipeline[-1] += order_qty

        demand = float(self.demand[self.t]) if self.t < len(self.demand) else 0.0
        lost_sales = max(0.0, demand - self.inventory)
        self.inventory = max(0.0, self.inventory - demand)
        stockout = 1.0 if lost_sales > 0 else 0.0

        hold = self.config.holding_cost * self.inventory
        penalty = self.config.stockout_penalty * lost_sales
        order_cost = self.config.fixed_order_cost * (order_qty > 0) + self.config.variable_order_cost * order_qty
        step_cost = float(hold + penalty + order_cost)
        self.total_cost += step_cost

        self.stockout_history = np.roll(self.stockout_history, -1)
        self.stockout_history[-1] = stockout

        self.history["inventory"].append(self.inventory)
        self.history["demand"].append(demand)
        self.history["order"].append(order_qty)
        self.history["cost"].append(step_cost)
        self.history["stockout"].append(stockout)

        self.t += 1
        terminated = self.t >= self.episode_length
        return self._get_obs(), -step_cost, terminated, False, {
            "step_cost": step_cost,
            "lost_sales": lost_sales,
            "stockout": stockout,
        }

    def render(self):
        print(f"t={self.t} inv={self.inventory:.1f} pipeline={self.pipeline} total_cost={self.total_cost:.1f}")


class CategoryInventoryEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        demand_series_list: list[np.ndarray],
        forecast_series_list: list[np.ndarray] | None = None,
        uid_list: list[str] | None = None,
        config: InventoryEnvConfig | None = None,
    ):
        super().__init__()
        if len(demand_series_list) == 0:
            raise ValueError("demand_series_list cannot be empty")
        self.config = config or InventoryEnvConfig()
        self.demand_list = [np.asarray(d, dtype=np.float32) for d in demand_series_list]
        if any(len(d) == 0 for d in self.demand_list):
            raise ValueError("all demand series must be non-empty")

        self.forecast_list = None
        if forecast_series_list is not None:
            if len(forecast_series_list) != len(demand_series_list):
                raise ValueError("forecast_series_list must align with demand_series_list")
            self.forecast_list = [
                _coerce_forecasts(d, np.asarray(f)) for d, f in zip(self.demand_list, forecast_series_list)
            ]

        self.uid_list = uid_list or [str(i) for i in range(len(self.demand_list))]
        self.n_products = len(self.demand_list)
        state_dim = 1 + 1 + self.config.lead_time + self.config.forecast_horizon + 2 + self.config.stockout_history_len
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.config.max_order + 1)
        self._current_idx = 0
        self.demand = self.demand_list[0]
        self.forecasts = self.forecast_list[0] if self.forecast_list else None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_idx = int(self.np_random.integers(0, self.n_products))
        self.demand = self.demand_list[self._current_idx]
        self.forecasts = self.forecast_list[self._current_idx] if self.forecast_list else None
        self._ep_len = min(self.config.episode_length, len(self.demand))
        self.t = 0
        self.inventory = float(self.config.initial_inventory)
        self.pipeline = np.zeros(self.config.lead_time, dtype=np.float32)
        self.stockout_history = np.zeros(self.config.stockout_history_len, dtype=np.float32)
        self.total_cost = 0.0
        self.history = {"inventory": [], "demand": [], "order": [], "cost": [], "stockout": []}
        return self._get_obs(), {}

    def _get_obs(self):
        fc = np.zeros(self.config.forecast_horizon, dtype=np.float32)
        src = self.forecasts if self.forecasts is not None else self.demand
        end = min(self.t + self.config.forecast_horizon, len(src))
        fc[: end - self.t] = src[self.t:end]
        product_norm = self._current_idx / max(self.n_products - 1, 1)
        return np.concatenate(
            [[product_norm], [self.inventory], self.pipeline, fc, [1.0, 1.0], self.stockout_history]
        ).astype(np.float32)

    def step(self, action):
        order_qty = int(action)
        received = self.pipeline[0]
        self.pipeline = np.roll(self.pipeline, -1)
        self.pipeline[-1] = 0
        self.inventory += float(received)

        if order_qty > 0:
            self.pipeline[-1] += order_qty

        demand = float(self.demand[self.t]) if self.t < len(self.demand) else 0.0
        lost_sales = max(0.0, demand - self.inventory)
        self.inventory = max(0.0, self.inventory - demand)
        stockout = 1.0 if lost_sales > 0 else 0.0

        hold = self.config.holding_cost * self.inventory
        penalty = self.config.stockout_penalty * lost_sales
        order_cost = self.config.fixed_order_cost * (order_qty > 0) + self.config.variable_order_cost * order_qty
        step_cost = float(hold + penalty + order_cost)
        self.total_cost += step_cost

        self.stockout_history = np.roll(self.stockout_history, -1)
        self.stockout_history[-1] = stockout
        self.history["inventory"].append(self.inventory)
        self.history["demand"].append(demand)
        self.history["order"].append(order_qty)
        self.history["cost"].append(step_cost)
        self.history["stockout"].append(stockout)

        self.t += 1
        terminated = self.t >= self._ep_len
        return self._get_obs(), -step_cost, terminated, False, {
            "step_cost": step_cost,
            "lost_sales": lost_sales,
            "stockout": stockout,
        }

    def current_uid(self):
        return self.uid_list[self._current_idx]

    def render(self):
        print(f"product={self.current_uid()} t={self.t} inv={self.inventory:.1f} total_cost={self.total_cost:.1f}")


def _get_daily_demand(hourly_df: pd.DataFrame, uid: str, split: str) -> np.ndarray:
    series = hourly_df[(hourly_df["unique_id"] == uid) & (hourly_df["split"] == split)].sort_values("ds").copy()
    if len(series) == 0:
        return np.array([], dtype=np.float32)
    series["date"] = series["ds"].dt.date
    daily = series.groupby("date", as_index=False)["y"].sum()
    return daily["y"].to_numpy(dtype=np.float32)


def _get_daily_forecast(forecast_df: pd.DataFrame | None, uid: str, split_dates: set) -> np.ndarray | None:
    if forecast_df is None:
        return None
    series = forecast_df[forecast_df["unique_id"] == uid].sort_values("ds").copy()
    if len(series) == 0:
        return None
    series["date"] = series["ds"].dt.date
    series = series[series["date"].isin(split_dates)]
    if len(series) == 0:
        return None
    daily = series.groupby("date", as_index=False)["forecast_median"].mean()
    return daily["forecast_median"].to_numpy(dtype=np.float32)


def make_env(
    uid: str,
    hourly_df: pd.DataFrame,
    split: str = "train",
    forecast_df: pd.DataFrame | None = None,
    config: InventoryEnvConfig | None = None,
) -> InventoryEnv:
    config = config or InventoryEnvConfig()
    demand = _get_daily_demand(hourly_df, uid, split)
    if len(demand) == 0:
        raise ValueError(f"No data for uid={uid}, split={split}")
    split_dates = set(hourly_df[(hourly_df["unique_id"] == uid) & (hourly_df["split"] == split)]["ds"].dt.date)
    forecasts = _get_daily_forecast(forecast_df, uid, split_dates)
    if forecast_df is not None and forecasts is None:
        raise ValueError(f"No forecast data for uid={uid}, split={split}")
    if forecasts is not None and len(forecasts) != len(demand):
        raise ValueError("forecast length mismatch for uid")
    return InventoryEnv(demand_series=demand, forecast_series=forecasts, config=config)


def make_category_env(
    category_id: int,
    hourly_df: pd.DataFrame,
    split: str = "train",
    forecast_df: pd.DataFrame | None = None,
    config: InventoryEnvConfig | None = None,
    min_series_days: int = 30,
) -> CategoryInventoryEnv:
    config = config or InventoryEnvConfig()
    cat_mask = hourly_df["first_category_id"] == category_id
    uids = hourly_df.loc[cat_mask, "unique_id"].unique().tolist()
    demand_list: list[np.ndarray] = []
    forecast_list: list[np.ndarray] = []
    uid_list: list[str] = []

    for uid in uids:
        demand = _get_daily_demand(hourly_df, uid, split)
        if len(demand) < min_series_days:
            continue
        split_dates = set(hourly_df[(hourly_df["unique_id"] == uid) & (hourly_df["split"] == split)]["ds"].dt.date)
        forecasts = _get_daily_forecast(forecast_df, uid, split_dates)
        if forecast_df is not None and forecasts is None:
            raise ValueError(f"No forecast data for uid={uid}, split={split}")
        if forecasts is not None and len(forecasts) != len(demand):
            raise ValueError(f"forecast length mismatch for uid={uid}")
        demand_list.append(demand)
        uid_list.append(uid)
        if forecasts is not None:
            forecast_list.append(forecasts)

    if len(demand_list) == 0:
        raise ValueError(f"No valid series for category {category_id}, split={split}")

    return CategoryInventoryEnv(
        demand_series_list=demand_list,
        forecast_series_list=forecast_list if len(forecast_list) == len(demand_list) else None,
        uid_list=uid_list,
        config=config,
    )


def evaluate_policy(env, policy_fn, n_episodes: int = 5) -> dict[str, float]:
    total_costs = []
    service_levels = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action = policy_fn(obs, env)
            obs, _, terminated, _, _ = env.step(action)
        total_costs.append(env.total_cost)
        service_levels.append(1.0 - float(np.mean(env.history["stockout"])))
    return {
        "avg_cost": float(np.mean(total_costs)),
        "std_cost": float(np.std(total_costs)),
        "avg_service_level": float(np.mean(service_levels)),
    }


def sS_policy(obs, env, s: int = 10, S: int = 30) -> int:
    inv_idx = 1 if isinstance(env, CategoryInventoryEnv) else 0
    inventory = obs[inv_idx]
    pipeline_total = obs[inv_idx + 1 : inv_idx + 1 + env.config.lead_time].sum()
    inventory_position = inventory + pipeline_total
    if inventory_position <= s:
        return min(int(S - inventory_position), env.config.max_order)
    return 0


def eoq_policy(obs, env, Q: int = 15, r: int = 8) -> int:
    inv_idx = 1 if isinstance(env, CategoryInventoryEnv) else 0
    inventory = obs[inv_idx]
    pipeline_total = obs[inv_idx + 1 : inv_idx + 1 + env.config.lead_time].sum()
    if inventory + pipeline_total <= r:
        return min(Q, env.config.max_order)
    return 0


def run_episode(env, policy_fn):
    obs, _ = env.reset()
    terminated = False
    while not terminated:
        action = policy_fn(obs, env)
        obs, _, terminated, _, _ = env.step(action)
    return env.history


def seed_replay_buffer(model: DQN, env, policy_fn, n_episodes: int = 10) -> None:
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action = policy_fn(obs, env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            model.replay_buffer.add(
                obs=np.array([obs]),
                next_obs=np.array([next_obs]),
                action=np.array([action]),
                reward=np.array([reward]),
                done=np.array([terminated]),
                infos=[info],
            )
            obs = next_obs


def build_dqn_agent(env, config: DQNConfig | None = None) -> DQN:
    config = config or DQNConfig()
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        policy_kwargs=dict(net_arch=list(config.net_arch)),
        seed=config.seed,
        verbose=config.verbose,
    )

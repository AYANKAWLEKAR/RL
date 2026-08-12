from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import DQN


@dataclass
class InventoryEnvConfig:
    """Costs are calibrated against the real FreshRetailNet demand scale.

    Selected series run 5-28 units/day (median ~7). The original defaults
    (fixed_order_cost=25, stockout_penalty=5) made NEVER ORDERING provably optimal
    below ~6.2 units/day - i.e. for 21% of series the correct policy was to order
    nothing and absorb every stockout, so a "broken-looking" agent was actually
    right. stockout_penalty now clearly exceeds the amortized cost of ordering, and
    the action space is expressed as order-up-to levels so it scales with demand
    instead of capping at a fixed 50 units (which 26% of series could not live within
    at lead_time=2).
    """

    lead_time: int = 2
    initial_inventory: int = 20
    holding_cost: float = 0.5
    stockout_penalty: float = 20.0
    fixed_order_cost: float = 10.0
    variable_order_cost: float = 1.0
    episode_length: int = 365
    forecast_horizon: int = 7
    stockout_history_len: int = 14

    # Action a in {0..n_order_levels} maps to a target inventory position of
    # a * order_level_step units; the order is the shortfall against that target.
    # This scales with demand, unlike a fixed max_order cap.
    n_order_levels: int = 20
    order_level_step: float = 5.0

    # Rewards are -cost. Raw episode cost runs to ~1e4, so an unscaled reward asks a
    # freshly-initialised DQN (Q~0) to regress onto targets of order -4000. Scaling
    # keeps Q in a sane range; total_cost is still tracked in raw units for reporting.
    reward_scale: float = 0.01

    # Sample a different episode window each reset so evaluation has real variance.
    # With a fixed start, a single-product env is deterministic and std_cost is
    # always exactly 0.0, making "average over N episodes" meaningless.
    random_start: bool = True

    @property
    def max_order(self) -> float:
        """Largest orderable quantity, implied by the order-up-to grid."""
        return self.n_order_levels * self.order_level_step


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

        # The old layout reserved 2 slots for price/discount that were hardcoded to
        # [1.0, 1.0] and never populated. Real exogenous signal belongs in
        # covariate_matrix, so the dead slots are gone.
        state_dim = 1 + self.config.lead_time + self.config.forecast_horizon + self.n_cov + self.config.stockout_history_len
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.config.n_order_levels + 1)

    def _order_from_action(self, action: int) -> float:
        """Order-up-to: action picks a TARGET inventory position, not a raw quantity.

        A fixed quantity grid capped at 50 units could not cover 26% of real series
        (demand up to 28/day at lead_time=2 needs ~84 units of cover). Targeting a
        position and ordering the shortfall makes the same action grid work across
        demand scales, and makes the (s,S) baseline expressible in the same space.
        """
        target = float(action) * self.config.order_level_step
        position = self.inventory + float(self.pipeline.sum())
        return max(0.0, target - position)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_length = min(self.config.episode_length, len(self.demand))
        self._start = 0
        if self.config.random_start:
            latest = len(self.demand) - self.episode_length
            if latest > 0:
                self._start = int(self.np_random.integers(0, latest + 1))
        self.t = 0
        self.inventory = float(self.config.initial_inventory)
        self.pipeline = np.zeros(self.config.lead_time, dtype=np.float32)
        self.stockout_history = np.zeros(self.config.stockout_history_len, dtype=np.float32)
        self.total_cost = 0.0
        self.history = {"inventory": [], "demand": [], "order": [], "cost": [], "stockout": []}
        return self._get_obs(), {}

    def _forecast_block(self) -> np.ndarray:
        """Forecast the agent sees. NEVER reads self.demand at or after self.t.

        The previous version fell back to `self.demand` when no forecast was supplied,
        which put the exact realised future demand into the observation — the agent
        got an oracle and any "RL beats baseline" result was meaningless. With no
        external forecast we now use a causal persistence proxy built only from
        demand strictly before t.
        """
        horizon = self.config.forecast_horizon
        now = self._start + self.t
        if self.forecasts is not None:
            fc = np.zeros(horizon, dtype=np.float32)
            end = min(now + horizon, len(self.forecasts))
            if end > now:
                fc[: end - now] = self.forecasts[now:end]
            return fc
        last_observed = float(self.demand[now - 1]) if now > 0 else 0.0
        return np.full(horizon, last_observed, dtype=np.float32)

    def _get_obs(self):
        fc = self._forecast_block()
        now = self._start + self.t

        cov = np.zeros(self.n_cov, dtype=np.float32)
        if self.covariates is not None and now < len(self.covariates):
            cov = self.covariates[now]

        return np.concatenate(
            [
                [self.inventory],
                self.pipeline,
                fc,
                cov,
                self.stockout_history,
            ]
        ).astype(np.float32)

    def step(self, action):
        order_qty = self._order_from_action(int(action))
        received = self.pipeline[0]
        self.pipeline = np.roll(self.pipeline, -1)
        self.pipeline[-1] = 0
        self.inventory += float(received)

        if order_qty > 0:
            self.pipeline[-1] += order_qty

        idx = self._start + self.t
        demand = float(self.demand[idx]) if idx < len(self.demand) else 0.0
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
        # The episode ends because we ran out of horizon, not because the world
        # genuinely ended, so this is a TRUNCATION. Reporting terminated=True made SB3
        # stop bootstrapping at the boundary (target = r instead of r + gamma*maxQ'),
        # systematically biasing values down.
        truncated = self.t >= self.episode_length
        return self._get_obs(), -step_cost * self.config.reward_scale, False, truncated, {
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
        state_dim = 1 + 1 + self.config.lead_time + self.config.forecast_horizon + self.config.stockout_history_len
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.config.n_order_levels + 1)
        self._current_idx = 0
        self._start = 0
        self.demand = self.demand_list[0]
        self.forecasts = self.forecast_list[0] if self.forecast_list else None

    _order_from_action = InventoryEnv._order_from_action

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_idx = int(self.np_random.integers(0, self.n_products))
        self.demand = self.demand_list[self._current_idx]
        self.forecasts = self.forecast_list[self._current_idx] if self.forecast_list else None
        self._ep_len = min(self.config.episode_length, len(self.demand))
        self._start = 0
        if self.config.random_start:
            latest = len(self.demand) - self._ep_len
            if latest > 0:
                self._start = int(self.np_random.integers(0, latest + 1))
        self.t = 0
        self.inventory = float(self.config.initial_inventory)
        self.pipeline = np.zeros(self.config.lead_time, dtype=np.float32)
        self.stockout_history = np.zeros(self.config.stockout_history_len, dtype=np.float32)
        self.total_cost = 0.0
        self.history = {"inventory": [], "demand": [], "order": [], "cost": [], "stockout": []}
        return self._get_obs(), {}

    _forecast_block = InventoryEnv._forecast_block

    def _get_obs(self):
        fc = self._forecast_block()
        product_norm = self._current_idx / max(self.n_products - 1, 1)
        return np.concatenate(
            [[product_norm], [self.inventory], self.pipeline, fc, self.stockout_history]
        ).astype(np.float32)

    def step(self, action):
        order_qty = self._order_from_action(int(action))
        received = self.pipeline[0]
        self.pipeline = np.roll(self.pipeline, -1)
        self.pipeline[-1] = 0
        self.inventory += float(received)

        if order_qty > 0:
            self.pipeline[-1] += order_qty

        idx = self._start + self.t
        demand = float(self.demand[idx]) if idx < len(self.demand) else 0.0
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
        truncated = self.t >= self._ep_len  # time limit, not a true terminal state
        return self._get_obs(), -step_cost * self.config.reward_scale, False, truncated, {
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
    # MUST be .sum() to match _get_daily_demand, which sums hourly y into a daily
    # total. Using .mean() here made the forecast the agent sees 24x smaller than the
    # demand it actually faces: a *perfect* hourly forecast of 2.0 units/hour became a
    # daily forecast of 2.0 against real daily demand of 48.0.
    daily = series.groupby("date", as_index=False)["forecast_median"].sum()
    return daily["forecast_median"].to_numpy(dtype=np.float32)


def make_env(
    uid: str,
    hourly_df: pd.DataFrame,
    split: str = "train",
    forecast_df: pd.DataFrame | None = None,
    config: InventoryEnvConfig | None = None,
    on_missing_forecast: str = "raise",
) -> InventoryEnv:
    """Build a single-product environment.

    `on_missing_forecast` controls what happens when `forecast_df` carries nothing for
    this uid/split. A model's forecasts usually only cover the held-out split it was
    evaluated on, so a training env legitimately has none — but silently proceeding
    would hide a genuine data-alignment bug, so the caller has to opt in:
      "raise" (default) - treat it as an error
      "naive"           - fall back to the env's causal persistence forecast
    Note the two are not interchangeable at evaluation time: training on naive
    forecasts and testing on model forecasts changes the observation distribution.
    """
    if on_missing_forecast not in {"raise", "naive"}:
        raise ValueError("on_missing_forecast must be 'raise' or 'naive'")
    config = config or InventoryEnvConfig()
    demand = _get_daily_demand(hourly_df, uid, split)
    if len(demand) == 0:
        raise ValueError(f"No data for uid={uid}, split={split}")
    split_dates = set(hourly_df[(hourly_df["unique_id"] == uid) & (hourly_df["split"] == split)]["ds"].dt.date)
    forecasts = _get_daily_forecast(forecast_df, uid, split_dates)
    if forecast_df is not None and forecasts is None and on_missing_forecast == "raise":
        raise ValueError(
            f"No forecast data for uid={uid}, split={split}. "
            "Pass on_missing_forecast='naive' to use the causal persistence fallback."
        )
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
        done = False
        while not done:
            action = policy_fn(obs, env)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        total_costs.append(env.total_cost)
        service_levels.append(1.0 - float(np.mean(env.history["stockout"])))
    return {
        "avg_cost": float(np.mean(total_costs)),
        "std_cost": float(np.std(total_costs)),
        "avg_service_level": float(np.mean(service_levels)),
    }


def _inventory_position(obs, env) -> float:
    inv_idx = 1 if isinstance(env, CategoryInventoryEnv) else 0
    inventory = float(obs[inv_idx])
    pipeline_total = float(obs[inv_idx + 1 : inv_idx + 1 + env.config.lead_time].sum())
    return inventory + pipeline_total


def _level_for_target(target: float, env) -> int:
    """Map a desired inventory position onto the env's order-up-to action grid."""
    level = int(round(target / env.config.order_level_step))
    return int(np.clip(level, 0, env.config.n_order_levels))


def sS_policy(obs, env, s: float = 10, S: float = 30) -> int:
    """Classic (s,S): when position drops to s or below, order up to S.

    Actions are order-up-to LEVELS now, so this returns the level encoding S rather
    than a raw quantity. The env computes the shortfall itself.
    """
    if _inventory_position(obs, env) <= s:
        return _level_for_target(S, env)
    return 0


def eoq_policy(obs, env, Q: float = 15, r: float = 8) -> int:
    """Reorder-point policy: at or below r, top up by roughly Q."""
    position = _inventory_position(obs, env)
    if position <= r:
        return _level_for_target(position + Q, env)
    return 0


def run_episode(env, policy_fn):
    obs, _ = env.reset()
    done = False
    while not done:
        action = policy_fn(obs, env)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.history


def seed_replay_buffer(model: DQN, env, policy_fn, n_episodes: int = 10) -> None:
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy_fn(obs, env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # `done` here is the BOOTSTRAP flag, so it must stay False on a pure time
            # limit: truncation is not a terminal state and the value function should
            # keep bootstrapping through it.
            model.replay_buffer.add(
                obs=np.array([obs]),
                next_obs=np.array([next_obs]),
                action=np.array([action]),
                reward=np.array([reward]),
                done=np.array([terminated]),
                infos=[info],
            )
            obs = next_obs
            done = terminated or truncated


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

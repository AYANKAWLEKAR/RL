"""Stage 4 (CPU): train and evaluate a single-product restocking DQN.

Deliberately CPU-only - an MlpPolicy on a 1-product env is minutes of work, and
paying for a GPU here would be waste.

Evaluation protocol, in order:
  1. Baselines FIRST, including never-order. Never-order is the control people skip,
     and it is the one that catches a miscalibrated cost function.
  2. Learning curve on a held-out split, so a flat/rising curve is visible rather
     than hidden behind a single final number.
  3. DQN vs best baseline over many episodes with random_start, reporting spread.
  4. Service level alongside cost - a policy can "win" on cost by refusing to stock.
  5. A rollout trace, because aggregate cost hides degenerate policies.

If DQN loses to (s,S) that is reported as-is. (s,S) is a strong industry-standard
baseline, though NOT provably optimal here - Scarf's result assumes backorders and
stationary demand, while this env has lost sales and seasonality, where (s,S) is known
not to be optimal. It is still a real bar for this
problem class under fairly general conditions, so it is a real bar.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from src.rl_pipeline import (  # noqa: E402
    DQNConfig,
    InventoryEnvConfig,
    build_dqn_agent,
    evaluate_policy,
    make_env,
    run_episode,
    sS_policy,
    eoq_policy,
)


def log(msg):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), msg), flush=True)


def pick_product(hourly, lo, hi):
    """Mid-range series so the result is not an artifact of a distribution edge."""
    daily = hourly.groupby(["unique_id", hourly.ds.dt.date])["y"].sum().groupby("unique_id").mean()
    band = daily[(daily >= lo) & (daily <= hi)].sort_values()
    if len(band) == 0:
        raise SystemExit("no series in demand band {}-{}".format(lo, hi))
    uid = band.index[len(band) // 2]
    return uid, float(daily.loc[uid])


def tune_sS(env, grid_s, grid_S, n_ep):
    best, best_p = None, None
    for s in grid_s:
        for S in grid_S:
            if S <= s:
                continue
            m = evaluate_policy(env, (lambda a, b: (lambda o, e: sS_policy(o, e, s=a, S=b)))(s, S), n_episodes=n_ep)
            if best is None or m["avg_cost"] < best["avg_cost"]:
                best, best_p = m, (s, S)
    return best, best_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=60000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--episode-length", type=int, default=10)
    ap.add_argument("--demand-lo", type=float, default=7.0)
    ap.add_argument("--demand-hi", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    hourly = pd.read_parquet("artifacts/hourly_features.parquet")
    fpath = "artifacts/rl_forecast_features.parquet"
    forecast = pd.read_parquet(fpath) if os.path.exists(fpath) else None
    if forecast is not None and "split" in forecast.columns:
        forecast = forecast.drop(columns=["split"])
    log("forecast rows={:,}".format(len(forecast) if forecast is not None else 0))

    uid, mean_daily = pick_product(hourly, args.demand_lo, args.demand_hi)
    log("product={} mean_daily={:.2f} units".format(uid, mean_daily))

    cfg = InventoryEnvConfig(episode_length=args.episode_length, random_start=True)
    log("costs: hold={} stockout={} fixed={} var={} | levels={} step={} (max_order={})".format(
        cfg.holding_cost, cfg.stockout_penalty, cfg.fixed_order_cost, cfg.variable_order_cost,
        cfg.n_order_levels, cfg.order_level_step, cfg.max_order))

    def build(split):
        return make_env(uid, hourly, split=split, forecast_df=forecast,
                        config=cfg, on_missing_forecast="naive")

    train_env, val_env, test_env = build("train"), build("val"), build("test")
    log("days: train={} val={} test={}".format(len(train_env.demand), len(val_env.demand), len(test_env.demand)))
    # Model selection must NOT see the test split. The learning curve and the
    # best-checkpoint choice run on val; test is evaluated exactly once, at the end.


    # ---- 1. baselines --------------------------------------------------------
    # (s,S) is TUNED ON VAL and scored on test, exactly like the DQN checkpoint.
    # Tuning it on test would hand the baseline an advantage the agent does not get.
    log("=" * 62)
    log("BASELINES (tuned on val, scored on test)")
    results = {}
    results["NeverOrder"] = evaluate_policy(test_env, lambda o, e: 0, n_episodes=args.eval_episodes)

    grid = list(range(0, 81, 10))
    _, sS_params = tune_sS(val_env, grid, grid, max(5, args.eval_episodes // 4))
    results["(s,S)"] = evaluate_policy(
        test_env, (lambda a, b: (lambda o, e: sS_policy(o, e, s=a, S=b)))(*sS_params),
        n_episodes=args.eval_episodes)
    log("  best (s,S) params = {}".format(sS_params))
    results["EOQ"] = evaluate_policy(
        test_env, lambda o, e: eoq_policy(o, e, Q=mean_daily * 4, r=mean_daily * 2),
        n_episodes=args.eval_episodes)

    for k, v in results.items():
        log("  {:12s} cost={:9.1f} +/-{:7.1f}  service={:.3f}".format(
            k, v["avg_cost"], v["std_cost"], v["avg_service_level"]))

    # ---- 2. DQN with learning curve ------------------------------------------
    log("=" * 62)
    log("TRAINING DQN ({:,} timesteps, eval every {:,})".format(args.timesteps, args.eval_every))
    dqn_cfg = DQNConfig(seed=args.seed, net_arch=(128, 128), learning_starts=1000,
                        exploration_fraction=0.4, target_update_interval=500)
    agent = build_dqn_agent(train_env, dqn_cfg)

    def dqn_policy(obs, env):
        a, _ = agent.predict(obs, deterministic=True)
        return int(a)

    curve = []
    best_path = "artifacts/dqn_best"

    class PeriodicEval(BaseCallback):
        """Evaluate inside a SINGLE learn() call.

        Calling learn(chunk, reset_num_timesteps=False) in a loop makes SB3 recompute
        _total_timesteps per call, so the epsilon schedule collapses to its floor
        inside the first chunk and the agent stops exploring for the rest of the run
        (measured: eps pinned at 0.010 across all 60k steps). One learn() call keeps
        the schedule intact.
        """

        def __init__(self, every):
            super().__init__()
            self.every = every
            self.best = None

        def _on_step(self):
            if self.num_timesteps % self.every:
                return True
            m = evaluate_policy(val_env, dqn_policy, n_episodes=args.eval_episodes)
            curve.append(dict(steps=self.num_timesteps, eps=float(self.model.exploration_rate), **m))
            star = ""
            if self.best is None or m["avg_cost"] < self.best:
                self.best = m["avg_cost"]; self.model.save(best_path); star = "  <- best so far"
            log("  steps={:>6,}  VAL cost={:9.1f} +/-{:7.1f}  service={:.3f}  eps={:.3f}{}".format(
                self.num_timesteps, m["avg_cost"], m["std_cost"], m["avg_service_level"],
                float(self.model.exploration_rate), star))
            return True

    cb = PeriodicEval(args.eval_every)
    agent.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=False)
    best_val = cb.best

    from stable_baselines3 import DQN as _DQN
    agent = _DQN.load(best_path)
    log("restored best-on-val checkpoint (val cost={:.1f})".format(best_val))
    results["DQN"] = evaluate_policy(test_env, dqn_policy, n_episodes=args.eval_episodes)

    # ---- 3. verdict -----------------------------------------------------------
    log("=" * 62)
    log("FINAL (test split, {} episodes, random_start)".format(args.eval_episodes))
    table = pd.DataFrame([
        dict(policy=k, avg_cost=v["avg_cost"], std_cost=v["std_cost"], service_level=v["avg_service_level"])
        for k, v in results.items()
    ]).sort_values("avg_cost").reset_index(drop=True)
    print(table.to_string(index=False), flush=True)

    best_base = min((k for k in results if k != "DQN"), key=lambda k: results[k]["avg_cost"])
    d, b = results["DQN"]["avg_cost"], results[best_base]["avg_cost"]
    log("best baseline = {} ({:.1f})".format(best_base, b))
    log("DQN = {:.1f}  ->  {}".format(d, "DQN WINS by {:.1f}%".format(100 * (1 - d / b)) if d < b
                                      else "DQN LOSES to {} by {:.1f}%".format(best_base, 100 * (d / b - 1))))

    # ---- 4. rollout inspection ------------------------------------------------
    log("=" * 62)
    log("DQN ROLLOUT (first 15 steps) - checking for degenerate policies")
    hist = pd.DataFrame(run_episode(test_env, dqn_policy))
    print(hist.head(15).to_string(index=False), flush=True)
    orders = hist["order"].to_numpy()
    log("orders: mean={:.1f} zero_share={:.1%} unique={}".format(
        orders.mean(), (orders == 0).mean(), len(np.unique(orders))))
    if len(np.unique(orders)) == 1:
        log("  !! DEGENERATE: identical action every step")

    os.makedirs("artifacts", exist_ok=True)
    table.to_csv("artifacts/rl_results.csv", index=False)
    pd.DataFrame(curve).to_csv("artifacts/rl_learning_curve.csv", index=False)
    hist.to_csv("artifacts/rl_rollout.csv", index=False)
    agent.save("artifacts/dqn_single_product")
    log("saved artifacts/rl_results.csv, rl_learning_curve.csv, rl_rollout.csv")
    log("STAGE4 DONE in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()

"""Stage 5 (CPU): is the DQN win robust across seeds, or one lucky initialization?

The headline result came from a single seed. RL results are notoriously
seed-sensitive, so a single-seed win is weak evidence. This re-runs the whole
protocol across N seeds and reports the distribution.

Protocol per seed (identical to stage 4):
  - (s,S) tuned on VAL, scored on test          <- baseline gets the same treatment
  - DQN checkpoint selected on VAL, scored on test exactly once
  - single learn() call so the epsilon schedule stays intact
  - training budget capped near the observed overfitting point; the curve degrades
    after ~15-25k steps because 61 training days is simply not much data
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import DQN as _DQN  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from src.rl_pipeline import (  # noqa: E402
    DQNConfig,
    InventoryEnvConfig,
    build_dqn_agent,
    evaluate_policy,
    make_env,
    sS_policy,
)


def log(m):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--eval-episodes", type=int, default=40)
    ap.add_argument("--episode-length", type=int, default=10)
    args = ap.parse_args()

    hourly = pd.read_parquet("artifacts/hourly_features.parquet")
    fc = pd.read_parquet("artifacts/rl_forecast_features.parquet")
    if "split" in fc.columns:
        fc = fc.drop(columns=["split"])
    daily = hourly.groupby(["unique_id", hourly.ds.dt.date])["y"].sum().groupby("unique_id").mean()
    band = daily[(daily >= 7.0) & (daily <= 15.0)].sort_values()
    uid = band.index[len(band) // 2]
    log("product={} mean_daily={:.2f}  seeds={}  budget={:,}".format(
        uid, daily.loc[uid], args.seeds, args.timesteps))

    cfg = InventoryEnvConfig(episode_length=args.episode_length, random_start=True)
    def build(s):
        return make_env(uid, hourly, split=s, forecast_df=fc, config=cfg, on_missing_forecast="naive")

    train_env, val_env, test_env = build("train"), build("val"), build("test")

    # baseline tuned on val, once - it is seed-independent
    grid = list(range(0, 81, 10))
    best, best_p = None, None
    for s_ in grid:
        for S_ in grid:
            if S_ <= s_:
                continue
            m = evaluate_policy(val_env, (lambda a, b: (lambda o, e: sS_policy(o, e, s=a, S=b)))(s_, S_),
                                n_episodes=10)
            if best is None or m["avg_cost"] < best:
                best, best_p = m["avg_cost"], (s_, S_)
    sS_test = evaluate_policy(test_env, (lambda a, b: (lambda o, e: sS_policy(o, e, s=a, S=b)))(*best_p),
                              n_episodes=args.eval_episodes)
    log("(s,S) tuned on val -> {} | TEST cost={:.1f} +/-{:.1f} service={:.3f}".format(
        best_p, sS_test["avg_cost"], sS_test["std_cost"], sS_test["avg_service_level"]))

    rows = []
    for seed in range(args.seeds):
        t0 = time.time()
        agent = build_dqn_agent(train_env, DQNConfig(
            seed=seed, net_arch=(128, 128), learning_starts=1000,
            exploration_fraction=0.4, target_update_interval=500))

        def dqn_policy(obs, env):
            a, _ = agent.predict(obs, deterministic=True)
            return int(a)

        path = "artifacts/dqn_seed{}".format(seed)

        class Eval(BaseCallback):
            def __init__(self, every):
                super().__init__()
                self.every, self.best = every, None

            def _on_step(self):
                if self.num_timesteps % self.every:
                    return True
                m = evaluate_policy(val_env, dqn_policy, n_episodes=20)
                if self.best is None or m["avg_cost"] < self.best:
                    self.best = m["avg_cost"]
                    self.model.save(path)
                return True

        cb = Eval(args.eval_every)
        agent.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=False)
        agent = _DQN.load(path)
        te = evaluate_policy(test_env, dqn_policy, n_episodes=args.eval_episodes)
        rows.append(dict(seed=seed, val_best=cb.best, test_cost=te["avg_cost"],
                         test_std=te["std_cost"], service=te["avg_service_level"]))
        log("  seed={} val_best={:7.1f}  TEST={:7.1f} +/-{:6.1f} service={:.3f}  ({:.0f}s)".format(
            seed, cb.best, te["avg_cost"], te["std_cost"], te["avg_service_level"], time.time() - t0))

    df = pd.DataFrame(rows)
    df.to_csv("artifacts/rl_seed_variance.csv", index=False)
    b = sS_test["avg_cost"]

    print("", flush=True)
    log("=" * 68)
    log("(s,S) test cost = {:.1f}".format(b))
    log("DQN across {} seeds: mean={:.1f} sd={:.1f} min={:.1f} max={:.1f}".format(
        len(df), df.test_cost.mean(), df.test_cost.std(), df.test_cost.min(), df.test_cost.max()))
    wins = int((df.test_cost < b).sum())
    log("seeds beating (s,S): {}/{}".format(wins, len(df)))
    log("mean improvement vs (s,S): {:+.1f}%".format(100 * (1 - df.test_cost.mean() / b)))
    if wins == len(df):
        log("VERDICT: DQN beats (s,S) on EVERY seed - result is robust")
    elif wins == 0:
        log("VERDICT: DQN loses on every seed - the single-seed win was luck")
    else:
        log("VERDICT: MIXED - {}/{} seeds win; the single-seed headline overstates it".format(wins, len(df)))
    log("SEEDS DONE")


if __name__ == "__main__":
    main()

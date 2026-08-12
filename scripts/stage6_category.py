"""Stage 6 (CPU): one shared DQN policy across MANY products and categories.

Motivation (design.md Part 8): the single-product agent overfits because 61 train days
at episode_length=10 is only ~51 distinct episode windows, replayed ~118 times over a
60k budget. No hyperparameter fixes that - the cure is more data. Training one policy
over hundreds of series multiplies the window count by the number of products.

Two things make a SHARED policy possible at all, both added for this stage:
  * scale_normalize - state in days of cover and actions in days of cover, because real
    series span 5.4-28.2 units/day. Unnormalized, one action means different things per
    product and no single policy can be right for all of them.
  * train-split scales - normalization constants come from TRAIN only, so an evaluation
    env never derives its own normalization from held-out demand.

Baseline is (s,S) in days of cover, tuned on val, so the baseline gets the same
cross-product treatment the agent does.
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
    compute_train_scales,
    evaluate_policy,
    make_multi_product_env,
    sS_policy_scaled,
)


def log(m):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--episode-length", type=int, default=10)
    ap.add_argument("--max-series", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    hourly = pd.read_parquet("artifacts/hourly_features.parquet")
    fc = pd.read_parquet("artifacts/rl_forecast_features.parquet")
    if "split" in fc.columns:
        fc = fc.drop(columns=["split"])

    cats = hourly.groupby("unique_id")["first_category_id"].first()
    daily = hourly.groupby(["unique_id", hourly.ds.dt.date])["y"].sum().groupby("unique_id").mean()
    rng = np.random.default_rng(args.seed)
    uids = list(cats.index)
    if len(uids) > args.max_series:
        uids = [uids[i] for i in sorted(rng.choice(len(uids), args.max_series, replace=False))]

    log("series={:,} across {} categories | demand {:.1f}-{:.1f} units/day".format(
        len(uids), cats.loc[uids].nunique(), daily.loc[uids].min(), daily.loc[uids].max()))

    scales = compute_train_scales(hourly, uids)
    cfg = InventoryEnvConfig(episode_length=args.episode_length, random_start=True,
                             scale_normalize=True, order_level_days=0.5, n_order_levels=20)
    log("scale_normalize=True order_level_days={} -> max cover {:.1f} days".format(
        cfg.order_level_days, cfg.n_order_levels * cfg.order_level_days))

    def build(split):
        return make_multi_product_env(hourly, uids, split=split, forecast_df=fc,
                                      config=cfg, min_series_days=12, scales=scales,
                                      on_missing_forecast="naive")

    train_env, val_env, test_env = build("train"), build("val"), build("test")
    windows = sum(max(1, len(d) - args.episode_length + 1) for d in train_env.demand_list)
    log("train products={:,}  distinct episode windows={:,}  (single product was ~51)".format(
        train_env.n_products, windows))

    # ---- baseline: (s,S) in days of cover, tuned on val ------------------------
    log("=" * 66)
    log("BASELINE (s,S) in days of cover - tuned on val, scored on test")
    best, best_p = None, None
    for s_d in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for S_d in [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]:
            if S_d <= s_d:
                continue
            m = evaluate_policy(val_env, (lambda a, b: (lambda o, e: sS_policy_scaled(o, e, a, b)))(s_d, S_d),
                                n_episodes=60)
            if best is None or m["avg_cost"] < best:
                best, best_p = m["avg_cost"], (s_d, S_d)
    log("  best (s,S) = {} days (val cost {:.1f})".format(best_p, best))

    results = {}
    results["NeverOrder"] = evaluate_policy(test_env, lambda o, e: 0, n_episodes=args.eval_episodes)
    results["(s,S) days"] = evaluate_policy(
        test_env, (lambda a, b: (lambda o, e: sS_policy_scaled(o, e, a, b)))(*best_p),
        n_episodes=args.eval_episodes)
    for k, v in results.items():
        log("  {:12s} cost={:8.1f} +/-{:7.1f} service={:.3f}".format(
            k, v["avg_cost"], v["std_cost"], v["avg_service_level"]))

    # ---- shared DQN ------------------------------------------------------------
    log("=" * 66)
    log("TRAINING SHARED DQN ({:,} steps)".format(args.timesteps))
    agent = build_dqn_agent(train_env, DQNConfig(
        seed=args.seed, net_arch=(256, 256), learning_starts=2000,
        exploration_fraction=0.3, target_update_interval=1000, buffer_size=100_000))

    def dqn_policy(obs, env):
        a, _ = agent.predict(obs, deterministic=True)
        return int(a)

    path, curve = "artifacts/dqn_category", []

    class Eval(BaseCallback):
        def __init__(self, every):
            super().__init__()
            self.every, self.best = every, None

        def _on_step(self):
            if self.num_timesteps % self.every:
                return True
            m = evaluate_policy(val_env, dqn_policy, n_episodes=100)
            curve.append(dict(steps=self.num_timesteps, eps=float(self.model.exploration_rate), **m))
            star = ""
            if self.best is None or m["avg_cost"] < self.best:
                self.best = m["avg_cost"]; self.model.save(path); star = "  <- best"
            log("  steps={:>7,} VAL={:8.1f} +/-{:6.1f} service={:.3f} eps={:.3f}{}".format(
                self.num_timesteps, m["avg_cost"], m["std_cost"], m["avg_service_level"],
                float(self.model.exploration_rate), star))
            return True

    cb = Eval(args.eval_every)
    agent.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=False)
    agent = _DQN.load(path)
    log("restored best-on-val checkpoint (val {:.1f})".format(cb.best))
    results["DQN (shared)"] = evaluate_policy(test_env, dqn_policy, n_episodes=args.eval_episodes)

    # ---- verdict ---------------------------------------------------------------
    log("=" * 66)
    log("FINAL - test split, {} episodes over {} products".format(args.eval_episodes, test_env.n_products))
    table = pd.DataFrame([
        dict(policy=k, avg_cost=v["avg_cost"], std_cost=v["std_cost"], service_level=v["avg_service_level"])
        for k, v in results.items()]).sort_values("avg_cost").reset_index(drop=True)
    print(table.to_string(index=False), flush=True)

    d = results["DQN (shared)"]["avg_cost"]
    b = results["(s,S) days"]["avg_cost"]
    log("DQN {:.1f} vs (s,S) {:.1f}  ->  {}".format(
        d, b, "DQN WINS by {:.1f}%".format(100 * (1 - d / b)) if d < b
        else "DQN LOSES by {:.1f}%".format(100 * (d / b - 1))))

    # overfitting check: did more data actually help?
    cur = pd.DataFrame(curve)
    if len(cur) > 1:
        degr = cur.iloc[-1]["avg_cost"] - cur["avg_cost"].min()
        log("val degradation from best to final: {:+.1f}  (single product was +86.9)".format(degr))

    table.to_csv("artifacts/rl_category_results.csv", index=False)
    cur.to_csv("artifacts/rl_category_curve.csv", index=False)
    log("STAGE6 DONE in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()

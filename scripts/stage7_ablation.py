"""Stage 7 (CPU): forecast ablation - does the TFT forecast actually help the DQN?

THE QUESTION THIS ANSWERS
The DQN reads a TFT forecast; (s,S) and EOQ do not. So "DQN beats (s,S) by 26%"
conflates a better ALGORITHM with more INFORMATION. Until the forecast is ablated, the
project's central claim - that transformer forecasting improves restocking - is asserted,
not tested.

ARMS (identical in every respect except what fills the 7 forecast slots)
  A  supplied     TFT walk-forward forecast, causal prefix   <- the real system
  B  zeros        slots blanked                              <- does the feature carry
                                                                ANY information?
  C  persistence  last observed demand                       <- does a LEARNED forecast
                                                                beat a NAIVE one?

A vs B asks "is this feature worth anything." A vs C is the sharper question and the one
a reviewer will care about: a real system always has *some* demand signal available, so
the honest comparison is against a naive forecast, not against nothing.

VALIDITY CHECK BUILT IN
(s,S) and never-order never read the forecast, so their test costs MUST be identical
across all three arms. If they are not, the harness is broken rather than the agent, and
the script says so loudly instead of reporting a difference that isn't real.

PROTOCOL
Paired and seeded throughout: every arm faces the identical episode windows, the (s,S)
baseline is tuned on val, and test is touched once per arm. Model selection uses
best-on-val checkpointing (design.md Part 8: training overfits a 61-day split).
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
    make_env,
    make_multi_product_env,
    sS_policy,
    sS_policy_scaled,
)

ARMS = ["supplied", "zeros", "persistence"]


def _seeds(args):
    """Explicit seed list enables sharding one arm across parallel workers."""
    if args.seed_list:
        return [int(x) for x in args.seed_list.split(",") if x.strip()]
    return list(range(args.seeds))


def log(m):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), m), flush=True)


def train_one(train_env, val_env, test_env, seed, timesteps, eval_every, eval_episodes, tag):
    """Train a DQN and score it once on test, selecting the checkpoint on val."""
    agent = build_dqn_agent(train_env, DQNConfig(
        seed=seed, net_arch=(128, 128), learning_starts=1000,
        exploration_fraction=0.4, target_update_interval=500))

    def policy(obs, env):
        a, _ = agent.predict(obs, deterministic=True)
        return int(a)

    path = "artifacts/dqn_abl_{}_{}".format(tag, seed)

    class Eval(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best = None

        def _on_step(self):
            if self.num_timesteps % eval_every:
                return True
            m = evaluate_policy(val_env, policy, n_episodes=20)
            if self.best is None or m["avg_cost"] < self.best:
                self.best = m["avg_cost"]
                self.model.save(path)
            return True

    cb = Eval()
    # ONE learn() call: chunked calls make SB3 recompute _total_timesteps and collapse
    # the epsilon schedule inside the first chunk (design.md Part 8).
    agent.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
    agent = _DQN.load(path)
    return evaluate_policy(test_env, policy, n_episodes=eval_episodes), cb.best


def single_product(hourly, forecast, args):
    daily = hourly.groupby(["unique_id", hourly.ds.dt.date])["y"].sum().groupby("unique_id").mean()
    band = daily[(daily >= 7.0) & (daily <= 15.0)].sort_values()
    uid = band.index[len(band) // 2]
    log("SINGLE PRODUCT: {}  mean_daily={:.2f}".format(uid, daily.loc[uid]))

    rows, checks = [], []
    for arm in ARMS:
        cfg = InventoryEnvConfig(episode_length=args.episode_length, random_start=True,
                                 forecast_mode=arm)

        def build(split):
            return make_env(uid, hourly, split=split, forecast_df=forecast,
                            config=cfg, on_missing_forecast="naive")

        tr, va, te = build("train"), build("val"), build("test")

        # baselines: tuned on val, scored on test. They ignore the forecast entirely,
        # so these numbers are the validity check.
        grid = list(range(0, 81, 10))
        best, best_p = None, None
        for s_ in grid:
            for S_ in grid:
                if S_ <= s_:
                    continue
                m = evaluate_policy(va, (lambda a, b: (lambda o, e: sS_policy(o, e, s=a, S=b)))(s_, S_),
                                    n_episodes=10)
                if best is None or m["avg_cost"] < best:
                    best, best_p = m["avg_cost"], (s_, S_)
        sS = evaluate_policy(te, (lambda a, b: (lambda o, e: sS_policy(o, e, s=a, S=b)))(*best_p),
                             n_episodes=args.eval_episodes)
        never = evaluate_policy(te, lambda o, e: 0, n_episodes=args.eval_episodes)
        checks.append(dict(arm=arm, sS=sS["avg_cost"], never=never["avg_cost"], params=str(best_p)))

        for seed in _seeds(args):
            t0 = time.time()
            m, valbest = train_one(tr, va, te, seed, args.timesteps, args.eval_every,
                                   args.eval_episodes, "sp_" + arm)
            rows.append(dict(scope="single", arm=arm, seed=seed, val_best=valbest,
                             test_cost=m["avg_cost"], test_std=m["std_cost"],
                             service=m["avg_service_level"]))
            log("  {:11s} seed={} val={:7.1f}  TEST={:7.1f} +/-{:5.1f} service={:.3f} ({:.0f}s)".format(
                arm, seed, valbest, m["avg_cost"], m["std_cost"], m["avg_service_level"],
                time.time() - t0))
    return pd.DataFrame(rows), pd.DataFrame(checks), sS["avg_cost"]


def shared_policy(hourly, forecast, args):
    cats = hourly.groupby("unique_id")["first_category_id"].first()
    rng = np.random.default_rng(args.seed)
    uids = list(cats.index)
    if len(uids) > args.max_series:
        uids = [uids[i] for i in sorted(rng.choice(len(uids), args.max_series, replace=False))]
    scales = compute_train_scales(hourly, uids)
    log("SHARED POLICY: {} series across {} categories".format(len(uids), cats.loc[uids].nunique()))

    rows, checks = [], []
    for arm in ARMS:
        cfg = InventoryEnvConfig(episode_length=args.episode_length, random_start=True,
                                 scale_normalize=True, order_level_days=0.5,
                                 n_order_levels=20, forecast_mode=arm)

        def build(split):
            return make_multi_product_env(hourly, uids, split=split, forecast_df=forecast,
                                          config=cfg, min_series_days=12, scales=scales,
                                          on_missing_forecast="naive")

        tr, va, te = build("train"), build("val"), build("test")

        best, best_p = None, None
        for s_d in [1.0, 2.0, 3.0]:
            for S_d in [2.0, 3.0, 4.0, 5.0, 6.0]:
                if S_d <= s_d:
                    continue
                m = evaluate_policy(va, (lambda a, b: (lambda o, e: sS_policy_scaled(o, e, a, b)))(s_d, S_d),
                                    n_episodes=40)
                if best is None or m["avg_cost"] < best:
                    best, best_p = m["avg_cost"], (s_d, S_d)
        sS = evaluate_policy(te, (lambda a, b: (lambda o, e: sS_policy_scaled(o, e, a, b)))(*best_p),
                             n_episodes=args.eval_episodes)
        never = evaluate_policy(te, lambda o, e: 0, n_episodes=args.eval_episodes)
        checks.append(dict(arm=arm, sS=sS["avg_cost"], never=never["avg_cost"], params=str(best_p)))

        t0 = time.time()
        m, valbest = train_one(tr, va, te, args.seed, args.cat_timesteps, args.eval_every,
                               args.eval_episodes, "cat_" + arm)
        rows.append(dict(scope="shared", arm=arm, seed=args.seed, val_best=valbest,
                         test_cost=m["avg_cost"], test_std=m["std_cost"],
                         service=m["avg_service_level"]))
        log("  {:11s} val={:7.1f}  TEST={:7.1f} +/-{:5.1f} service={:.3f} ({:.0f}s)".format(
            arm, valbest, m["avg_cost"], m["std_cost"], m["avg_service_level"], time.time() - t0))
    return pd.DataFrame(rows), pd.DataFrame(checks)


def report_validity(checks, label):
    log("-" * 70)
    log("VALIDITY CHECK ({}) - (s,S) and never-order ignore the forecast,".format(label))
    log("so these MUST match across arms. A mismatch means the harness is broken.")
    print(checks.to_string(index=False), flush=True)
    ok = np.allclose(checks["sS"], checks["sS"].iloc[0], rtol=1e-9) and \
        np.allclose(checks["never"], checks["never"].iloc[0], rtol=1e-9)
    log("  -> {}".format("PASS: baselines identical across arms" if ok
                         else "FAIL: baselines differ - do NOT trust the arm comparison"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-list", default="", help="explicit comma-separated seeds; enables sharding")
    ap.add_argument("--out", default="artifacts/ablation_results.csv")
    ap.add_argument("--timesteps", type=int, default=30000)
    ap.add_argument("--cat-timesteps", type=int, default=150000)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--eval-episodes", type=int, default=40)
    ap.add_argument("--episode-length", type=int, default=10)
    ap.add_argument("--max-series", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scope", default="single,shared")
    args = ap.parse_args()
    scopes = [s.strip() for s in args.scope.split(",")]

    t0 = time.time()
    hourly = pd.read_parquet("artifacts/hourly_features.parquet")
    forecast = pd.read_parquet("artifacts/rl_forecast_features.parquet")
    if "split" in forecast.columns:
        forecast = forecast.drop(columns=["split"])
    log("arms: {}".format(ARMS))

    all_rows, valid = [], True
    if "single" in scopes:
        log("=" * 70)
        rows, checks, _ = single_product(hourly, forecast, args)
        valid &= report_validity(checks, "single product")
        all_rows.append(rows)
    if "shared" in scopes:
        log("=" * 70)
        rows, checks = shared_policy(hourly, forecast, args)
        valid &= report_validity(checks, "shared policy")
        all_rows.append(rows)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(args.out, index=False)

    log("=" * 70)
    log("ABLATION SUMMARY (test cost, lower is better)")
    for scope, g in df.groupby("scope"):
        print("\n  --- {} ---".format(scope), flush=True)
        summ = g.groupby("arm").agg(n=("test_cost", "size"), mean=("test_cost", "mean"),
                                    sd=("test_cost", "std"), service=("service", "mean"))
        summ = summ.reindex([a for a in ARMS if a in summ.index])
        print(summ.to_string(), flush=True)
        base = summ.loc["supplied", "mean"]
        for arm in summ.index:
            if arm == "supplied":
                continue
            delta = summ.loc[arm, "mean"] - base
            pct = 100 * delta / base
            verdict = ("TFT HELPS by {:.1f}%".format(pct) if delta > 0
                       else "TFT HURTS by {:.1f}%".format(-pct))
            log("  supplied vs {:11s}: {:+8.1f} cost  -> {}".format(arm, delta, verdict))
        if len(g[g.arm == "supplied"]) > 1:
            a = g[g.arm == "supplied"]["test_cost"].to_numpy()
            for arm in [x for x in ARMS if x != "supplied"]:
                b = g[g.arm == arm]["test_cost"].to_numpy()
                if len(b) < 2:
                    continue
                se = math_se(a, b)
                t = (a.mean() - b.mean()) / se if se > 0 else 0.0
                log("  Welch t (supplied vs {}): {:+.2f}  -> {}".format(
                    arm, t, "significant" if abs(t) > 2 else "NOT significant"))

    if not valid:
        log("!! VALIDITY CHECK FAILED - arm comparison is not trustworthy")
    log("saved {}".format(args.out))
    log("ABLATION DONE in {:.0f}s".format(time.time() - t0))


def math_se(a, b):
    import math
    return math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


if __name__ == "__main__":
    main()

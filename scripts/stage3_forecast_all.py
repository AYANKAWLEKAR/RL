"""Stage 3 (GPU): TFT forecasts covering train, val AND test, for the RL state.

The RL agent needs a forecast signal at every timestep it trains on, but stage 2 only
exported the held-out test split. Training on a causal-persistence proxy and evaluating
on TFT forecasts would be a distribution shift, so we produce model forecasts everywhere.

HONESTY NOTE: train-period forecasts are partly IN-SAMPLE - the model was fitted on
those rows. They will look better than the val/test forecasts, which are genuinely
out-of-sample. That is a known and accepted asymmetry: the purpose here is to give the
agent a consistent *kind* of observation, not to claim train-period forecast accuracy.
Any forecast-accuracy claim must come from the test split only (see stage 2).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.forecasting import (  # noqa: E402
    ForecastConfig,
    NeuralForecastAdapter,
    prepare_forecast_frames,
)


def log(msg):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--windows-batch", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default="artifacts/rl_forecast_features.parquet")
    args = ap.parse_args()

    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import TFT

    t0 = time.time()
    hourly = pd.read_parquet("artifacts/hourly_features.parquet")
    cfg = ForecastConfig()
    b = prepare_forecast_frames(hourly, cfg)
    log("train={:,} val={:,} test={:,} series={:,}".format(
        len(b.train_df), len(b.val_df), len(b.test_df), hourly["unique_id"].nunique()))

    model = TFT(
        h=cfg.horizon, input_size=cfg.input_size, hidden_size=args.hidden, n_head=4, dropout=0.15,
        futr_exog_list=b.future_cols, hist_exog_list=b.historic_cols, stat_exog_list=list(cfg.static_cols),
        loss=MAE(), valid_loss=MAE(), max_steps=args.max_steps, batch_size=32, learning_rate=3e-4,
        early_stop_patience_steps=40, val_check_steps=100, scaler_type="standard",
        windows_batch_size=args.windows_batch, inference_windows_batch_size=args.windows_batch,
        dataloader_kwargs={"num_workers": args.num_workers}, random_seed=42,
    )
    adapter = NeuralForecastAdapter(model, "TFT")

    log("fitting TFT on train...")
    adapter.fit(b.train_df, b.val_df, b)
    log("fit done in {:.0f}s".format(time.time() - t0))

    # Split train into a seed history (the first input_size steps per series, which can
    # never be forecast because there is nothing before them) and the forecastable rest.
    per_series = b.train_df.sort_values(["unique_id", "ds"])
    seed = per_series.groupby("unique_id", sort=False, group_keys=False).head(cfg.input_size)
    train_future = per_series.drop(seed.index)
    log("train seed={:,} rows (unforecastable warmup), train_future={:,}".format(len(seed), len(train_future)))

    jobs = [
        ("train", seed, train_future),
        ("val", b.train_df, b.val_df),
        ("test", pd.concat([b.train_df, b.val_df], ignore_index=True), b.test_df),
    ]

    parts = []
    for name, hist, fut in jobs:
        if len(fut) == 0:
            log("{}: empty, skipping".format(name))
            continue
        log("forecasting {} ({:,} rows)...".format(name, len(fut)))
        t1 = time.time()
        p = adapter.predict(hist, fut, b)
        p["split"] = name
        parts.append(p)
        log("  {} -> {:,} preds in {:.0f}s  range {} .. {}".format(
            name, len(p), time.time() - t1, p.ds.min(), p.ds.max()))
        pd.concat(parts, ignore_index=True).to_parquet(args.out + ".partial", index=False)

    allp = pd.concat(parts, ignore_index=True)
    # predictions are in log space when use_log_target; convert back for the RL env,
    # which reasons in real units
    if cfg.use_log_target:
        allp["prediction"] = np.expm1(allp["prediction"].to_numpy(dtype=float))
    allp["prediction"] = allp["prediction"].clip(lower=0.0)

    out = allp[["unique_id", "ds", "prediction", "split"]].rename(columns={"prediction": "forecast_median"})
    out = out.sort_values(["unique_id", "ds"]).drop_duplicates(["unique_id", "ds"]).reset_index(drop=True)
    out.to_parquet(args.out, index=False)
    if os.path.exists(args.out + ".partial"):
        os.remove(args.out + ".partial")

    log("saved {} rows={:,}".format(args.out, len(out)))
    log("coverage by split:\n{}".format(out.groupby("split").size().to_string()))
    log("forecast_median: mean={:.3f} median={:.3f} max={:.2f} negatives={}".format(
        out.forecast_median.mean(), out.forecast_median.median(), out.forecast_median.max(),
        int((out.forecast_median < 0).sum())))
    log("STAGE3 DONE in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()

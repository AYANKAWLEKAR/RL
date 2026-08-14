"""Stage 2 (GPU): train + evaluate forecasters with skill-relative metrics.

Every model is scored against TWO controls:
  * Zero        - predicts 0 everywhere. On intermittent demand MAE is minimized by
                  the conditional median (= 0 when most hours are zero), so a model
                  that cannot beat this has learned nothing regardless of its MAE.
  * Persistence - lag-24. The standard "did you beat the naive baseline" bar.

rel_zero / rel_persist below 1.0 mean the model is genuinely better than the control.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.forecasting import (  # noqa: E402
    ForecastConfig,
    NeuralForecastAdapter,
    PersistenceForecaster,
    ZeroForecaster,
    evaluate_forecaster,
    export_rl_forecast_features,
    prepare_forecast_frames,
)


def log(msg):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), msg), flush=True)


def build_models(cfg, bundle, args):
    from neuralforecast.losses.pytorch import MAE, MQLoss
    from neuralforecast.models import LSTM as NF_LSTM
    from neuralforecast.models import TFT

    static_cols = list(cfg.static_cols)
    dl = {"num_workers": args.num_workers}
    # windows_batch_size is the real GPU-memory driver for window-based models like
    # TFT: memory scales with windows_batch_size * input_size * hidden_size. The
    # NeuralForecast default is 1024, which OOMs a 14.5GB T4 at input_size=168 and
    # hidden_size=128. Set it explicitly rather than inheriting the default.
    common = dict(
        h=cfg.horizon,
        input_size=cfg.input_size,
        futr_exog_list=bundle.future_cols,
        hist_exog_list=bundle.historic_cols,
        stat_exog_list=static_cols,
        scaler_type="standard",
        dataloader_kwargs=dl,
        windows_batch_size=args.windows_batch,
        inference_windows_batch_size=args.windows_batch,
        random_seed=42,
    )
    out = []
    if "tft_mae" in args.models:
        out.append(NeuralForecastAdapter(TFT(
            hidden_size=args.hidden, n_head=4, dropout=0.15, loss=MAE(), valid_loss=MAE(),
            max_steps=args.max_steps, batch_size=32, learning_rate=3e-4,
            early_stop_patience_steps=args.patience, val_check_steps=args.val_check, **common
        ), "TFT-MAE"))
    if "tft_mqloss" in args.models:
        # MQLoss optimizes the whole predictive distribution instead of just the
        # median, which is the actual fix for intermittent demand: the median is
        # zero, but the upper quantiles carry the signal inventory decisions need.
        out.append(NeuralForecastAdapter(TFT(
            hidden_size=args.hidden, n_head=4, dropout=0.15,
            loss=MQLoss(level=[80]), valid_loss=MQLoss(level=[80]),
            max_steps=args.max_steps, batch_size=32, learning_rate=3e-4,
            early_stop_patience_steps=args.patience, val_check_steps=args.val_check, **common
        ), "TFT-MQLoss"))
    if "lstm" in args.models:
        out.append(NeuralForecastAdapter(NF_LSTM(
            encoder_hidden_size=128, encoder_n_layers=2, encoder_dropout=0.25,
            decoder_hidden_size=128, decoder_layers=2, loss=MAE(), valid_loss=MAE(),
            max_steps=args.max_steps, batch_size=64, learning_rate=5e-4,
            early_stop_patience_steps=args.patience, val_check_steps=args.val_check, **common
        ), "LSTM-MAE"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tft_mae,tft_mqloss", help="comma list: tft_mae,tft_mqloss,lstm")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--val-check", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--windows-batch", type=int, default=128, help="GPU memory driver for TFT")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    t0 = time.time()
    log("loading features...")
    hourly = pd.read_parquet("artifacts/hourly_features.parquet")
    log("hourly rows={:,} series={:,}".format(len(hourly), hourly["unique_id"].nunique()))

    cfg = ForecastConfig()
    bundle = prepare_forecast_frames(hourly, cfg)
    log("train={:,} val={:,} test={:,}".format(len(bundle.train_df), len(bundle.val_df), len(bundle.test_df)))
    log("use_log_target={} horizon={} input_size={}".format(cfg.use_log_target, cfg.horizon, cfg.input_size))

    adapters = [ZeroForecaster(), PersistenceForecaster()] + build_models(cfg, bundle, args)

    rows = []
    preds_store = {}
    for split in splits:
        log("=" * 60)
        log("EVALUATION SPLIT: {}".format(split))
        for ad in adapters:
            log("  running {} ...".format(ad.name))
            try:
                m, preds = evaluate_forecaster(ad, bundle, use_log_target=cfg.use_log_target, evaluation_split=split)
                rows.append(dict(split=split, model=m.model_name, mae=m.mae, rmse=m.rmse, bias=m.bias, n_obs=m.n_obs))
                preds_store[(split, m.model_name)] = preds
                log("    mae={:.4f} rmse={:.4f} bias={:+.4f} n={:,}".format(m.mae, m.rmse, m.bias, m.n_obs))
            except Exception as e:  # keep going so one model failing doesn't lose the rest
                log("    FAILED: {}: {}".format(type(e).__name__, e))
                rows.append(dict(split=split, model=ad.name, mae=float("nan"), rmse=float("nan"),
                                 bias=float("nan"), n_obs=0))
            pd.DataFrame(rows).to_csv("artifacts/forecast_metrics.csv", index=False)

    res = pd.DataFrame(rows)
    for split in splits:
        sub = res[res.split == split]
        zero = sub.loc[sub.model == "Zero", "mae"]
        per = sub.loc[sub.model == "Persistence", "mae"]
        if len(zero):
            res.loc[res.split == split, "rel_zero"] = res.loc[res.split == split, "mae"] / zero.iloc[0]
        if len(per):
            res.loc[res.split == split, "rel_persist"] = res.loc[res.split == split, "mae"] / per.iloc[0]

    res.to_csv("artifacts/forecast_metrics.csv", index=False)
    log("=" * 60)
    log("FINAL RESULTS (rel_* < 1.0 means better than that control)")
    print(res.to_string(index=False), flush=True)

    # export best non-control model on test for the RL stage
    test = res[(res.split == "test") & (~res.model.isin(["Zero", "Persistence"]))].dropna(subset=["mae"])
    if len(test):
        best = test.sort_values("mae").iloc[0]["model"]
        log("best learned model on test: {}".format(best))
        p = preds_store.get(("test", best))
        if p is not None:
            export_rl_forecast_features(p).to_parquet("artifacts/rl_forecast_features.parquet", index=False)
            log("exported artifacts/rl_forecast_features.parquet")
    log("STAGE2 DONE in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()

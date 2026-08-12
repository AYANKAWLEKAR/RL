"""Stage 1 (CPU): select non-degenerate series, reconstruct demand, build features.

Why series selection is necessary, not a shortcut:
  * SCALE - the full dataset is 4.85M daily rows -> 116.4M hourly rows after
    expand_to_hourly(), which builds a Python list of dicts via .iterrows().
    That does not fit in this box's 15GB RAM.
  * DEGENERACY - median demand is 0.00 units/hour and 77.8% of hours are exactly
    zero. On those series MAE is minimized by predicting zero, so a "good" MAE
    measures nothing. Restricting to higher-volume series makes the forecasting
    problem non-trivial in the first place.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import (  # noqa: E402
    FeatureConfig,
    LatentDemandConfig,
    build_hourly_features,
    reconstruct_latent_demand,
    split_and_scale_features,
)


def log(msg):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-daily", type=float, default=5.0, help="min mean daily units for a series to qualify")
    ap.add_argument("--max-series", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/hourly_features.parquet")
    args = ap.parse_args()

    t0 = time.time()
    log("loading cached parquet...")
    train = pd.read_parquet("data_cache/train.parquet")
    ev = pd.read_parquet("data_cache/eval.parquet")
    raw = pd.concat([train, ev], ignore_index=True)
    del train, ev
    raw["dt"] = pd.to_datetime(raw["dt"])
    log("raw daily rows={:,}  series={:,}".format(len(raw), raw.groupby(["product_id", "store_id"]).ngroups))

    # --- select non-degenerate series by mean daily volume ---
    log("scoring series by mean daily volume...")
    vol = raw.groupby(["product_id", "store_id"])["sale_amount"].mean()
    qualifying = vol[vol >= args.min_daily]
    log("series with mean daily >= {}: {:,} ({:.2%} of all)".format(args.min_daily, len(qualifying), len(qualifying) / len(vol)))
    if len(qualifying) == 0:
        raise SystemExit("no series meet the volume threshold; lower --min-daily")

    rng = np.random.default_rng(args.seed)
    keys = list(qualifying.index)
    if len(keys) > args.max_series:
        sel = rng.choice(len(keys), args.max_series, replace=False)
        keys = [keys[i] for i in sorted(sel)]
    keep = pd.MultiIndex.from_tuples(keys, names=["product_id", "store_id"])
    raw = raw.set_index(["product_id", "store_id"]).loc[keep].reset_index()
    log("selected series={:,}  daily rows={:,}".format(len(keys), len(raw)))
    log("projected hourly rows={:,}".format(len(raw) * 24))

    dsel = raw.groupby(["product_id", "store_id"])["sale_amount"].mean()
    log("selected demand: mean={:.1f} median={:.1f} min={:.1f} max={:.1f} units/day".format(
        dsel.mean(), dsel.median(), dsel.min(), dsel.max()))

    # --- causal reconstruction cutoff (position-based; .quantile can return float) ---
    fc = FeatureConfig()
    cutoff_frac = fc.train_frac + fc.val_frac
    sorted_dt = raw["dt"].sort_values().reset_index(drop=True)
    cutoff = sorted_dt.iloc[min(int(len(sorted_dt) * cutoff_frac), len(sorted_dt) - 1)]
    log("reconstruction_cutoff={} (imputation model sees no rows at/after this)".format(cutoff))

    log("reconstructing latent demand (CPU, slowest step)...")
    recon, diag = reconstruct_latent_demand(raw, LatentDemandConfig(), reconstruction_cutoff=cutoff)
    log("reconstruction done in {:.0f}s".format(time.time() - t0))
    log("  model_used={}  total_stockout_hours={:,}".format(diag.model_used, diag.total_stockout_hours))
    log("  imputation_source_counts={}".format(diag.imputation_source_counts))
    log("  masked_backtest: heuristic_mae={} model_mae={}".format(
        diag.masked_backtest_mae_heuristic, diag.masked_backtest_mae_model))
    stockout_share = diag.total_stockout_hours / float(len(raw) * 24)
    log("  >> {:.1%} of all hours are IMPUTED, not observed".format(stockout_share))

    log("building hourly features...")
    hourly = build_hourly_features(recon, fc)
    log("hourly rows={:,} cols={}".format(len(hourly), len(hourly.columns)))

    log("splitting + scaling...")
    hourly, scaler = split_and_scale_features(hourly, fc)
    log("split counts:\n{}".format(hourly["split"].value_counts().to_string()))

    y = hourly["y"]
    log("target y: mean={:.3f} median={:.3f} p90={:.2f} max={:.1f} zero_share={:.1%}".format(
        y.mean(), y.median(), y.quantile(0.9), y.max(), (y == 0).mean()))

    os.makedirs("artifacts", exist_ok=True)
    hourly.to_parquet(args.out, index=False)
    import pickle
    with open("artifacts/feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    log("saved {} ({:.0f} MB)".format(args.out, os.path.getsize(args.out) / 1e6))
    log("STAGE1 DONE in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()

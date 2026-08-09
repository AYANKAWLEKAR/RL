from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    FeatureConfig,
    LatentDemandConfig,
    backtest_reconstruction,
    build_hourly_features,
    parse_sequence,
    reconstruct_latent_demand,
    split_and_scale_features,
)


def test_parse_sequence_handles_string_and_list():
    assert np.array_equal(parse_sequence("[1, 2, 3]", expected_length=3), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(parse_sequence([1, 2, 3], expected_length=3), np.array([1.0, 2.0, 3.0]))


def test_parse_sequence_rejects_wrong_length():
    with pytest.raises(ValueError):
        parse_sequence("[1, 2]", expected_length=3)


def test_reconstruct_latent_demand_imputes_short_and_long_stockouts(synthetic_daily_df: pd.DataFrame):
    reconstructed, diagnostics = reconstruct_latent_demand(
        synthetic_daily_df,
        LatentDemandConfig(model_min_training_rows=64, short_stockout_max_hours=2),
    )
    short_row = reconstructed[(reconstructed["product_id"] == 101) & (reconstructed["dt"] == pd.Timestamp("2024-01-13"))].iloc[0]
    long_row = reconstructed[(reconstructed["product_id"] == 202) & (reconstructed["dt"] == pd.Timestamp("2024-01-21"))].iloc[0]

    short_latent = np.asarray(short_row["latent_demand"])
    long_latent = np.asarray(long_row["latent_demand"])
    assert np.all(short_latent[8:10] > 0)
    assert np.all(long_latent[6:12] > 0)
    assert diagnostics.total_stockout_hours == 8
    assert diagnostics.model_used is True
    assert diagnostics.heuristic_imputed_hours >= 2
    assert diagnostics.model_imputed_hours >= 1


def test_reconstruction_does_not_leak_future_days_into_past_stockouts():
    def make_df(future_sales_value: float) -> pd.DataFrame:
        rows = []
        start = pd.Timestamp("2024-01-01")
        for day_idx in range(30):
            hours = np.arange(24)
            base = 4 + 1.5 * np.sin((hours - 6) / 24 * 2 * np.pi)
            sales = np.clip(base, 0.2, None).round(3)
            stock = np.zeros(24, dtype=int)
            if day_idx == 5:
                stock[10:12] = 1
                sales[10:12] = 0
            if day_idx == 10:
                sales[:] = future_sales_value
            rows.append(
                {
                    "city_id": 1,
                    "store_id": 1,
                    "management_group_id": 1,
                    "first_category_id": 10,
                    "second_category_id": 100,
                    "third_category_id": 1000,
                    "product_id": 101,
                    "dt": start + pd.Timedelta(days=day_idx),
                    "sale_amount": float(sales.sum()),
                    "hours_sale": sales.tolist(),
                    "hours_stock_status": stock.tolist(),
                    "discount": 0.0,
                    "holiday_flag": 0,
                    "activity_flag": 0,
                    "precpt": 0.0,
                    "avg_temperature": 60.0,
                    "avg_humidity": 40.0,
                    "avg_wind_level": 5.0,
                }
            )
        return pd.DataFrame(rows)

    config = LatentDemandConfig(model_min_training_rows=10_000)  # isolate the heuristic path
    recon_low, _ = reconstruct_latent_demand(make_df(1.0), config)
    recon_high, _ = reconstruct_latent_demand(make_df(50.0), config)

    def imputed_hours(reconstructed: pd.DataFrame) -> np.ndarray:
        row = reconstructed[reconstructed["dt"] == pd.Timestamp("2024-01-06")].iloc[0]
        return np.asarray(row["latent_demand"])[10:12]

    # Day 10 is chronologically AFTER the day-5 stockout, so changing only day 10's
    # sales must not change how the day-5 stockout gets reconstructed.
    assert np.allclose(imputed_hours(recon_low), imputed_hours(recon_high))


def test_backtest_reconstruction_reports_metrics(synthetic_daily_df: pd.DataFrame):
    diagnostics = backtest_reconstruction(synthetic_daily_df, LatentDemandConfig(model_min_training_rows=64))
    assert diagnostics.masked_backtest_rows > 0
    assert diagnostics.masked_backtest_mae_heuristic is not None
    assert diagnostics.masked_backtest_mae_model is not None


def test_build_hourly_features_expands_rows_and_generates_features(reconstructed_daily_df: pd.DataFrame):
    hourly_df = build_hourly_features(reconstructed_daily_df, FeatureConfig(lag_periods=(1, 24), rolling_windows=(6, 24)))
    assert len(hourly_df) == len(reconstructed_daily_df) * 24
    assert hourly_df["unique_id"].nunique() == 3
    assert {"lag_1", "lag_24", "rolling_mean_6h", "store_mean_lag24", "category_mean_lag24"}.issubset(hourly_df.columns)
    first_uid = hourly_df["unique_id"].iloc[0]
    uid_frame = hourly_df[hourly_df["unique_id"] == first_uid].sort_values("ds")
    diffs = uid_frame["ds"].diff().dropna().unique()
    assert len(diffs) == 1
    assert diffs[0] == pd.Timedelta(hours=1)


def test_split_and_scale_features_is_groupwise_and_keeps_target_raw(hourly_feature_df: pd.DataFrame):
    processed, scaler = split_and_scale_features(
        hourly_feature_df,
        FeatureConfig(lag_periods=(1, 24), rolling_windows=(6, 24), scale_binary_features=False),
    )
    assert "split" in processed.columns
    assert processed["split"].isin({"train", "val", "test"}).all()
    assert processed["y"].min() >= 0
    assert set(processed["stockout_flag"].dropna().unique()).issubset({0, 1})
    train_mean = processed.loc[processed["split"] == "train", "lag_1"].mean()
    assert abs(train_mean) < 1e-6
    assert hasattr(scaler, "mean_")

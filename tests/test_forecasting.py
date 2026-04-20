from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting import (
    ForecastConfig,
    PersistenceForecaster,
    choose_best_forecaster,
    export_rl_forecast_features,
    prepare_forecast_frames,
)


class PerfectForecaster:
    name = "Perfect"

    def fit(self, train_df, val_df, bundle):
        return None

    def predict(self, history_df, future_df, bundle):
        return future_df[["unique_id", "ds", bundle.target_col]].rename(columns={bundle.target_col: "prediction"})


def test_prepare_forecast_frames_keeps_alignment(split_hourly_df: pd.DataFrame):
    bundle = prepare_forecast_frames(split_hourly_df, ForecastConfig())
    assert {"unique_id", "ds", "y", "y_model"}.issubset(bundle.full_df.columns)
    assert len(bundle.train_df) > len(bundle.val_df) > 0
    assert len(bundle.test_df) > 0
    assert set(bundle.future_cols) == {"hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "holiday_flag", "discount"}


def test_choose_best_forecaster_prefers_lower_mae(split_hourly_df: pd.DataFrame):
    bundle = prepare_forecast_frames(split_hourly_df, ForecastConfig())
    metrics, predictions, adapter = choose_best_forecaster(
        [PersistenceForecaster(), PerfectForecaster()],
        bundle,
        use_log_target=True,
    )
    assert adapter.name == "Perfect"
    assert metrics.mae == pytest.approx(0.0, abs=1e-12)
    assert len(predictions) == len(bundle.val_df)


def test_export_rl_forecast_features_produces_stable_schema(split_hourly_df: pd.DataFrame):
    bundle = prepare_forecast_frames(split_hourly_df, ForecastConfig())
    future = bundle.test_df[["unique_id", "ds", bundle.raw_target_col]].rename(columns={bundle.raw_target_col: "model_prediction"})
    future["lo"] = future["model_prediction"] * 0.9
    future["hi"] = future["model_prediction"] * 1.1
    exported = export_rl_forecast_features(future, prediction_col="model_prediction", lo_col="lo", hi_col="hi")
    assert list(exported.columns) == ["unique_id", "ds", "forecast_median", "forecast_lo_10", "forecast_hi_90"]
    assert not exported.duplicated(["unique_id", "ds"]).any()

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    ForecastConfig,
    ForecastDatasetBundle,
    NeuralForecastAdapter,
    PersistenceForecaster,
    choose_best_forecaster,
    evaluate_forecaster,
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


def test_neural_forecast_adapter_covers_full_evaluation_split():
    """Regression test: a single nf.predict() only forecasts h steps right after the
    fitted data, so a naive adapter mislabels an early window as the true (later,
    longer) evaluation split and silently drops most of it. The fixed adapter must
    walk forward and cover the *entire* requested split with correctly aligned dates.
    """
    neuralforecast = pytest.importorskip("neuralforecast")
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import MLP

    np.random.seed(0)
    n = 960
    ds = pd.date_range("2024-01-01", periods=n, freq="h")
    frames = []
    for uid, bias in [("A", 0.0), ("B", 2.0)]:
        y = 5 + bias + 2 * np.sin(2 * np.pi * ds.hour / 24) + np.random.normal(0, 0.3, n)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": ds,
                    "y": y,
                    "y_model": y,
                    "hour_sin": np.sin(2 * np.pi * ds.hour / 24),
                    "hour_cos": np.cos(2 * np.pi * ds.hour / 24),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)

    train_end, val_end = int(n * 0.70), int(n * 0.85)
    parts = []
    for _, g in df.groupby("unique_id"):
        g = g.sort_values("ds").reset_index(drop=True)
        g["split"] = "train"
        g.loc[train_end:val_end, "split"] = "val"
        g.loc[val_end:, "split"] = "test"
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)

    bundle = ForecastDatasetBundle(
        full_df=df,
        train_df=df[df.split == "train"].drop(columns="split").reset_index(drop=True),
        val_df=df[df.split == "val"].drop(columns="split").reset_index(drop=True),
        test_df=df[df.split == "test"].drop(columns="split").reset_index(drop=True),
        static_df=df.groupby("unique_id", as_index=False).first()[["unique_id"]],
        future_cols=["hour_sin", "hour_cos"],
        historic_cols=[],
        target_col="y_model",
        raw_target_col="y",
    )

    model = MLP(
        h=24,
        input_size=48,
        futr_exog_list=["hour_sin", "hour_cos"],
        loss=MAE(),
        max_steps=5,
        val_check_steps=100,
        enable_progress_bar=False,
    )
    adapter = NeuralForecastAdapter(model, "MLP")

    _, preds = evaluate_forecaster(adapter, bundle, use_log_target=False, evaluation_split="test")

    assert preds["ds"].min() == bundle.test_df["ds"].min()
    assert preds["ds"].max() == bundle.test_df["ds"].max()
    assert len(preds) == len(bundle.test_df)


def test_export_rl_forecast_features_produces_stable_schema(split_hourly_df: pd.DataFrame):
    bundle = prepare_forecast_frames(split_hourly_df, ForecastConfig())
    future = bundle.test_df[["unique_id", "ds", bundle.raw_target_col]].rename(columns={bundle.raw_target_col: "model_prediction"})
    future["lo"] = future["model_prediction"] * 0.9
    future["hi"] = future["model_prediction"] * 1.1
    exported = export_rl_forecast_features(future, prediction_col="model_prediction", lo_col="lo", hi_col="hi")
    assert list(exported.columns) == ["unique_id", "ds", "forecast_median", "forecast_lo_10", "forecast_hi_90"]
    assert not exported.duplicated(["unique_id", "ds"]).any()

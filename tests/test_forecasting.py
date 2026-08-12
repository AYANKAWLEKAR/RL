from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    ForecastConfig,
    ForecastDatasetBundle,
    NeuralForecastAdapter,
    PersistenceForecaster,
    ZeroForecaster,
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


def test_val_size_is_per_series_not_total_rows():
    """NeuralForecast's val_size counts PER-SERIES timesteps, not dataframe rows.

    Passing len(val_df) happens to work with 1-2 series and fails hard on a real
    panel: with 502 series x ~1629 train steps, len(val_df)=175198 made
    fit() compute 1629 - 175198 = -173569 available steps and raise.
    """
    class _Model:
        input_size = 168
        h = 24

    adapter = NeuralForecastAdapter(_Model(), "TFT")
    n_series, train_steps, val_steps = 60, 1000, 200
    train_df = pd.DataFrame({"unique_id": np.repeat([f"s{i}" for i in range(n_series)], train_steps)})
    val_df = pd.DataFrame({"unique_id": np.repeat([f"s{i}" for i in range(n_series)], val_steps)})

    val_size = adapter._val_size(train_df, val_df)

    assert val_size == val_steps, "should be per-series steps, not len(val_df)"
    assert val_size < len(val_df), "using the raw row count is the bug this guards"
    # every series must retain enough history to form at least one training window
    assert train_steps - val_size >= _Model.input_size + _Model.h

    # clamped when the requested holdout would starve training
    greedy_val = pd.DataFrame({"unique_id": np.repeat([f"s{i}" for i in range(n_series)], 990)})
    assert adapter._val_size(train_df, greedy_val) == train_steps - _Model.input_size - _Model.h


def test_zero_forecaster_is_the_control_for_sparse_targets(split_hourly_df: pd.DataFrame):
    """On intermittent demand, MAE is minimized by the conditional median, which is
    zero when most periods are zero. ZeroForecaster makes that floor explicit so a
    model's MAE is never mistaken for skill.
    """
    bundle = prepare_forecast_frames(split_hourly_df, ForecastConfig())
    metrics, preds = evaluate_forecaster(ZeroForecaster(), bundle, use_log_target=True, evaluation_split="val")

    assert (preds["model_prediction"] == 0.0).all()
    # MAE of the all-zeros forecast is exactly the mean of |actual|.
    assert metrics.mae == pytest.approx(preds["actual"].abs().mean(), rel=1e-9)
    # It never over-forecasts, so bias is the negative of the mean level.
    assert metrics.bias <= 0.0


def test_adapter_ignores_spurious_index_column_from_reset_index():
    """NeuralForecast >=3.2 returns unique_id as a COLUMN with a RangeIndex.

    Calling .reset_index() on that injects an "index" column holding row numbers.
    It sorts first, so a positional pick scores the ROW NUMBER as the forecast -
    values 0..n-1, which overflow expm1 under a log target and produced mae=inf on
    real data while every local test still passed on the older nf layout.
    """
    bundle = ForecastDatasetBundle(
        full_df=pd.DataFrame(), train_df=pd.DataFrame(), val_df=pd.DataFrame(), test_df=pd.DataFrame(),
        static_df=pd.DataFrame(), future_cols=[], historic_cols=[], target_col="y_model", raw_target_col="y",
    )

    class _NF32:
        """Mimics neuralforecast >= 3.2: unique_id is a column, index is a RangeIndex."""

        def make_future_dataframe(self, df=None):
            return pd.DataFrame({"unique_id": ["A", "B"], "ds": [pd.Timestamp("2024-01-02")] * 2})

        def predict(self, df=None, static_df=None, futr_df=None):
            return pd.DataFrame(
                {"unique_id": ["A", "B"], "ds": [pd.Timestamp("2024-01-02")] * 2, "TFT": [0.25, 0.75]},
                index=pd.RangeIndex(2),
            )

    class _Model:
        h = 1

    adapter = NeuralForecastAdapter(_Model(), "TFT")
    adapter._nf = _NF32()
    history = pd.DataFrame({"unique_id": ["A", "B"], "ds": [pd.Timestamp("2024-01-01")] * 2, "y_model": [1.0, 2.0]})
    future = pd.DataFrame({"unique_id": ["A", "B"], "ds": [pd.Timestamp("2024-01-02")] * 2, "y_model": [1.5, 2.5]})

    out = adapter.predict(history, future, bundle)

    assert "index" not in out.columns
    assert list(out["prediction"]) == [0.25, 0.75], "must score the TFT column, not row numbers"


def test_adapter_prefers_median_column_under_quantile_loss():
    """Under MQLoss the output has median/lo/hi columns in no guaranteed order, so
    taking the first column can score a tail quantile as the point forecast.
    """
    bundle = ForecastDatasetBundle(
        full_df=pd.DataFrame(), train_df=pd.DataFrame(), val_df=pd.DataFrame(), test_df=pd.DataFrame(),
        static_df=pd.DataFrame(), future_cols=[], historic_cols=[], target_col="y_model", raw_target_col="y",
    )

    class _FakeNF:
        def make_future_dataframe(self, df=None):
            return pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2024-01-02")]})

        def predict(self, df=None, static_df=None, futr_df=None):
            # deliberately puts a tail quantile first
            return pd.DataFrame({
                "unique_id": ["A"], "ds": [pd.Timestamp("2024-01-02")],
                "TFT-lo-80": [1.0], "TFT-median": [5.0], "TFT-hi-80": [9.0],
            })

    class _Model:
        h = 1

    adapter = NeuralForecastAdapter(_Model(), "TFT")
    adapter._nf = _FakeNF()
    history = pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2024-01-01")], "y_model": [3.0]})
    future = pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2024-01-02")], "y_model": [4.0]})

    out = adapter.predict(history, future, bundle)
    assert out["prediction"].iloc[0] == 5.0, "should select TFT-median, not the first column"


def test_export_rl_forecast_features_produces_stable_schema(split_hourly_df: pd.DataFrame):
    bundle = prepare_forecast_frames(split_hourly_df, ForecastConfig())
    future = bundle.test_df[["unique_id", "ds", bundle.raw_target_col]].rename(columns={bundle.raw_target_col: "model_prediction"})
    future["lo"] = future["model_prediction"] * 0.9
    future["hi"] = future["model_prediction"] * 1.1
    exported = export_rl_forecast_features(future, prediction_col="model_prediction", lo_col="lo", hi_col="hi")
    assert list(exported.columns) == ["unique_id", "ds", "forecast_median", "forecast_lo_10", "forecast_hi_90"]
    assert not exported.duplicated(["unique_id", "ds"]).any()

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecastConfig:
    horizon: int = 24
    input_size: int = 168
    use_log_target: bool = True
    futr_cols: tuple[str, ...] = (
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "holiday_flag",
        "discount",
    )
    hist_cols: tuple[str, ...] = (
        "stockout_flag",
        "store_mean_lag24",
        "category_mean_lag24",
    )
    static_cols: tuple[str, ...] = (
        "product_id",
        "store_id",
        "city_id",
        "first_category_id",
    )


@dataclass
class ForecastDatasetBundle:
    full_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    static_df: pd.DataFrame
    future_cols: list[str]
    historic_cols: list[str]
    target_col: str
    raw_target_col: str


@dataclass
class ForecastMetrics:
    model_name: str
    mae: float
    rmse: float
    bias: float
    n_obs: int


class ForecastModelAdapter(Protocol):
    name: str

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, bundle: ForecastDatasetBundle) -> None:
        ...

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame, bundle: ForecastDatasetBundle) -> pd.DataFrame:
        ...


class PersistenceForecaster:
    name = "Persistence"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, bundle: ForecastDatasetBundle) -> None:
        return None

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame, bundle: ForecastDatasetBundle) -> pd.DataFrame:
        history = history_df[["unique_id", "ds", bundle.target_col]].copy()
        lookup = history.rename(columns={"ds": "lag_ds", bundle.target_col: "prediction"})
        preds = future_df[["unique_id", "ds"]].copy()
        preds["lag_ds"] = preds["ds"] - pd.Timedelta(hours=24)
        preds = preds.merge(lookup, on=["unique_id", "lag_ds"], how="left").drop(columns=["lag_ds"])
        if preds["prediction"].isna().any():
            fallback = history.groupby("unique_id")[bundle.target_col].last().rename("fallback")
            preds = preds.merge(fallback, on="unique_id", how="left")
            preds["prediction"] = preds["prediction"].fillna(preds["fallback"]).fillna(0.0)
            preds = preds.drop(columns=["fallback"])
        return preds


def prepare_forecast_frames(
    hourly_df: pd.DataFrame,
    config: ForecastConfig | None = None,
) -> ForecastDatasetBundle:
    config = config or ForecastConfig()
    raw_target_col = "y"
    target_col = "y_model"

    frame = hourly_df[["unique_id", "ds", "split", raw_target_col] + list(config.futr_cols) + list(config.hist_cols)].copy()
    frame[target_col] = np.log1p(frame[raw_target_col].clip(lower=0)) if config.use_log_target else frame[raw_target_col]

    static_df = (
        hourly_df.groupby("unique_id", as_index=False)[list(config.static_cols)]
        .first()
        .reset_index(drop=True)
    )

    return ForecastDatasetBundle(
        full_df=frame,
        train_df=frame[frame["split"] == "train"].drop(columns=["split"]).reset_index(drop=True),
        val_df=frame[frame["split"] == "val"].drop(columns=["split"]).reset_index(drop=True),
        test_df=frame[frame["split"] == "test"].drop(columns=["split"]).reset_index(drop=True),
        static_df=static_df,
        future_cols=list(config.futr_cols),
        historic_cols=list(config.hist_cols),
        target_col=target_col,
        raw_target_col=raw_target_col,
    )


def inverse_transform_predictions(predictions: pd.Series | np.ndarray, use_log_target: bool) -> np.ndarray:
    preds = np.asarray(predictions, dtype=float)
    return np.expm1(preds) if use_log_target else preds


def evaluate_forecaster(
    model_adapter: ForecastModelAdapter,
    bundle: ForecastDatasetBundle,
    use_log_target: bool = True,
    evaluation_split: str = "val",
) -> tuple[ForecastMetrics, pd.DataFrame]:
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation_split must be 'val' or 'test'")

    future_df = bundle.val_df if evaluation_split == "val" else bundle.test_df
    history_df = bundle.train_df if evaluation_split == "val" else pd.concat([bundle.train_df, bundle.val_df], ignore_index=True)
    model_adapter.fit(bundle.train_df, bundle.val_df, bundle)
    preds = model_adapter.predict(history_df, future_df, bundle)

    merged = future_df[["unique_id", "ds", bundle.target_col, bundle.raw_target_col]].merge(
        preds.rename(columns={"prediction": "model_prediction"}),
        on=["unique_id", "ds"],
        how="inner",
    )
    if use_log_target:
        merged["model_prediction"] = inverse_transform_predictions(merged["model_prediction"], use_log_target=True)
    merged["actual"] = merged[bundle.raw_target_col]

    clean = merged[["actual", "model_prediction"]].replace([np.inf, -np.inf], np.nan).dropna()
    metrics = ForecastMetrics(
        model_name=model_adapter.name,
        mae=float(mean_absolute_error(clean["actual"], clean["model_prediction"])) if len(clean) else np.nan,
        rmse=float(np.sqrt(mean_squared_error(clean["actual"], clean["model_prediction"]))) if len(clean) else np.nan,
        bias=float((clean["model_prediction"] - clean["actual"]).mean()) if len(clean) else np.nan,
        n_obs=int(len(clean)),
    )
    return metrics, merged[["unique_id", "ds", "actual", "model_prediction"]]


def choose_best_forecaster(
    adapters: list[ForecastModelAdapter],
    bundle: ForecastDatasetBundle,
    use_log_target: bool = True,
) -> tuple[ForecastMetrics, pd.DataFrame, ForecastModelAdapter]:
    best_metrics: ForecastMetrics | None = None
    best_predictions: pd.DataFrame | None = None
    best_adapter: ForecastModelAdapter | None = None
    for adapter in adapters:
        metrics, predictions = evaluate_forecaster(adapter, bundle, use_log_target=use_log_target, evaluation_split="val")
        if best_metrics is None or metrics.mae < best_metrics.mae:
            best_metrics = metrics
            best_predictions = predictions
            best_adapter = adapter
    assert best_metrics is not None and best_predictions is not None and best_adapter is not None
    return best_metrics, best_predictions, best_adapter


def export_rl_forecast_features(
    predictions_df: pd.DataFrame,
    prediction_col: str = "model_prediction",
    lo_col: str | None = None,
    hi_col: str | None = None,
) -> pd.DataFrame:
    export = predictions_df[["unique_id", "ds", prediction_col]].copy()
    export = export.rename(columns={prediction_col: "forecast_median"})
    if lo_col and lo_col in predictions_df.columns:
        export["forecast_lo_10"] = predictions_df[lo_col].values
    if hi_col and hi_col in predictions_df.columns:
        export["forecast_hi_90"] = predictions_df[hi_col].values
    export = export.sort_values(["unique_id", "ds"]).drop_duplicates(["unique_id", "ds"]).reset_index(drop=True)
    return export


class NeuralForecastAdapter:
    """Thin optional adapter so notebooks can still train NeuralForecast models."""

    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self._nf = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, bundle: ForecastDatasetBundle) -> None:
        from neuralforecast import NeuralForecast

        self._nf = NeuralForecast(models=[self.model], freq="h")
        self._nf.fit(df=train_df[["unique_id", "ds", bundle.target_col] + bundle.future_cols + bundle.historic_cols], val_size=len(val_df))

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame, bundle: ForecastDatasetBundle) -> pd.DataFrame:
        if self._nf is None:
            raise RuntimeError("Adapter must be fit before predict")
        futr_df = future_df[["unique_id", "ds"] + bundle.future_cols].copy()
        preds = self._nf.predict(futr_df=futr_df).reset_index()
        model_cols = [c for c in preds.columns if c not in {"unique_id", "ds"}]
        pred_col = model_cols[0]
        return preds[["unique_id", "ds", pred_col]].rename(columns={pred_col: "prediction"})

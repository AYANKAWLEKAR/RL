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


class ZeroForecaster:
    """Predicts zero everywhere. The control for intermittent demand.

    On sparse series (FreshRetailNet is ~78% zero hours, mean 0.042 units/hour)
    MAE is minimized by the conditional median, which is zero. A model can post an
    impressive-looking MAE while having learned nothing. Always report this
    alongside real models: if a model cannot beat predicting zero, its MAE is
    measuring the sparsity of the data, not the skill of the model.
    """

    name = "Zero"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, bundle: ForecastDatasetBundle) -> None:
        return None

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame, bundle: ForecastDatasetBundle) -> pd.DataFrame:
        preds = future_df[["unique_id", "ds"]].copy()
        preds["prediction"] = 0.0
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
    """Thin optional adapter so notebooks can still train NeuralForecast models.

    NOTE: for static covariates (`bundle.static_cols`) to actually influence
    predictions, the wrapped `model` must itself be constructed with
    `stat_exog_list=list(bundle.static_cols)` — passing `static_df` into `fit()`
    only supplies the values, the model still has to be told to use them.
    """

    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self._nf = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, bundle: ForecastDatasetBundle) -> None:
        from neuralforecast import NeuralForecast

        self._nf = NeuralForecast(models=[self.model], freq="h")
        cols = ["unique_id", "ds", bundle.target_col] + bundle.future_cols + bundle.historic_cols
        # NeuralForecast looks for a column literally named "y" unless told otherwise;
        # bundle.target_col is "y_model" whenever a log1p transform is applied, so it
        # must be renamed here or fit() raises KeyError('y').
        nf_train = train_df[cols].rename(columns={bundle.target_col: "y"})
        self._nf.fit(df=nf_train, static_df=bundle.static_df, val_size=len(val_df))

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame, bundle: ForecastDatasetBundle) -> pd.DataFrame:
        if self._nf is None:
            raise RuntimeError("Adapter must be fit before predict")

        # A single nf.predict() call only forecasts the h steps immediately after the
        # data it was fit/re-based on — it does NOT know how to reach an arbitrary,
        # possibly far-future `future_df` window on its own. Walk forward, feeding
        # back the true observed values (already present in future_df) as new history
        # at each step, so the full requested span gets covered with dates that
        # actually line up with future_df. Each window's expected (unique_id, ds)
        # combinations come from `make_future_dataframe(df=...)` rather than a
        # global date union, because per-series history can end at slightly
        # different timestamps (e.g. differing NaN-drop counts from lag features).
        cols = ["unique_id", "ds", bundle.target_col] + bundle.future_cols + bundle.historic_cols
        rolling_history = history_df[cols].rename(columns={bundle.target_col: "y"})
        future_exog = future_df[["unique_id", "ds"] + bundle.future_cols]
        future_actuals = future_df[cols].rename(columns={bundle.target_col: "y"})
        max_ds = future_df["ds"].max()

        forecasts: list[pd.DataFrame] = []
        max_iterations = 500
        for _ in range(max_iterations):
            if rolling_history["ds"].max() >= max_ds:
                break
            expected = self._nf.make_future_dataframe(df=rolling_history)
            futr_input = expected.merge(future_exog, on=["unique_id", "ds"], how="left")
            futr_input[bundle.future_cols] = futr_input[bundle.future_cols].fillna(0.0)

            preds = self._nf.predict(df=rolling_history, static_df=bundle.static_df, futr_df=futr_input).reset_index()
            forecasts.append(preds)

            next_actuals = future_actuals.merge(expected[["unique_id", "ds"]], on=["unique_id", "ds"], how="inner")
            if next_actuals.empty:
                break
            rolling_history = pd.concat([rolling_history, next_actuals], ignore_index=True)

        if not forecasts:
            return pd.DataFrame(columns=["unique_id", "ds", "prediction"])
        all_preds = pd.concat(forecasts, ignore_index=True)
        model_cols = [c for c in all_preds.columns if c not in {"unique_id", "ds"}]
        if not model_cols:
            raise RuntimeError("NeuralForecast returned no prediction columns")
        # Under a point loss (MAE) there is exactly one output column. Under a
        # quantile loss (MQLoss) there are several — "<Model>-median", "<Model>-lo-90",
        # "<Model>-hi-90" ... — and column order is NOT guaranteed to put the median
        # first, so taking model_cols[0] can silently score a tail quantile as if it
        # were the point forecast. Prefer an explicit median column when present.
        median_cols = [c for c in model_cols if "median" in c.lower()]
        pred_col = median_cols[0] if median_cols else model_cols[0]
        return all_preds[["unique_id", "ds", pred_col]].rename(columns={pred_col: "prediction"})

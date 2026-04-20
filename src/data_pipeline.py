from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class LatentDemandConfig:
    sequence_length: int = 24
    same_hour_window_days: int = 7
    min_same_hour_points: int = 2
    short_stockout_max_hours: int = 3
    model_min_training_rows: int = 128
    random_state: int = 42


@dataclass
class FeatureConfig:
    lag_periods: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 24, 48, 168, 336, 720)
    rolling_windows: tuple[int, ...] = (6, 24, 168)
    max_ffill_hours: int = 6
    train_frac: float = 0.70
    val_frac: float = 0.15
    scale_binary_features: bool = False


@dataclass
class ReconstructionDiagnostics:
    model_used: bool
    total_stockout_hours: int
    heuristic_imputed_hours: int
    model_imputed_hours: int
    fallback_mean_hours: int
    masked_backtest_rows: int = 0
    masked_backtest_mae_heuristic: float | None = None
    masked_backtest_mae_model: float | None = None
    imputation_source_counts: dict[str, int] = field(default_factory=dict)


def parse_sequence(val: Any, expected_length: int | None = 24, dtype=np.float64) -> np.ndarray:
    """Parse a list-like value into a numeric numpy array."""
    if isinstance(val, np.ndarray):
        arr = val.astype(dtype, copy=False)
    elif isinstance(val, list):
        arr = np.asarray(val, dtype=dtype)
    elif isinstance(val, str):
        arr = np.asarray(ast.literal_eval(val), dtype=dtype)
    else:
        raise TypeError(f"Unsupported sequence type: {type(val)!r}")

    if expected_length is not None and len(arr) != expected_length:
        raise ValueError(f"Expected sequence length {expected_length}, got {len(arr)}")
    return arr


def prepare_daily_dataframe(
    df: pd.DataFrame,
    sale_col: str = "hours_sale",
    stock_col: str = "hours_stock_status",
    config: LatentDemandConfig | None = None,
) -> pd.DataFrame:
    config = config or LatentDemandConfig()
    prepared = df.copy()
    prepared["dt"] = pd.to_datetime(prepared["dt"])
    prepared[sale_col] = prepared[sale_col].apply(
        lambda val: parse_sequence(val, expected_length=config.sequence_length, dtype=np.float64)
    )
    prepared[stock_col] = prepared[stock_col].apply(
        lambda val: parse_sequence(val, expected_length=config.sequence_length, dtype=np.int32)
    )
    return prepared


def _same_hour_heuristic(
    sales: np.ndarray,
    status: np.ndarray,
    day_idx: int,
    hour_idx: int,
    config: LatentDemandConfig,
) -> tuple[float | None, str]:
    window_start = max(0, day_idx - config.same_hour_window_days)
    window_end = min(len(sales), day_idx + config.same_hour_window_days + 1)
    same_hour_values: list[float] = []
    for offset in range(window_start, window_end):
        if offset == day_idx:
            continue
        if status[offset, hour_idx] == 0:
            same_hour_values.append(float(sales[offset, hour_idx]))
    if len(same_hour_values) >= config.min_same_hour_points:
        return float(np.mean(same_hour_values)), "heuristic_same_hour"

    in_stock_today = sales[day_idx, status[day_idx] == 0]
    if len(in_stock_today) > 0:
        return float(np.mean(in_stock_today)), "fallback_day_mean"

    global_in_stock = sales[status == 0]
    if len(global_in_stock) > 0:
        return float(np.mean(global_in_stock)), "fallback_global_mean"
    return 0.0, "fallback_zero"


def _build_reconstruction_training_frame(
    prepared_df: pd.DataFrame,
    config: LatentDemandConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in prepared_df.sort_values(["product_id", "store_id", "dt"]).iterrows():
        sales = row["hours_sale"]
        status = row["hours_stock_status"]
        for hour in range(config.sequence_length):
            if status[hour] != 0:
                continue
            records.append(
                {
                    "target": float(sales[hour]),
                    "hour": hour,
                    "day_of_week": row["dt"].dayofweek,
                    "month": row["dt"].month,
                    "discount": float(row.get("discount", 0.0)),
                    "holiday_flag": int(row.get("holiday_flag", 0)),
                    "activity_flag": int(row.get("activity_flag", 0)),
                    "precpt": float(row.get("precpt", 0.0)),
                    "avg_temperature": float(row.get("avg_temperature", 0.0)),
                    "avg_humidity": float(row.get("avg_humidity", 0.0)),
                    "avg_wind_level": float(row.get("avg_wind_level", 0.0)),
                    "sale_amount": float(row.get("sale_amount", np.sum(sales))),
                    "store_id": int(row.get("store_id", 0)),
                    "product_id": int(row.get("product_id", 0)),
                    "first_category_id": int(row.get("first_category_id", 0)),
                }
            )
    return pd.DataFrame.from_records(records)


def _model_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "hour",
        "day_of_week",
        "month",
        "discount",
        "holiday_flag",
        "activity_flag",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "sale_amount",
        "store_id",
        "product_id",
        "first_category_id",
    ]
    features = frame.reindex(columns=cols, fill_value=0.0).copy()
    return features.astype(float)


def backtest_reconstruction(
    prepared_df: pd.DataFrame,
    config: LatentDemandConfig | None = None,
    max_samples: int = 256,
) -> ReconstructionDiagnostics:
    config = config or LatentDemandConfig()
    train_frame = _build_reconstruction_training_frame(prepared_df, config)
    if train_frame.empty:
        return ReconstructionDiagnostics(
            model_used=False,
            total_stockout_hours=0,
            heuristic_imputed_hours=0,
            model_imputed_hours=0,
            fallback_mean_hours=0,
        )

    model = None
    if len(train_frame) >= config.model_min_training_rows:
        model = HistGradientBoostingRegressor(random_state=config.random_state)
        model.fit(_model_feature_frame(train_frame), train_frame["target"])

    sampled = train_frame.head(max_samples).copy()
    heuristic_preds = np.repeat(train_frame["target"].median(), len(sampled))
    model_preds = None
    if model is not None and len(sampled) > 0:
        model_preds = model.predict(_model_feature_frame(sampled))

    return ReconstructionDiagnostics(
        model_used=model is not None,
        total_stockout_hours=0,
        heuristic_imputed_hours=0,
        model_imputed_hours=0,
        fallback_mean_hours=0,
        masked_backtest_rows=len(sampled),
        masked_backtest_mae_heuristic=float(np.mean(np.abs(sampled["target"] - heuristic_preds))),
        masked_backtest_mae_model=(
            float(np.mean(np.abs(sampled["target"] - model_preds))) if model_preds is not None else None
        ),
    )


def reconstruct_latent_demand(
    df: pd.DataFrame,
    config: LatentDemandConfig | None = None,
) -> tuple[pd.DataFrame, ReconstructionDiagnostics]:
    config = config or LatentDemandConfig()
    prepared = prepare_daily_dataframe(df, config=config)
    train_frame = _build_reconstruction_training_frame(prepared, config)

    model = None
    if len(train_frame) >= config.model_min_training_rows:
        model = HistGradientBoostingRegressor(random_state=config.random_state)
        model.fit(_model_feature_frame(train_frame), train_frame["target"])

    total_stockout_hours = 0
    heuristic_imputed_hours = 0
    model_imputed_hours = 0
    fallback_mean_hours = 0
    source_counts: dict[str, int] = {}
    reconstructed_groups: list[pd.DataFrame] = []

    grouped = prepared.groupby(["product_id", "store_id"], sort=False)
    for _, group in grouped:
        group = group.sort_values("dt").reset_index(drop=True)
        sales = np.stack(group["hours_sale"].values)
        status = np.stack(group["hours_stock_status"].values)
        latent = sales.copy()

        for day_idx in range(len(group)):
            stockout_hours = np.where(status[day_idx] == 1)[0]
            total_stockout_hours += len(stockout_hours)
            for hour_idx in stockout_hours:
                stockout_len = int(np.sum(status[day_idx] == 1))
                estimate, source = _same_hour_heuristic(sales, status, day_idx, hour_idx, config)
                if model is not None and stockout_len > config.short_stockout_max_hours:
                    model_row = pd.DataFrame(
                        [
                            {
                                "hour": hour_idx,
                                "day_of_week": group.loc[day_idx, "dt"].dayofweek,
                                "month": group.loc[day_idx, "dt"].month,
                                "discount": float(group.loc[day_idx].get("discount", 0.0)),
                                "holiday_flag": int(group.loc[day_idx].get("holiday_flag", 0)),
                                "activity_flag": int(group.loc[day_idx].get("activity_flag", 0)),
                                "precpt": float(group.loc[day_idx].get("precpt", 0.0)),
                                "avg_temperature": float(group.loc[day_idx].get("avg_temperature", 0.0)),
                                "avg_humidity": float(group.loc[day_idx].get("avg_humidity", 0.0)),
                                "avg_wind_level": float(group.loc[day_idx].get("avg_wind_level", 0.0)),
                                "sale_amount": float(group.loc[day_idx].get("sale_amount", np.sum(sales[day_idx]))),
                                "store_id": int(group.loc[day_idx].get("store_id", 0)),
                                "product_id": int(group.loc[day_idx].get("product_id", 0)),
                                "first_category_id": int(group.loc[day_idx].get("first_category_id", 0)),
                            }
                        ]
                    )
                    estimate = float(model.predict(_model_feature_frame(model_row))[0])
                    source = "model_fallback"

                latent[day_idx, hour_idx] = max(0.0, estimate if estimate is not None else 0.0)
                source_counts[source] = source_counts.get(source, 0) + 1
                if source == "heuristic_same_hour":
                    heuristic_imputed_hours += 1
                elif source == "model_fallback":
                    model_imputed_hours += 1
                else:
                    fallback_mean_hours += 1

        out = group.copy()
        out["latent_demand"] = list(latent)
        reconstructed_groups.append(out)

    diagnostics = backtest_reconstruction(prepared, config=config)
    diagnostics.model_used = model is not None
    diagnostics.total_stockout_hours = total_stockout_hours
    diagnostics.heuristic_imputed_hours = heuristic_imputed_hours
    diagnostics.model_imputed_hours = model_imputed_hours
    diagnostics.fallback_mean_hours = fallback_mean_hours
    diagnostics.imputation_source_counts = source_counts
    return pd.concat(reconstructed_groups, ignore_index=True), diagnostics


def expand_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        base_dt = pd.Timestamp(row["dt"])
        latent = np.asarray(row["latent_demand"], dtype=float)
        stock_status = np.asarray(row["hours_stock_status"], dtype=int)
        observed_sale = np.asarray(row["hours_sale"], dtype=float)
        for hour in range(len(latent)):
            records.append(
                {
                    "unique_id": f"{row['product_id']}_{row['store_id']}",
                    "ds": base_dt + timedelta(hours=hour),
                    "y": float(latent[hour]),
                    "observed_sale": float(observed_sale[hour]),
                    "stockout_flag": int(stock_status[hour]),
                    "product_id": row["product_id"],
                    "store_id": row["store_id"],
                    "city_id": row.get("city_id"),
                    "first_category_id": row.get("first_category_id"),
                    "second_category_id": row.get("second_category_id"),
                    "third_category_id": row.get("third_category_id"),
                    "discount": float(row.get("discount", 0.0)),
                    "holiday_flag": int(row.get("holiday_flag", 0)),
                    "activity_flag": int(row.get("activity_flag", 0)),
                    "precpt": float(row.get("precpt", 0.0)),
                    "avg_temperature": float(row.get("avg_temperature", 0.0)),
                    "avg_humidity": float(row.get("avg_humidity", 0.0)),
                    "avg_wind_level": float(row.get("avg_wind_level", 0.0)),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _days_since_until(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    vals = series.to_numpy()
    n = len(vals)
    since = np.full(n, np.nan)
    until = np.full(n, np.nan)
    last_idx = -1
    for idx in range(n):
        if vals[idx] == 1:
            last_idx = idx
        if last_idx >= 0:
            since[idx] = idx - last_idx
    next_idx = -1
    for idx in range(n - 1, -1, -1):
        if vals[idx] == 1:
            next_idx = idx
        if next_idx >= 0:
            until[idx] = next_idx - idx
    return since, until


def build_hourly_features(
    daily_df: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
) -> pd.DataFrame:
    feature_config = feature_config or FeatureConfig()
    hourly_df = expand_to_hourly(daily_df)
    hourly_df["hour"] = hourly_df["ds"].dt.hour
    hourly_df["day_of_week"] = hourly_df["ds"].dt.dayofweek
    hourly_df["month"] = hourly_df["ds"].dt.month
    hourly_df["week_of_year"] = hourly_df["ds"].dt.isocalendar().week.astype(int)
    hourly_df["day"] = hourly_df["ds"].dt.day
    hourly_df["is_weekend"] = (hourly_df["day_of_week"] >= 5).astype(int)

    for col, period in [("hour", 24), ("day_of_week", 7), ("month", 12), ("week_of_year", 52)]:
        hourly_df[f"{col}_sin"] = np.sin(2 * np.pi * hourly_df[col] / period)
        hourly_df[f"{col}_cos"] = np.cos(2 * np.pi * hourly_df[col] / period)

    for flag, prefix in [("activity_flag", "promo"), ("stockout_flag", "stockout")]:
        since_parts = []
        until_parts = []
        for _, grp in hourly_df.groupby("unique_id", sort=False):
            since, until = _days_since_until(grp[flag])
            since_parts.append(pd.Series(since, index=grp.index))
            until_parts.append(pd.Series(until, index=grp.index))
        hourly_df[f"hours_since_{prefix}"] = pd.concat(since_parts).sort_index()
        hourly_df[f"hours_until_{prefix}"] = pd.concat(until_parts).sort_index()

    for lag in feature_config.lag_periods:
        hourly_df[f"lag_{lag}"] = hourly_df.groupby("unique_id", sort=False)["y"].shift(lag)

    for window in feature_config.rolling_windows:
        grp = hourly_df.groupby("unique_id", sort=False)["y"]
        hourly_df[f"rolling_mean_{window}h"] = grp.transform(
            lambda series: series.shift(1).rolling(window, min_periods=1).mean()
        )
        hourly_df[f"rolling_std_{window}h"] = grp.transform(
            lambda series: series.shift(1).rolling(window, min_periods=1).std()
        )

    grp24 = hourly_df.groupby("unique_id", sort=False)["y"]
    hourly_df["rolling_min_24h"] = grp24.transform(lambda series: series.shift(1).rolling(24, min_periods=1).min())
    hourly_df["rolling_max_24h"] = grp24.transform(lambda series: series.shift(1).rolling(24, min_periods=1).max())

    for lag_offset in (24, 168):
        for window in (24, 168):
            col = f"rolling_mean_{window}h"
            if col in hourly_df.columns:
                hourly_df[f"{col}_lag{lag_offset}"] = hourly_df.groupby("unique_id", sort=False)[col].shift(lag_offset)

    hourly_df["discount_x_holiday"] = hourly_df["discount"] * hourly_df["holiday_flag"]
    hourly_df["temp_x_hour_sin"] = hourly_df["avg_temperature"] * hourly_df["hour_sin"]
    hourly_df["precpt_x_weekend"] = hourly_df["precpt"] * hourly_df["is_weekend"]
    hourly_df["prev_day_sale_x_discount"] = hourly_df["lag_24"] * hourly_df["discount"]

    store_agg = hourly_df.groupby(["store_id", "ds"], as_index=False)["y"].mean()
    store_agg["ds"] = store_agg["ds"] + pd.Timedelta(hours=24)
    store_agg = store_agg.rename(columns={"y": "store_mean_lag24"})
    hourly_df = hourly_df.merge(store_agg, on=["store_id", "ds"], how="left")

    category_agg = hourly_df.groupby(["first_category_id", "ds"], as_index=False)["y"].mean()
    category_agg["ds"] = category_agg["ds"] + pd.Timedelta(hours=24)
    category_agg = category_agg.rename(columns={"y": "category_mean_lag24"})
    hourly_df = hourly_df.merge(category_agg, on=["first_category_id", "ds"], how="left")
    hourly_df["store_mean_lag24"] = hourly_df["store_mean_lag24"].fillna(0.0)
    hourly_df["category_mean_lag24"] = hourly_df["category_mean_lag24"].fillna(0.0)
    return hourly_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _apply_groupwise_ffill(hourly_df: pd.DataFrame, limit: int) -> pd.DataFrame:
    filled = hourly_df.copy()
    cols = [col for col in filled.columns if col != "unique_id"]
    filled.loc[:, cols] = hourly_df.groupby("unique_id", sort=False)[cols].ffill(limit=limit)
    return filled.reset_index(drop=True)


def split_and_scale_features(
    hourly_df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    config = config or FeatureConfig()
    processed = hourly_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    processed = _apply_groupwise_ffill(processed, limit=config.max_ffill_hours)
    processed = processed.dropna(subset=["y"]).reset_index(drop=True)

    def chrono_split(group: pd.DataFrame) -> pd.DataFrame:
        n = len(group)
        train_end = int(n * config.train_frac)
        val_end = int(n * (config.train_frac + config.val_frac))
        group = group.sort_values("ds").copy()
        group["split"] = "train"
        group.iloc[train_end:val_end, group.columns.get_loc("split")] = "val"
        group.iloc[val_end:, group.columns.get_loc("split")] = "test"
        return group

    split_groups: list[pd.DataFrame] = []
    for uid, grp in processed.groupby("unique_id", sort=False):
        split_grp = chrono_split(grp)
        split_grp["unique_id"] = uid
        split_groups.append(split_grp)
    processed = pd.concat(split_groups, ignore_index=True)

    id_cols = {
        "unique_id",
        "ds",
        "product_id",
        "store_id",
        "city_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "split",
    }
    protected = {"y", "observed_sale"}
    binary_cols = {
        col
        for col in processed.columns
        if set(processed[col].dropna().unique()).issubset({0, 1})
    }
    numeric_cols = processed.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in id_cols | protected]
    if not config.scale_binary_features:
        feature_cols = [col for col in feature_cols if col not in binary_cols]

    scaler = StandardScaler()
    train_mask = processed["split"] == "train"
    if feature_cols:
        scaler.fit(processed.loc[train_mask, feature_cols])
        scaled = pd.DataFrame(
            scaler.transform(processed[feature_cols]),
            columns=feature_cols,
            index=processed.index,
        )
        for col in feature_cols:
            processed[col] = scaled[col].astype(float)
    return processed, scaler

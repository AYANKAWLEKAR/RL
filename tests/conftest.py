from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import (
    FeatureConfig,
    LatentDemandConfig,
    build_hourly_features,
    reconstruct_latent_demand,
    split_and_scale_features,
)


def _daily_hours(day_idx: int, product_bias: float) -> np.ndarray:
    hours = np.arange(24)
    base = 4 + product_bias + 1.5 * np.sin((hours - 6) / 24 * 2 * np.pi) + 0.05 * day_idx
    return np.clip(base, 0.2, None).round(3)


@pytest.fixture()
def synthetic_daily_df() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01")
    for product_id, store_id, category_id, bias in [(101, 1, 10, 0.0), (202, 2, 10, 1.2), (303, 3, 20, 2.0)]:
        for day_idx in range(42):
            sales = _daily_hours(day_idx, bias)
            stock = np.zeros(24, dtype=int)
            if product_id == 101 and day_idx == 12:
                stock[8:10] = 1
                sales[8:10] = 0
            if product_id == 202 and day_idx == 20:
                stock[6:12] = 1
                sales[6:12] = 0
            rows.append(
                {
                    "city_id": store_id,
                    "store_id": store_id,
                    "management_group_id": 1,
                    "first_category_id": category_id,
                    "second_category_id": category_id * 10,
                    "third_category_id": category_id * 100,
                    "product_id": product_id,
                    "dt": start + pd.Timedelta(days=day_idx),
                    "sale_amount": float(sales.sum()),
                    "hours_sale": sales.tolist(),
                    "hours_stock_status": stock.tolist(),
                    "discount": 0.1 * (day_idx % 3),
                    "holiday_flag": int(day_idx % 7 == 0),
                    "activity_flag": int(day_idx % 5 == 0),
                    "precpt": float(day_idx % 4),
                    "avg_temperature": 60 + day_idx % 8,
                    "avg_humidity": 40 + day_idx % 10,
                    "avg_wind_level": 5 + day_idx % 3,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def reconstructed_daily_df(synthetic_daily_df: pd.DataFrame) -> pd.DataFrame:
    reconstructed, _ = reconstruct_latent_demand(synthetic_daily_df, LatentDemandConfig(model_min_training_rows=64))
    return reconstructed


@pytest.fixture()
def hourly_feature_df(reconstructed_daily_df: pd.DataFrame) -> pd.DataFrame:
    return build_hourly_features(reconstructed_daily_df, FeatureConfig(lag_periods=(1, 24), rolling_windows=(6, 24)))


@pytest.fixture()
def split_hourly_df(hourly_feature_df: pd.DataFrame) -> pd.DataFrame:
    processed, _ = split_and_scale_features(hourly_feature_df, FeatureConfig(lag_periods=(1, 24), rolling_windows=(6, 24)))
    return processed

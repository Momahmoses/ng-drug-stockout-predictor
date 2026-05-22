import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import pytest

from src.features.feature_engineering import (
    add_lag_features, add_rolling_features,
    add_seasonality_features, create_target, FEATURE_COLS
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 120
    return pd.DataFrame({
        "facility_id": ["FAC-001"] * 60 + ["FAC-002"] * 60,
        "drug_code": ["ACT-AL-24"] * 120,
        "period": pd.date_range("2022-01-01", periods=60, freq="MS").tolist() * 2,
        "month": list(range(1, 13)) * 10,
        "quantity_consumed": np.random.randint(50, 200, n),
        "closing_stock": np.random.randint(100, 500, n),
        "opening_stock": np.random.randint(200, 600, n),
        "quantity_received": np.random.randint(0, 300, n),
        "stockout_flag": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "true_demand": np.random.randint(60, 210, n),
        "lead_time_days": [21] * n,
        "population_catchment": [5000] * n,
        "malaria_burden": [0.85] * n,
        "drug_tier": [1] * n,
    })


def test_lag_features_created(sample_df):
    result = add_lag_features(sample_df, lags=[1, 3])
    assert "consumption_lag_1m" in result.columns
    assert "consumption_lag_3m" in result.columns
    assert "stock_lag_1m" in result.columns


def test_rolling_features_created(sample_df):
    df = add_lag_features(sample_df)
    result = add_rolling_features(df)
    assert "consumption_roll_mean_3m" in result.columns
    assert "consumption_roll_mean_6m" in result.columns
    assert "consumption_trend_3m" in result.columns


def test_seasonality_features(sample_df):
    result = add_seasonality_features(sample_df)
    assert "malaria_season_index" in result.columns
    assert "month_sin" in result.columns
    assert result["month_sin"].between(-1, 1).all()
    assert result["month_cos"].between(-1, 1).all()


def test_target_creation(sample_df):
    df = add_lag_features(sample_df)
    result = create_target(df, horizon=3)
    assert "target_stockout_next_45d" in result.columns
    assert result["target_stockout_next_45d"].isin([0, 1]).all()


def test_no_data_leakage(sample_df):
    df = add_lag_features(sample_df, lags=[1])
    first_record = df.sort_values(["facility_id", "period"]).groupby("facility_id").first()
    assert first_record["consumption_lag_1m"].isna().all(), \
        "First record for each facility should have NaN lag (no leakage)"

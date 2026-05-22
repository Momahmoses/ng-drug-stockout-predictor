"""
Feature engineering pipeline for drug stock-out prediction.
Generates lag features, rolling statistics, seasonality encodings,
and supply chain stress indicators from DHIS2 consumption history.
"""

import pandas as pd
import numpy as np
from pathlib import Path


MALARIA_SEASON = {1:0.6,2:0.65,3:0.85,4:1.3,5:1.5,6:1.4,
                  7:1.1,8:1.0,9:1.35,10:1.4,11:1.1,12:0.7}


def load_data(processed_dir: str = "data/processed") -> pd.DataFrame:
    p = Path(processed_dir)
    df = pd.read_csv(p / "consumption_history.csv", parse_dates=["period"])
    facilities = pd.read_csv(p / "facilities.csv")
    disease_cal = pd.read_csv(p / "disease_calendar.csv")
    lead_times = pd.read_csv(p / "lead_times.csv")

    df = df.merge(facilities[["facility_id", "facility_type"]], on="facility_id", how="left")
    df = df.merge(disease_cal, on="month", how="left")
    df = df.merge(
        lead_times[["state", "drug_code", "avg_lead_time_days", "lead_time_std_days"]],
        on=["state", "drug_code"], how="left"
    )
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3, 6]) -> pd.DataFrame:
    df = df.sort_values(["facility_id", "drug_code", "period"])
    group = ["facility_id", "drug_code"]

    for lag in lags:
        df[f"consumption_lag_{lag}m"] = (
            df.groupby(group)["quantity_consumed"].shift(lag)
        )
        df[f"stock_lag_{lag}m"] = (
            df.groupby(group)["closing_stock"].shift(lag)
        )
        df[f"stockout_lag_{lag}m"] = (
            df.groupby(group)["stockout_flag"].shift(lag)
        )

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    group = ["facility_id", "drug_code"]
    df = df.sort_values(["facility_id", "drug_code", "period"])

    for window in [3, 6, 12]:
        grp = df.groupby(group)["quantity_consumed"]
        df[f"consumption_roll_mean_{window}m"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )
        df[f"consumption_roll_std_{window}m"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).std()
        )

    df["consumption_trend_3m"] = (
        df["consumption_roll_mean_3m"] / df["consumption_roll_mean_6m"].replace(0, np.nan)
    ).clip(0.5, 2.5)

    grp_stock = df.groupby(group)["closing_stock"]
    df["stock_roll_min_3m"] = grp_stock.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).min()
    )
    df["stockout_freq_6m"] = df.groupby(group)["stockout_flag"].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean()
    )

    return df


def add_supply_chain_features(df: pd.DataFrame) -> pd.DataFrame:
    df["days_of_stock"] = np.where(
        df["consumption_roll_mean_3m"] > 0,
        df["closing_stock"] / (df["consumption_roll_mean_3m"] / 30),
        999
    ).clip(0, 365)

    df["stock_coverage_ratio"] = (
        df["days_of_stock"] / df["avg_lead_time_days"].fillna(21)
    ).clip(0, 10)

    df["delivery_shortfall"] = np.where(
        df["quantity_received"] > 0,
        (df["true_demand"] - df["quantity_received"]) / df["true_demand"].replace(0, 1),
        0
    ).clip(-2, 2)

    df["reorder_point"] = (
        df["consumption_roll_mean_3m"] / 30
        * df["avg_lead_time_days"].fillna(21)
        * 1.25  # 25% safety stock
    )
    df["below_reorder_point"] = (df["closing_stock"] < df["reorder_point"]).astype(int)

    return df


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    df["malaria_season_index"] = df["month"].map(MALARIA_SEASON)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_peak_malaria"] = df["month"].isin([4, 5, 6, 9, 10]).astype(int)
    df["is_dry_season"] = df["month"].isin([11, 12, 1, 2, 3]).astype(int)
    df["quarter"] = df["month"].apply(lambda m: (m - 1) // 3 + 1)
    return df


def create_target(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """
    Binary target: will a stockout occur within the next `horizon` months?
    Uses forward-looking window — this is the prediction target.
    """
    group = ["facility_id", "drug_code"]
    df = df.sort_values(["facility_id", "drug_code", "period"])
    df["target_stockout_next_45d"] = df.groupby(group)["stockout_flag"].transform(
        lambda x: x.shift(-1).rolling(horizon, min_periods=1).max()
    ).fillna(0).astype(int)
    return df


def build_feature_matrix(processed_dir: str = "data/processed") -> pd.DataFrame:
    df = load_data(processed_dir)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_supply_chain_features(df)
    df = add_seasonality_features(df)
    df = create_target(df)
    df = df.dropna(subset=["consumption_lag_3m", "consumption_roll_mean_6m"])
    return df


FEATURE_COLS = [
    "consumption_lag_1m", "consumption_lag_2m", "consumption_lag_3m", "consumption_lag_6m",
    "stock_lag_1m", "stock_lag_3m",
    "stockout_lag_1m", "stockout_lag_3m",
    "consumption_roll_mean_3m", "consumption_roll_mean_6m", "consumption_roll_mean_12m",
    "consumption_roll_std_3m", "consumption_roll_std_6m",
    "consumption_trend_3m",
    "stock_roll_min_3m", "stockout_freq_6m",
    "days_of_stock", "stock_coverage_ratio", "delivery_shortfall",
    "below_reorder_point",
    "malaria_season_index", "month_sin", "month_cos",
    "is_peak_malaria", "is_dry_season",
    "malaria_incidence_index", "diarrhea_index",
    "avg_lead_time_days", "lead_time_std_days",
    "population_catchment", "malaria_burden",
    "drug_tier",
]

TARGET_COL = "target_stockout_next_45d"


if __name__ == "__main__":
    print("Building feature matrix...")
    features = build_feature_matrix()
    out = Path("data/processed")
    features.to_csv(out / "feature_matrix.csv", index=False)
    pos_rate = features[TARGET_COL].mean() * 100
    print(f"  Feature matrix: {features.shape}")
    print(f"  Positive rate (stockouts): {pos_rate:.1f}%")
    print(f"  Features: {len(FEATURE_COLS)}")
    print("  Saved to data/processed/feature_matrix.csv")

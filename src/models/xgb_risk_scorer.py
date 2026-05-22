"""
XGBoost stock-out risk classifier.
Predicts probability of drug stockout within next 45 days per PHC per drug line.
Uses SHAP for explainability — every prediction is accompanied by top contributing factors.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from src.features.feature_engineering import (FEATURE_COLS, TARGET_COL,
                                               build_feature_matrix)


MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["state", "lga", "facility_type", "drug_code"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    return df


def train(processed_dir: str = "data/processed") -> dict:
    print("Loading feature matrix...")
    df = build_feature_matrix(processed_dir)
    df = encode_categoricals(df)

    extended_features = FEATURE_COLS + [
        c for c in ["state_enc", "facility_type_enc", "drug_code_enc"] if c in df.columns
    ]

    X = df[extended_features].fillna(0)
    y = df[TARGET_COL]
    groups = df["facility_id"]

    pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"  Class balance — positive rate: {y.mean()*100:.1f}%, pos_weight: {pos_weight:.2f}")

    params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": pos_weight,
        "random_state": 42,
        "eval_metric": "auc",
        "use_label_encoder": False,
    }

    cv = StratifiedGroupKFold(n_splits=5)
    fold_metrics = []

    print("Running 5-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.45).astype(int)

        fold_metrics.append({
            "fold": fold + 1,
            "auc": roc_auc_score(y_val, y_prob),
            "f1": f1_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
        })
        print(f"  Fold {fold+1}: AUC={fold_metrics[-1]['auc']:.3f} "
              f"F1={fold_metrics[-1]['f1']:.3f} "
              f"Precision={fold_metrics[-1]['precision']:.3f} "
              f"Recall={fold_metrics[-1]['recall']:.3f}")

    print("\nTraining final model on full dataset...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)

    calibrated = CalibratedClassifierCV(final_model, method="sigmoid", cv="prefit")
    calibrated.fit(X, y)

    joblib.dump(calibrated, MODEL_DIR / "xgb_stockout_model.pkl")
    joblib.dump(extended_features, MODEL_DIR / "feature_names.pkl")

    print("Computing SHAP values on 2000 sample subset...")
    sample = X.sample(min(2000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(sample)
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=extended_features
    ).sort_values(ascending=False)

    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")

    avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics]))
                   for k in ["auc", "f1", "precision", "recall"]}

    print("\n=== Cross-Validation Results ===")
    for k, v in avg_metrics.items():
        print(f"  Mean {k.upper()}: {v:.3f}")

    print("\n=== Top 10 Most Important Features ===")
    for feat, imp in mean_shap.head(10).items():
        print(f"  {feat:<40} {imp:.4f}")

    results = {
        "model": "XGBoost + Sigmoid Calibration",
        "cv_metrics": avg_metrics,
        "fold_details": fold_metrics,
        "top_features": mean_shap.head(15).to_dict(),
        "training_samples": len(X),
        "positive_rate": float(y.mean()),
    }
    with open(MODEL_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel saved to {MODEL_DIR}/xgb_stockout_model.pkl")
    return results


def predict_single(features: dict) -> dict:
    model = joblib.load(MODEL_DIR / "xgb_stockout_model.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")

    X = pd.DataFrame([features])[feature_names].fillna(0)
    prob = model.predict_proba(X)[0, 1]
    risk_score = round(prob * 100, 1)

    raw_model = model.estimator if hasattr(model, 'estimator') else model
    shap_vals = explainer.shap_values(X)[0]
    top_factors = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]), reverse=True
    )[:5]

    days_of_stock = features.get("days_of_stock", 45)
    predicted_stockout_days = max(0, int(days_of_stock * (1 - prob * 0.6)))

    return {
        "risk_score": risk_score,
        "risk_level": "HIGH" if risk_score >= 70 else "MEDIUM" if risk_score >= 40 else "LOW",
        "stockout_probability": round(float(prob), 4),
        "days_until_predicted_stockout": predicted_stockout_days,
        "top_risk_factors": [
            {"factor": feat, "contribution": round(float(val), 4)}
            for feat, val in top_factors
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    if args.train:
        train(args.processed_dir)
    else:
        print("Pass --train to train the model.")

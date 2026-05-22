"""
FastAPI application for the NG Drug Stock-out Predictor.
Exposes endpoints for single-facility risk scoring, batch scoring,
and reorder recommendations — integrates with DHIS2 and NPHCDA systems.
"""

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (BatchRequest, HealthResponse, RiskFactor,
                              StockoutRequest, StockoutResponse)
from src.models.reorder_optimizer import compute_reorder_plan

START_TIME = time.time()
MODEL_VERSION = "1.0.0"

app = FastAPI(
    title="NG Drug Stock-out Predictor API",
    description="Predicts essential medicine stock-out risk 45 days ahead for Nigerian PHCs.",
    version=MODEL_VERSION,
    contact={
        "name": "MOMAH MOSES .C.",
        "url": "https://github.com/Momahmoses",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_feature_names = None

DRUG_UNIT_COST = {
    "ACT-AL-24": 1200, "OXY-10IU": 850, "MgSO4-50PCT": 1500,
    "AMOX-500MG": 45, "ARV-TDF3TC": 3500, "RDT-MALARIA": 350,
}

MALARIA_SEASON = {
    1:0.6, 2:0.65, 3:0.85, 4:1.3, 5:1.5, 6:1.4,
    7:1.1, 8:1.0, 9:1.35, 10:1.4, 11:1.1, 12:0.7
}


def _load_model():
    global _model, _feature_names
    model_path = Path("models/saved/xgb_stockout_model.pkl")
    if model_path.exists():
        import joblib
        _model = joblib.load(model_path)
        _feature_names = joblib.load("models/saved/feature_names.pkl")
    return _model is not None


def _build_feature_vector(req: StockoutRequest) -> dict:
    today = date.today()
    month = today.month
    avg_daily = req.avg_3m_consumption / 30
    days_of_stock = req.current_stock_units / avg_daily if avg_daily > 0 else 999

    return {
        "consumption_lag_1m": req.last_month_consumption,
        "consumption_lag_2m": req.avg_3m_consumption,
        "consumption_lag_3m": req.avg_3m_consumption,
        "consumption_lag_6m": req.avg_3m_consumption * 0.9,
        "stock_lag_1m": req.current_stock_units,
        "stock_lag_3m": req.current_stock_units * 1.1,
        "stockout_lag_1m": 0,
        "stockout_lag_3m": 0,
        "consumption_roll_mean_3m": req.avg_3m_consumption,
        "consumption_roll_mean_6m": req.avg_3m_consumption * 0.95,
        "consumption_roll_mean_12m": req.avg_3m_consumption * 0.9,
        "consumption_roll_std_3m": req.avg_3m_consumption * 0.15,
        "consumption_roll_std_6m": req.avg_3m_consumption * 0.18,
        "consumption_trend_3m": req.last_month_consumption / max(1, req.avg_3m_consumption),
        "stock_roll_min_3m": req.current_stock_units * 0.8,
        "stockout_freq_6m": 0.0,
        "days_of_stock": min(days_of_stock, 365),
        "stock_coverage_ratio": days_of_stock / req.lead_time_days,
        "delivery_shortfall": 0.0,
        "below_reorder_point": 1 if req.current_stock_units < avg_daily * req.lead_time_days * 1.25 else 0,
        "malaria_season_index": MALARIA_SEASON.get(month, 1.0),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "is_peak_malaria": 1 if month in [4, 5, 6, 9, 10] else 0,
        "is_dry_season": 1 if month in [11, 12, 1, 2, 3] else 0,
        "malaria_incidence_index": MALARIA_SEASON.get(month, 1.0),
        "diarrhea_index": 0.85,
        "avg_lead_time_days": req.lead_time_days,
        "lead_time_std_days": 5.0,
        "population_catchment": req.population_catchment or 4500,
        "malaria_burden": req.malaria_burden_index or 0.85,
        "drug_tier": 1 if req.drug_code in ["ACT-AL-24", "OXY-10IU", "MgSO4-50PCT", "ARV-TDF3TC"] else 2,
        "state_enc": 0,
        "facility_type_enc": 0,
        "drug_code_enc": 0,
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if _load_model() else "no_model_loaded",
        model_version=MODEL_VERSION,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.post("/predict/stockout-risk", response_model=StockoutResponse)
def predict_stockout_risk(req: StockoutRequest):
    model_loaded = _load_model()

    features = _build_feature_vector(req)
    avg_daily = req.avg_3m_consumption / 30

    if model_loaded and _feature_names:
        try:
            X = pd.DataFrame([features])[[c for c in _feature_names if c in features]]
            prob = float(_model.predict_proba(X)[0, 1])
        except Exception:
            prob = _heuristic_risk(features)
    else:
        prob = _heuristic_risk(features)

    risk_score = round(prob * 100, 1)
    risk_level = (
        "CRITICAL" if risk_score >= 85 else
        "HIGH" if risk_score >= 70 else
        "MEDIUM" if risk_score >= 40 else "LOW"
    )

    days_of_stock = features["days_of_stock"]
    days_until_stockout = max(0, int(days_of_stock * (1 - prob * 0.5)))
    reorder_date = date.today() + timedelta(days=max(0, days_until_stockout - req.lead_time_days - 7))

    plan = compute_reorder_plan(
        drug_code=req.drug_code,
        facility_id=req.facility_id,
        avg_daily_demand=max(0.01, avg_daily),
        demand_std_daily=max(0.01, avg_daily * 0.2),
        current_stock=req.current_stock_units,
        lead_time_days=req.lead_time_days,
    )

    top_factors = [
        RiskFactor(factor="days_of_stock_remaining", contribution=round(-0.35 * (days_of_stock / 60 - 1), 3), direction="increases_risk" if days_of_stock < 30 else "decreases_risk"),
        RiskFactor(factor="seasonal_demand_index", contribution=round(features["malaria_season_index"] * 0.22, 3), direction="increases_risk" if features["is_peak_malaria"] else "decreases_risk"),
        RiskFactor(factor="stock_coverage_ratio", contribution=round(-0.18 * features["stock_coverage_ratio"], 3), direction="increases_risk" if features["stock_coverage_ratio"] < 1.5 else "decreases_risk"),
        RiskFactor(factor="consumption_trend", contribution=round((features["consumption_trend_3m"] - 1) * 0.15, 3), direction="increases_risk" if features["consumption_trend_3m"] > 1.1 else "decreases_risk"),
        RiskFactor(factor="lead_time_days", contribution=round(req.lead_time_days / 100 * 0.12, 3), direction="increases_risk" if req.lead_time_days > 28 else "decreases_risk"),
    ]

    alert_msgs = {
        "CRITICAL": f"URGENT: {req.drug_code} at {req.facility_id} will stock out in ~{days_until_stockout} days. Immediate procurement required.",
        "HIGH": f"HIGH RISK: {req.drug_code} at {req.facility_id} — stock-out likely within 45 days. Reorder {plan.reorder_quantity_units} units by {reorder_date}.",
        "MEDIUM": f"MONITOR: {req.drug_code} at {req.facility_id} — moderate stock-out risk. Consider reorder within 3 weeks.",
        "LOW": f"OK: {req.drug_code} at {req.facility_id} — sufficient stock. Next review in 30 days.",
    }

    return StockoutResponse(
        facility_id=req.facility_id,
        drug_code=req.drug_code,
        risk_score=risk_score,
        risk_level=risk_level,
        stockout_probability=round(prob, 4),
        days_until_predicted_stockout=days_until_stockout,
        recommended_reorder_units=plan.reorder_quantity_units,
        reorder_trigger_date=reorder_date if risk_level in ["HIGH", "CRITICAL"] else None,
        estimated_reorder_cost_naira=plan.estimated_cost_naira,
        top_risk_factors=top_factors,
        alert_message=alert_msgs[risk_level],
    )


def _heuristic_risk(features: dict) -> float:
    score = 0.0
    dos = features.get("days_of_stock", 60)
    if dos < 14: score += 0.55
    elif dos < 30: score += 0.35
    elif dos < 45: score += 0.15

    if features.get("is_peak_malaria"): score += 0.15
    if features.get("consumption_trend_3m", 1.0) > 1.2: score += 0.12
    if features.get("stock_coverage_ratio", 2.0) < 1.0: score += 0.18
    return min(0.95, score)


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

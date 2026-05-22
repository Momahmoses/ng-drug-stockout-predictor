from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class StockoutRequest(BaseModel):
    facility_id: str = Field(..., example="NG-KAN-0042")
    drug_code: str = Field(..., example="ACT-AL-24")
    state: str = Field(..., example="Kano")
    lga: str = Field(..., example="Kano Municipal")
    current_stock_units: int = Field(..., ge=0, example=120)
    last_delivery_date: date = Field(..., example="2024-11-15")
    last_month_consumption: int = Field(..., ge=0, example=95)
    avg_3m_consumption: float = Field(..., ge=0, example=88.5)
    lead_time_days: int = Field(default=21, ge=1, le=90)
    population_catchment: Optional[int] = Field(default=4500)
    malaria_burden_index: Optional[float] = Field(default=0.85)


class RiskFactor(BaseModel):
    factor: str
    contribution: float
    direction: str  # "increases_risk" or "decreases_risk"


class StockoutResponse(BaseModel):
    facility_id: str
    drug_code: str
    risk_score: float = Field(..., description="0-100 risk score")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    stockout_probability: float
    days_until_predicted_stockout: int
    recommended_reorder_units: int
    reorder_trigger_date: Optional[date]
    estimated_reorder_cost_naira: float
    top_risk_factors: List[RiskFactor]
    alert_message: str


class BatchRequest(BaseModel):
    facility_ids: List[str]
    drug_codes: Optional[List[str]] = None
    state: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_seconds: float

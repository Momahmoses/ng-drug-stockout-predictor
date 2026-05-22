"""
EOQ-based reorder quantity optimizer.
Computes Economic Order Quantity, reorder point, and safety stock
for each drug-facility pair, factoring in demand uncertainty and lead time.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ReorderPlan:
    drug_code: str
    facility_id: str
    reorder_point_units: int
    reorder_quantity_units: int
    safety_stock_units: int
    days_until_reorder: int
    urgency: str
    estimated_cost_naira: float


DRUG_UNIT_COST_NAIRA = {
    "ACT-AL-24":    1200,
    "OXY-10IU":     850,
    "MgSO4-50PCT":  1500,
    "MISO-200MCG":  600,
    "AMOX-500MG":   45,
    "COTRIM-480MG": 35,
    "METRO-400MG":  30,
    "ORS-SACHET":   25,
    "ZINC-20MG":    40,
    "RDT-MALARIA":  350,
    "ARV-TDF3TC":   3500,
    "RDT-HIV":      450,
    "DEPO-PROVEN":  1100,
    "COTRIM-PEDS":  38,
    "VIT-A-200K":   120,
}

ORDERING_COST_NAIRA = 15000
HOLDING_COST_RATE = 0.25
SERVICE_LEVEL_Z = 1.65  # 95% service level


def compute_reorder_plan(
    drug_code: str,
    facility_id: str,
    avg_daily_demand: float,
    demand_std_daily: float,
    current_stock: int,
    lead_time_days: int,
    lead_time_std_days: float = 5.0,
    annual_demand: float = None,
) -> ReorderPlan:
    if annual_demand is None:
        annual_demand = avg_daily_demand * 365

    unit_cost = DRUG_UNIT_COST_NAIRA.get(drug_code, 500)
    holding_cost = unit_cost * HOLDING_COST_RATE

    eoq = max(
        1,
        int(np.sqrt((2 * annual_demand * ORDERING_COST_NAIRA) / holding_cost))
    )

    demand_during_lt_std = np.sqrt(
        lead_time_days * demand_std_daily**2
        + avg_daily_demand**2 * lead_time_std_days**2
    )
    safety_stock = max(0, int(SERVICE_LEVEL_Z * demand_during_lt_std))

    reorder_point = int(avg_daily_demand * lead_time_days + safety_stock)
    days_until_reorder = max(0, int(
        (current_stock - reorder_point) / avg_daily_demand
        if avg_daily_demand > 0 else 999
    ))

    if days_until_reorder <= 7:
        urgency = "URGENT"
    elif days_until_reorder <= 21:
        urgency = "HIGH"
    elif days_until_reorder <= 45:
        urgency = "MEDIUM"
    else:
        urgency = "LOW"

    estimated_cost = eoq * unit_cost

    return ReorderPlan(
        drug_code=drug_code,
        facility_id=facility_id,
        reorder_point_units=reorder_point,
        reorder_quantity_units=eoq,
        safety_stock_units=safety_stock,
        days_until_reorder=days_until_reorder,
        urgency=urgency,
        estimated_cost_naira=estimated_cost,
    )


def batch_reorder_plans(consumption_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    latest = (
        consumption_df.sort_values("period")
        .groupby(["facility_id", "drug_code"])
        .tail(6)
    )

    for (fac_id, drug_code), grp in latest.groupby(["facility_id", "drug_code"]):
        avg_daily = grp["quantity_consumed"].mean() / 30
        std_daily = grp["quantity_consumed"].std() / 30
        current_stock = grp.iloc[-1]["closing_stock"]
        lead_time = int(grp["lead_time_days"].mean())

        plan = compute_reorder_plan(
            drug_code=drug_code,
            facility_id=fac_id,
            avg_daily_demand=max(0.01, avg_daily),
            demand_std_daily=max(0.01, std_daily if not np.isnan(std_daily) else avg_daily * 0.2),
            current_stock=int(current_stock),
            lead_time_days=lead_time,
        )

        results.append({
            "facility_id": plan.facility_id,
            "drug_code": plan.drug_code,
            "reorder_point_units": plan.reorder_point_units,
            "reorder_quantity_units": plan.reorder_quantity_units,
            "safety_stock_units": plan.safety_stock_units,
            "days_until_reorder": plan.days_until_reorder,
            "urgency": plan.urgency,
            "estimated_cost_naira": plan.estimated_cost_naira,
        })

    return pd.DataFrame(results)

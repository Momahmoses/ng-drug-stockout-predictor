"""
Generates realistic synthetic DHIS2-format drug consumption data
for 500 Nigerian PHCs across 10 states, 36 months, 20 drug lines.
Mirrors actual HMIS consumption patterns including:
  - Malaria season peaks (April-June, September-November)
  - Supply chain disruptions (random delivery delays)
  - Stockout events (ground truth labels)
  - Population-scaled demand variation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

np.random.seed(42)

STATES = {
    "Kano":    {"pop_factor": 1.8, "malaria_burden": 0.9, "facilities": 60},
    "Lagos":   {"pop_factor": 2.1, "malaria_burden": 0.7, "facilities": 70},
    "Oyo":     {"pop_factor": 1.4, "malaria_burden": 0.8, "facilities": 50},
    "Rivers":  {"pop_factor": 1.2, "malaria_burden": 1.0, "facilities": 45},
    "Kaduna":  {"pop_factor": 1.3, "malaria_burden": 0.85, "facilities": 48},
    "Katsina": {"pop_factor": 1.1, "malaria_burden": 0.75, "facilities": 40},
    "Ogun":    {"pop_factor": 1.0, "malaria_burden": 0.65, "facilities": 38},
    "Benue":   {"pop_factor": 0.9, "malaria_burden": 0.95, "facilities": 42},
    "Plateau": {"pop_factor": 0.85, "malaria_burden": 0.7, "facilities": 36},
    "Niger":   {"pop_factor": 0.8, "malaria_burden": 0.8, "facilities": 35},
}

DRUGS = {
    "ACT-AL-24":   {"base_monthly": 180, "seasonal": "malaria", "tier": 1},
    "OXY-10IU":    {"base_monthly": 45,  "seasonal": "flat",    "tier": 1},
    "MgSO4-50PCT": {"base_monthly": 20,  "seasonal": "flat",    "tier": 1},
    "MISO-200MCG": {"base_monthly": 38,  "seasonal": "flat",    "tier": 1},
    "AMOX-500MG":  {"base_monthly": 320, "seasonal": "mild",    "tier": 2},
    "COTRIM-480MG":{"base_monthly": 200, "seasonal": "mild",    "tier": 2},
    "METRO-400MG": {"base_monthly": 150, "seasonal": "flat",    "tier": 2},
    "ORS-SACHET":  {"base_monthly": 280, "seasonal": "malaria", "tier": 2},
    "ZINC-20MG":   {"base_monthly": 140, "seasonal": "malaria", "tier": 2},
    "RDT-MALARIA": {"base_monthly": 220, "seasonal": "malaria", "tier": 3},
    "ARV-TDF3TC":  {"base_monthly": 55,  "seasonal": "flat",    "tier": 1},
    "RDT-HIV":     {"base_monthly": 65,  "seasonal": "flat",    "tier": 3},
    "DEPO-PROVEN": {"base_monthly": 80,  "seasonal": "flat",    "tier": 2},
    "COTRIM-PEDS": {"base_monthly": 95,  "seasonal": "mild",    "tier": 2},
    "VIT-A-200K":  {"base_monthly": 110, "seasonal": "mild",    "tier": 2},
}

MALARIA_SEASONALITY = {
    1: 0.6, 2: 0.65, 3: 0.85, 4: 1.3, 5: 1.5,  6: 1.4,
    7: 1.1, 8: 1.0,  9: 1.35, 10: 1.4, 11: 1.1, 12: 0.7
}


def generate_facilities(states: dict) -> pd.DataFrame:
    records = []
    fac_id = 1
    for state, props in states.items():
        for i in range(props["facilities"]):
            lga = f"{state}-LGA-{(i % 8) + 1:02d}"
            records.append({
                "facility_id": f"NG-{state[:3].upper()}-{fac_id:04d}",
                "facility_name": f"{lga} PHC {chr(65 + (i % 26))}",
                "state": state,
                "lga": lga,
                "facility_type": np.random.choice(
                    ["PHC", "Comprehensive PHC", "Health Post"],
                    p=[0.5, 0.3, 0.2]
                ),
                "population_catchment": int(
                    np.random.normal(4500, 1200) * props["pop_factor"]
                ),
                "malaria_burden_index": props["malaria_burden"],
            })
            fac_id += 1
    return pd.DataFrame(records)


def seasonal_factor(month: int, pattern: str) -> float:
    if pattern == "malaria":
        return MALARIA_SEASONALITY[month]
    elif pattern == "mild":
        return 1.0 + 0.15 * MALARIA_SEASONALITY[month] - 0.15
    return 1.0


def generate_consumption(
    facilities: pd.DataFrame,
    start: str = "2022-01-01",
    months: int = 36
) -> pd.DataFrame:
    date_range = pd.date_range(start=start, periods=months, freq="MS")
    records = []

    for _, fac in facilities.iterrows():
        pop_scale = fac["population_catchment"] / 4500
        burden = fac["malaria_burden_index"]

        for drug_code, drug_props in DRUGS.items():
            stock = int(drug_props["base_monthly"] * pop_scale * 3.5)
            lead_time = np.random.randint(14, 35)
            stockout_count = 0

            for dt in date_range:
                month = dt.month
                sf = seasonal_factor(month, drug_props["seasonal"])
                if drug_props["seasonal"] == "malaria":
                    sf *= burden

                true_demand = max(1, int(
                    drug_props["base_monthly"] * pop_scale * sf
                    * np.random.lognormal(0, 0.12)
                ))

                actual_consumption = min(true_demand, stock)
                stockout_this_month = 1 if stock < true_demand * 0.85 else 0
                if stockout_this_month:
                    stockout_count += 1

                delivery = 0
                if stock < drug_props["base_monthly"] * pop_scale * 1.5:
                    if np.random.random() < 0.72:
                        delivery = int(
                            drug_props["base_monthly"] * pop_scale * 3.0
                            * np.random.uniform(0.85, 1.15)
                        )
                        delivery_delay = np.random.choice(
                            [0, lead_time, lead_time * 2],
                            p=[0.55, 0.30, 0.15]
                        )
                    else:
                        delivery = 0
                        delivery_delay = 0
                else:
                    delivery_delay = 0

                records.append({
                    "facility_id": fac["facility_id"],
                    "state": fac["state"],
                    "lga": fac["lga"],
                    "drug_code": drug_code,
                    "drug_tier": drug_props["tier"],
                    "period": dt.strftime("%Y-%m"),
                    "month": month,
                    "year": dt.year,
                    "opening_stock": stock,
                    "quantity_received": delivery,
                    "quantity_consumed": actual_consumption,
                    "closing_stock": max(0, stock + delivery - actual_consumption),
                    "true_demand": true_demand,
                    "days_out_of_stock": int(stockout_this_month * np.random.randint(3, 25)),
                    "stockout_flag": stockout_this_month,
                    "lead_time_days": lead_time + delivery_delay,
                    "population_catchment": fac["population_catchment"],
                    "malaria_burden": fac["malaria_burden_index"],
                })

                stock = max(0, stock + delivery - actual_consumption)

    return pd.DataFrame(records)


def generate_disease_calendar() -> pd.DataFrame:
    months = range(1, 13)
    records = []
    for month in months:
        records.append({
            "month": month,
            "malaria_incidence_index": MALARIA_SEASONALITY[month],
            "diarrhea_index": 0.8 + 0.4 * np.sin((month - 4) * np.pi / 6),
            "pneumonia_index": 1.2 if month in [11, 12, 1, 2, 3] else 0.85,
            "meningitis_index": 2.5 if month in [1, 2, 3, 4] else 0.4,
            "cholera_index": 1.8 if month in [5, 6, 7, 8, 9] else 0.5,
        })
    return pd.DataFrame(records)


def generate_lead_times() -> pd.DataFrame:
    states = list(STATES.keys())
    records = []
    drugs = list(DRUGS.keys())
    for state in states:
        for drug in drugs:
            records.append({
                "state": state,
                "drug_code": drug,
                "avg_lead_time_days": np.random.randint(14, 42),
                "lead_time_std_days": np.random.randint(3, 12),
                "stockout_during_lead_time_pct": round(
                    np.random.uniform(0.05, 0.35), 3
                ),
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    print("Generating facilities...")
    facilities = generate_facilities(STATES)
    facilities.to_csv(out / "facilities.csv", index=False)
    print(f"  {len(facilities)} facilities created")

    print("Generating consumption history (36 months)...")
    consumption = generate_consumption(facilities, months=36)
    consumption.to_csv(out / "consumption_history.csv", index=False)
    total_stockouts = consumption["stockout_flag"].sum()
    stockout_rate = total_stockouts / len(consumption) * 100
    print(f"  {len(consumption):,} records | {total_stockouts:,} stockout events ({stockout_rate:.1f}%)")

    print("Generating disease calendar...")
    disease_cal = generate_disease_calendar()
    disease_cal.to_csv(out / "disease_calendar.csv", index=False)

    print("Generating lead time data...")
    lead_times = generate_lead_times()
    lead_times.to_csv(out / "lead_times.csv", index=False)

    summary = {
        "facilities": len(facilities),
        "states": len(STATES),
        "drugs": len(DRUGS),
        "total_records": len(consumption),
        "stockout_events": int(total_stockouts),
        "stockout_rate_pct": round(stockout_rate, 2),
        "date_range": "2022-01 to 2024-12",
    }
    with open(out / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Files written to data/processed/")
    print(json.dumps(summary, indent=2))

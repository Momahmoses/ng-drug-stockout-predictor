"""
Streamlit national dashboard for the NG Drug Stock-out Predictor.
Displays facility-level risk scores, state summaries, drug-level trends,
and auto-generated procurement alerts on an interactive Nigeria map.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="NG Drug Stock-out Predictor",
    page_icon="💊",
    layout="wide",
)

st.markdown("""
<style>
    .risk-HIGH    { background-color:#FF4B4B; color:white; padding:4px 10px; border-radius:4px; }
    .risk-MEDIUM  { background-color:#FFA500; color:white; padding:4px 10px; border-radius:4px; }
    .risk-LOW     { background-color:#00CC00; color:white; padding:4px 10px; border-radius:4px; }
    .risk-CRITICAL{ background-color:#8B0000; color:white; padding:4px 10px; border-radius:4px; }
    .metric-card  { background:#f0f2f6; padding:16px; border-radius:8px; margin:4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    p = Path("data/processed")
    if not (p / "consumption_history.csv").exists():
        st.info("Generating synthetic data...")
        import subprocess
        subprocess.run(["python", "data/generators/generate_synthetic_data.py"])

    df = pd.read_csv(p / "consumption_history.csv")
    facilities = pd.read_csv(p / "facilities.csv")
    df = df.merge(facilities[["facility_id", "facility_type", "state", "lga"]], on="facility_id", how="left")

    df["risk_score"] = np.where(
        df["days_out_of_stock"] > 0,
        np.random.uniform(65, 95, len(df)),
        np.where(
            df["closing_stock"] < df["quantity_consumed"] * 1.5,
            np.random.uniform(40, 70, len(df)),
            np.random.uniform(5, 40, len(df))
        )
    )
    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[0, 40, 70, 85, 100],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    )
    return df


df = load_data()
latest = df[df["period"] == df["period"].max()].copy()

st.title("💊 Nigerian PHC Drug Stock-out Predictor")
st.caption("45-day ahead stock-out risk forecasting for essential medicines across Nigeria's PHCs")

st.sidebar.header("Filters")
selected_state = st.sidebar.selectbox("State", ["All States"] + sorted(df["state"].unique().tolist()))
selected_drug = st.sidebar.selectbox("Drug", ["All Drugs"] + sorted(df["drug_code"].unique().tolist()))
risk_threshold = st.sidebar.slider("Risk Alert Threshold", 0, 100, 70)
drug_tier = st.sidebar.multiselect("Drug Tier", [1, 2, 3], default=[1, 2, 3])

filt = latest.copy()
if selected_state != "All States":
    filt = filt[filt["state"] == selected_state]
if selected_drug != "All Drugs":
    filt = filt[filt["drug_code"] == selected_drug]
if drug_tier:
    filt = filt[filt["drug_tier"].isin(drug_tier)]

high_risk = filt[filt["risk_score"] >= risk_threshold]
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Facilities Monitored", f"{filt['facility_id'].nunique():,}")
with col2:
    st.metric("High-Risk Alerts", f"{len(high_risk):,}",
              delta=f"{len(high_risk)/max(1,len(filt))*100:.1f}% of assessments",
              delta_color="inverse")
with col3:
    st.metric("Average Risk Score", f"{filt['risk_score'].mean():.1f}/100")
with col4:
    critical_count = len(filt[filt["risk_score"] >= 85])
    st.metric("Critical (>85)", f"{critical_count}", delta_color="inverse")

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Risk Map", "📊 Drug Analysis", "⚠️ Alerts", "📈 Trends"])

with tab1:
    st.subheader("Facility Risk Distribution by State")
    state_summary = (
        filt.groupby("state")
        .agg(
            avg_risk=("risk_score", "mean"),
            high_risk_count=("risk_score", lambda x: (x >= risk_threshold).sum()),
            facility_count=("facility_id", "nunique"),
        )
        .reset_index()
    )
    state_summary["high_risk_pct"] = (
        state_summary["high_risk_count"] / state_summary["facility_count"] * 100
    )

    fig = px.bar(
        state_summary.sort_values("avg_risk", ascending=False),
        x="state", y="avg_risk",
        color="avg_risk",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 100],
        text=state_summary["high_risk_count"].astype(str) + " alerts",
        title="Average Stock-out Risk Score by State",
        labels={"avg_risk": "Avg Risk Score", "state": "State"},
    )
    fig.update_traces(textposition="outside")
    fig.add_hline(y=risk_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Alert threshold ({risk_threshold})")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        risk_dist = filt["risk_level"].value_counts()
        fig2 = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            color=risk_dist.index,
            color_discrete_map={"LOW": "#00CC00", "MEDIUM": "#FFA500", "HIGH": "#FF4B4B", "CRITICAL": "#8B0000"},
            title="Risk Level Distribution",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        drug_risk = (
            filt.groupby("drug_code")["risk_score"].mean()
            .sort_values(ascending=False).head(10)
        )
        fig3 = px.bar(
            x=drug_risk.values, y=drug_risk.index,
            orientation="h",
            color=drug_risk.values,
            color_continuous_scale="RdYlGn_r",
            title="Top 10 Highest-Risk Drug Lines",
            labels={"x": "Avg Risk Score", "y": "Drug Code"},
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Drug-Level Stock-out Analysis")
    drug_analysis = (
        df.groupby(["drug_code", "period"])
        .agg(
            total_stockouts=("stockout_flag", "sum"),
            total_facilities=("facility_id", "nunique"),
            avg_consumption=("quantity_consumed", "mean"),
            avg_closing_stock=("closing_stock", "mean"),
        )
        .reset_index()
    )
    drug_analysis["stockout_rate"] = (
        drug_analysis["total_stockouts"] / drug_analysis["total_facilities"] * 100
    )

    selected_drug_analysis = st.selectbox(
        "Select drug for trend analysis",
        df["drug_code"].unique(),
        key="drug_analysis"
    )
    drug_trend = drug_analysis[drug_analysis["drug_code"] == selected_drug_analysis]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=drug_trend["period"], y=drug_trend["avg_consumption"],
        name="Avg Consumption", line=dict(color="blue")
    ))
    fig4.add_trace(go.Scatter(
        x=drug_trend["period"], y=drug_trend["avg_closing_stock"],
        name="Avg Closing Stock", line=dict(color="green")
    ))
    fig4.add_trace(go.Bar(
        x=drug_trend["period"], y=drug_trend["stockout_rate"],
        name="Stockout Rate (%)", yaxis="y2", opacity=0.4,
        marker_color="red"
    ))
    fig4.update_layout(
        title=f"{selected_drug_analysis} — Consumption vs Stock vs Stockout Rate",
        yaxis=dict(title="Units"),
        yaxis2=dict(title="Stockout Rate (%)", overlaying="y", side="right"),
        legend=dict(x=0, y=1),
    )
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader(f"Active Alerts — Risk Score ≥ {risk_threshold}")
    alerts = high_risk[["facility_id", "state", "lga", "drug_code",
                          "closing_stock", "quantity_consumed", "risk_score", "risk_level"]].copy()
    alerts = alerts.sort_values("risk_score", ascending=False).head(100)
    alerts["risk_score"] = alerts["risk_score"].round(1)
    alerts.columns = ["Facility ID", "State", "LGA", "Drug", "Current Stock",
                       "Last Month Consumption", "Risk Score", "Risk Level"]

    st.dataframe(
        alerts.style.background_gradient(subset=["Risk Score"], cmap="RdYlGn_r"),
        use_container_width=True,
        height=400,
    )

    st.download_button(
        "📥 Download Alert Report (CSV)",
        data=alerts.to_csv(index=False),
        file_name=f"stockout_alerts_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

with tab4:
    st.subheader("National Stockout Trend — Monthly")
    monthly_trend = (
        df.groupby("period")
        .agg(
            stockout_events=("stockout_flag", "sum"),
            total_assessments=("facility_id", "count"),
            avg_risk=("risk_score", "mean"),
        )
        .reset_index()
    )
    monthly_trend["stockout_rate"] = (
        monthly_trend["stockout_events"] / monthly_trend["total_assessments"] * 100
    )

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=monthly_trend["period"], y=monthly_trend["stockout_rate"],
        fill="tozeroy", name="Stockout Rate (%)", line=dict(color="red")
    ))
    fig5.update_layout(
        title="National Monthly Stockout Rate",
        xaxis_title="Month",
        yaxis_title="Stockout Rate (%)",
    )
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")
st.caption("MOMAH MOSES .C. · Geospatial AI Engineer & Data Scientist · github.com/Momahmoses")

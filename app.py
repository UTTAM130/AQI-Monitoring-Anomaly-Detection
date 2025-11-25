"""
app.py - Streamlit dashboard for AQI Monitoring & Anomaly Detection
Run: pipenv run streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests

OUTPUT_DIR = "./output"
PROCESSED = os.path.join(OUTPUT_DIR, "processed_with_flags.csv")
API_BASE = "http://127.0.0.1:8000"  # change if running elsewhere

st.set_page_config(layout="wide", page_title="AQI Dashboard")
st.title("AQI Monitoring & Anomaly Detection")

if not os.path.exists(PROCESSED):
    st.warning("Processed data with flags not found. Run aqi_models.py first.")
else:
    df = pd.read_csv(PROCESSED, parse_dates=["timestamp"]).set_index("timestamp")
    # Identify pollutants available
    pollutant_cols = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3"] if c in df.columns]
    if len(pollutant_cols) == 0:
        pollutant_cols = [c for c in df.select_dtypes('number').columns.tolist() if not c.startswith("anom")][:6]

    # Sidebar
    st.sidebar.header("Filters")
    col = st.sidebar.selectbox("Pollutant", pollutant_cols, index=0)
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    start = st.sidebar.date_input("Start", min_date)
    end = st.sidebar.date_input("End", max_date)
    view = df[(df.index.date >= start) & (df.index.date <= end)]

    # Top-level stats
    st.markdown(f"**Data range:** {df.index.min().date()}  →  {df.index.max().date()}")
    st.markdown(f"**Records in view:** {len(view)}  \n**Anomalies (any) in view:** {int(view.get('anom_any', False).sum() if 'anom_any' in view.columns else 0)}")

    # Time series with anomalies
    st.subheader(f"{col} over time")
    fig = px.line(view.reset_index(), x="timestamp", y=col, title=f"{col} time series")
    if "anom_any" in view.columns:
        anoms = view[view["anom_any"] == True]
        if not anoms.empty and col in anoms.columns:
            fig.add_scatter(x=anoms.index, y=anoms[col], mode="markers", marker=dict(color="red", size=7), name="Anomaly")
    st.plotly_chart(fig, use_container_width=True)

    # Anomalies table
    st.subheader("Anomalies")
    if "anom_any" in view.columns:
        anom_table = view[view["anom_any"] == True].sort_index(ascending=False)
        st.write(f"Total anomalies in view: {len(anom_table)}")
        st.dataframe(anom_table.head(300))
    else:
        st.write("No anomaly column 'anom_any' found in processed file.")

    # Download processed CSV view
    st.download_button("Download processed CSV (view)", data=view.reset_index().to_csv(index=False).encode('utf-8'), file_name="processed_view.csv")

    # API test panel
    st.sidebar.header("API: Predict single row")
    pm25 = st.sidebar.number_input("PM2.5", value=0.0, step=0.1)
    pm10 = st.sidebar.number_input("PM10", value=0.0, step=0.1)
    if st.sidebar.button("Call /predict"):
        payload = {"PM25": pm25, "PM10": pm10}
        try:
            r = requests.post(API_BASE + "/predict", json=payload, timeout=4)
            st.sidebar.write(r.json())
        except Exception as e:
            st.sidebar.error("Could not contact API: " + str(e))

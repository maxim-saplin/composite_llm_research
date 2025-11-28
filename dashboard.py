import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

st.set_page_config(page_title="Composite LLM Dashboard", layout="wide")

st.title("Composite LLM Observability")

LOG_FILE = "llm_logs.jsonl"

@st.cache_data(ttl=5)
def load_data():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    
    data = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(data)

df = load_data()

if df.empty:
    st.warning("No logs found yet. Run some LLM calls!")
else:
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Calls", len(df))
    with col2:
        st.metric("Total Tokens", df["total_tokens"].sum() if "total_tokens" in df.columns else 0)
    with col3:
        st.metric("Avg Latency (s)", f"{df['duration_seconds'].mean():.2f}")
    with col4:
        error_rate = (len(df[df['status'] == 'failure']) / len(df)) * 100
        st.metric("Error Rate", f"{error_rate:.1f}%")

    # Time Series
    st.subheader("Calls over Time")
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        calls_over_time = df.set_index('timestamp').resample('1min').size()
        st.line_chart(calls_over_time)

    # Model Breakdown
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Calls by Model")
        fig_model = px.bar(df['model'].value_counts())
        st.plotly_chart(fig_model, use_container_width=True)
        
    with col_b:
        st.subheader("Latency Distribution")
        fig_lat = px.histogram(df, x="duration_seconds", nbins=20)
        st.plotly_chart(fig_lat, use_container_width=True)

    # Recent Logs Table
    st.subheader("Recent Logs")
    st.dataframe(df.sort_values("timestamp", ascending=False).head(50))


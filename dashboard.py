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

    # Trace Explorer
    if "trace" in df.columns:
        st.subheader("Trace Explorer")
        traced_df = df[df["trace"].notna()]

        if traced_df.empty:
            st.info("No trace data found yet. Run composite models (e.g., MoA, Council, CoT, ThinkTool).")
        else:
            # Ensure trace column is parsed JSON if serialized as string
            def _parse_trace(val):
                if isinstance(val, dict) or pd.isna(val):
                    return val
                try:
                    return json.loads(val)
                except Exception:
                    return None

            traced_df = traced_df.copy()
            traced_df["trace"] = traced_df["trace"].apply(_parse_trace)
            traced_df = traced_df[traced_df["trace"].notna()]

            if not traced_df.empty:
                # Build selection options
                def _format_row(idx):
                    row = traced_df.loc[idx]
                    ts = row.get("timestamp", "")
                    model = row.get("model", "")
                    trace = row.get("trace", {}) or {}
                    strategy = trace.get("strategy", "")
                    trace_id = trace.get("trace_id", "")
                    return f"{ts} | {model} | {strategy} | {trace_id}"

                indices = traced_df.index.tolist()
                selected_idx = st.selectbox(
                    "Select a traced call",
                    indices,
                    format_func=_format_row,
                )

                trace = traced_df.loc[selected_idx, "trace"]

                # Render trace as a tree using nested expanders
                nodes = {n["id"]: n for n in trace.get("nodes", [])}
                children_map = {node_id: [] for node_id in nodes.keys()}
                for node in nodes.values():
                    parent_id = node.get("parent_id")
                    if parent_id is not None and parent_id in children_map:
                        children_map[parent_id].append(node["id"])

                root_id = trace.get("root_node_id")
                if not root_id and nodes:
                    # Fallback: pick a node without a parent
                    root_candidates = [n["id"] for n in nodes.values() if n.get("parent_id") is None]
                    root_id = root_candidates[0] if root_candidates else None

                st.markdown("**Trace Metadata**")
                meta_cols = st.columns(3)
                with meta_cols[0]:
                    st.write(f"**Trace ID:** {trace.get('trace_id', '')}")
                with meta_cols[1]:
                    st.write(f"**Strategy:** {trace.get('strategy', '')}")
                with meta_cols[2]:
                    st.write(f"**Root Model:** {trace.get('root_model', '')}")
                st.write(f"**User Request Preview:** {trace.get('user_request_preview', '')}")

                def render_node(node_id: str, depth: int = 0):
                    node = nodes.get(node_id)
                    if not node:
                        return
                    label_parts = [node.get("step_type", "")]
                    model_name = node.get("model")
                    if model_name:
                        label_parts.append(f"model={model_name}")
                    duration = node.get("duration_seconds")
                    if duration is not None:
                        label_parts.append(f"{duration:.2f}s")
                    label = " | ".join(label_parts)

                    with st.expander(("  " * depth) + label, expanded=(depth == 0)):
                        st.write(f"**Step Type:** {node.get('step_type', '')}")
                        st.write(f"**Model:** {model_name or 'N/A'}")
                        st.write(f"**Role:** {node.get('role', '')}")
                        if duration is not None:
                            st.write(f"**Duration:** {duration:.2f} s")
                        if "prompt_tokens" in node or "completion_tokens" in node:
                            st.write(
                                f"**Tokens:** prompt={node.get('prompt_tokens', 0)}, "
                                f"completion={node.get('completion_tokens', 0)}, "
                                f"total={node.get('total_tokens', 0)}"
                            )
                        content_preview = node.get("content_preview")
                        if content_preview:
                            st.write("**Content Preview:**")
                            st.code(content_preview)
                        extra = node.get("extra")
                        if extra:
                            st.write("**Extra:**")
                            st.json(extra)

                        for child_id in children_map.get(node_id, []):
                            render_node(child_id, depth + 1)

                if root_id:
                    render_node(root_id)
                else:
                    st.info("No root node found in trace.")


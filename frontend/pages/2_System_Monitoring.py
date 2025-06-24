"""
System Monitoring Page
Displays real-time health and performance metrics from the backend API.
"""
import streamlit as st
import time
from api_client import get_health, get_metrics

st.set_page_config(page_title="System Monitoring", page_icon="📊", layout="wide")

st.title("📊 System Monitoring Dashboard")

if 'health_data' not in st.session_state:
    st.session_state.health_data = None
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = None

def fetch_data():
    with st.spinner("Fetching latest data from backend..."):
        st.session_state.health_data = get_health()
        st.session_state.metrics_data = get_metrics()

# Initial data fetch
if st.session_state.health_data is None:
    fetch_data()

st.button("🔄 Refresh Data", on_click=fetch_data, use_container_width=True)
st.divider()

health_data = st.session_state.health_data
metrics_data = st.session_state.metrics_data

if health_data and health_data.get('status') != 'unhealthy':
    # --- Key Metrics ---
    st.subheader("Key Performance Indicators (KPIs)")
    
    api_metrics = metrics_data.get('api_metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "API Status", 
            health_data.get('status', 'Unknown').title(), 
            help="Overall health of the API server."
        )
    with col2:
        st.metric(
            "Ollama Status", 
            health_data.get('ollama_status', 'Unknown').title(),
            help="Health of the local LLM service."
        )
    with col3:
        uptime_seconds = health_data.get('uptime_seconds', 0)
        uptime_str = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m"
        st.metric("Server Uptime", uptime_str)
    with col4:
        st.metric(
            "Error Rate", 
            f"{api_metrics.get('error_rate', 0):.2f}%",
            help="Percentage of requests that resulted in an error."
        )
        
    st.divider()

    # --- System Usage ---
    st.subheader("System & Orchestration Usage")
    
    orch_metrics = metrics_data.get('orchestration_metrics', {})
    mem_usage = health_data.get('memory_usage', {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total API Requests", api_metrics.get('total_requests', 0))
    with col2:
        st.metric("Active Forecasts", api_metrics.get('active_forecasts', 0))
    with col3:
        st.metric(
            "Memory Usage", 
            f"{mem_usage.get('used_percent', 0)}%", 
            f"{(mem_usage.get('available_gb', 0)):.1f} GB free",
            delta_color="inverse"
        )
        
    st.divider()
    
    # --- Raw Data Expanders ---
    st.subheader("Raw API Data")
    with st.expander("Health Check Data"):
        st.json(health_data)
    with st.expander("Metrics Data"):
        st.json(metrics_data)

else:
    st.error("🔴 **Backend API is offline or unreachable.**")
    st.warning("Please ensure the backend server is running. You can start it with the command:")
    st.code("ts-ensemble serve")
    if health_data and 'error' in health_data:
        st.code(f"Connection Error: {health_data['error']}")

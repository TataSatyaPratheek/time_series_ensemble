"""
Main Streamlit application file.
This file sets up the main page, sidebar, and initializes session state.
"""
import streamlit as st
import pandas as pd
from api_client import get_health

# --- Page Configuration ---
st.set_page_config(
    page_title="Time Series Ensemble Forecaster",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    state_defaults = {
        'api_status': 'unknown',
        'api_health_data': None,
        'uploaded_file': None,
        'time_series_data': None,
        'forecast_results': None,
        'forecast_id': None,
        'forecast_running': False
    }
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Main App ---

# Sidebar
with st.sidebar:
    st.title("🔮 TS Ensemble")
    st.write("Multi-Agent Time Series Forecasting")

    # API Health Check
    if st.button("Check API Status"):
        with st.spinner("Pinging backend..."):
            health_data = get_health()
            st.session_state.api_health_data = health_data
            st.session_state.api_status = health_data.get("status", "unhealthy")
    
    if st.session_state.api_status == "healthy":
        st.success(f"API Status: {st.session_state.api_status.title()}")
    elif st.session_state.api_status == "degraded":
        st.warning(f"API Status: {st.session_state.api_status.title()}")
    else:
        st.error(f"API Status: {st.session_state.api_status.title()}")
        if st.session_state.api_health_data and 'error' in st.session_state.api_health_data:
            st.caption(f"Error: {st.session_state.api_health_data['error']}")
    
    st.divider()
    
    st.page_link("app.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/1_Forecasting_Studio.py", label="Forecasting Studio", icon="🔮")
    st.page_link("pages/2_System_Monitoring.py", label="System Monitoring", icon="📊")
    st.page_link("pages/3_Configuration_Viewer.py", label="Configuration", icon="⚙️")
    
    st.divider()
    st.info("Built with a multi-agent system powered by CrewAI and local LLMs.")

# Home Page Content
st.title("Welcome to the Time Series Ensemble Forecaster")
st.markdown(
    """
    This application leverages a sophisticated multi-agent system to deliver high-quality time series forecasts.
    Navigate to the **Forecasting Studio** to get started.
    """
)
st.image("https://raw.githubusercontent.com/tatasatyapratheek/time_series_ensemble/main/docs/architecture.png", caption="System Architecture")

st.header("About This Project")
st.markdown(
    """
    This project is a demonstration of an advanced forecasting system that uses specialized AI agents for different aspects of time series analysis:
    - **Trend Agent:** Analyzes long-term patterns and structural changes.
    - **Seasonality Agent:** Detects and models seasonal cycles and holiday effects.
    - **Anomaly Agent:** Identifies outliers and unusual data points that could impact the forecast.
    - **Ensemble Coordinator:** Intelligently combines the insights from all agents to produce a final, robust forecast with uncertainty estimates.
    
    The entire system is powered by local Large Language Models (LLMs) via Ollama for strategic reasoning and runs on a FastAPI backend.
    """
)

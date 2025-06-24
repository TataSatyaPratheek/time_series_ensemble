"""
Forecasting Studio Page
This is the main interactive page for uploading data and generating forecasts.
"""
import streamlit as st
import pandas as pd
import numpy as np
from api_client import post_forecast, list_workflows
from ui_components import (
    display_forecast_chart,
    display_llm_insights,
    display_agent_analysis,
    display_recommendations
)

st.title("🔮 Forecasting Studio")

# --- 1. Data Input Section ---
st.header("1. Provide Your Time Series Data")

input_col, preview_col = st.columns([1, 2])

with input_col:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    use_sample_data = st.toggle("Or use sample data", value=not uploaded_file)

    if use_sample_data:
        # Generate sample data
        date_rng = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
        data = np.random.randn(len(date_rng)).cumsum() + 50
        trend = np.linspace(0, 20, len(date_rng))
        seasonal = 10 * np.sin(np.arange(len(date_rng)) * 2 * np.pi / 365.25)
        series = pd.Series(data + trend + seasonal, index=date_rng)
        st.session_state.time_series_data = series
        st.info("Using sample sales data.")
    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Basic validation
            if len(df.columns) < 1:
                st.error("CSV must have at least one column.")
            else:
                # Assume first column is date/index, second is value
                if len(df.columns) >= 2:
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    series = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0], name="Uploaded Data")
                else:
                    series = pd.Series(df.iloc[:, 0].values, name="Uploaded Data")
                
                # Check for non-numeric data
                if not pd.api.types.is_numeric_dtype(series):
                    st.error("The value column must be numeric.")
                else:
                    st.session_state.time_series_data = series.sort_index()

        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")
            st.session_state.time_series_data = None

with preview_col:
    if st.session_state.time_series_data is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.time_series_data.head())
        st.line_chart(st.session_state.time_series_data)
    else:
        st.info("Upload a CSV or select sample data to see a preview.")

# --- 2. Configuration Section ---
if st.session_state.time_series_data is not None:
    st.header("2. Configure Your Forecast")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        # Get available workflows from API
        workflows = list_workflows()
        workflow_names = [wf['name'] for wf in workflows] if workflows else ["standard", "fast", "comprehensive"]
        workflow_type = st.selectbox(
            "Select Workflow",
            options=workflow_names,
            index=0,
            help="Choose the analysis depth. 'Fast' is quicker, 'Comprehensive' is more thorough."
        )

        forecast_horizon = st.number_input(
            "Forecast Horizon (periods)",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="How many future periods to predict."
        )

    with config_col2:
        confidence_level = st.slider(
            "Confidence Level",
            min_value=0.50,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="The confidence level for prediction intervals."
        )
    
    if st.button("✨ Generate Forecast", type="primary", use_container_width=True):
        st.session_state.forecast_results = None # Clear previous results
        with st.spinner("🤖 The multi-agent crew is analyzing your data... This may take several minutes."):
            results = post_forecast(
                series=st.session_state.time_series_data,
                workflow_type=workflow_type,
                forecast_horizon=forecast_horizon,
                confidence_level=confidence_level
            )
            st.session_state.forecast_results = results
            
        if results.get("status") == "success":
            st.success("Forecast generated successfully!")
            st.balloons()
        else:
            st.error(f"Forecast generation failed: {results.get('error', 'An unknown error occurred.')}")

# --- 3. Results Section ---
if st.session_state.forecast_results:
    results = st.session_state.forecast_results
    st.header("3. Forecast Results & Insights")
    
    if results.get("status") == "success":
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast Plot", "🧠 LLM Insights", "🕵️ Agent Analysis", "💡 Recommendations"])
        
        with tab1:
            display_forecast_chart(st.session_state.time_series_data, results)
        
        with tab2:
            display_llm_insights(results.get('metadata', {}))

        with tab3:
            display_agent_analysis(results.get('metadata', {}))

        with tab4:
            display_recommendations(results.get('metadata', {}))
    
    else:
        st.error("Could not display results due to a previous error.")


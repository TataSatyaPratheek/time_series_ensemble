"""
Updated Forecasting Studio with Intelligent Data Processing
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
from data_processor import IntelligentDataProcessor

st.title("🔮 Forecasting Studio")

# Initialize data processor
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = IntelligentDataProcessor()

# --- 1. Enhanced Data Input Section ---
st.header("1. Upload Your Dataset")

upload_col, info_col = st.columns([2, 1])

with upload_col:
    st.subheader("📁 File Upload")
    
    # Multiple file upload for complex datasets
    uploaded_files = st.file_uploader(
        "Upload CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload single CSV or multiple related files (e.g., features.csv, stores.csv, sales.csv)"
    )
    
    use_sample_data = st.toggle("Or use Walmart sample data", value=not uploaded_files)

with info_col:
    st.subheader("💡 Supported Formats")
    st.info("""
    **Retail/Sales Data:**
    - Walmart-style datasets
    - Store sales with features
    - Economic indicators
    
    **Single File:**
    - Date + Value columns
    - Multiple time series
    
    **Multi-File:**
    - features.csv
    - stores.csv  
    - sales.csv (optional)
    """)

# Process data
processed_data = None

if use_sample_data and not uploaded_files:
    # Create sample Walmart-style data
    st.info("🏪 Using Walmart sample dataset")
    
    # Generate sample data that matches Walmart structure
    dates = pd.date_range('2019-01-01', '2023-12-31', freq='W')
    n_periods = len(dates)
    
    # Create realistic Walmart-style sample data
    np.random.seed(42)
    base_sales = 15000
    trend = np.linspace(0, 2000, n_periods)
    seasonal = 3000 * np.sin(2 * np.pi * np.arange(n_periods) / 52.0)  # Yearly seasonality
    weekly_pattern = 1000 * np.sin(2 * np.pi * np.arange(n_periods) / 4.3)  # Monthly pattern
    holiday_effect = np.random.choice([0, 2000, 5000], n_periods, p=[0.85, 0.10, 0.05])
    noise = np.random.normal(0, 800, n_periods)
    
    weekly_sales = base_sales + trend + seasonal + weekly_pattern + holiday_effect + noise
    weekly_sales = np.maximum(weekly_sales, 1000)  # Ensure positive sales
    
    # Create features
    sample_data = pd.DataFrame({
        'Date': dates,
        'Weekly_Sales': weekly_sales,
        'Temperature': 65 + 20 * np.sin(2 * np.pi * np.arange(n_periods) / 52.0) + np.random.normal(0, 5, n_periods),
        'Fuel_Price': 2.8 + 0.5 * np.sin(2 * np.pi * np.arange(n_periods) / 52.0) + np.random.normal(0, 0.2, n_periods),
        'CPI': 210 + np.cumsum(np.random.normal(0.02, 0.5, n_periods)),
        'Unemployment': 8 + 2 * np.sin(2 * np.pi * np.arange(n_periods) / 52.0) + np.random.normal(0, 0.3, n_periods),
        'IsHoliday': np.random.choice([0, 1], n_periods, p=[0.92, 0.08])
    })
    
    uploaded_files_dict = {'sample_data.csv': sample_data}
    processed_data = st.session_state.data_processor.create_forecasting_dataset(uploaded_files_dict)

elif uploaded_files:
    try:
        # Process uploaded files
        uploaded_files_dict = {}
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            uploaded_files_dict[uploaded_file.name] = df
            st.success(f"✅ Loaded {uploaded_file.name}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Process with intelligent data processor
        with st.spinner("🤖 Intelligently processing your data..."):
            processed_data = st.session_state.data_processor.create_forecasting_dataset(uploaded_files_dict)
            
        if not processed_data.empty:
            st.success("🎉 Data processed successfully!")
        else:
            st.error("❌ Failed to process data. Please check your file format.")
            
    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")

# Display processed data info
if processed_data is not None and not processed_data.empty:
    st.header("2. Data Processing Results")
    
    # Display summary
    st.session_state.data_processor.display_data_summary()
    
    # Show processed data sample
    with st.expander("📋 View Processed Data Sample"):
        st.dataframe(processed_data.head(20), use_container_width=True)
    
    # Data visualization
    with st.expander("📈 Data Visualization", expanded=True):
        st.session_state.data_processor.create_data_visualization()
    
    # Extract forecasting series
    forecasting_series = st.session_state.data_processor.get_forecasting_series()
    
    if not forecasting_series.empty:
        st.session_state.time_series_data = forecasting_series
        
        # --- 3. Configuration Section ---
        st.header("3. Configure Your Forecast")
        
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
                max_value=104,  # Up to 2 years for weekly data
                value=12,  # Default to 3 months for weekly data
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
            
            include_features = st.checkbox(
                "Include External Features",
                value=True,
                help="Use additional features for forecasting (recommended for complex datasets)"
            )
        
        # Show forecasting series info
        st.info(f"""
        📊 **Forecasting Series Ready:**
        - **Target:** {st.session_state.data_processor.metadata['target_column']}
        - **Frequency:** {pd.infer_freq(forecasting_series.index) or 'Auto-detected'}
        - **Data Points:** {len(forecasting_series):,}
        - **Date Range:** {forecasting_series.index.min().strftime('%Y-%m-%d')} to {forecasting_series.index.max().strftime('%Y-%m-%d')}
        """)
        
        if st.button("✨ Generate Forecast", type="primary", use_container_width=True):
            st.session_state.forecast_results = None  # Clear previous results
            
            with st.spinner("🤖 The multi-agent crew is analyzing your retail data... This may take several minutes."):
                results = post_forecast(
                    series=forecasting_series,
                    workflow_type=workflow_type,
                    forecast_horizon=forecast_horizon,
                    confidence_level=confidence_level
                )
                st.session_state.forecast_results = results
                
            if results.get("status") == "success":
                st.success("🎉 Forecast generated successfully!")
                st.balloons()
            else:
                st.error(f"❌ Forecast generation failed: {results.get('error', 'An unknown error occurred.')}")

# --- 4. Results Section ---
if hasattr(st.session_state, 'forecast_results') and st.session_state.forecast_results:
    results = st.session_state.forecast_results
    st.header("4. Forecast Results & Insights")
    
    if results.get("status") == "success":
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast Plot", "🧠 LLM Insights", "🕵️ Agent Analysis", "💡 Recommendations"])
        
        with tab1:
            if hasattr(st.session_state, 'time_series_data'):
                display_forecast_chart(st.session_state.time_series_data, results)
        
        with tab2:
            display_llm_insights(results.get('metadata', {}))

        with tab3:
            display_agent_analysis(results.get('metadata', {}))

        with tab4:
            display_recommendations(results.get('metadata', {}))
    
    else:
        st.error("Could not display results due to a previous error.")

else:
    st.info("👆 Upload your dataset and configure settings to generate a forecast.")

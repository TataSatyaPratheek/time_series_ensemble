"""
Reusable UI components for the Streamlit frontend.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List

def display_forecast_chart(
    historical_data: pd.Series,
    forecast_data: Dict[str, Any]
):
    """Renders an interactive plot of the forecast results."""
    st.subheader("📈 Forecast Visualization")

    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data.values,
        mode='lines',
        name='Historical Data',
        line=dict(color='royalblue')
    ))

    # Create future index for forecast
    last_date = historical_data.index[-1]
    freq = pd.infer_freq(historical_data.index) or 'D'
    future_index = pd.date_range(start=last_date, periods=len(forecast_data['forecast']) + 1, freq=freq)[1:]

    # Add forecast data
    fig.add_trace(go.Scatter(
        x=future_index,
        y=forecast_data['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='crimson', dash='dot')
    ))

    # Add confidence intervals if available
    if forecast_data.get('confidence_intervals'):
        lower_bound = [ci[0] for ci in forecast_data['confidence_intervals']]
        upper_bound = [ci[1] for ci in forecast_data['confidence_intervals']]
        fig.add_trace(go.Scatter(
            x=future_index,
            y=upper_bound,
            mode='lines',
            name='Upper Confidence Bound',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=future_index,
            y=lower_bound,
            mode='lines',
            name='Lower Confidence Bound',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            showlegend=False
        ))

    fig.update_layout(
        title="Time Series Forecast",
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_llm_insights(results: Dict[str, Any]):
    """Renders the LLM-generated strategic insights."""
    st.subheader("🧠 LLM Strategic Insights")
    insights = results.get('coordination_results', {}).get('comprehensive_report', {}).get('llm_strategic_insights', {})
    
    if insights and insights.get('strategic_analysis'):
        st.markdown(insights['strategic_analysis'])
    else:
        st.info("No LLM insights were generated for this forecast.")

def display_agent_analysis(results: Dict[str, Any]):
    """Renders a detailed breakdown of each agent's analysis."""
    st.subheader("🕵️ Agent Analysis Breakdown")

    agent_results = results.get('coordination_results', {})

    if not agent_results:
        st.warning("No detailed agent analysis available.")
        return

    # Trend Agent Analysis
    with st.expander("📈 Trend Agent Analysis", expanded=True):
        trend_analysis = agent_results.get('TrendAnalysis_results', {}).get('trend_analysis', {})
        if trend_analysis:
            strength = trend_analysis.get('trend_strength', {})
            validation = trend_analysis.get('trend_validation', {})
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Trend Direction", strength.get('trend_direction', 'N/A').title())
            col2.metric("Is Significant?", str(validation.get('is_significant', 'N/A')))
            col3.metric("R² Score", f"{strength.get('r_squared', 0):.2f}")
            if trend_analysis.get('llm_insights', {}).get('analysis'):
                st.info("LLM Trend Summary:")
                st.markdown(trend_analysis['llm_insights']['analysis'])
        else:
            st.write("No trend analysis data available.")
            
    # Seasonality Agent Analysis
    with st.expander("📅 Seasonality Agent Analysis"):
        seasonality_analysis = agent_results.get('SeasonalityDetection_results', {}).get('seasonality_analysis', {})
        if seasonality_analysis:
            periods = seasonality_analysis.get('period_detection', {}).get('detected_periods', [])
            holiday_impact = seasonality_analysis.get('holiday_effects', {}).get('holiday_impact', 0)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Periods Detected", len(periods))
            col2.metric("Strongest Period", f"{periods[0]['period']}" if periods else "N/A")
            col3.metric("Holiday Impact", f"{holiday_impact:.2%}")

            if periods:
                st.write("**Detected Periods & Strengths:**")
                period_df = pd.DataFrame(periods).head(5)
                st.dataframe(period_df[['period', 'strength', 'confidence', 'methods']])
            if seasonality_analysis.get('llm_insights', {}).get('analysis'):
                 st.info("LLM Seasonality Summary:")
                 st.markdown(seasonality_analysis['llm_insights']['analysis'])
        else:
            st.write("No seasonality analysis data available.")

    # Anomaly Agent Analysis
    with st.expander("🚨 Anomaly Agent Analysis"):
        anomaly_analysis = agent_results.get('AnomalyDetection_results', {}).get('anomaly_analysis', {})
        if anomaly_analysis:
            anomalies = anomaly_analysis.get('detected_anomalies', [])
            impact = anomaly_analysis.get('impact_assessment', {}).get('impact_assessment', {})
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Anomalies Detected", len(anomalies))
            col2.metric("Business Risk", impact.get('business_impact', {}).get('business_risk', 'N/A').title())
            col3.metric("Data Quality Score", f"{impact.get('data_quality_impact', {}).get('quality_score', 0):.2f}")
            if anomaly_analysis.get('llm_insights', {}).get('analysis'):
                 st.info("LLM Anomaly Summary:")
                 st.markdown(anomaly_analysis['llm_insights']['analysis'])
        else:
            st.write("No anomaly analysis data available.")

def display_recommendations(results: Dict[str, Any]):
    """Displays actionable recommendations from the forecast."""
    st.subheader("💡 Recommendations")
    recommendations = results.get('coordination_results', {}).get('comprehensive_report', {}).get('recommendations', [])
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.warning(f"**Recommendation {i}:** {rec}")
    else:
        st.info("No specific recommendations were generated.")


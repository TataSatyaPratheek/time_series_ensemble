"""
API Client for communicating with the Time Series Ensemble Forecasting backend.
"""
import requests
import pandas as pd
from typing import Dict, Any, List

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"  # Assumes the backend is running locally

# --- API Functions ---

def get_health() -> Dict[str, Any]:
    """Checks the health of the backend API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}

def list_workflows() -> List[Dict[str, Any]]:
    """Fetches the list of available forecasting workflows."""
    try:
        response = requests.get(f"{API_BASE_URL}/workflows", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching workflows: {e}")
        return []

def get_metrics() -> Dict[str, Any]:
    """Fetches system metrics from the backend API."""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching workflows: {e}")
        return {}

def post_forecast(
    series: pd.Series,
    workflow_type: str,
    forecast_horizon: int,
    confidence_level: float
) -> Dict[str, Any]:
    """
    Submits a forecast request to the backend API.

    Args:
        series: The time series data as a pandas Series.
        workflow_type: The selected workflow (e.g., 'standard').
        forecast_horizon: The number of periods to forecast.
        confidence_level: The confidence level for prediction intervals.

    Returns:
        The JSON response from the API.
    """
    try:
        payload = {
            "data": {
                "values": series.tolist(),
                "timestamps": series.index.strftime('%Y-%m-%dT%H:%M:%S').tolist() if isinstance(series.index, pd.DatetimeIndex) else None
            },
            "workflow_type": workflow_type,
            "forecast_horizon": forecast_horizon,
            "confidence_level": confidence_level,
            "enable_explanations": True  # Always enable for frontend
        }
        
        response = requests.post(f"{API_BASE_URL}/forecast", json=payload, timeout=1800) # 30 min timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_detail = "Failed to connect to backend or request timed out."
        try:
            # Try to get more specific error from response if available
            error_detail = e.response.json().get("detail", error_detail)
        except:
            pass
        return {"status": "error", "error": f"API Request Failed: {error_detail}"}

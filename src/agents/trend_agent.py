"""
Async Trend Analysis Agent with CrewAI and Local LLM Integration
Handles long-term pattern detection and trend forecasting using statistical and ML models.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

# CrewAI imports
from crewai import Agent, Task
from src.llm.direct_interface import DirectOllamaInterface

# Project imports
from src.config import settings
from src.models.statistical import (
    AsyncSTLDecomposition, 
    AsyncExponentialSmoothing, 
    AsyncARIMAModel,
    auto_arima_selection
)
from src.models.ml_models import AsyncLinearModel, create_time_series_features
from src.utils.exceptions import AgentError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TrendAnalysisAgent:
    """
    Specialized agent for trend analysis using multiple methods and local LLM reasoning.
    Integrates with CrewAI for multi-agent orchestration.
    """
    
    def __init__(self, 
                 agent_config: Optional[Dict[str, Any]] = None,
                 llm_model: str = settings.TREND_ANALYSIS_MODEL,
                 timeout: Optional[int] = None,
                 **kwargs):
        """
        Initialize trend analysis agent.
        
        Args:
            agent_config: Agent configuration from YAML
            llm_model: Local LLM model for reasoning
            timeout: Task timeout in seconds
        """
        self.agent_config = agent_config or {}
        self.llm_model = llm_model
        self.timeout = timeout or 180  # Default 3 minutes
        
        # Initialize models
        self.stl_model = AsyncSTLDecomposition()
        self.exp_smoothing_model = AsyncExponentialSmoothing()
        self.arima_model = None
        self.linear_model = AsyncLinearModel('linear')
        
        # Agent state
        self.is_fitted = False
        self.trend_results = {}
        self.analysis_metadata = {}
        
        # Function definitions for LLM
        self.functions = [
            {
                "name": "analyze_trend_strength",
                "description": "Analyze the strength and direction of trend in time series data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trend_data": {"type": "array", "description": "Trend component values"},
                        "method": {"type": "string", "enum": ["linear", "polynomial", "seasonal"]},
                        "significance_threshold": {"type": "number", "description": "Threshold for trend significance"}
                    },
                    "required": ["trend_data", "method"]
                }
            },
            {
                "name": "detect_change_points",
                "description": "Identify structural breaks and change points in trend",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "series": {"type": "array", "description": "Time series data"},
                        "min_size": {"type": "integer", "description": "Minimum segment size"},
                        "penalty": {"type": "number", "description": "Change point detection penalty"}
                    },
                    "required": ["series"]
                }
            },
            {
                "name": "extrapolate_trend",
                "description": "Extrapolate trend component for forecasting",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trend_data": {"type": "array", "description": "Historical trend values"},
                        "steps": {"type": "integer", "description": "Number of steps to forecast"},
                        "method": {"type": "string", "enum": ["linear", "polynomial", "exponential"]}
                    },
                    "required": ["trend_data", "steps"]
                }
            },
            {
                "name": "validate_trend_significance",
                "description": "Validate statistical significance of detected trend",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trend_data": {"type": "array", "description": "Trend component"},
                        "original_series": {"type": "array", "description": "Original time series"},
                        "confidence_level": {"type": "number", "description": "Confidence level for validation"}
                    },
                    "required": ["trend_data", "original_series"]
                }
            }
        ]
        
        # Initialize LLM generator
        try:
            self.llm_interface = DirectOllamaInterface(model_name=self.llm_model)
            logger.info(f"TrendAgent initialized with LLM: {self.llm_model}")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {str(e)}, using fallback mode")
            self.llm_interface = None
    
    def get_crewai_agent(self) -> Agent:
        """
        Create CrewAI agent instance for orchestration.
        
        Returns:
            CrewAI Agent configured for trend analysis
        """
        return Agent(
            role="Time Series Trend Analyst",
            goal="Identify and analyze long-term patterns, trends, and structural changes in temporal data using statistical and ML methods",
            backstory="""You are an expert time series analyst specializing in trend detection and long-term pattern analysis.
            You have deep knowledge of statistical decomposition, change point detection, and trend extrapolation methods.
            Your strength lies in identifying subtle long-term patterns that other models might miss.""",
            tools=list(self.functions),
            llm=self.llm_model,
            verbose=True,
            allow_delegation=False, # Keep as False unless specific delegation is intended
            max_execution_time=self.timeout, # Pass the timeout to the CrewAI Agent
            memory=True
        )
    
    async def analyze_trend_strength(self, 
                                   trend_data: np.ndarray,
                                   method: str = "linear",
                                   significance_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Analyze trend strength and statistical significance.
        
        Args:
            trend_data: Trend component values
            method: Analysis method ('linear', 'polynomial', 'seasonal')
            significance_threshold: P-value threshold for significance
            
        Returns:
            Dictionary with trend strength analysis
        """
        try:
            if method == "linear":
                # Linear trend analysis
                x = np.arange(len(trend_data))
                coeffs = np.polyfit(x, trend_data, 1)
                slope = coeffs[0]
                
                # Calculate R-squared
                trend_fit = np.polyval(coeffs, x)
                ss_res = np.sum((trend_data - trend_fit) ** 2)
                ss_tot = np.sum((trend_data - np.mean(trend_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Statistical significance test
                from scipy import stats
                slope_stat, p_value = stats.linregress(x, trend_data)[:2]
                is_significant = p_value < significance_threshold
                
                return {
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat',
                    'trend_slope': float(slope),
                    'trend_strength': float(abs(slope)),
                    'r_squared': float(r_squared),
                    'p_value': float(p_value),
                    'is_significant': bool(is_significant),
                    'method': method
                }
            
            elif method == "polynomial":
                # Polynomial trend analysis (degree 2)
                x = np.arange(len(trend_data))
                coeffs = np.polyfit(x, trend_data, 2)
                
                # Calculate curvature
                curvature = 2 * coeffs[0]
                
                return {
                    'trend_type': 'accelerating' if curvature > 0 else 'decelerating' if curvature < 0 else 'linear',
                    'curvature': float(curvature),
                    'coefficients': coeffs.tolist(),
                    'method': method
                }
            
            else:  # seasonal method
                # Seasonal trend analysis
                seasonal_strength = np.var(trend_data) / (np.var(trend_data) + np.var(trend_data - np.mean(trend_data)))
                
                return {
                    'seasonal_strength': float(seasonal_strength),
                    'trend_variance': float(np.var(trend_data)),
                    'method': method
                }
                
        except Exception as e:
            logger.error(f"Trend strength analysis failed: {e}")
            raise AgentError(f"Trend analysis error: {str(e)}") from e
    
    async def detect_change_points(self, 
                                 series: np.ndarray,
                                 min_size: int = 10,
                                 penalty: float = 1.0) -> Dict[str, Any]:
        """
        Detect structural breaks and change points in the trend.
        
        Args:
            series: Time series data
            min_size: Minimum segment size
            penalty: Change point detection penalty
            
        Returns:
            Dictionary with detected change points
        """
        try:
            # Simple change point detection using variance
            n = len(series)
            change_points = []
            
            # Sliding window approach
            window_size = max(min_size, n // 10)
            
            for i in range(window_size, n - window_size):
                left_segment = series[i-window_size:i]
                right_segment = series[i:i+window_size]
                
                # Calculate variance difference
                left_var = np.var(left_segment)
                right_var = np.var(right_segment)
                var_ratio = abs(left_var - right_var) / (left_var + right_var + 1e-8)
                
                # Mean difference
                left_mean = np.mean(left_segment)
                right_mean = np.mean(right_segment)
                mean_diff = abs(left_mean - right_mean)
                
                # Combined score
                score = var_ratio + mean_diff / np.std(series)
                
                if score > penalty:
                    change_points.append({
                        'index': int(i),
                        'score': float(score),
                        'mean_before': float(left_mean),
                        'mean_after': float(right_mean),
                        'variance_before': float(left_var),
                        'variance_after': float(right_var)
                    })
            
            return {
                'change_points': change_points,
                'n_change_points': len(change_points),
                'method': 'variance_based',
                'parameters': {'min_size': min_size, 'penalty': penalty}
            }
            
        except Exception as e:
            logger.error(f"Change point detection failed: {e}")
            raise AgentError(f"Change point detection error: {str(e)}") from e
    
    async def extrapolate_trend(self, 
                              trend_data: np.ndarray,
                              steps: int = 10,
                              method: str = "linear") -> Dict[str, Any]:
        """
        Extrapolate trend component for forecasting.
        
        Args:
            trend_data: Historical trend values
            steps: Number of steps to forecast
            method: Extrapolation method
            
        Returns:
            Dictionary with trend forecast
        """
        try:
            x = np.arange(len(trend_data))
            future_x = np.arange(len(trend_data), len(trend_data) + steps)
            
            if method == "linear":
                # Linear extrapolation
                coeffs = np.polyfit(x, trend_data, 1)
                trend_forecast = np.polyval(coeffs, future_x)
                
                # Calculate confidence intervals
                residuals = trend_data - np.polyval(coeffs, x)
                std_error = np.std(residuals)
                confidence_intervals = [
                    (float(trend_forecast[i] - 1.96 * std_error), 
                     float(trend_forecast[i] + 1.96 * std_error))
                    for i in range(len(trend_forecast))
                ]
                
            elif method == "polynomial":
                # Polynomial extrapolation (degree 2)
                coeffs = np.polyfit(x, trend_data, 2)
                trend_forecast = np.polyval(coeffs, future_x)
                confidence_intervals = None  # More complex for polynomial
                
            elif method == "exponential":
                # Exponential trend extrapolation
                log_trend = np.log(np.maximum(trend_data, 1e-8))
                coeffs = np.polyfit(x, log_trend, 1)
                log_forecast = np.polyval(coeffs, future_x)
                trend_forecast = np.exp(log_forecast)
                confidence_intervals = None
                
            else:
                raise ValueError(f"Unknown extrapolation method: {method}")
            
            return {
                'trend_forecast': trend_forecast.tolist(),
                'confidence_intervals': confidence_intervals,
                'method': method,
                'steps': steps,
                'extrapolation_coefficients': coeffs.tolist()
            }
            
        except Exception as e:
            logger.error(f"Trend extrapolation failed: {e}")
            raise AgentError(f"Trend extrapolation error: {str(e)}") from e
    
    async def validate_trend_significance(self, 
                                        trend_data: np.ndarray,
                                        original_series: np.ndarray,
                                        confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Validate statistical significance of detected trend.
        
        Args:
            trend_data: Trend component
            original_series: Original time series
            confidence_level: Confidence level for validation
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Calculate trend as percentage of total variance
            trend_variance = np.var(trend_data)
            total_variance = np.var(original_series)
            trend_contribution = trend_variance / total_variance if total_variance > 0 else 0
            
            # Perform statistical tests
            from scipy import stats
            
            # Test for trend presence (Mann-Kendall test alternative)
            x = np.arange(len(trend_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, trend_data)
            
            # Calculate confidence intervals for slope
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, len(trend_data) - 2)
            slope_ci_lower = slope - t_critical * std_err
            slope_ci_upper = slope + t_critical * std_err
            
            # Determine significance
            is_significant = p_value < alpha
            
            return {
                'is_significant': bool(is_significant),
                'p_value': float(p_value),
                'trend_slope': float(slope),
                'slope_confidence_interval': [float(slope_ci_lower), float(slope_ci_upper)],
                'r_squared': float(r_value ** 2),
                'trend_contribution_pct': float(trend_contribution * 100),
                'confidence_level': confidence_level,
                'standard_error': float(std_err)
            }
            
        except Exception as e:
            logger.error(f"Trend validation failed: {e}")
            raise AgentError(f"Trend validation error: {str(e)}") from e
    
    async def analyze_comprehensive_trend(self, 
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive trend analysis using multiple methods and LLM reasoning.
        
        Args:
            context: Dictionary containing 'series' (pd.Series) and 'use_llm_reasoning' (bool)
            
        Returns:
            Complete trend analysis results
        """
        series = context.get('series')
        use_llm_reasoning = context.get('use_llm_reasoning', True)
        if series is None:
            raise ValueError("Time series data ('series') not found in context for TrendAnalysisAgent.")

        start_time = asyncio.get_event_loop().time() # type: ignore
        
        try:
            logger.info(f"TrendAgent: Starting comprehensive trend analysis for {len(series)} observations")
            
            # 1. STL Decomposition
            stl_results = await self.stl_model.fit(series)
            components = self.stl_model.get_components()
            trend_component = components['trend'].dropna().values
            
            # 2. Exponential Smoothing
            exp_results = await self.exp_smoothing_model.fit(series)
            
            # 3. ARIMA model selection and fitting
            best_order, best_seasonal_order = await auto_arima_selection(series)
            self.arima_model = AsyncARIMAModel(order=best_order, seasonal_order=best_seasonal_order)
            arima_results = await self.arima_model.fit(series)
            
            # 4. Linear trend model on features
            X, y = await create_time_series_features(series.to_frame('value'), 'value')
            if not X.empty:
                linear_results = await self.linear_model.fit(X, y)
            else:
                linear_results = {}
            
            # 5. Detailed trend analysis using functions
            trend_strength = await self.analyze_trend_strength(trend_component)
            change_points = await self.detect_change_points(series.values)
            trend_extrapolation = await self.extrapolate_trend(trend_component, steps=30)
            trend_validation = await self.validate_trend_significance(trend_component, series.values)
            
            # 6. LLM-based reasoning (if available)
            llm_insights = {}
            if use_llm_reasoning and self.llm_interface:
                try:
                    # Generate insights using LLM
                    prompt = f"""
                    Analyze the following trend analysis results and provide insights:
                    
                    Trend Direction: {trend_strength['trend_direction']}
                    Trend Strength: {trend_strength['trend_strength']:.4f}
                    Statistical Significance: {trend_validation['is_significant']}
                    R-squared: {trend_strength['r_squared']:.4f}
                    Change Points: {len(change_points['change_points'])}
                    
                    Provide a comprehensive analysis including:
                    1. Overall trend assessment
                    2. Reliability of the trend
                    3. Forecasting implications
                    4. Potential risks or opportunities
                    """
                    
                    system_prompt = """You are an expert time series trend analyst with deep knowledge of statistical analysis 
                and business implications. Provide strategic insights for forecasting and business decision-making."""
                
                    llm_response = await self.llm_interface.query_llm_async(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.1
                )

                    llm_insights = {'analysis': llm_response, 'timestamp': datetime.now().isoformat()}
                    
                except Exception as e:
                    logger.warning(f"LLM reasoning failed: {e}")
                    llm_insights = {'error': str(e)}
            
            # Compile results
            self.trend_results = {
                'stl_decomposition': stl_results,
                'components': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in components.items()},
                'exponential_smoothing': exp_results,
                'arima_model': arima_results,
                'linear_model': linear_results,
                'trend_strength': trend_strength,
                'change_points': change_points,
                'trend_extrapolation': trend_extrapolation,
                'trend_validation': trend_validation,
                'llm_insights': llm_insights
            }
            
            self.analysis_metadata = { # type: ignore
                'analysis_time': asyncio.get_event_loop().time() - start_time,
                'series_length': len(series),
                'models_used': ['STL', 'ExponentialSmoothing', 'ARIMA', 'Linear'],
                'llm_model': self.llm_model,
                'timestamp': datetime.now().isoformat()
            }
            
            self.is_fitted = True
            
            logger.info(f"TrendAgent: Comprehensive analysis completed in {self.analysis_metadata['analysis_time']:.2f}s")
            
            return {
                'trend_analysis': self.trend_results,
                'metadata': self.analysis_metadata
            }
            
        except Exception as e:
            error_msg = f"Comprehensive trend analysis failed: {e}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e
    
    async def forecast_trend(self, steps: int = 30) -> Dict[str, Any]:
        """
        Generate trend forecasts using multiple methods.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Dictionary with trend forecasts
        """
        if not self.is_fitted:
            raise AgentError("TrendAgent must be fitted before forecasting")

        try:
            forecasts = {}
            
            # STL trend extrapolation
            if self.stl_model.is_fitted:
                stl_forecast = await self.stl_model.forecast_trend(steps)
                forecasts['stl_trend'] = stl_forecast
            
            # Exponential smoothing forecast
            if self.exp_smoothing_model.is_fitted:
                exp_forecast = await self.exp_smoothing_model.predict(steps)
                forecasts['exponential_smoothing'] = exp_forecast.predictions
            
            # ARIMA forecast
            if self.arima_model and self.arima_model.is_fitted:
                arima_forecast = await self.arima_model.predict(steps)
                forecasts['arima'] = arima_forecast.predictions
            
            # Ensemble trend forecast (simple average)
            if len(forecasts) > 1:
                forecast_arrays = [np.array(f) for f in forecasts.values()]
                ensemble_forecast = np.mean(forecast_arrays, axis=0)
                forecasts['ensemble_trend'] = ensemble_forecast.tolist()
            
            return {
                'forecasts': forecasts,
                'steps': steps,
                'forecast_metadata': {
                    'methods_used': list(forecasts.keys()),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_msg = f"Trend forecasting failed: {e}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of trend analysis results."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        trend_strength = self.trend_results.get('trend_strength', {})
        trend_validation = self.trend_results.get('trend_validation', {})
        change_points = self.trend_results.get('change_points', {})
        
        return {
            'status': 'fitted',
            'trend_direction': trend_strength.get('trend_direction', 'unknown'),
            'trend_significant': trend_validation.get('is_significant', False),
            'trend_strength': trend_strength.get('trend_strength', 0.0),
            'r_squared': trend_strength.get('r_squared', 0.0),
            'change_points_detected': change_points.get('n_change_points', 0),
            'analysis_time': self.analysis_metadata.get('analysis_time', 0.0),
            'models_used': self.analysis_metadata.get('models_used', [])
        }

"""
Async Seasonality Detection Agent with CrewAI and Local LLM Integration
Handles detection and modeling of seasonal patterns in time series data.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import signal
from scipy.fft import fft, fftfreq

# CrewAI imports
from crewai import Agent, Task
from local_llm_function_calling import Generator

# Project imports
from src.config import settings
from src.models.statistical import (
    AsyncProphetModel,
    AsyncSTLDecomposition,
    seasonal_decompose
)
from src.utils.exceptions import AgentError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SeasonalityAgent:
    """
    Specialized agent for seasonality detection and modeling using multiple methods.
    Integrates with CrewAI for multi-agent orchestration.
    """
    
    def __init__(self, 
                 agent_config: Optional[Dict[str, Any]] = None,
                 llm_model: str = settings.SEASONALITY_MODEL):
        """
        Initialize seasonality detection agent.
        
        Args:
            agent_config: Agent configuration from YAML
            llm_model: Local LLM model for reasoning
        """
        self.agent_config = agent_config or {}
        self.llm_model = llm_model
        
        # Initialize models
        self.prophet_model = None
        self.stl_model = AsyncSTLDecomposition()
        
        # Agent state
        self.is_fitted = False
        self.seasonality_results = {}
        self.detected_periods = []
        self.seasonal_components = {}
        
        # Function definitions for LLM
        self.functions = [
            {
                "name": "detect_seasonality_periods",
                "description": "Detect seasonal periods using spectral analysis and autocorrelation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "series": {"type": "array", "description": "Time series data"},
                        "max_period": {"type": "integer", "description": "Maximum period to consider"},
                        "significance_threshold": {"type": "number", "description": "Threshold for period significance"}
                    },
                    "required": ["series"]
                }
            },
            {
                "name": "analyze_multiple_seasons",
                "description": "Analyze multiple overlapping seasonal patterns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "series": {"type": "array", "description": "Time series data"},
                        "periods": {"type": "array", "description": "List of potential periods"},
                        "method": {"type": "string", "enum": ["stl", "fourier", "prophet"]}
                    },
                    "required": ["series", "periods"]
                }
            },
            {
                "name": "model_holiday_effects",
                "description": "Model and analyze holiday and special event effects",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "series": {"type": "array", "description": "Time series data"},
                        "dates": {"type": "array", "description": "Date index"},
                        "country": {"type": "string", "description": "Country code for holidays"},
                        "custom_events": {"type": "array", "description": "Custom event dates"}
                    },
                    "required": ["series", "dates"]
                }
            },
            {
                "name": "validate_seasonal_stability",
                "description": "Validate stability of seasonal patterns over time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seasonal_components": {"type": "array", "description": "Seasonal component values"},
                        "period": {"type": "integer", "description": "Seasonal period"},
                        "method": {"type": "string", "enum": ["variance", "correlation", "trend"]}
                    },
                    "required": ["seasonal_components", "period"]
                }
            }
        ]
        
        # Initialize LLM generator
        try:
            self.llm_generator = Generator.hf(self.functions, self.llm_model)
            logger.info(f"SeasonalityAgent initialized with LLM: {self.llm_model}")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {str(e)}, using fallback mode")
            self.llm_generator = None
    
    def get_crewai_agent(self) -> Agent:
        """Create CrewAI agent instance for orchestration."""
        return Agent(
            role="Seasonality Pattern Specialist",
            goal="Detect, analyze, and model seasonal patterns including multiple seasonalities, holidays, and cyclical behaviors",
            backstory="""You are a specialist in seasonal pattern recognition with expertise in Fourier analysis, 
            decomposition techniques, and holiday effect modeling. You excel at identifying both 
            obvious and subtle seasonal patterns across different time scales.""",
            tools=list(self.functions),
            llm=self.llm_model,
            verbose=True,
            allow_delegation=False,
            memory=True
        )
    
    async def detect_seasonality_periods(self, 
                                       series: np.ndarray,
                                       max_period: Optional[int] = None,
                                       significance_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Detect seasonal periods using multiple methods.
        
        Args:
            series: Time series data
            max_period: Maximum period to consider
            significance_threshold: Threshold for period significance
            
        Returns:
            Dictionary with detected periods and their characteristics
        """
        try:
            if max_period is None:
                max_period = min(len(series) // 4, 365)  # Max 1 year or quarter of data
            
            detected_periods = []
            
            # Method 1: Autocorrelation analysis
            autocorr_periods = await self._detect_periods_autocorr(series, max_period, significance_threshold)
            detected_periods.extend(autocorr_periods)
            
            # Method 2: FFT/Spectral analysis
            fft_periods = await self._detect_periods_fft(series, max_period, significance_threshold)
            detected_periods.extend(fft_periods)
            
            # Method 3: Seasonal decomposition residuals
            decomp_periods = await self._detect_periods_decomposition(series, max_period)
            detected_periods.extend(decomp_periods)
            
            # Consolidate and rank periods
            consolidated_periods = await self._consolidate_periods(detected_periods, significance_threshold)
            
            # Common seasonal periods for validation
            common_periods = {
                7: 'weekly',
                30: 'monthly', 
                31: 'monthly',
                90: 'quarterly',
                91: 'quarterly',
                365: 'yearly',
                366: 'yearly'
            }
            
            # Enhance with known patterns
            for period in consolidated_periods:
                period_val = period['period']
                if period_val in common_periods:
                    period['pattern_type'] = common_periods[period_val]
                elif 28 <= period_val <= 31:
                    period['pattern_type'] = 'monthly'
                elif 88 <= period_val <= 93:
                    period['pattern_type'] = 'quarterly'
                elif 360 <= period_val <= 370:
                    period['pattern_type'] = 'yearly'
                else:
                    period['pattern_type'] = 'custom'
            
            return {
                'detected_periods': consolidated_periods,
                'n_periods': len(consolidated_periods),
                'methods_used': ['autocorrelation', 'fft', 'decomposition'],
                'significance_threshold': significance_threshold,
                'max_period_searched': max_period
            }
            
        except Exception as e:
            logger.error(f"Period detection failed: {str(e)}")
            raise AgentError(f"Period detection error: {str(e)}") from e
    
    async def _detect_periods_autocorr(self, 
                                     series: np.ndarray, 
                                     max_period: int,
                                     threshold: float) -> List[Dict[str, Any]]:
        """Detect periods using autocorrelation analysis."""
        try:
            # Calculate autocorrelation
            autocorr = np.correlate(series, series, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find peaks in autocorrelation
            peaks, properties = signal.find_peaks(
                autocorr[1:max_period], 
                height=threshold,
                distance=5  # Minimum distance between peaks
            )
            
            periods = []
            for i, peak in enumerate(peaks):
                periods.append({
                    'period': int(peak + 1),
                    'strength': float(autocorr[peak + 1]),
                    'method': 'autocorrelation',
                    'rank': i + 1
                })
            
            # Sort by strength
            periods.sort(key=lambda x: x['strength'], reverse=True)
            return periods
            
        except Exception as e:
            logger.warning(f"Autocorrelation period detection failed: {str(e)}")
            return []
    
    async def _detect_periods_fft(self, 
                                series: np.ndarray, 
                                max_period: int,
                                threshold: float) -> List[Dict[str, Any]]:
        """Detect periods using FFT spectral analysis."""
        try:
            # Apply FFT
            fft_values = fft(series - np.mean(series))
            freqs = fftfreq(len(series))
            
            # Get power spectrum
            power = np.abs(fft_values) ** 2
            
            # Convert to periods (avoid division by zero)
            positive_freqs = freqs[freqs > 0]
            periods = 1.0 / positive_freqs
            power_positive = power[freqs > 0]
            
            # Filter periods within range
            valid_mask = (periods >= 2) & (periods <= max_period)
            valid_periods = periods[valid_mask]
            valid_power = power_positive[valid_mask]
            
            # Normalize power
            if len(valid_power) > 0:
                valid_power = valid_power / np.max(valid_power)
                
                # Find significant periods
                significant_mask = valid_power > threshold
                significant_periods = valid_periods[significant_mask]
                significant_power = valid_power[significant_mask]
                
                # Create period list
                periods = []
                for i, (period, power) in enumerate(zip(significant_periods, significant_power)):
                    periods.append({
                        'period': int(round(period)),
                        'strength': float(power),
                        'method': 'fft',
                        'rank': i + 1
                    })
                
                # Sort by strength
                periods.sort(key=lambda x: x['strength'], reverse=True)
                return periods
            
            return []
            
        except Exception as e:
            logger.warning(f"FFT period detection failed: {str(e)}")
            return []
    
    async def _detect_periods_decomposition(self, 
                                          series: np.ndarray, 
                                          max_period: int) -> List[Dict[str, Any]]:
        """Detect periods using seasonal decomposition."""
        try:
            periods = []
            
            # Try common periods
            test_periods = [7, 30, 31, 90, 91, 365, 366]
            test_periods = [p for p in test_periods if p <= max_period and p < len(series) // 2]
            
            for period in test_periods:
                try:
                    # Perform seasonal decomposition
                    if len(series) >= 2 * period:
                        decomposition = seasonal_decompose(
                            pd.Series(series), 
                            model='additive', 
                            period=period,
                            extrapolate_trend='freq'
                        )
                        
                        # Calculate seasonal strength
                        seasonal_var = np.var(decomposition.seasonal.dropna())
                        residual_var = np.var(decomposition.resid.dropna())
                        
                        if residual_var > 0:
                            seasonal_strength = seasonal_var / (seasonal_var + residual_var)
                            
                            if seasonal_strength > 0.1:  # Minimum threshold
                                periods.append({
                                    'period': period,
                                    'strength': float(seasonal_strength),
                                    'method': 'decomposition',
                                    'rank': 0
                                })
                
                except Exception:
                    continue
            
            # Sort by strength
            periods.sort(key=lambda x: x['strength'], reverse=True)
            
            # Update ranks
            for i, period in enumerate(periods):
                period['rank'] = i + 1
            
            return periods
            
        except Exception as e:
            logger.warning(f"Decomposition period detection failed: {str(e)}")
            return []
    
    async def _consolidate_periods(self, 
                                 periods: List[Dict[str, Any]], 
                                 threshold: float) -> List[Dict[str, Any]]:
        """Consolidate periods from different methods."""
        if not periods:
            return []
        
        # Group similar periods (within ±2)
        period_groups = {}
        
        for period_info in periods:
            period = period_info['period']
            
            # Find existing group
            found_group = False
            for key in period_groups.keys():
                if abs(period - key) <= 2:
                    period_groups[key].append(period_info)
                    found_group = True
                    break
            
            if not found_group:
                period_groups[period] = [period_info]
        
        # Consolidate groups
        consolidated = []
        for group_periods in period_groups.values():
            if len(group_periods) >= 1:  # At least detected by one method
                # Average period and strength
                avg_period = int(np.mean([p['period'] for p in group_periods]))
                avg_strength = np.mean([p['strength'] for p in group_periods])
                methods = list(set([p['method'] for p in group_periods]))
                
                if avg_strength >= threshold:
                    consolidated.append({
                        'period': avg_period,
                        'strength': float(avg_strength),
                        'methods': methods,
                        'detection_count': len(group_periods),
                        'confidence': float(min(avg_strength * len(group_periods) / 3, 1.0))
                    })
        
        # Sort by confidence and strength
        consolidated.sort(key=lambda x: (x['confidence'], x['strength']), reverse=True)
        
        return consolidated
    
    async def analyze_multiple_seasons(self, 
                                     series: np.ndarray,
                                     periods: List[int],
                                     method: str = "stl") -> Dict[str, Any]:
        """
        Analyze multiple overlapping seasonal patterns.
        
        Args:
            series: Time series data
            periods: List of seasonal periods to analyze
            method: Analysis method ('stl', 'fourier', 'prophet')
            
        Returns:
            Dictionary with multiple seasonal analysis
        """
        try:
            seasonal_components = {}
            seasonal_strengths = {}
            
            if method == "stl":
                # Use STL for each period
                for period in periods:
                    if period < len(series) // 2:
                        try:
                            stl = AsyncSTLDecomposition(period=period)
                            await stl.fit(pd.Series(series))
                            components = stl.get_components()
                            
                            seasonal_comp = components['seasonal'].values
                            seasonal_components[f'period_{period}'] = seasonal_comp.tolist()
                            
                            # Calculate strength
                            seasonal_var = np.var(seasonal_comp)
                            residual_var = np.var(components['residual'].values)
                            strength = seasonal_var / (seasonal_var + residual_var)
                            seasonal_strengths[f'period_{period}'] = float(strength)
                            
                        except Exception as e:
                            logger.warning(f"STL analysis failed for period {period}: {str(e)}")
            
            elif method == "fourier":
                # Fourier analysis for each period
                for period in periods:
                    try:
                        # Extract Fourier components for specific period
                        freq = 1.0 / period
                        fft_values = fft(series)
                        freqs = fftfreq(len(series))
                        
                        # Find frequency bin closest to target
                        freq_idx = np.argmin(np.abs(freqs - freq))
                        
                        # Reconstruct seasonal component
                        seasonal_fft = np.zeros_like(fft_values)
                        seasonal_fft[freq_idx] = fft_values[freq_idx]
                        seasonal_fft[-freq_idx] = fft_values[-freq_idx]  # Symmetric
                        
                        seasonal_comp = np.real(np.fft.ifft(seasonal_fft))
                        seasonal_components[f'period_{period}'] = seasonal_comp.tolist()
                        
                        # Calculate strength
                        strength = np.var(seasonal_comp) / np.var(series)
                        seasonal_strengths[f'period_{period}'] = float(strength)
                        
                    except Exception as e:
                        logger.warning(f"Fourier analysis failed for period {period}: {str(e)}")
            
            return {
                'seasonal_components': seasonal_components,
                'seasonal_strengths': seasonal_strengths,
                'periods_analyzed': periods,
                'method': method,
                'total_seasonal_strength': float(sum(seasonal_strengths.values()))
            }
            
        except Exception as e:
            logger.error(f"Multiple seasons analysis failed: {str(e)}")
            raise AgentError(f"Multiple seasons analysis error: {str(e)}") from e
    
    async def model_holiday_effects(self, 
                                  series: np.ndarray,
                                  dates: List[str],
                                  country: str = "US",
                                  custom_events: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Model and analyze holiday and special event effects.
        
        Args:
            series: Time series data
            dates: Date index as strings
            country: Country code for holidays
            custom_events: Custom event dates
            
        Returns:
            Dictionary with holiday effects analysis
        """
        try:
            # Create DataFrame for Prophet
            df = pd.DataFrame({
                'ds': pd.to_datetime(dates),
                'y': series
            })
            
            # Initialize Prophet with holidays
            self.prophet_model = AsyncProphetModel(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            
            # Add country holidays if Prophet is available
            try:
                if hasattr(self.prophet_model.model, 'add_country_holidays'):
                    self.prophet_model.model.add_country_holidays(country_name=country)
            except Exception as e:
                logger.warning(f"Could not add country holidays: {str(e)}")
            
            # Add custom events
            if custom_events:
                custom_holidays = pd.DataFrame({
                    'holiday': 'custom_event',
                    'ds': pd.to_datetime(custom_events),
                })
                if hasattr(self.prophet_model.model, 'add_holidays'):
                    self.prophet_model.model.add_holidays(custom_holidays)
            
            # Fit model
            prophet_results = await self.prophet_model.fit(df)
            
            # Generate predictions to get components
            future = await asyncio.to_thread(
                self.prophet_model.model.make_future_dataframe, 
                periods=0
            )
            forecast = await asyncio.to_thread(
                self.prophet_model.model.predict, 
                future
            )
            
            # Extract holiday effects
            holiday_effects = {}
            for col in forecast.columns:
                if 'holiday' in col.lower() or col in ['holidays']:
                    holiday_effects[col] = forecast[col].tolist()
            
            # Calculate holiday impact
            holiday_impact = 0.0
            if holiday_effects:
                combined_effects = np.sum([np.array(effects) for effects in holiday_effects.values()], axis=0)
                holiday_impact = np.var(combined_effects) / np.var(series)
            
            return {
                'holiday_effects': holiday_effects,
                'holiday_impact': float(holiday_impact),
                'country': country,
                'custom_events_count': len(custom_events) if custom_events else 0,
                'prophet_results': prophet_results
            }
            
        except Exception as e:
            logger.error(f"Holiday effects modeling failed: {str(e)}")
            raise AgentError(f"Holiday effects modeling error: {str(e)}") from e
    
    async def validate_seasonal_stability(self, 
                                        seasonal_components: np.ndarray,
                                        period: int,
                                        method: str = "variance") -> Dict[str, Any]:
        """
        Validate stability of seasonal patterns over time.
        
        Args:
            seasonal_components: Seasonal component values
            period: Seasonal period
            method: Validation method
            
        Returns:
            Dictionary with stability validation results
        """
        try:
            if len(seasonal_components) < 2 * period:
                return {
                    'is_stable': False,
                    'reason': 'Insufficient data for stability analysis',
                    'stability_score': 0.0
                }
            
            stability_metrics = {}
            
            if method == "variance":
                # Split into seasonal cycles and compare variance
                cycles = []
                for i in range(0, len(seasonal_components) - period + 1, period):
                    cycle = seasonal_components[i:i + period]
                    if len(cycle) == period:
                        cycles.append(cycle)
                
                if len(cycles) >= 2:
                    cycle_variances = [np.var(cycle) for cycle in cycles]
                    variance_stability = 1.0 - (np.std(cycle_variances) / np.mean(cycle_variances))
                    stability_metrics['variance_stability'] = float(max(0, variance_stability))
            
            elif method == "correlation":
                # Calculate correlation between consecutive cycles
                correlations = []
                for i in range(period, len(seasonal_components) - period + 1, period):
                    cycle1 = seasonal_components[i-period:i]
                    cycle2 = seasonal_components[i:i+period]
                    
                    if len(cycle1) == len(cycle2) == period:
                        corr = np.corrcoef(cycle1, cycle2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    stability_metrics['correlation_stability'] = float(avg_correlation)
            
            elif method == "trend":
                # Check for trend in seasonal amplitude
                amplitudes = []
                for i in range(0, len(seasonal_components) - period + 1, period):
                    cycle = seasonal_components[i:i + period]
                    if len(cycle) == period:
                        amplitude = np.max(cycle) - np.min(cycle)
                        amplitudes.append(amplitude)
                
                if len(amplitudes) >= 3:
                    # Check if amplitude is changing over time
                    x = np.arange(len(amplitudes))
                    slope = np.polyfit(x, amplitudes, 1)[0]
                    relative_slope = abs(slope) / np.mean(amplitudes) if np.mean(amplitudes) > 0 else 0
                    trend_stability = 1.0 - min(relative_slope, 1.0)
                    stability_metrics['trend_stability'] = float(trend_stability)
            
            # Overall stability score
            if stability_metrics:
                overall_stability = np.mean(list(stability_metrics.values()))
                is_stable = overall_stability > 0.7  # Threshold for stability
            else:
                overall_stability = 0.0
                is_stable = False
            
            return {
                'is_stable': bool(is_stable),
                'stability_score': float(overall_stability),
                'stability_metrics': stability_metrics,
                'method': method,
                'period': period,
                'threshold': 0.7
            }
            
        except Exception as e:
            logger.error(f"Seasonal stability validation failed: {str(e)}")
            raise AgentError(f"Seasonal stability validation error: {str(e)}") from e
    
    async def analyze_comprehensive_seasonality(self, 
                                              series: pd.Series,
                                              use_llm_reasoning: bool = True) -> Dict[str, Any]:
        """
        Comprehensive seasonality analysis using multiple methods and LLM reasoning.
        
        Args:
            series: Time series data
            use_llm_reasoning: Whether to use LLM for analysis reasoning
            
        Returns:
            Complete seasonality analysis results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"SeasonalityAgent: Starting comprehensive seasonality analysis for {len(series)} observations")
            
            # 1. Detect seasonal periods
            period_detection = await self.detect_seasonality_periods(series.values)
            self.detected_periods = period_detection['detected_periods']
            
            # 2. Analyze multiple seasons if periods detected
            multiple_seasons = {}
            if self.detected_periods:
                periods = [p['period'] for p in self.detected_periods[:5]]  # Top 5 periods
                multiple_seasons = await self.analyze_multiple_seasons(series.values, periods)
                self.seasonal_components = multiple_seasons.get('seasonal_components', {})
            
            # 3. Model holiday effects (if date index available)
            holiday_effects = {}
            if isinstance(series.index, pd.DatetimeIndex):
                try:
                    dates = series.index.strftime('%Y-%m-%d').tolist()
                    holiday_effects = await self.model_holiday_effects(series.values, dates)
                except Exception as e:
                    logger.warning(f"Holiday effects analysis failed: {str(e)}")
                    holiday_effects = {'error': str(e)}
            
            # 4. Validate seasonal stability
            stability_results = {}
            for period_info in self.detected_periods[:3]:  # Top 3 periods
                period = period_info['period']
                period_key = f'period_{period}'
                
                if period_key in self.seasonal_components:
                    seasonal_comp = np.array(self.seasonal_components[period_key])
                    stability = await self.validate_seasonal_stability(seasonal_comp, period)
                    stability_results[period_key] = stability
            
            # 5. LLM-based reasoning (if available)
            llm_insights = {}
            if use_llm_reasoning and self.llm_generator:
                try:
                    # Generate insights using LLM
                    n_periods = len(self.detected_periods)
                    main_periods = [p['period'] for p in self.detected_periods[:3]]
                    main_strengths = [p['strength'] for p in self.detected_periods[:3]]
                    
                    prompt = f"""
                    Analyze the following seasonality detection results and provide insights:
                    
                    Number of Seasonal Periods Detected: {n_periods}
                    Main Periods: {main_periods}
                    Period Strengths: {main_strengths}
                    Holiday Impact: {holiday_effects.get('holiday_impact', 0):.4f}
                    
                    Provide a comprehensive analysis including:
                    1. Seasonal pattern assessment
                    2. Business implications of detected seasonality
                    3. Forecasting considerations
                    4. Seasonal adjustment recommendations
                    """
                    
                    llm_response = await asyncio.to_thread(self.llm_generator.generate, prompt)
                    llm_insights = {'analysis': llm_response, 'timestamp': datetime.now().isoformat()}
                    
                except Exception as e:
                    logger.warning(f"LLM reasoning failed: {str(e)}")
                    llm_insights = {'error': str(e)}
            
            # Compile results
            self.seasonality_results = {
                'period_detection': period_detection,
                'multiple_seasons': multiple_seasons,
                'holiday_effects': holiday_effects,
                'stability_results': stability_results,
                'llm_insights': llm_insights
            }
            
            analysis_time = asyncio.get_event_loop().time() - start_time
            
            self.is_fitted = True
            
            logger.info(f"SeasonalityAgent: Comprehensive analysis completed in {analysis_time:.2f}s")
            
            return {
                'seasonality_analysis': self.seasonality_results,
                'metadata': {
                    'analysis_time': analysis_time,
                    'series_length': len(series),
                    'periods_detected': len(self.detected_periods),
                    'llm_model': self.llm_model,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_msg = f"Comprehensive seasonality analysis failed: {str(e)}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e
    
    async def forecast_seasonality(self, steps: int = 30) -> Dict[str, Any]:
        """
        Generate seasonal forecasts for detected patterns.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Dictionary with seasonal forecasts
        """
        if not self.is_fitted:
            raise AgentError("SeasonalityAgent must be fitted before forecasting")
        
        try:
            seasonal_forecasts = {}
            
            # Generate forecasts for each detected seasonal component
            for period_info in self.detected_periods:
                period = period_info['period']
                period_key = f'period_{period}'
                
                if period_key in self.seasonal_components:
                    seasonal_comp = np.array(self.seasonal_components[period_key])
                    
                    # Simple seasonal forecast: repeat the pattern
                    n_complete_cycles = steps // period
                    remaining_steps = steps % period
                    
                    forecast = []
                    
                    # Add complete cycles
                    if len(seasonal_comp) >= period:
                        last_cycle = seasonal_comp[-period:]
                        for _ in range(n_complete_cycles):
                            forecast.extend(last_cycle)
                        
                        # Add remaining steps
                        if remaining_steps > 0:
                            forecast.extend(last_cycle[:remaining_steps])
                    
                    seasonal_forecasts[period_key] = {
                        'forecast': forecast,
                        'period': period,
                        'strength': period_info['strength']
                    }
            
            # Combined seasonal forecast (weighted by strength)
            if seasonal_forecasts:
                combined_forecast = np.zeros(steps)
                total_weight = 0.0
                
                for period_key, forecast_data in seasonal_forecasts.items():
                    weight = forecast_data['strength']
                    forecast = np.array(forecast_data['forecast'][:steps])
                    
                    if len(forecast) == steps:
                        combined_forecast += weight * forecast
                        total_weight += weight
                
                if total_weight > 0:
                    combined_forecast /= total_weight
                    seasonal_forecasts['combined_seasonal'] = {
                        'forecast': combined_forecast.tolist(),
                        'method': 'weighted_combination',
                        'total_weight': total_weight
                    }
            
            return {
                'seasonal_forecasts': seasonal_forecasts,
                'steps': steps,
                'forecast_metadata': {
                    'n_seasonal_components': len(seasonal_forecasts) - (1 if 'combined_seasonal' in seasonal_forecasts else 0),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_msg = f"Seasonal forecasting failed: {str(e)}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of seasonality analysis results."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        period_detection = self.seasonality_results.get('period_detection', {})
        holiday_effects = self.seasonality_results.get('holiday_effects', {})
        
        return {
            'status': 'fitted',
            'periods_detected': len(self.detected_periods),
            'main_periods': [p['period'] for p in self.detected_periods[:3]],
            'strongest_period': self.detected_periods[0]['period'] if self.detected_periods else None,
            'seasonal_strength': self.detected_periods[0]['strength'] if self.detected_periods else 0.0,
            'holiday_impact': holiday_effects.get('holiday_impact', 0.0),
            'has_multiple_seasonality': len(self.detected_periods) > 1
        }

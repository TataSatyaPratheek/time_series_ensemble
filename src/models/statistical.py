"""
Statistical models for time series forecasting with async support.
Includes ARIMA, SARIMA, Prophet, and seasonal decomposition with comprehensive configuration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
import warnings

# Statistical libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    Prophet = None
    HAS_PROPHET = False

from src.config import settings
from src.utils.exceptions import ModelError
from src.utils.logging import get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore')  # Suppress statsmodels warnings


class ModelConfig(BaseModel):
    """Base configuration for statistical models."""
    model_type: str
    model_params: Dict[str, Any] = Field(default_factory=dict)
    fit_params: Dict[str, Any] = Field(default_factory=dict)
    validation_enabled: bool = True
    uncertainty_quantification: bool = True


class ModelResult(BaseModel):
    """Standard result format for all models."""
    model_name: str
    predictions: List[float]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None
    model_metadata: Dict[str, Any] = Field(default_factory=dict)
    fit_metrics: Dict[str, float] = Field(default_factory=dict)
    training_time: float = 0.0
    prediction_time: float = 0.0


class AsyncARIMAModel:
    """
    Async ARIMA/SARIMA model with comprehensive configuration and validation.
    Optimized for concurrent execution in multi-agent systems.
    """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 trend: Optional[str] = None,
                 enforce_stationarity: bool = True,
                 enforce_invertibility: bool = True,
                 **kwargs):
        """
        Initialize ARIMA model with configuration.
        
        Args:
            order: (p, d, q) parameters for ARIMA
            seasonal_order: (P, D, Q, s) parameters for seasonal ARIMA
            trend: Trend component ('n', 'c', 't', 'ct')
            enforce_stationarity: Enforce stationarity of autoregressive component
            enforce_invertibility: Enforce invertibility of moving average component
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.kwargs = kwargs
        
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.fit_summary = None
        
    async def fit(self, 
                  series: pd.Series, 
                  validate_input: bool = True,
                  **fit_kwargs) -> Dict[str, Any]:
        """
        Asynchronously fit ARIMA model to time series data.
        
        Args:
            series: Time series data
            validate_input: Whether to validate input data
            **fit_kwargs: Additional fitting parameters
            
        Returns:
            Dictionary with fitting results and metrics
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            if validate_input:
                await self._validate_series(series)
            
            logger.info(f"Fitting ARIMA{self.order} model to {len(series)} observations")
            
            # Determine if seasonal ARIMA is needed
            is_seasonal = any(self.seasonal_order[:3])
            
            if is_seasonal:
                model_class = SARIMAX
                model_args = {
                    'order': self.order,
                    'seasonal_order': self.seasonal_order,
                    'trend': self.trend,
                    'enforce_stationarity': self.enforce_stationarity,
                    'enforce_invertibility': self.enforce_invertibility,
                    **self.kwargs
                }
            else:
                model_class = ARIMA
                model_args = {
                    'order': self.order,
                    'trend': self.trend,
                    'enforce_stationarity': self.enforce_stationarity,
                    'enforce_invertibility': self.enforce_invertibility,
                    **self.kwargs
                }
            
            # Fit model in thread pool to avoid blocking
            self.model = await asyncio.to_thread(model_class, series, **model_args)
            
            # Default fitting parameters
            default_fit_params = {
                'method': 'lbfgs', # Default method
                # 'maxiter': 50, # Removed as it causes issues in some statsmodels versions
                'disp': False
            }
            default_fit_params.update(fit_kwargs)
            
            self.fitted_model = await asyncio.to_thread(
                self.model.fit, **default_fit_params
            )
            
            self.is_fitted = True
            self.fit_summary = self.fitted_model.summary()
            
            # Calculate fit metrics
            fit_time = asyncio.get_event_loop().time() - start_time
            
            fit_metrics = {
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'hqic': float(self.fitted_model.hqic),
                'llf': float(self.fitted_model.llf),
                'fit_time': fit_time,
                'params': self.fitted_model.params.to_dict(),
                'converged': bool(self.fitted_model.mle_retvals['converged'])
            }
            
            logger.info(f"ARIMA model fitted successfully. AIC: {fit_metrics['aic']:.2f}, BIC: {fit_metrics['bic']:.2f}")
            return fit_metrics
            
        except Exception as e:
            error_msg = f"ARIMA model fitting failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    async def predict(self, 
                     steps: int = 1,
                     start: Optional[int] = None,
                     end: Optional[int] = None,
                     alpha: float = 0.05,
                     return_conf_int: bool = True) -> ModelResult:
        """
        Generate predictions with confidence intervals.
        
        Args:
            steps: Number of steps to forecast
            start: Start index for prediction
            end: End index for prediction
            alpha: Significance level for confidence intervals
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            ModelResult with predictions and metadata
        """
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Generate forecast
            if start is not None and end is not None:
                forecast = await asyncio.to_thread(
                    self.fitted_model.predict, start=start, end=end
                )
                if return_conf_int:
                    forecast_ci = await asyncio.to_thread(
                        self.fitted_model.get_prediction, start=start, end=end
                    )
                    conf_int = forecast_ci.conf_int(alpha=alpha)
            else:
                forecast = await asyncio.to_thread(
                    self.fitted_model.forecast, steps=steps
                )
                if return_conf_int:
                    forecast_result = await asyncio.to_thread(
                        self.fitted_model.get_forecast, steps=steps
                    )
                    conf_int = forecast_result.conf_int(alpha=alpha)
            
            prediction_time = asyncio.get_event_loop().time() - start_time
            
            # Format confidence intervals
            confidence_intervals = None
            if return_conf_int:
                confidence_intervals = [
                    (float(conf_int.iloc[i, 0]), float(conf_int.iloc[i, 1]))
                    for i in range(len(conf_int))
                ]
            
            result = ModelResult(
                model_name=f"ARIMA{self.order}",
                predictions=forecast.tolist(),
                confidence_intervals=confidence_intervals,
                model_metadata={
                    'order': self.order,
                    'seasonal_order': self.seasonal_order,
                    'trend': self.trend,
                    'n_predictions': len(forecast),
                    'confidence_level': 1 - alpha
                },
                fit_metrics=self._get_current_metrics(),
                prediction_time=prediction_time
            )
            
            logger.info(f"Generated {len(forecast)} ARIMA predictions in {prediction_time:.3f}s")
            return result
            
        except Exception as e:
            error_msg = f"ARIMA prediction failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    async def _validate_series(self, series: pd.Series) -> None:
        """Validate input time series data."""
        if series.empty:
            raise ValueError("Time series cannot be empty")
        
        if series.isnull().any():
            logger.warning("Time series contains missing values")
        
        if len(series) < max(self.order) + max(self.seasonal_order[:3]) + 1:
            raise ValueError("Insufficient data for ARIMA model parameters")
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current model metrics if fitted."""
        if not self.is_fitted:
            return {}
        
        return {
            'aic': float(self.fitted_model.aic),
            'bic': float(self.fitted_model.bic),
            'hqic': float(self.fitted_model.hqic),
            'llf': float(self.fitted_model.llf)
        }


class AsyncProphetModel:
    """
    Async Prophet model wrapper with comprehensive configuration.
    """
    
    def __init__(self, 
                 growth: str = 'linear',
                 seasonality_mode: str = 'additive',
                 n_changepoints: int = 25,
                 yearly_seasonality: Union[bool, str] = 'auto',
                 weekly_seasonality: Union[bool, str] = 'auto',
                 daily_seasonality: Union[bool, str] = 'auto',
                 **kwargs):
        """
        Initialize Prophet model with configuration.
        
        Args:
            growth: 'linear' or 'logistic' growth
            seasonality_mode: 'additive' or 'multiplicative'
            n_changepoints: Number of potential changepoints
            yearly_seasonality: Yearly seasonality handling
            weekly_seasonality: Weekly seasonality handling  
            daily_seasonality: Daily seasonality handling
        """
        if not HAS_PROPHET:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")
        
        self.config = {
            'growth': growth,
            'seasonality_mode': seasonality_mode,
            'n_changepoints': n_changepoints,
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            **kwargs
        }
        
        self.model = None
        self.is_fitted = False
        self.fit_metrics = {}
    
    async def fit(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Fit Prophet model to data.
        
        Args:
            df: DataFrame with 'ds' (date) and 'y' (value) columns
            **kwargs: Additional fitting parameters
            
        Returns:
            Dictionary with fitting results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate input format
            if 'ds' not in df.columns or 'y' not in df.columns:
                raise ValueError("DataFrame must have 'ds' and 'y' columns")
            
            logger.info(f"Fitting Prophet model to {len(df)} observations")
            
            # Initialize model
            self.model = Prophet(**self.config)
            
            # Fit model in thread pool
            await asyncio.to_thread(self.model.fit, df, **kwargs)
            
            self.is_fitted = True
            fit_time = asyncio.get_event_loop().time() - start_time
            
            # Calculate basic metrics
            self.fit_metrics = {
                'fit_time': fit_time,
                'n_observations': len(df),
                'changepoints': len(self.model.changepoints),
                'seasonality_components': list(self.model.seasonalities.keys())
            }
            
            logger.info(f"Prophet model fitted successfully in {fit_time:.2f}s")
            return self.fit_metrics
            
        except Exception as e:
            error_msg = f"Prophet model fitting failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    async def predict(self, 
                     periods: int = 30,
                     freq: str = 'D',
                     include_history: bool = True) -> ModelResult:
        """
        Generate Prophet predictions.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of predictions
            include_history: Include historical predictions
            
        Returns:
            ModelResult with predictions and components
        """
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create future dataframe
            future = await asyncio.to_thread(
                self.model.make_future_dataframe, periods=periods, freq=freq, include_history=include_history
            )
            
            # Generate predictions
            forecast = await asyncio.to_thread(self.model.predict, future)
            
            prediction_time = asyncio.get_event_loop().time() - start_time
            
            # Extract predictions and confidence intervals
            if include_history:
                predictions = forecast['yhat'].tolist()
                conf_intervals = [
                    (float(forecast.iloc[i]['yhat_lower']), float(forecast.iloc[i]['yhat_upper']))
                    for i in range(len(forecast))
                ]
            else:
                predictions = forecast['yhat'][-periods:].tolist()
                conf_intervals = [
                    (float(forecast.iloc[i]['yhat_lower']), float(forecast.iloc[i]['yhat_upper']))
                    for i in range(len(forecast) - periods, len(forecast))
                ]
            
            result = ModelResult(
                model_name="Prophet",
                predictions=predictions,
                confidence_intervals=conf_intervals,
                model_metadata={
                    'growth': self.config['growth'],
                    'seasonality_mode': self.config['seasonality_mode'],
                    'n_changepoints': self.config['n_changepoints'],
                    'periods_forecasted': periods,
                    'frequency': freq,
                    'components': {
                        'trend': forecast['trend'].tolist()[-periods:] if not include_history else forecast['trend'].tolist(),
                        'yearly': forecast.get('yearly', pd.Series([0] * len(forecast))).tolist()[-periods:] if not include_history else forecast.get('yearly', pd.Series([0] * len(forecast))).tolist(),
                        'weekly': forecast.get('weekly', pd.Series([0] * len(forecast))).tolist()[-periods:] if not include_history else forecast.get('weekly', pd.Series([0] * len(forecast))).tolist()
                    }
                },
                fit_metrics=self.fit_metrics,
                prediction_time=prediction_time
            )
            
            logger.info(f"Generated {periods} Prophet predictions in {prediction_time:.3f}s")
            return result
            
        except Exception as e:
            error_msg = f"Prophet prediction failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e


class AsyncSTLDecomposition:
    """
    Async STL decomposition with trend and seasonal component analysis.
    """
    
    def __init__(self, 
                 period: Optional[int] = None,
                 seasonal: int = 7,
                 trend: Optional[int] = None,
                 robust: bool = False):
        """
        Initialize STL decomposition.
        
        Args:
            period: Period of seasonal component
            seasonal: Length of seasonal smoother
            trend: Length of trend smoother
            robust: Use robust fitting
        """
        self.period = period
        self.seasonal = seasonal
        self.trend = trend
        self.robust = robust
        
        self.decomposition = None
        self.is_fitted = False
    
    async def fit(self, series: pd.Series) -> Dict[str, Any]:
        """
        Fit STL decomposition to time series.
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary with decomposition results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Performing STL decomposition on {len(series)} observations")
            
            # Perform STL decomposition
            stl = STL(
                series, 
                period=self.period,
                seasonal=self.seasonal,
                trend=self.trend,
                robust=self.robust
            )
            
            self.decomposition = await asyncio.to_thread(stl.fit)
            self.is_fitted = True
            
            fit_time = asyncio.get_event_loop().time() - start_time
            
            # Calculate component statistics
            seasonal_strength = np.var(self.decomposition.seasonal) / np.var(series)
            trend_strength = np.var(self.decomposition.trend) / np.var(series)
            
            results = {
                'fit_time': fit_time,
                'seasonal_strength': float(seasonal_strength),
                'trend_strength': float(trend_strength),
                'residual_variance': float(np.var(self.decomposition.resid)),
                'period_detected': self.period
            }
            
            logger.info(f"STL decomposition completed in {fit_time:.2f}s")
            return results
            
        except Exception as e:
            error_msg = f"STL decomposition failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    def get_components(self) -> Dict[str, pd.Series]:
        """Get decomposition components."""
        if not self.is_fitted:
            raise ModelError("STL decomposition must be fitted first")
        
        return {
            'trend': self.decomposition.trend,
            'seasonal': self.decomposition.seasonal,
            'residual': self.decomposition.resid,
            'observed': self.decomposition.observed
        }
    
    async def forecast_trend(self, steps: int = 10) -> List[float]:
        """Simple trend extrapolation."""
        if not self.is_fitted:
            raise ModelError("STL decomposition must be fitted first")
        
        trend = self.decomposition.trend.dropna()
        
        # Simple linear extrapolation
        x = np.arange(len(trend))
        y = trend.values
        
        # Fit linear trend
        coeffs = await asyncio.to_thread(np.polyfit, x, y, 1)
        
        # Extrapolate
        future_x = np.arange(len(trend), len(trend) + steps)
        trend_forecast = np.polyval(coeffs, future_x)
        
        return trend_forecast.tolist()


class AsyncExponentialSmoothing:
    """
    Async Exponential Smoothing (Holt-Winters) model.
    """
    
    def __init__(self,
                 trend: Optional[str] = 'add',
                 seasonal: Optional[str] = 'add',
                 seasonal_periods: Optional[int] = None,
                 damped_trend: bool = False,
                 **kwargs):
        """
        Initialize Exponential Smoothing model.
        
        Args:
            trend: Trend component ('add', 'mul', None)
            seasonal: Seasonal component ('add', 'mul', None)
            seasonal_periods: Number of periods in season
            damped_trend: Use damped trend
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.kwargs = kwargs
        
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
    
    async def fit(self, series: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit Exponential Smoothing model."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Fitting Exponential Smoothing model to {len(series)} observations")
            
            self.model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend,
                **self.kwargs
            )
            
            self.fitted_model = await asyncio.to_thread(self.model.fit, **kwargs)
            self.is_fitted = True
            
            fit_time = asyncio.get_event_loop().time() - start_time
            
            results = {
                'fit_time': fit_time,
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'sse': float(self.fitted_model.sse),
                'params': self.fitted_model.params.to_dict() if isinstance(self.fitted_model.params, pd.Series) else self.fitted_model.params
            }
            
            logger.info(f"Exponential Smoothing fitted successfully in {fit_time:.2f}s")
            return results
            
        except Exception as e:
            error_msg = f"Exponential Smoothing fitting failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    async def predict(self, steps: int = 10) -> ModelResult:
        """Generate predictions."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before prediction")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            forecast = await asyncio.to_thread(self.fitted_model.forecast, steps)
            prediction_time = asyncio.get_event_loop().time() - start_time
            
            result = ModelResult(
                model_name="ExponentialSmoothing",
                predictions=forecast.tolist(),
                model_metadata={
                    'trend': self.trend,
                    'seasonal': self.seasonal,
                    'seasonal_periods': self.seasonal_periods,
                    'damped_trend': self.damped_trend,
                    'steps_forecasted': steps
                },
                fit_metrics={'aic': float(self.fitted_model.aic), 'bic': float(self.fitted_model.bic)},
                prediction_time=prediction_time
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Exponential Smoothing prediction failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e


# Utility functions for model selection and validation
async def auto_arima_selection(series: pd.Series, 
                              max_p: int = 5, 
                              max_d: int = 2, 
                              max_q: int = 5,
                              seasonal: bool = True,
                              m: int = 1) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """
    Automatic ARIMA order selection using information criteria.
    
    Args:
        series: Time series data
        max_p: Maximum AR order
        max_d: Maximum differencing order
        max_q: Maximum MA order
        seasonal: Include seasonal terms
        m: Seasonal period
        
    Returns:
        Best ARIMA orders (non-seasonal, seasonal)
    """
    try:
        # Use pmdarima if available, fallback to grid search
        try:
            from pmdarima import auto_arima
            
            auto_model = await asyncio.to_thread(
                auto_arima,
                series,
                start_p=0, start_q=0,
                max_p=max_p, max_d=max_d, max_q=max_q,
                seasonal=seasonal, m=m,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            return auto_model.order, auto_model.seasonal_order
            
        except ImportError:
            logger.warning("pmdarima not available, using grid search")
            return await _grid_search_arima(series, max_p, max_d, max_q, seasonal, m)
            
    except Exception as e:
        logger.error(f"Auto ARIMA selection failed: {str(e)}")
        return (1, 1, 1), (0, 0, 0, 0)


async def _grid_search_arima(series: pd.Series, 
                           max_p: int, 
                           max_d: int, 
                           max_q: int,
                           seasonal: bool,
                           m: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """Grid search for best ARIMA parameters."""
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal_order = (0, 0, 0, 0)
    
    # Grid search over parameter space
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if seasonal:
                    for P in range(2):
                        for D in range(2):
                            for Q in range(2):
                                try:
                                    model = AsyncARIMAModel(
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, m)
                                    )
                                    metrics = await model.fit(series)
                                    
                                    if metrics['aic'] < best_aic:
                                        best_aic = metrics['aic']
                                        best_order = (p, d, q)
                                        best_seasonal_order = (P, D, Q, m)
                                        
                                except:
                                    continue
                else:
                    try:
                        model = AsyncARIMAModel(order=(p, d, q))
                        metrics = await model.fit(series)
                        
                        if metrics['aic'] < best_aic:
                            best_aic = metrics['aic']
                            best_order = (p, d, q)
                            
                    except:
                        continue
    
    return best_order, best_seasonal_order

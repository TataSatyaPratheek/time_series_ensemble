"""
Async Time Series Preprocessor
Comprehensive preprocessing pipeline for time series data with async support.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.signal import savgol_filter
import warnings
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from ..config import settings
from ..utils.exceptions import PreprocessingError, ValidationError
from ..utils.logging import get_logger
from .loader import TimeSeriesMetadata

logger = get_logger(__name__)


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing operations."""
    handle_missing: bool = True
    missing_method: str = Field(default='interpolation', 
                               regex=r'^(interpolation|forward_fill|backward_fill|mean|median|drop)$')
    
    detect_outliers: bool = True
    outlier_method: str = Field(default='iqr', 
                               regex=r'^(iqr|zscore|isolation_forest|modified_zscore)$')
    outlier_threshold: float = Field(default=1.5, gt=0)
    outlier_action: str = Field(default='clip', 
                               regex=r'^(clip|remove|transform|flag)$')
    
    scale_data: bool = True
    scaling_method: str = Field(default='standard', 
                               regex=r'^(standard|minmax|robust|none)$')
    
    smooth_data: bool = False
    smoothing_method: str = Field(default='savgol', 
                                 regex=r'^(savgol|rolling|exponential)$')
    smoothing_window: int = Field(default=7, gt=0)
    
    resample_data: bool = False
    target_frequency: Optional[str] = None
    aggregation_method: str = Field(default='mean', 
                                   regex=r'^(mean|sum|median|min|max|first|last)$')
    
    feature_engineering: bool = True
    lag_features: List[int] = Field(default=[1, 7, 30])
    rolling_features: List[int] = Field(default=[7, 14, 30])
    diff_features: List[int] = Field(default=[1])


class PreprocessingReport(BaseModel):
    """Report of preprocessing operations performed."""
    timestamp: datetime = Field(default_factory=datetime.now)
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    operations_performed: List[str] = Field(default_factory=list)
    missing_values_handled: int = 0
    outliers_detected: int = 0
    outliers_handled: int = 0
    data_quality_before: float = 0.0
    data_quality_after: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_seconds: float = 0.0


class AsyncTimeSeriesPreprocessor:
    """
    Comprehensive async time series preprocessor with configurable operations.
    Optimized for concurrent processing of multiple time series.
    """
    
    def __init__(self, 
                 config: Optional[PreprocessingConfig] = None,
                 max_concurrent_operations: int = 4):
        """
        Initialize the async preprocessor.
        
        Args:
            config: Preprocessing configuration
            max_concurrent_operations: Maximum concurrent operations
        """
        self.config = config or PreprocessingConfig()
        self.max_concurrent_operations = min(max_concurrent_operations, settings.MAX_WORKERS)
        self.semaphore = asyncio.Semaphore(self.max_concurrent_operations)
        self.scalers: Dict[str, Any] = {}
        
    async def handle_missing_values(self, 
                                  df: pd.DataFrame, 
                                  method: str = 'interpolation',
                                  **kwargs) -> Tuple[pd.DataFrame, int]:
        """
        Handle missing values in time series data.
        
        Args:
            df: Input DataFrame
            method: Method for handling missing values
            **kwargs: Additional method-specific parameters
            
        Returns:
            Tuple of (processed DataFrame, number of missing values handled)
        """
        async with self.semaphore:
            try:
                original_missing = df.isnull().sum().sum()
                
                if original_missing == 0:
                    return df.copy(), 0
                
                logger.info(f"Handling {original_missing} missing values using method: {method}")
                
                if method == 'interpolation':
                    # Time series interpolation
                    interpolation_method = kwargs.get('interpolation_method', 'time')
                    limit = kwargs.get('limit', None)
                    processed_df = await asyncio.to_thread(
                        df.interpolate,
                        method=interpolation_method,
                        limit=limit,
                        limit_direction='both'
                    )
                    
                elif method == 'forward_fill':
                    limit = kwargs.get('limit', None)
                    processed_df = await asyncio.to_thread(df.fillna, method='ffill', limit=limit)
                    
                elif method == 'backward_fill':
                    limit = kwargs.get('limit', None)
                    processed_df = await asyncio.to_thread(df.fillna, method='bfill', limit=limit)
                    
                elif method == 'mean':
                    processed_df = await asyncio.to_thread(df.fillna, df.mean())
                    
                elif method == 'median':
                    processed_df = await asyncio.to_thread(df.fillna, df.median())
                    
                elif method == 'drop':
                    processed_df = await asyncio.to_thread(df.dropna)
                    
                else:
                    raise ValueError(f"Unsupported missing value method: {method}")
                
                handled_missing = original_missing - processed_df.isnull().sum().sum()
                logger.info(f"Successfully handled {handled_missing} missing values")
                
                return processed_df, handled_missing
                
            except Exception as e:
                logger.error(f"Failed to handle missing values: {str(e)}")
                raise PreprocessingError(f"Missing value handling failed: {str(e)}") from e
    
    async def detect_outliers(self, 
                            df: pd.DataFrame, 
                            method: str = 'iqr',
                            threshold: float = 1.5,
                            **kwargs) -> pd.DataFrame:
        """
        Detect outliers in time series data.
        
        Args:
            df: Input DataFrame
            method: Outlier detection method
            threshold: Threshold for outlier detection
            **kwargs: Additional method-specific parameters
            
        Returns:
            Boolean DataFrame indicating outliers
        """
        async with self.semaphore:
            try:
                logger.info(f"Detecting outliers using method: {method}")
                
                if method == 'iqr':
                    def iqr_outliers(data, threshold):
                        Q1 = data.quantile(0.25)
                        Q3 = data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        return (data < lower_bound) | (data > upper_bound)
                    
                    outlier_mask = await asyncio.to_thread(
                        df.apply, lambda col: iqr_outliers(col, threshold) if col.dtype in ['int64', 'float64'] else False
                    )
                    
                elif method == 'zscore':
                    z_threshold = kwargs.get('z_threshold', 3)
                    z_scores = await asyncio.to_thread(
                        df.apply, lambda col: np.abs(stats.zscore(col.dropna())) if col.dtype in ['int64', 'float64'] else 0
                    )
                    outlier_mask = z_scores > z_threshold
                    
                elif method == 'modified_zscore':
                    def modified_zscore(data):
                        median = np.median(data)
                        mad = np.median(np.abs(data - median))
                        modified_z_scores = 0.6745 * (data - median) / mad
                        return np.abs(modified_z_scores)
                    
                    mad_threshold = kwargs.get('mad_threshold', 3.5)
                    mad_scores = await asyncio.to_thread(
                        df.apply, lambda col: modified_zscore(col.dropna()) if col.dtype in ['int64', 'float64'] else 0
                    )
                    outlier_mask = mad_scores > mad_threshold
                    
                elif method == 'isolation_forest':
                    from sklearn.ensemble import IsolationForest
                    contamination = kwargs.get('contamination', 0.1)
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
                    
                    for col in numeric_cols:
                        if not df[col].isnull().all():
                            iso_forest = IsolationForest(contamination=contamination, random_state=42)
                            col_data = df[col].dropna().values.reshape(-1, 1)
                            outliers = await asyncio.to_thread(iso_forest.fit_predict, col_data)
                            outlier_mask.loc[df[col].dropna().index, col] = outliers == -1
                
                else:
                    raise ValueError(f"Unsupported outlier detection method: {method}")
                
                total_outliers = outlier_mask.sum().sum()
                logger.info(f"Detected {total_outliers} outliers using {method} method")
                
                return outlier_mask
                
            except Exception as e:
                logger.error(f"Failed to detect outliers: {str(e)}")
                raise PreprocessingError(f"Outlier detection failed: {str(e)}") from e
    
    async def handle_outliers(self, 
                            df: pd.DataFrame, 
                            outlier_mask: pd.DataFrame,
                            action: str = 'clip',
                            **kwargs) -> Tuple[pd.DataFrame, int]:
        """
        Handle detected outliers.
        
        Args:
            df: Input DataFrame
            outlier_mask: Boolean mask indicating outliers
            action: Action to take for outliers
            **kwargs: Additional action-specific parameters
            
        Returns:
            Tuple of (processed DataFrame, number of outliers handled)
        """
        async with self.semaphore:
            try:
                total_outliers = outlier_mask.sum().sum()
                
                if total_outliers == 0:
                    return df.copy(), 0
                
                logger.info(f"Handling {total_outliers} outliers using action: {action}")
                processed_df = df.copy()
                
                if action == 'clip':
                    # Clip outliers to reasonable bounds
                    for col in df.select_dtypes(include=[np.number]).columns:
                        if outlier_mask[col].any():
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            processed_df.loc[outlier_mask[col], col] = np.where(
                                df.loc[outlier_mask[col], col] < lower_bound,
                                lower_bound,
                                upper_bound
                            )
                
                elif action == 'remove':
                    # Remove rows with outliers
                    rows_with_outliers = outlier_mask.any(axis=1)
                    processed_df = processed_df[~rows_with_outliers]
                
                elif action == 'transform':
                    # Log transform or other transformation
                    transform_method = kwargs.get('transform_method', 'log')
                    
                    for col in df.select_dtypes(include=[np.number]).columns:
                        if outlier_mask[col].any():
                            if transform_method == 'log':
                                # Ensure positive values for log transform
                                min_val = df[col].min()
                                if min_val <= 0:
                                    processed_df[col] = processed_df[col] - min_val + 1
                                processed_df.loc[outlier_mask[col], col] = np.log(
                                    processed_df.loc[outlier_mask[col], col]
                                )
                            elif transform_method == 'sqrt':
                                processed_df.loc[outlier_mask[col], col] = np.sqrt(
                                    np.abs(processed_df.loc[outlier_mask[col], col])
                                )
                
                elif action == 'flag':
                    # Add outlier flag columns
                    for col in df.columns:
                        if outlier_mask[col].any():
                            processed_df[f'{col}_outlier_flag'] = outlier_mask[col]
                
                else:
                    raise ValueError(f"Unsupported outlier action: {action}")
                
                handled_outliers = total_outliers
                if action == 'remove':
                    handled_outliers = (outlier_mask.any(axis=1)).sum()
                
                logger.info(f"Successfully handled {handled_outliers} outliers")
                return processed_df, handled_outliers
                
            except Exception as e:
                logger.error(f"Failed to handle outliers: {str(e)}")
                raise PreprocessingError(f"Outlier handling failed: {str(e)}") from e
    
    async def scale_data(self, 
                        df: pd.DataFrame, 
                        method: str = 'standard',
                        fit_scaler: bool = True,
                        scaler_key: Optional[str] = None) -> pd.DataFrame:
        """
        Scale time series data.
        
        Args:
            df: Input DataFrame
            method: Scaling method
            fit_scaler: Whether to fit the scaler
            scaler_key: Key for storing/retrieving scaler
            
        Returns:
            Scaled DataFrame
        """
        async with self.semaphore:
            try:
                if method == 'none':
                    return df.copy()
                
                logger.info(f"Scaling data using method: {method}")
                
                # Select numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    logger.warning("No numeric columns found for scaling")
                    return df.copy()
                
                # Initialize scaler
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                elif method == 'robust':
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"Unsupported scaling method: {method}")
                
                processed_df = df.copy()
                
                # Use existing scaler or fit new one
                if scaler_key and scaler_key in self.scalers and not fit_scaler:
                    scaler = self.scalers[scaler_key]
                    scaled_values = await asyncio.to_thread(
                        scaler.transform, 
                        df[numeric_cols].values
                    )
                else:
                    scaled_values = await asyncio.to_thread(
                        scaler.fit_transform, 
                        df[numeric_cols].values
                    )
                    if scaler_key:
                        self.scalers[scaler_key] = scaler
                
                # Update DataFrame with scaled values
                processed_df[numeric_cols] = scaled_values
                
                logger.info(f"Successfully scaled {len(numeric_cols)} columns")
                return processed_df
                
            except Exception as e:
                logger.error(f"Failed to scale data: {str(e)}")
                raise PreprocessingError(f"Data scaling failed: {str(e)}") from e
    
    async def smooth_data(self, 
                         df: pd.DataFrame, 
                         method: str = 'savgol',
                         window: int = 7,
                         **kwargs) -> pd.DataFrame:
        """
        Smooth time series data to reduce noise.
        
        Args:
            df: Input DataFrame
            method: Smoothing method
            window: Smoothing window size
            **kwargs: Additional method-specific parameters
            
        Returns:
            Smoothed DataFrame
        """
        async with self.semaphore:
            try:
                logger.info(f"Smoothing data using method: {method}")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    logger.warning("No numeric columns found for smoothing")
                    return df.copy()
                
                processed_df = df.copy()
                
                for col in numeric_cols:
                    if method == 'savgol':
                        polyorder = kwargs.get('polyorder', min(3, window - 1))
                        if len(df) > window:
                            smoothed = await asyncio.to_thread(
                                savgol_filter,
                                df[col].fillna(df[col].median()),
                                window,
                                polyorder
                            )
                            processed_df[col] = smoothed
                    
                    elif method == 'rolling':
                        processed_df[col] = await asyncio.to_thread(
                            df[col].rolling(window=window, center=True).mean
                        )
                    
                    elif method == 'exponential':
                        alpha = kwargs.get('alpha', 0.3)
                        processed_df[col] = await asyncio.to_thread(
                            df[col].ewm(alpha=alpha).mean
                        )
                    
                    else:
                        raise ValueError(f"Unsupported smoothing method: {method}")
                
                logger.info(f"Successfully smoothed {len(numeric_cols)} columns")
                return processed_df
                
            except Exception as e:
                logger.error(f"Failed to smooth data: {str(e)}")
                raise PreprocessingError(f"Data smoothing failed: {str(e)}") from e
    
    async def resample_data(self, 
                          df: pd.DataFrame, 
                          target_frequency: str,
                          aggregation_method: str = 'mean') -> pd.DataFrame:
        """
        Resample time series data to different frequency.
        
        Args:
            df: Input DataFrame
            target_frequency: Target frequency for resampling
            aggregation_method: Method for aggregating data
            
        Returns:
            Resampled DataFrame
        """
        async with self.semaphore:
            try:
                logger.info(f"Resampling data to frequency: {target_frequency}")
                
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError("DataFrame must have DatetimeIndex for resampling")
                
                # Perform resampling
                resampler = df.resample(target_frequency)
                
                if aggregation_method == 'mean':
                    processed_df = await asyncio.to_thread(resampler.mean)
                elif aggregation_method == 'sum':
                    processed_df = await asyncio.to_thread(resampler.sum)
                elif aggregation_method == 'median':
                    processed_df = await asyncio.to_thread(resampler.median)
                elif aggregation_method == 'min':
                    processed_df = await asyncio.to_thread(resampler.min)
                elif aggregation_method == 'max':
                    processed_df = await asyncio.to_thread(resampler.max)
                elif aggregation_method == 'first':
                    processed_df = await asyncio.to_thread(resampler.first)
                elif aggregation_method == 'last':
                    processed_df = await asyncio.to_thread(resampler.last)
                else:
                    raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
                
                logger.info(f"Successfully resampled data from {len(df)} to {len(processed_df)} records")
                return processed_df.dropna()
                
            except Exception as e:
                logger.error(f"Failed to resample data: {str(e)}")
                raise PreprocessingError(f"Data resampling failed: {str(e)}") from e
    
    async def engineer_features(self, 
                              df: pd.DataFrame,
                              lag_features: List[int] = [1, 7, 30],
                              rolling_features: List[int] = [7, 14, 30],
                              diff_features: List[int] = [1]) -> pd.DataFrame:
        """
        Engineer time series features.
        
        Args:
            df: Input DataFrame
            lag_features: List of lag periods
            rolling_features: List of rolling window sizes
            diff_features: List of differencing orders
            
        Returns:
            DataFrame with engineered features
        """
        async with self.semaphore:
            try:
                logger.info("Engineering time series features")
                
                processed_df = df.copy()
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # Lag features
                for lag in lag_features:
                    for col in numeric_cols:
                        processed_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
                # Rolling features
                for window in rolling_features:
                    for col in numeric_cols:
                        processed_df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                        processed_df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                        processed_df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                        processed_df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
                
                # Differencing features
                for order in diff_features:
                    for col in numeric_cols:
                        processed_df[f'{col}_diff_{order}'] = df[col].diff(order)
                
                # Time-based features
                if isinstance(df.index, pd.DatetimeIndex):
                    processed_df['hour'] = df.index.hour
                    processed_df['day'] = df.index.day
                    processed_df['month'] = df.index.month
                    processed_df['year'] = df.index.year
                    processed_df['dayofweek'] = df.index.dayofweek
                    processed_df['dayofyear'] = df.index.dayofyear
                
                feature_count = len(processed_df.columns) - len(df.columns)
                logger.info(f"Successfully engineered {feature_count} new features")
                
                return processed_df
                
            except Exception as e:
                logger.error(f"Failed to engineer features: {str(e)}")
                raise PreprocessingError(f"Feature engineering failed: {str(e)}") from e
    
    async def preprocess_pipeline(self, 
                                df: pd.DataFrame, 
                                name: str = "unnamed",
                                config: Optional[PreprocessingConfig] = None) -> Tuple[pd.DataFrame, PreprocessingReport]:
        """
        Execute complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            name: Name of the time series
            config: Preprocessing configuration
            
        Returns:
            Tuple of (processed DataFrame, preprocessing report)
        """
        start_time = asyncio.get_event_loop().time()
        config = config or self.config
        
        # Initialize report
        report = PreprocessingReport(
            original_shape=df.shape,
            processed_shape=df.shape,
            data_quality_before=1.0 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        )
        
        try:
            logger.info(f"Starting preprocessing pipeline for {name}")
            processed_df = df.copy()
            
            # Handle missing values
            if config.handle_missing:
                processed_df, missing_handled = await self.handle_missing_values(
                    processed_df, config.missing_method
                )
                report.missing_values_handled = missing_handled
                report.operations_performed.append(f"missing_values_{config.missing_method}")
            
            # Detect and handle outliers
            if config.detect_outliers:
                outlier_mask = await self.detect_outliers(
                    processed_df, config.outlier_method, config.outlier_threshold
                )
                report.outliers_detected = outlier_mask.sum().sum()
                
                processed_df, outliers_handled = await self.handle_outliers(
                    processed_df, outlier_mask, config.outlier_action
                )
                report.outliers_handled = outliers_handled
                report.operations_performed.append(f"outliers_{config.outlier_method}_{config.outlier_action}")
            
            # Scale data
            if config.scale_data:
                processed_df = await self.scale_data(processed_df, config.scaling_method, scaler_key=name)
                report.operations_performed.append(f"scaling_{config.scaling_method}")
            
            # Smooth data
            if config.smooth_data:
                processed_df = await self.smooth_data(
                    processed_df, config.smoothing_method, config.smoothing_window
                )
                report.operations_performed.append(f"smoothing_{config.smoothing_method}")
            
            # Resample data
            if config.resample_data and config.target_frequency:
                processed_df = await self.resample_data(
                    processed_df, config.target_frequency, config.aggregation_method
                )
                report.operations_performed.append(f"resample_{config.target_frequency}")
            
            # Engineer features
            if config.feature_engineering:
                processed_df = await self.engineer_features(
                    processed_df, config.lag_features, config.rolling_features, config.diff_features
                )
                report.operations_performed.append("feature_engineering")
            
            # Update report
            report.processed_shape = processed_df.shape
            report.data_quality_after = 1.0 - (processed_df.isnull().sum().sum() / (processed_df.shape[0] * processed_df.shape[1]))
            report.processing_time_seconds = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Preprocessing pipeline completed for {name} in {report.processing_time_seconds:.2f}s")
            return processed_df, report
            
        except Exception as e:
            error_msg = f"Preprocessing pipeline failed for {name}: {str(e)}"
            logger.error(error_msg)
            report.errors.append(error_msg)
            report.processing_time_seconds = asyncio.get_event_loop().time() - start_time
            raise PreprocessingError(error_msg) from e
    
    async def preprocess_multiple(self, 
                                dataframes: List[Tuple[pd.DataFrame, str]],
                                config: Optional[PreprocessingConfig] = None) -> List[Tuple[pd.DataFrame, PreprocessingReport]]:
        """
        Preprocess multiple time series concurrently.
        
        Args:
            dataframes: List of (DataFrame, name) tuples
            config: Preprocessing configuration
            
        Returns:
            List of (processed DataFrame, report) tuples
        """
        logger.info(f"Starting concurrent preprocessing of {len(dataframes)} time series")
        
        # Create tasks for concurrent processing
        tasks = [
            self.preprocess_pipeline(df, name, config)
            for df, name in dataframes
        ]
        
        # Execute with error handling
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate successful results from errors
            successful_results = []
            errors = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append((dataframes[i][1], result))
                else:
                    successful_results.append(result)
            
            if errors:
                logger.warning(f"Failed to preprocess {len(errors)} time series: {[e[0] for e in errors]}")
            
            logger.info(f"Successfully preprocessed {len(successful_results)} out of {len(dataframes)} time series")
            return successful_results
            
        except Exception as e:
            logger.error(f"Critical error in concurrent preprocessing: {str(e)}")
            raise PreprocessingError(f"Concurrent preprocessing failed: {str(e)}") from e
    
    def get_scaler(self, key: str) -> Optional[Any]:
        """Get a fitted scaler by key."""
        return self.scalers.get(key)
    
    def clear_scalers(self):
        """Clear all fitted scalers."""
        self.scalers.clear()
        logger.info("All scalers cleared")


# Utility functions
async def create_preprocessing_config(
    missing_method: str = 'interpolation',
    outlier_method: str = 'iqr',
    scaling_method: str = 'standard',
    enable_feature_engineering: bool = True
) -> PreprocessingConfig:
    """Create a preprocessing configuration with common settings."""
    return PreprocessingConfig(
        missing_method=missing_method,
        outlier_method=outlier_method,
        scaling_method=scaling_method,
        feature_engineering=enable_feature_engineering
    )


async def quick_preprocess(df: pd.DataFrame, name: str = "unnamed") -> pd.DataFrame:
    """Quick preprocessing with default settings."""
    preprocessor = AsyncTimeSeriesPreprocessor()
    processed_df, _ = await preprocessor.preprocess_pipeline(df, name)
    return processed_df

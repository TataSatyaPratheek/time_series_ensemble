"""
Async Time Series Data Loader
Handles loading and initial validation of time series data from various sources.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import aiofiles
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from datetime import datetime
import structlog

# Configuration imports
from ..config import settings
from ..utils.exceptions import DataLoadError, ValidationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TimeSeriesMetadata(BaseModel):
    """Metadata for time series data validation and processing."""
    name: str
    columns: List[str]
    datetime_column: str
    value_columns: List[str]
    frequency: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    missing_count: int = 0
    total_records: int = 0
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @validator('frequency')
    def validate_frequency(cls, v):
        """Validate pandas frequency strings."""
        valid_frequencies = ['D', 'H', 'T', 'S', 'W', 'M', 'Q', 'Y', 'B', 'BH']
        if v and v not in valid_frequencies:
            logger.warning(f"Unusual frequency detected: {v}")
        return v


class AsyncTimeSeriesLoader:
    """
    Async time series data loader with comprehensive validation and metadata extraction.
    Optimized for concurrent loading of multiple time series datasets.
    """
    
    def __init__(self, 
                 data_path: Optional[Union[str, Path]] = None,
                 max_concurrent_loads: int = 4,
                 chunk_size: int = 10000,
                 enable_caching: bool = True):
        """
        Initialize the async time series loader.
        
        Args:
            data_path: Base path for data files
            max_concurrent_loads: Maximum concurrent file loads (optimized for M1 Air)
            chunk_size: Chunk size for large file processing
            enable_caching: Enable result caching
        """
        self.data_path = Path(data_path) if data_path else Path(settings.DATA_DIR)
        self.max_concurrent_loads = min(max_concurrent_loads, settings.MAX_WORKERS)
        self.chunk_size = chunk_size
        self.enable_caching = enable_caching
        self.cache: Dict[str, pd.DataFrame] = {}
        self.metadata_cache: Dict[str, TimeSeriesMetadata] = {}
        self.semaphore = asyncio.Semaphore(self.max_concurrent_loads)
        
        # Ensure data directories exist
        self.raw_data_path = self.data_path / "raw"
        self.processed_data_path = self.data_path / "processed"
        self.external_data_path = self.data_path / "external"
        
        for path in [self.raw_data_path, self.processed_data_path, self.external_data_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    async def load_csv_async(self, 
                           file_path: Union[str, Path], 
                           datetime_column: str = 'date',
                           **kwargs) -> pd.DataFrame:
        """
        Asynchronously load CSV file with time series optimization.
        
        Args:
            file_path: Path to CSV file
            datetime_column: Name of datetime column
            **kwargs: Additional pandas.read_csv arguments
            
        Returns:
            Loaded DataFrame with datetime index
            
        Raises:
            DataLoadError: If file loading fails
        """
        async with self.semaphore:
            try:
                file_path = Path(file_path)
                cache_key = str(file_path.absolute())
                
                # Check cache first
                if self.enable_caching and cache_key in self.cache:
                    logger.info(f"Loading from cache: {file_path.name}")
                    return self.cache[cache_key].copy()
                
                logger.info(f"Loading CSV file: {file_path.name}")
                
                # Default CSV reading parameters optimized for time series
                default_kwargs = {
                    'parse_dates': [datetime_column] if datetime_column else None,
                    'index_col': datetime_column if datetime_column else None,
                    'infer_datetime_format': True,
                    'low_memory': False,
                    'engine': 'c',  # Faster C engine
                }
                default_kwargs.update(kwargs)
                
                # Load data in thread pool to avoid blocking
                df = await asyncio.to_thread(pd.read_csv, file_path, **default_kwargs)
                
                # Validate and process datetime index
                if datetime_column and datetime_column in df.columns:
                    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
                    df.set_index(datetime_column, inplace=True)
                    df.sort_index(inplace=True)
                
                # Cache result
                if self.enable_caching:
                    self.cache[cache_key] = df.copy()
                
                logger.info(f"Successfully loaded {len(df)} records from {file_path.name}")
                return df
                
            except Exception as e:
                error_msg = f"Failed to load CSV file {file_path}: {str(e)}"
                logger.error(error_msg)
                raise DataLoadError(error_msg) from e
    
    async def load_multiple_csv(self, 
                              file_paths: List[Union[str, Path]], 
                              datetime_column: str = 'date',
                              **kwargs) -> List[pd.DataFrame]:
        """
        Load multiple CSV files concurrently.
        
        Args:
            file_paths: List of file paths to load
            datetime_column: Name of datetime column
            **kwargs: Additional pandas.read_csv arguments
            
        Returns:
            List of loaded DataFrames
        """
        logger.info(f"Loading {len(file_paths)} CSV files concurrently")
        
        # Create tasks for concurrent loading
        tasks = [
            self.load_csv_async(fp, datetime_column, **kwargs) 
            for fp in file_paths
        ]
        
        # Execute with progress tracking
        try:
            dataframes = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate successful loads from errors
            successful_dfs = []
            errors = []
            
            for i, result in enumerate(dataframes):
                if isinstance(result, Exception):
                    errors.append((file_paths[i], result))
                else:
                    successful_dfs.append(result)
            
            if errors:
                logger.warning(f"Failed to load {len(errors)} files: {[str(e[0]) for e in errors]}")
            
            logger.info(f"Successfully loaded {len(successful_dfs)} out of {len(file_paths)} files")
            return successful_dfs
            
        except Exception as e:
            logger.error(f"Critical error in concurrent loading: {str(e)}")
            raise DataLoadError(f"Concurrent loading failed: {str(e)}") from e
    
    async def extract_metadata(self, df: pd.DataFrame, name: str) -> TimeSeriesMetadata:
        """
        Extract comprehensive metadata from time series DataFrame.
        
        Args:
            df: Time series DataFrame
            name: Name identifier for the time series
            
        Returns:
            TimeSeriesMetadata object
        """
        try:
            # Basic information
            total_records = len(df)
            columns = df.columns.tolist()
            
            # Detect datetime column and value columns
            datetime_column = df.index.name if hasattr(df.index, 'name') else None
            value_columns = [col for col in columns if df[col].dtype in ['int64', 'float64']]
            
            # Time series specific metadata
            start_date = df.index.min() if hasattr(df.index, 'min') else None
            end_date = df.index.max() if hasattr(df.index, 'max') else None
            
            # Detect frequency
            frequency = None
            if hasattr(df.index, 'freq') and df.index.freq:
                frequency = str(df.index.freq)
            elif hasattr(df.index, 'inferred_freq') and df.index.inferred_freq:
                frequency = df.index.inferred_freq
            else:
                # Try to infer frequency
                try:
                    frequency = await asyncio.to_thread(pd.infer_freq, df.index)
                except:
                    frequency = None
            
            # Data quality assessment
            missing_count = df.isnull().sum().sum()
            data_quality_score = max(0.0, 1.0 - (missing_count / (total_records * len(columns))))
            
            metadata = TimeSeriesMetadata(
                name=name,
                columns=columns,
                datetime_column=datetime_column or 'index',
                value_columns=value_columns,
                frequency=frequency,
                start_date=start_date,
                end_date=end_date,
                missing_count=missing_count,
                total_records=total_records,
                data_quality_score=data_quality_score
            )
            
            logger.info(f"Extracted metadata for {name}: {total_records} records, quality score: {data_quality_score:.3f}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata for {name}: {str(e)}")
            raise ValidationError(f"Metadata extraction failed: {str(e)}") from e
    
    async def load_with_metadata(self, 
                               file_path: Union[str, Path], 
                               name: Optional[str] = None,
                               datetime_column: str = 'date',
                               **kwargs) -> Tuple[pd.DataFrame, TimeSeriesMetadata]:
        """
        Load time series data with metadata extraction.
        
        Args:
            file_path: Path to data file
            name: Name for the time series (defaults to filename)
            datetime_column: Name of datetime column
            **kwargs: Additional loading arguments
            
        Returns:
            Tuple of (DataFrame, TimeSeriesMetadata)
        """
        file_path = Path(file_path)
        name = name or file_path.stem
        
        # Load data
        df = await self.load_csv_async(file_path, datetime_column, **kwargs)
        
        # Extract metadata
        metadata = await self.extract_metadata(df, name)
        
        # Cache metadata
        if self.enable_caching:
            self.metadata_cache[name] = metadata
        
        return df, metadata
    
    async def discover_data_files(self, 
                                pattern: str = "*.csv",
                                recursive: bool = True) -> List[Path]:
        """
        Discover data files in the data directory.
        
        Args:
            pattern: File pattern to match
            recursive: Search recursively in subdirectories
            
        Returns:
            List of discovered file paths
        """
        try:
            if recursive:
                files = list(self.data_path.rglob(pattern))
            else:
                files = list(self.data_path.glob(pattern))
            
            logger.info(f"Discovered {len(files)} files matching pattern '{pattern}'")
            return files
            
        except Exception as e:
            logger.error(f"Failed to discover files: {str(e)}")
            return []
    
    async def validate_time_series(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """
        Comprehensive validation of time series data.
        
        Args:
            df: Time series DataFrame
            name: Name of the time series
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'name': name,
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Check if DataFrame is empty
            if df.empty:
                validation_results['is_valid'] = False
                validation_results['errors'].append("DataFrame is empty")
                return validation_results
            
            # Check datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                validation_results['warnings'].append("Index is not DatetimeIndex")
                validation_results['recommendations'].append("Convert index to datetime for time series analysis")
            
            # Check for missing values
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_pct > 50:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Too many missing values: {missing_pct:.1f}%")
            elif missing_pct > 10:
                validation_results['warnings'].append(f"High missing values: {missing_pct:.1f}%")
            
            # Check for duplicated timestamps
            if hasattr(df.index, 'duplicated') and df.index.duplicated().any():
                validation_results['warnings'].append("Duplicated timestamps detected")
                validation_results['recommendations'].append("Remove or aggregate duplicated timestamps")
            
            # Check data types
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                validation_results['warnings'].append("No numeric columns found")
                validation_results['recommendations'].append("Ensure value columns are numeric")
            
            # Check for constant values
            for col in numeric_columns:
                if df[col].nunique() == 1:
                    validation_results['warnings'].append(f"Column '{col}' has constant values")
            
            logger.info(f"Validation completed for {name}: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
            return validation_results
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Validation failed for {name}: {str(e)}")
            return validation_results
    
    async def clear_cache(self):
        """Clear all cached data and metadata."""
        self.cache.clear()
        self.metadata_cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        return {
            'cached_datasets': len(self.cache),
            'cached_metadata': len(self.metadata_cache),
            'cache_enabled': self.enable_caching
        }


# Utility functions for common loading patterns
async def load_monash_dataset(dataset_name: str, data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load a dataset from Monash Time Series Repository format.
    
    Args:
        dataset_name: Name of the Monash dataset
        data_path: Path to data directory
        
    Returns:
        Loaded DataFrame
    """
    loader = AsyncTimeSeriesLoader(data_path)
    file_path = loader.external_data_path / f"{dataset_name}.csv"
    
    if not file_path.exists():
        raise DataLoadError(f"Monash dataset not found: {file_path}")
    
    return await loader.load_csv_async(file_path)


async def load_sample_data(n_samples: int = 1000, 
                         frequency: str = 'D', 
                         start_date: str = '2020-01-01') -> pd.DataFrame:
    """
    Generate sample time series data for testing.
    
    Args:
        n_samples: Number of samples to generate
        frequency: Pandas frequency string
        start_date: Start date for the series
        
    Returns:
        Generated sample DataFrame
    """
    try:
        # Generate datetime index
        date_range = pd.date_range(start=start_date, periods=n_samples, freq=frequency)
        
        # Generate sample data with trend and seasonality
        np.random.seed(42)  # For reproducibility
        trend = np.linspace(100, 200, n_samples)
        seasonality = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
        noise = np.random.normal(0, 5, n_samples)
        
        values = trend + seasonality + noise
        
        # Create DataFrame
        df = pd.DataFrame({
            'value': values,
            'trend': trend,
            'seasonal': seasonality,
            'noise': noise
        }, index=date_range)
        
        logger.info(f"Generated sample data: {n_samples} samples with frequency {frequency}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to generate sample data: {str(e)}")
        raise DataLoadError(f"Sample data generation failed: {str(e)}") from e

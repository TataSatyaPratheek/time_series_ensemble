"""
Data module initialization.
Exports main classes and functions for easy importing.
"""

from .loader import (
    AsyncTimeSeriesLoader,
    TimeSeriesMetadata,
    load_monash_dataset,
    load_sample_data
)

from .preprocessor import (
    AsyncTimeSeriesPreprocessor,
    PreprocessingConfig,
    PreprocessingReport,
    create_preprocessing_config,
    quick_preprocess
)

__all__ = [
    'AsyncTimeSeriesLoader',
    'TimeSeriesMetadata',
    'AsyncTimeSeriesPreprocessor',
    'PreprocessingConfig',
    'PreprocessingReport',
    'load_monash_dataset',
    'load_sample_data',
    'create_preprocessing_config',
    'quick_preprocess'
]

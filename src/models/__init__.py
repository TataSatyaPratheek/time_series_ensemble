"""
Model module initialization.
Exports statistical, machine learning, and ensemble models with registry.
"""

from typing import Optional, List, Dict, Any

# Statistical models
from .statistical import (
    AsyncARIMAModel,
    AsyncProphetModel, 
    AsyncSTLDecomposition,
    AsyncExponentialSmoothing,
    ModelResult,
    ModelConfig,
    auto_arima_selection
)

# Machine learning models
from .ml_models import (
    AsyncMLModel,
    AsyncXGBoostModel,
    AsyncLightGBMModel,
    AsyncCatBoostModel,
    AsyncRandomForestModel,
    AsyncLinearModel,
    MLModelConfig,
    create_time_series_features,
    select_best_ml_model
)

# Ensemble methods
from .ensemble import (
    BaseEnsemble,
    AsyncSimpleAverageEnsemble,
    AsyncWeightedAverageEnsemble,
    AsyncStackingEnsemble,
    AsyncBayesianModelAveraging,
    ModelPrediction,
    EnsembleFactory,
    EnsembleConfig,
    evaluate_ensemble_performance
)

__all__ = [
    # Statistical models
    'AsyncARIMAModel',
    'AsyncProphetModel',
    'AsyncSTLDecomposition',
    'AsyncExponentialSmoothing',
    'ModelResult',
    'ModelConfig',
    'auto_arima_selection',
    
    # ML models
    'AsyncMLModel',
    'AsyncXGBoostModel',
    'AsyncLightGBMModel',
    'AsyncCatBoostModel',
    'AsyncRandomForestModel',
    'AsyncLinearModel',
    'MLModelConfig',
    'create_time_series_features',
    'select_best_ml_model',
    
    # Ensemble methods
    'BaseEnsemble',
    'AsyncSimpleAverageEnsemble',
    'AsyncWeightedAverageEnsemble',
    'AsyncStackingEnsemble',
    'AsyncBayesianModelAveraging',
    'ModelPrediction',
    'EnsembleFactory',
    'EnsembleConfig',
    'evaluate_ensemble_performance'
]

# Model registry for dynamic model loading
STATISTICAL_MODELS = {
    'arima': AsyncARIMAModel,
    'prophet': AsyncProphetModel,
    'stl': AsyncSTLDecomposition,
    'exponential_smoothing': AsyncExponentialSmoothing,
    'holt_winters': AsyncExponentialSmoothing  # Alias
}

ML_MODELS = {
    'linear': AsyncLinearModel,
    'ridge': lambda **kwargs: AsyncLinearModel('ridge', **kwargs),
    'lasso': lambda **kwargs: AsyncLinearModel('lasso', **kwargs),
    'elasticnet': lambda **kwargs: AsyncLinearModel('elasticnet', **kwargs),
    'random_forest': AsyncRandomForestModel,
    'xgboost': AsyncXGBoostModel,
    'lightgbm': AsyncLightGBMModel,
    'catboost': AsyncCatBoostModel
}

ENSEMBLE_METHODS = {
    'simple_average': AsyncSimpleAverageEnsemble,
    'weighted_average': AsyncWeightedAverageEnsemble,
    'stacking': AsyncStackingEnsemble,
    'bayesian_averaging': AsyncBayesianModelAveraging
}

# Complete model registry
MODEL_REGISTRY = {
    **STATISTICAL_MODELS,
    **ML_MODELS,
    **ENSEMBLE_METHODS
}

# Model categories for organized access
MODEL_CATEGORIES = {
    'statistical': list(STATISTICAL_MODELS.keys()),
    'machine_learning': list(ML_MODELS.keys()),
    'ensemble': list(ENSEMBLE_METHODS.keys())
}

# Utility functions
def get_model_by_name(model_name: str, **kwargs):
    """
    Get model instance by name from registry.
    
    Args:
        model_name: Name of the model
        **kwargs: Model initialization parameters
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model name not found
    """
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)

def list_available_models(category: Optional[str] = None) -> List[str]:
    """
    List available models by category.
    
    Args:
        category: Model category ('statistical', 'machine_learning', 'ensemble', or None for all)
        
    Returns:
        List of available model names
    """
    if category is None:
        return list(MODEL_REGISTRY.keys())
    
    if category not in MODEL_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Available: {list(MODEL_CATEGORIES.keys())}")
    
    return MODEL_CATEGORIES[category]

def get_model_info() -> Dict[str, Any]:
    """Get comprehensive information about available models."""
    return {
        'total_models': len(MODEL_REGISTRY),
        'categories': MODEL_CATEGORIES,
        'statistical_models': len(STATISTICAL_MODELS),
        'ml_models': len(ML_MODELS),
        'ensemble_methods': len(ENSEMBLE_METHODS)
    }

# Version and metadata
__version__ = "0.1.0"
__author__ = "Time Series Ensemble Team"
__description__ = "Comprehensive async time series forecasting models with ensemble support"

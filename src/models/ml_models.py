"""
Machine learning models for time series forecasting with async support.
Optimized for M1 Air performance and multi-agent orchestration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pydantic import BaseModel
import joblib
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    cb = None
    HAS_CATBOOST = False

from src.config import settings
from src.utils.exceptions import ModelError
from src.utils.logging import get_logger
from .statistical import ModelResult

logger = get_logger(__name__)


class MLModelConfig(BaseModel):
    """Configuration for ML models."""
    model_type: str
    model_params: Dict[str, Any] = {}
    fit_params: Dict[str, Any] = {}
    cv_folds: int = 5
    enable_feature_selection: bool = True
    enable_hyperparameter_tuning: bool = False
    random_state: int = 42


class AsyncMLModel:
    """
    Base class for async ML models with time series optimization.
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 model_name: str,
                 config: Optional[MLModelConfig] = None):
        """
        Initialize async ML model.
        
        Args:
            model: Sklearn-compatible model
            model_name: Name of the model
            config: Model configuration
        """
        self.model = model
        self.model_name = model_name
        self.config = config or MLModelConfig(model_type=model_name)
        
        self.is_fitted = False
        self.feature_names = None
        self.fit_metrics = {}
        self.cv_scores = {}
    
    async def fit(self, 
                  X: pd.DataFrame, 
                  y: pd.Series,
                  validate_input: bool = True,
                  **kwargs) -> Dict[str, Any]:
        """
        Fit ML model with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target series
            validate_input: Whether to validate inputs
            **kwargs: Additional fitting parameters
            
        Returns:
            Dictionary with fitting results and metrics
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            if validate_input:
                await self._validate_inputs(X, y)
            
            logger.info(f"Fitting {self.model_name} model with {X.shape[0]} samples and {X.shape[1]} features")
            
            self.feature_names = X.columns.tolist()
            
            # Fit model in thread pool
            await asyncio.to_thread(self.model.fit, X, y, **kwargs)
            self.is_fitted = True
            
            # Perform cross-validation
            if self.config.cv_folds > 1:
                self.cv_scores = await self._cross_validate(X, y)
            
            # Calculate fit metrics
            fit_time = asyncio.get_event_loop().time() - start_time
            train_predictions = await asyncio.to_thread(self.model.predict, X)
            
            self.fit_metrics = {
                'fit_time': fit_time,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'train_mae': float(mean_absolute_error(y, train_predictions)),
                'train_rmse': float(np.sqrt(mean_squared_error(y, train_predictions))),
                'train_r2': float(r2_score(y, train_predictions))
            }
            
            # Add model-specific metrics
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
                self.fit_metrics['feature_importance'] = feature_importance
            
            logger.info(f"{self.model_name} fitted successfully. Train R²: {self.fit_metrics['train_r2']:.3f}")
            return self.fit_metrics
            
        except Exception as e:
            error_msg = f"{self.model_name} model fitting failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    async def predict(self, 
                     X: pd.DataFrame,
                     return_std: bool = False) -> ModelResult:
        """
        Generate predictions.
        
        Args:
            X: Feature matrix for prediction
            return_std: Return prediction standard deviation (if supported)
            
        Returns:
            ModelResult with predictions and metadata
        """
        if not self.is_fitted:
            raise ModelError(f"{self.model_name} model must be fitted before prediction")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Generate predictions
            predictions = await asyncio.to_thread(self.model.predict, X)
            prediction_time = asyncio.get_event_loop().time() - start_time
            
            # Try to get prediction intervals (for ensemble models)
            confidence_intervals = None
            if return_std and hasattr(self.model, 'predict') and hasattr(self.model, 'estimators_'):
                try:
                    # Get predictions from all estimators
                    all_predictions = []
                    for estimator in self.model.estimators_:
                        pred = await asyncio.to_thread(estimator.predict, X)
                        all_predictions.append(pred)
                    
                    all_predictions = np.array(all_predictions)
                    pred_std = np.std(all_predictions, axis=0)
                    
                    # Create confidence intervals (±2σ ≈ 95% confidence)
                    confidence_intervals = [
                        (float(predictions[i] - 2 * pred_std[i]), 
                         float(predictions[i] + 2 * pred_std[i]))
                        for i in range(len(predictions))
                    ]
                except:
                    logger.warning("Could not calculate prediction intervals")
            
            result = ModelResult(
                model_name=self.model_name,
                predictions=predictions.tolist(),
                confidence_intervals=confidence_intervals,
                model_metadata={
                    'feature_names': self.feature_names,
                    'n_features': len(self.feature_names),
                    'model_type': self.config.model_type,
                    'n_predictions': len(predictions)
                },
                fit_metrics=self.fit_metrics,
                prediction_time=prediction_time
            )
            
            logger.info(f"Generated {len(predictions)} {self.model_name} predictions in {prediction_time:.3f}s")
            return result
            
        except Exception as e:
            error_msg = f"{self.model_name} prediction failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    async def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate input data."""
        if X.empty or y.empty:
            raise ValueError("Input data cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if X.isnull().any().any():
            logger.warning("Feature matrix contains missing values")
        
        if y.isnull().any():
            logger.warning("Target series contains missing values")
    
    async def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform time series cross-validation."""
        try:
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            
            # Perform CV in thread pool
            scores = await asyncio.to_thread(
                cross_val_score,
                self.model, X, y,
                cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=1  # Single job for stability
            )
            
            return {
                'cv_mae_mean': float(-scores.mean()),
                'cv_mae_std': float(scores.std()),
                'cv_scores': (-scores).tolist()
            }
            
        except Exception as e:
            logger.warning(f"Cross-validation failed for {self.model_name}: {str(e)}")
            return {}


# Specific ML Model Implementations
class AsyncXGBoostModel(AsyncMLModel):
    """XGBoost model optimized for time series and M1 Mac."""
    
    def __init__(self, **kwargs):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed")
        
        # M1 Mac optimized parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': min(4, settings.MAX_WORKERS),  # Optimize for M1 Air
            'tree_method': 'hist',  # Faster on M1
            'objective': 'reg:squarederror'
        }
        default_params.update(kwargs)
        
        model = xgb.XGBRegressor(**default_params)
        super().__init__(model, "XGBoost")


class AsyncLightGBMModel(AsyncMLModel):
    """LightGBM model optimized for M1 Mac."""
    
    def __init__(self, **kwargs):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is not installed")
        
        # M1 Mac optimized parameters
        default_params = {
            'n_estimators': 100,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'random_state': 42,
            'n_jobs': min(4, settings.MAX_WORKERS),
            'force_col_wise': True,  # Better for M1
            'objective': 'regression'
        }
        default_params.update(kwargs)
        
        model = lgb.LGBMRegressor(**default_params)
        super().__init__(model, "LightGBM")


class AsyncCatBoostModel(AsyncMLModel):
    """CatBoost model with silent training."""
    
    def __init__(self, **kwargs):
        if not HAS_CATBOOST:
            raise ImportError("CatBoost is not installed")
        
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': False,
            'thread_count': min(4, settings.MAX_WORKERS)
        }
        default_params.update(kwargs)
        
        model = cb.CatBoostRegressor(**default_params)
        super().__init__(model, "CatBoost")


class AsyncRandomForestModel(AsyncMLModel):
    """Random Forest model optimized for time series."""
    
    def __init__(self, **kwargs):
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': min(4, settings.MAX_WORKERS)
        }
        default_params.update(kwargs)
        
        model = RandomForestRegressor(**default_params)
        super().__init__(model, "RandomForest")


class AsyncLinearModel(AsyncMLModel):
    """Linear regression models (Linear, Ridge, Lasso, ElasticNet)."""
    
    def __init__(self, model_type: str = 'linear', **kwargs):
        model_map = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'elasticnet': ElasticNet
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unsupported linear model type: {model_type}")
        
        default_params = {'random_state': 42} if model_type != 'linear' else {}
        default_params.update(kwargs)
        
        model = model_map[model_type](**default_params)
        super().__init__(model, model_type.title())


# Time series specific feature engineering for ML models
async def create_time_series_features(df: pd.DataFrame, 
                                    target_col: str,
                                    lag_features: List[int] = [1, 7, 30],
                                    rolling_features: List[int] = [7, 14, 30],
                                    diff_features: List[int] = [1]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create time series features for ML models.
    
    Args:
        df: DataFrame with time series data
        target_col: Name of target column
        lag_features: Lag periods to create
        rolling_features: Rolling window sizes
        diff_features: Differencing orders
        
    Returns:
        Tuple of (feature DataFrame, target Series)
    """
    try:
        features_df = df.copy()
        
        # Create lag features
        for lag in lag_features:
            features_df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Create rolling features
        for window in rolling_features:
            features_df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            features_df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
            features_df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
            features_df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
        
        # Create differencing features
        for order in diff_features:
            features_df[f'{target_col}_diff_{order}'] = df[target_col].diff(order)
        
        # Create time-based features if datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            features_df['hour'] = df.index.hour
            features_df['day'] = df.index.day
            features_df['month'] = df.index.month
            features_df['year'] = df.index.year
            features_df['dayofweek'] = df.index.dayofweek
            features_df['dayofyear'] = df.index.dayofyear
            features_df['quarter'] = df.index.quarter
        
        # Remove target column from features
        X = features_df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove rows with NaN values
        X = X.dropna()
        y = y.loc[X.index]
        
        logger.info(f"Created {X.shape[1]} features from time series data")
        return X, y
        
    except Exception as e:
        error_msg = f"Feature engineering failed: {str(e)}"
        logger.error(error_msg)
        raise ModelError(error_msg) from e


# Model selection and validation utilities
async def select_best_ml_model(X: pd.DataFrame, 
                              y: pd.Series,
                              models: Optional[List[str]] = None,
                              cv_folds: int = 5) -> Tuple[AsyncMLModel, Dict[str, Any]]:
    """
    Select best ML model using cross-validation.
    
    Args:
        X: Feature matrix
        y: Target series
        models: List of model names to test
        cv_folds: Number of CV folds
        
    Returns:
        Tuple of (best model, comparison results)
    """
    if models is None:
        models = ['linear', 'ridge', 'random_forest']
        if HAS_XGBOOST:
            models.append('xgboost')
        if HAS_LIGHTGBM:
            models.append('lightgbm')
    
    model_results = {}
    best_score = float('inf')
    best_model = None
    
    for model_name in models:
        try:
            # Create model
            if model_name == 'xgboost':
                model = AsyncXGBoostModel()
            elif model_name == 'lightgbm':
                model = AsyncLightGBMModel()
            elif model_name == 'catboost':
                model = AsyncCatBoostModel()
            elif model_name == 'random_forest':
                model = AsyncRandomForestModel()
            elif model_name in ['linear', 'ridge', 'lasso', 'elasticnet']:
                model = AsyncLinearModel(model_name)
            else:
                logger.warning(f"Unknown model type: {model_name}")
                continue
            
            # Set CV folds
            model.config.cv_folds = cv_folds
            
            # Fit model
            fit_results = await model.fit(X, y)
            
            # Get CV score
            cv_score = model.cv_scores.get('cv_mae_mean', float('inf'))
            model_results[model_name] = {
                'model': model,
                'cv_mae': cv_score,
                'fit_results': fit_results
            }
            
            # Update best model
            if cv_score < best_score:
                best_score = cv_score
                best_model = model
                
            logger.info(f"{model_name}: CV MAE = {cv_score:.4f}")
            
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {str(e)}")
            continue
    
    if best_model is None:
        raise ModelError("No models could be fitted successfully")
    
    logger.info(f"Best model: {best_model.model_name} with CV MAE: {best_score:.4f}")
    return best_model, model_results

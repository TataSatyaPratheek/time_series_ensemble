"""
Advanced ensemble methods for combining multiple forecasting models.
Includes sophisticated combination strategies with async support.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from pydantic import BaseSettings, Field
from src.config import settings
from src.utils.exceptions import ModelError
from src.utils.logging import get_logger
from .statistical import ModelResult

logger = get_logger(__name__)

warnings.filterwarnings('ignore')


class EnsembleConfig(BaseSettings):
    """Configuration for ensemble methods."""
    combination_method: str = 'weighted_average'
    weighting_strategy: str = 'performance_based'
    min_models: int = 2
    max_models: int = 10
    validation_split: float = 0.2
    reweight_frequency: int = 50  # Recompute weights every N predictions
    enable_meta_learning: bool = True


@dataclass
class ModelPrediction:
    """Container for individual model predictions."""
    model_name: str
    predictions: np.ndarray
    confidence_intervals: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    prediction_time: float = 0.0


class BaseEnsemble(ABC):
    """Abstract base class for ensemble methods."""
    
    def __init__(self, name: str, config: Optional[EnsembleConfig] = None):
        self.name = name
        self.config = config or EnsembleConfig()
        self.is_fitted = False
        self.model_weights = {}
        self.performance_history = {}
    
    @abstractmethod
    async def fit(self, predictions: List[ModelPrediction], y_true: np.ndarray) -> Dict[str, Any]:
        """Fit the ensemble method."""
        pass
    
    @abstractmethod
    async def combine(self, predictions: List[ModelPrediction]) -> ModelResult:
        """Combine model predictions."""
        pass


class AsyncSimpleAverageEnsemble(BaseEnsemble):
    """Simple averaging ensemble with async support."""
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        super().__init__("SimpleAverage", config)
    
    async def fit(self, predictions: List[ModelPrediction], y_true: np.ndarray) -> Dict[str, Any]:
        """Fit simple average ensemble (no training needed)."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.is_fitted = True
            
            # Equal weights for all models
            n_models = len(predictions)
            self.model_weights = {pred.model_name: 1.0 / n_models for pred in predictions}
            
            fit_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Simple average ensemble fitted with {n_models} models")
            return {
                'fit_time': fit_time,
                'n_models': n_models,
                'weights': self.model_weights
            }
            
        except Exception as e:
            error_msg = f"Simple average ensemble fitting failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    async def combine(self, predictions: List[ModelPrediction]) -> ModelResult:
        """Combine predictions using simple averaging."""
        if not self.is_fitted:
            raise ModelError("Ensemble must be fitted before prediction")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Simple average of all predictions
            pred_arrays = [pred.predictions for pred in predictions]
            combined_predictions = np.mean(pred_arrays, axis=0)
            
            # Combine confidence intervals if available
            combined_ci = None
            ci_arrays = [pred.confidence_intervals for pred in predictions if pred.confidence_intervals is not None]
            if ci_arrays:
                ci_lower = np.mean([ci[:, 0] for ci in ci_arrays], axis=0)
                ci_upper = np.mean([ci[:, 1] for ci in ci_arrays], axis=0)
                combined_ci = [(ci_lower[i], ci_upper[i]) for i in range(len(ci_lower))]
            
            prediction_time = asyncio.get_event_loop().time() - start_time
            
            result = ModelResult(
                model_name="SimpleAverageEnsemble",
                predictions=combined_predictions.tolist(),
                confidence_intervals=combined_ci,
                model_metadata={
                    'ensemble_method': 'simple_average',
                    'n_models': len(predictions),
                    'model_names': [pred.model_name for pred in predictions],
                    'weights': self.model_weights
                },
                prediction_time=prediction_time
            )
            
            logger.info(f"Combined {len(predictions)} predictions using simple average")
            return result
            
        except Exception as e:
            error_msg = f"Simple average combination failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e


class AsyncWeightedAverageEnsemble(BaseEnsemble):
    """Weighted averaging ensemble with dynamic weight computation."""
    
    def __init__(self, 
                 weighting_strategy: str = 'performance_based',
                 config: Optional[EnsembleConfig] = None):
        super().__init__("WeightedAverage", config)
        self.weighting_strategy = weighting_strategy
        self.validation_errors = {}
    
    async def fit(self, predictions: List[ModelPrediction], y_true: np.ndarray) -> Dict[str, Any]:
        """Fit weighted ensemble by computing performance-based weights."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if len(predictions) < self.config.min_models:
                raise ValueError(f"Need at least {self.config.min_models} models for ensemble")
            
            # Calculate weights based on strategy
            if self.weighting_strategy == 'performance_based':
                self.model_weights = await self._compute_performance_weights(predictions, y_true)
            elif self.weighting_strategy == 'inverse_error':
                self.model_weights = await self._compute_inverse_error_weights(predictions, y_true)
            elif self.weighting_strategy == 'rank_based':
                self.model_weights = await self._compute_rank_weights(predictions, y_true)
            else:
                # Equal weights fallback
                n_models = len(predictions)
                self.model_weights = {pred.model_name: 1.0 / n_models for pred in predictions}
            
            self.is_fitted = True
            fit_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Weighted ensemble fitted with strategy: {self.weighting_strategy}")
            logger.info(f"Model weights: {self.model_weights}")
            
            return {
                'fit_time': fit_time,
                'weighting_strategy': self.weighting_strategy,
                'weights': self.model_weights,
                'validation_errors': self.validation_errors
            }
            
        except Exception as e:
            error_msg = f"Weighted ensemble fitting failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    async def combine(self, predictions: List[ModelPrediction]) -> ModelResult:
        """Combine predictions using weighted averaging."""
        if not self.is_fitted:
            raise ModelError("Ensemble must be fitted before prediction")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Weighted combination
            combined_predictions = np.zeros_like(predictions[0].predictions)
            total_weight = 0.0
            
            for pred in predictions:
                weight = self.model_weights.get(pred.model_name, 0.0)
                combined_predictions += weight * pred.predictions
                total_weight += weight
            
            # Normalize if weights don't sum to 1
            if total_weight > 0:
                combined_predictions /= total_weight
            
            # Weighted confidence intervals
            combined_ci = None
            if all(pred.confidence_intervals is not None for pred in predictions):
                weighted_lower = np.zeros(len(predictions[0].predictions))
                weighted_upper = np.zeros(len(predictions[0].predictions))
                
                for pred in predictions:
                    weight = self.model_weights.get(pred.model_name, 0.0)
                    ci_array = np.array(pred.confidence_intervals)
                    weighted_lower += weight * ci_array[:, 0]
                    weighted_upper += weight * ci_array[:, 1]
                
                if total_weight > 0:
                    weighted_lower /= total_weight
                    weighted_upper /= total_weight
                
                combined_ci = [(weighted_lower[i], weighted_upper[i]) for i in range(len(weighted_lower))]
            
            prediction_time = asyncio.get_event_loop().time() - start_time
            
            result = ModelResult(
                model_name="WeightedAverageEnsemble",
                predictions=combined_predictions.tolist(),
                confidence_intervals=combined_ci,
                model_metadata={
                    'ensemble_method': 'weighted_average',
                    'weighting_strategy': self.weighting_strategy,
                    'n_models': len(predictions),
                    'model_names': [pred.model_name for pred in predictions],
                    'weights': self.model_weights,
                    'total_weight': total_weight
                },
                prediction_time=prediction_time
            )
            
            logger.info(f"Combined {len(predictions)} predictions using weighted average")
            return result
            
        except Exception as e:
            error_msg = f"Weighted average combination failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    async def _compute_performance_weights(self, predictions: List[ModelPrediction], y_true: np.ndarray) -> Dict[str, float]:
        """Compute weights based on model performance."""
        errors = {}
        
        for pred in predictions:
            mae = mean_absolute_error(y_true, pred.predictions)
            errors[pred.model_name] = mae
            self.validation_errors[pred.model_name] = mae
        
        # Convert errors to weights (inverse relationship)
        max_error = max(errors.values())
        weights = {}
        total_weight = 0.0
        
        for model_name, error in errors.items():
            # Use inverse error with small epsilon to avoid division by zero
            weight = 1.0 / (error + 1e-8)
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight
        
        return weights
    
    async def _compute_inverse_error_weights(self, predictions: List[ModelPrediction], y_true: np.ndarray) -> Dict[str, float]:
        """Compute weights as inverse of validation errors."""
        return await self._compute_performance_weights(predictions, y_true)
    
    async def _compute_rank_weights(self, predictions: List[ModelPrediction], y_true: np.ndarray) -> Dict[str, float]:
        """Compute weights based on performance ranking."""
        errors = {}
        
        for pred in predictions:
            mae = mean_absolute_error(y_true, pred.predictions)
            errors[pred.model_name] = mae
        
        # Rank models by performance (lower error = higher rank)
        sorted_models = sorted(errors.items(), key=lambda x: x[1])
        n_models = len(sorted_models)
        
        weights = {}
        total_weight = 0.0
        
        for i, (model_name, _) in enumerate(sorted_models):
            # Linear rank-based weights (best model gets highest weight)
            weight = (n_models - i) / n_models
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight
        
        return weights


class AsyncStackingEnsemble(BaseEnsemble):
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, 
                 meta_learner: Optional[BaseEstimator] = None,
                 config: Optional[EnsembleConfig] = None):
        super().__init__("Stacking", config)
        self.meta_learner = meta_learner or Ridge(alpha=1.0)
        self.meta_features = None
        self.is_meta_fitted = False
    
    async def fit(self, predictions: List[ModelPrediction], y_true: np.ndarray) -> Dict[str, Any]:
        """Fit stacking ensemble with meta-learner."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if len(predictions) < self.config.min_models:
                raise ValueError(f"Need at least {self.config.min_models} models for stacking")
            
            # Create meta-features matrix
            self.meta_features = np.column_stack([pred.predictions for pred in predictions])
            
            # Fit meta-learner
            await asyncio.to_thread(self.meta_learner.fit, self.meta_features, y_true)
            
            self.is_fitted = True
            self.is_meta_fitted = True
            fit_time = asyncio.get_event_loop().time() - start_time
            
            # Calculate meta-learner performance
            meta_predictions = await asyncio.to_thread(self.meta_learner.predict, self.meta_features)
            meta_mae = mean_absolute_error(y_true, meta_predictions)
            
            logger.info(f"Stacking ensemble fitted. Meta-learner MAE: {meta_mae:.4f}")
            
            return {
                'fit_time': fit_time,
                'n_base_models': len(predictions),
                'meta_learner_type': type(self.meta_learner).__name__,
                'meta_mae': meta_mae,
                'base_model_names': [pred.model_name for pred in predictions]
            }
            
        except Exception as e:
            error_msg = f"Stacking ensemble fitting failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    async def combine(self, predictions: List[ModelPrediction]) -> ModelResult:
        """Combine predictions using meta-learner."""
        if not self.is_fitted or not self.is_meta_fitted:
            raise ModelError("Stacking ensemble must be fitted before prediction")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create meta-features for prediction
            meta_features = np.column_stack([pred.predictions for pred in predictions])
            
            # Generate meta-predictions
            combined_predictions = await asyncio.to_thread(self.meta_learner.predict, meta_features)
            
            prediction_time = asyncio.get_event_loop().time() - start_time
            
            result = ModelResult(
                model_name="StackingEnsemble",
                predictions=combined_predictions.tolist(),
                model_metadata={
                    'ensemble_method': 'stacking',
                    'meta_learner': type(self.meta_learner).__name__,
                    'n_base_models': len(predictions),
                    'base_model_names': [pred.model_name for pred in predictions]
                },
                prediction_time=prediction_time
            )
            
            logger.info(f"Combined {len(predictions)} predictions using stacking")
            return result
            
        except Exception as e:
            error_msg = f"Stacking combination failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e


class AsyncBayesianModelAveraging(BaseEnsemble):
    """Bayesian Model Averaging ensemble."""
    
    def __init__(self, 
                 prior_weights: Optional[Dict[str, float]] = None,
                 config: Optional[EnsembleConfig] = None):
        super().__init__("BayesianAveraging", config)
        self.prior_weights = prior_weights or {}
        self.posterior_weights = {}
        self.model_likelihoods = {}
    
    async def fit(self, predictions: List[ModelPrediction], y_true: np.ndarray) -> Dict[str, Any]:
        """Fit Bayesian ensemble by computing posterior weights."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Compute likelihoods for each model
            for pred in predictions:
                likelihood = await self._compute_likelihood(pred.predictions, y_true)
                self.model_likelihoods[pred.model_name] = likelihood
            
            # Compute posterior weights
            self.posterior_weights = await self._compute_posterior_weights(predictions)
            
            self.is_fitted = True
            fit_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Bayesian ensemble fitted with posterior weights: {self.posterior_weights}")
            
            return {
                'fit_time': fit_time,
                'n_models': len(predictions),
                'posterior_weights': self.posterior_weights,
                'model_likelihoods': self.model_likelihoods
            }
            
        except Exception as e:
            error_msg = f"Bayesian ensemble fitting failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    async def combine(self, predictions: List[ModelPrediction]) -> ModelResult:
        """Combine predictions using Bayesian averaging."""
        if not self.is_fitted:
            raise ModelError("Bayesian ensemble must be fitted before prediction")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Bayesian weighted combination
            combined_predictions = np.zeros_like(predictions[0].predictions)
            
            for pred in predictions:
                weight = self.posterior_weights.get(pred.model_name, 0.0)
                combined_predictions += weight * pred.predictions
            
            prediction_time = asyncio.get_event_loop().time() - start_time
            
            result = ModelResult(
                model_name="BayesianAverageEnsemble",
                predictions=combined_predictions.tolist(),
                model_metadata={
                    'ensemble_method': 'bayesian_averaging',
                    'n_models': len(predictions),
                    'posterior_weights': self.posterior_weights,
                    'model_likelihoods': self.model_likelihoods,
                    'model_names': [pred.model_name for pred in predictions]
                },
                prediction_time=prediction_time
            )
            
            logger.info(f"Combined {len(predictions)} predictions using Bayesian averaging")
            return result
            
        except Exception as e:
            error_msg = f"Bayesian combination failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    async def _compute_likelihood(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Compute likelihood of predictions given true values."""
        # Use Gaussian likelihood with empirical variance
        residuals = y_true - predictions
        variance = np.var(residuals)
        
        # Prevent division by zero
        variance = max(variance, 1e-8)
        
        # Log-likelihood for Gaussian distribution
        n = len(predictions)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * variance) - 0.5 * np.sum(residuals**2) / variance
        
        return float(log_likelihood)
    
    async def _compute_posterior_weights(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        """Compute posterior weights using Bayes' theorem."""
        # Convert log-likelihoods to probabilities
        log_likelihoods = list(self.model_likelihoods.values())
        max_log_likelihood = max(log_likelihoods)
        
        # Normalize to prevent overflow
        normalized_likelihoods = {}
        for pred in predictions:
            log_likelihood = self.model_likelihoods[pred.model_name]
            normalized_likelihoods[pred.model_name] = np.exp(log_likelihood - max_log_likelihood)
        
        # Apply priors and compute posteriors
        posteriors = {}
        total_posterior = 0.0
        
        for pred in predictions:
            prior = self.prior_weights.get(pred.model_name, 1.0)  # Uniform prior by default
            likelihood = normalized_likelihoods[pred.model_name]
            posterior = prior * likelihood
            posteriors[pred.model_name] = posterior
            total_posterior += posterior
        
        # Normalize to sum to 1
        if total_posterior > 0:
            for model_name in posteriors:
                posteriors[model_name] /= total_posterior
        else:
            # Fallback to equal weights
            n_models = len(predictions)
            posteriors = {pred.model_name: 1.0 / n_models for pred in predictions}
        
        return posteriors


# Ensemble factory and utilities
class EnsembleFactory:
    """Factory for creating ensemble methods."""
    
    @staticmethod
    def create_ensemble(ensemble_type: str, **kwargs) -> BaseEnsemble:
        """
        Create ensemble method by type.
        
        Args:
            ensemble_type: Type of ensemble ('simple', 'weighted', 'stacking', 'bayesian')
            **kwargs: Additional arguments for ensemble initialization
            
        Returns:
            Ensemble instance
        """
        ensemble_map = {
            'simple': AsyncSimpleAverageEnsemble,
            'simple_average': AsyncSimpleAverageEnsemble,
            'weighted': AsyncWeightedAverageEnsemble,
            'weighted_average': AsyncWeightedAverageEnsemble,
            'stacking': AsyncStackingEnsemble,
            'bayesian': AsyncBayesianModelAveraging,
            'bayesian_averaging': AsyncBayesianModelAveraging
        }
        
        if ensemble_type not in ensemble_map:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        return ensemble_map[ensemble_type](**kwargs)


async def evaluate_ensemble_performance(ensemble: BaseEnsemble,
                                       test_predictions: List[ModelPrediction],
                                       y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate ensemble performance on test data.
    
    Args:
        ensemble: Fitted ensemble model
        test_predictions: Test predictions from base models
        y_test: True test values
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        # Get ensemble prediction
        ensemble_result = await ensemble.combine(test_predictions)
        ensemble_pred = np.array(ensemble_result.predictions)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        
        # Calculate baseline metrics (best individual model)
        best_mae = float('inf')
        for pred in test_predictions:
            pred_mae = mean_absolute_error(y_test, pred.predictions)
            best_mae = min(best_mae, pred_mae)
        
        improvement = (best_mae - mae) / best_mae * 100
        
        return {
            'ensemble_mae': float(mae),
            'ensemble_rmse': float(rmse),
            'ensemble_mape': float(mape),
            'best_individual_mae': float(best_mae),
            'improvement_pct': float(improvement)
        }
        
    except Exception as e:
        logger.error(f"Ensemble evaluation failed: {str(e)}")
        return {}

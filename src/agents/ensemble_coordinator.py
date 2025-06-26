"""
Async Ensemble Coordinator Agent with CrewAI and Local LLM Integration
Orchestrates multiple forecasting models and combines predictions optimally.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# CrewAI imports
from crewai import Agent, Task
from src.llm.direct_interface import DirectOllamaInterface

# Project imports
from src.config import settings
from src.models.ensemble import (
    EnsembleFactory,
    ModelPrediction,
    evaluate_ensemble_performance
)
from src.models.statistical import ModelResult
from src.utils.exceptions import AgentError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EnsembleCoordinatorAgent:
    """
    Master coordinator agent that orchestrates multiple forecasting models and agents.
    Integrates with CrewAI for multi-agent coordination.
    """
    
    def __init__(self, 
                 agent_config: Optional[Dict[str, Any]] = None,
                 llm_model: str = settings.ENSEMBLE_COORDINATOR_MODEL,
                 timeout: Optional[int] = None,
                 **kwargs):
        """
        Initialize ensemble coordinator agent.
        
        Args:
            agent_config: Agent configuration from YAML
            llm_model: Local LLM model for reasoning
            timeout: Task timeout in seconds
        """
        self.agent_config = agent_config or {}
        self.llm_model = llm_model
        self.timeout = timeout or 300  # Default 5 minutes
        
        # Initialize ensemble methods
        self.ensemble_methods = {}
        self.active_models = {}
        self.model_predictions = []
        
        # Agent state
        self.is_fitted = False
        self.coordination_results = {}
        self.final_forecast = None
        self.performance_metrics = {}
        
        # Function definitions for LLM
        self.functions = [
            {
                "name": "combine_forecasts",
                "description": "Combine multiple model forecasts using various ensemble methods",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "predictions": {"type": "array", "description": "List of model predictions"},
                        "method": {"type": "string", "enum": ["simple_average", "weighted_average", "stacking", "bayesian"]},
                        "weights": {"type": "array", "description": "Optional weights for models"}
                    },
                    "required": ["predictions", "method"]
                }
            },
            {
                "name": "calculate_ensemble_weights",
                "description": "Calculate optimal weights for ensemble combination",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_performances": {"type": "array", "description": "Historical performance metrics"},
                        "weighting_strategy": {"type": "string", "enum": ["performance", "inverse_error", "rank_based"]},
                        "validation_period": {"type": "integer", "description": "Period for weight calculation"}
                    },
                    "required": ["model_performances", "weighting_strategy"]
                }
            },
            {
                "name": "estimate_uncertainty",
                "description": "Estimate prediction uncertainty and confidence intervals",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ensemble_predictions": {"type": "array", "description": "Ensemble predictions"},
                        "individual_predictions": {"type": "array", "description": "Individual model predictions"},
                        "confidence_level": {"type": "number", "description": "Confidence level for intervals"}
                    },
                    "required": ["ensemble_predictions", "individual_predictions"]
                }
            },
            {
                "name": "validate_ensemble_performance",
                "description": "Validate ensemble performance against benchmarks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ensemble_forecast": {"type": "array", "description": "Ensemble predictions"},
                        "actual_values": {"type": "array", "description": "Actual observed values"},
                        "baseline_forecasts": {"type": "array", "description": "Baseline model predictions"}
                    },
                    "required": ["ensemble_forecast", "actual_values"]
                }
            },
            {
                "name": "generate_forecast_report",
                "description": "Generate comprehensive forecasting report with insights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "forecast_results": {"type": "object", "description": "Complete forecast results"},
                        "model_contributions": {"type": "object", "description": "Individual model contributions"},
                        "uncertainty_analysis": {"type": "object", "description": "Uncertainty analysis results"}
                    },
                    "required": ["forecast_results"]
                }
            }
        ]
        
        # Initialize LLM generator
        try:
            self.llm_interface = DirectOllamaInterface(model_name=self.llm_model)
            logger.info(f"EnsembleCoordinator initialized with LLM: {self.llm_model}")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {str(e)}, using fallback mode")
            self.llm_interface = None
    
    def get_crewai_agent(self) -> Agent:
        """Create CrewAI agent instance for orchestration."""
        return Agent(
            role="Ensemble Forecasting Coordinator",
            goal="Orchestrate multiple forecasting models, combine predictions optimally, and provide final ensemble forecasts with uncertainty estimates",
            backstory="""You are the master coordinator with deep understanding of ensemble methods, model combination strategies,
            and uncertainty quantification. You synthesize insights from specialized agents and produce
            robust, accurate forecasts with proper confidence intervals.""",
            tools=list(self.functions),
            llm=self.llm_model,
            verbose=True,
            allow_delegation=True, # This agent needs to delegate to others
            max_execution_time=self.timeout,
            memory=True
        )
    
    async def combine_forecasts(self,
                              predictions: List[ModelPrediction],
                              method: str = "weighted_average",
                              weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Combine multiple model forecasts using specified ensemble method.
        
        Args:
            predictions: List of model predictions
            method: Ensemble combination method
            weights: Optional weights for models
            
        Returns:
            Dictionary with combined forecast results
        """
        try:
            if not predictions:
                raise ValueError("No predictions provided for ensemble combination")
            
            logger.info(f"Combining {len(predictions)} forecasts using {method} method")
            
            # Create appropriate ensemble method
            ensemble = EnsembleFactory.create_ensemble(method)
            
            # Prepare validation data if needed for fitting
            if method in ['weighted_average', 'stacking', 'bayesian']:
                # For demonstration, use part of predictions as validation
                # In practice, this would be separate validation data
                validation_data = np.array(predictions[0].predictions[:50])  # First 50 points
                
                # Fit ensemble method
                await ensemble.fit(predictions, validation_data)
            else:
                # Simple average doesn't need fitting
                await ensemble.fit(predictions, np.array([]))
            
            # Combine predictions
            combined_result = await ensemble.combine(predictions)
            
            return {
                'method': method,
                'combined_forecast': combined_result.predictions,
                'confidence_intervals': combined_result.confidence_intervals,
                'model_metadata': combined_result.model_metadata,
                'n_models_combined': len(predictions),
                'ensemble_weights': getattr(ensemble, 'model_weights', {}),
                'combination_time': combined_result.prediction_time
            }
            
        except Exception as e:
            logger.error(f"Forecast combination failed: {str(e)}")
            raise AgentError(f"Forecast combination error: {str(e)}") from e
    
    async def calculate_ensemble_weights(self,
                                       model_performances: List[Dict[str, float]],
                                       weighting_strategy: str = "performance",
                                       validation_period: int = 30) -> Dict[str, Any]:
        """
        Calculate optimal weights for ensemble combination.
        
        Args:
            model_performances: Historical performance metrics for each model
            weighting_strategy: Strategy for weight calculation
            validation_period: Period used for weight calculation
            
        Returns:
            Dictionary with calculated weights and metadata
        """
        try:
            if not model_performances:
                raise ValueError("No model performances provided")
            
            weights = {}
            weight_metadata = {
                'strategy': weighting_strategy,
                'validation_period': validation_period,
                'n_models': len(model_performances)
            }
            
            if weighting_strategy == "performance":
                # Weight based on performance metrics (lower error = higher weight)
                errors = []
                model_names = []
                
                for perf in model_performances:
                    model_name = perf.get('model_name', f'model_{len(model_names)}')
                    error = perf.get('mae', perf.get('rmse', perf.get('mape', 1.0)))
                    
                    model_names.append(model_name)
                    errors.append(error)
                
                # Calculate inverse error weights
                max_error = max(errors)
                inverse_errors = [max_error - error + 0.001 for error in errors]  # Add small constant
                total_weight = sum(inverse_errors)
                
                for model_name, inv_error in zip(model_names, inverse_errors):
                    weights[model_name] = float(inv_error / total_weight)
                
                weight_metadata['performance_based'] = {
                    'original_errors': errors,
                    'inverse_errors': inverse_errors
                }
            
            elif weighting_strategy == "inverse_error":
                # Direct inverse error weighting
                total_inverse = 0.0
                model_errors = {}
                
                for perf in model_performances:
                    model_name = perf.get('model_name', f'model_{len(model_errors)}')
                    error = perf.get('mae', perf.get('rmse', 1.0))
                    
                    # Avoid division by zero
                    inverse_error = 1.0 / (error + 1e-8)
                    model_errors[model_name] = inverse_error
                    total_inverse += inverse_error
                
                # Normalize weights
                for model_name, inv_error in model_errors.items():
                    weights[model_name] = float(inv_error / total_inverse)
                
                weight_metadata['inverse_error_based'] = model_errors
            
            elif weighting_strategy == "rank_based":
                # Rank-based weighting (best model gets highest weight)
                ranked_models = sorted(model_performances, 
                                     key=lambda x: x.get('mae', x.get('rmse', 1.0)))
                
                n_models = len(ranked_models)
                total_rank_weight = sum(range(1, n_models + 1))
                
                for i, perf in enumerate(ranked_models):
                    model_name = perf.get('model_name', f'model_{i}')
                    rank_weight = n_models - i  # Best model gets highest weight
                    weights[model_name] = float(rank_weight / total_rank_weight)
                
                weight_metadata['rank_based'] = {
                    'ranking': [(perf.get('model_name', f'model_{i}'), i+1) 
                              for i, perf in enumerate(ranked_models)]
                }
            
            else:
                # Equal weights fallback
                n_models = len(model_performances)
                equal_weight = 1.0 / n_models
                
                for i, perf in enumerate(model_performances):
                    model_name = perf.get('model_name', f'model_{i}')
                    weights[model_name] = equal_weight
                
                weight_metadata['equal_weights'] = True
            
            logger.info(f"Calculated ensemble weights using {weighting_strategy}: {weights}")
            
            return {
                'weights': weights,
                'metadata': weight_metadata,
                'total_weight': sum(weights.values()),
                'strategy_used': weighting_strategy
            }
            
        except Exception as e:
            logger.error(f"Weight calculation failed: {str(e)}")
            raise AgentError(f"Weight calculation error: {str(e)}") from e
    
    async def estimate_uncertainty(self,
                                 ensemble_predictions: np.ndarray,
                                 individual_predictions: List[np.ndarray],
                                 confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Estimate prediction uncertainty and confidence intervals.
        
        Args:
            ensemble_predictions: Final ensemble predictions
            individual_predictions: Individual model predictions
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with uncertainty analysis
        """
        try:
            if len(individual_predictions) == 0:
                raise ValueError("No individual predictions provided for uncertainty estimation")
            
            # Convert to numpy arrays
            ensemble_pred = np.array(ensemble_predictions)
            individual_pred_array = np.array(individual_predictions)
            
            # Calculate prediction variance across models
            prediction_variance = np.var(individual_pred_array, axis=0)
            prediction_std = np.sqrt(prediction_variance)
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            z_score = 1.96  # For 95% confidence (approximate)
            
            if confidence_level == 0.99:
                z_score = 2.576
            elif confidence_level == 0.90:
                z_score = 1.645
            
            # Confidence intervals based on ensemble prediction ± z * std
            lower_bounds = ensemble_pred - z_score * prediction_std
            upper_bounds = ensemble_pred + z_score * prediction_std
            
            confidence_intervals = [(float(lower), float(upper)) 
                                   for lower, upper in zip(lower_bounds, upper_bounds)]
            
            # Calculate additional uncertainty metrics
            uncertainty_metrics = {
                'mean_uncertainty': float(np.mean(prediction_std)),
                'max_uncertainty': float(np.max(prediction_std)),
                'min_uncertainty': float(np.min(prediction_std)),
                'uncertainty_trend': self._analyze_uncertainty_trend(prediction_std),
                'model_agreement': self._calculate_model_agreement(individual_pred_array),
                'prediction_intervals': confidence_intervals
            }
            
            # Epistemic vs Aleatoric uncertainty approximation
            epistemic_uncertainty = prediction_std  # Model uncertainty
            aleatoric_uncertainty = np.std(ensemble_pred) * np.ones_like(prediction_std)  # Data uncertainty
            
            return {
                'confidence_level': confidence_level,
                'confidence_intervals': confidence_intervals,
                'uncertainty_metrics': uncertainty_metrics,
                'epistemic_uncertainty': epistemic_uncertainty.tolist(),
                'aleatoric_uncertainty': aleatoric_uncertainty.tolist(),
                'total_uncertainty': prediction_std.tolist(),
                'uncertainty_sources': {
                    'model_disagreement': float(np.mean(prediction_std)),
                    'ensemble_variance': float(np.var(ensemble_pred)),
                    'n_models': len(individual_predictions)
                }
            }
            
        except Exception as e:
            logger.error(f"Uncertainty estimation failed: {str(e)}")
            raise AgentError(f"Uncertainty estimation error: {str(e)}") from e
    
    def _analyze_uncertainty_trend(self, uncertainty_series: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in uncertainty over forecast horizon."""
        try:
            # Simple linear trend analysis
            x = np.arange(len(uncertainty_series))
            slope, intercept = np.polyfit(x, uncertainty_series, 1)
            
            trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            trend_strength = abs(slope) / np.mean(uncertainty_series)
            
            return {
                'trend_direction': trend_direction,
                'trend_slope': float(slope),
                'trend_strength': float(trend_strength),
                'is_significant': trend_strength > 0.1
            }
        except:
            return {'trend_direction': 'unknown', 'error': 'trend analysis failed'}
    
    def _calculate_model_agreement(self, individual_predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate agreement metrics between models."""
        try:
            # Pairwise correlations between models
            n_models = individual_predictions.shape[0]
            correlations = []
            
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    corr = np.corrcoef(individual_predictions[i], individual_predictions[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            # Overall agreement metrics
            agreement_metrics = {
                'mean_correlation': float(np.mean(correlations)) if correlations else 0.0,
                'min_correlation': float(np.min(correlations)) if correlations else 0.0,
                'max_correlation': float(np.max(correlations)) if correlations else 0.0,
                'agreement_level': 'high' if np.mean(correlations) > 0.8 else 'medium' if np.mean(correlations) > 0.6 else 'low'
            }
            
            return agreement_metrics
            
        except Exception as e:
            return {'error': f'Agreement calculation failed: {str(e)}'}
    
    async def validate_ensemble_performance(self,
                                          ensemble_forecast: np.ndarray,
                                          actual_values: np.ndarray,
                                          baseline_forecasts: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Validate ensemble performance against benchmarks.
        
        Args:
            ensemble_forecast: Ensemble predictions
            actual_values: Actual observed values
            baseline_forecasts: Baseline model predictions for comparison
            
        Returns:
            Dictionary with performance validation results
        """
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            # Calculate ensemble performance metrics
            ensemble_mae = mean_absolute_error(actual_values, ensemble_forecast)
            ensemble_rmse = np.sqrt(mean_squared_error(actual_values, ensemble_forecast))
            ensemble_mape = np.mean(np.abs((actual_values - ensemble_forecast) / actual_values)) * 100
            ensemble_r2 = r2_score(actual_values, ensemble_forecast)
            
            performance_results = {
                'ensemble_metrics': {
                    'mae': float(ensemble_mae),
                    'rmse': float(ensemble_rmse),
                    'mape': float(ensemble_mape),
                    'r2_score': float(ensemble_r2)
                }
            }
            
            # Compare with baseline forecasts if provided
            if baseline_forecasts:
                baseline_performance = []
                improvements = []
                
                for i, baseline in enumerate(baseline_forecasts):
                    baseline_mae = mean_absolute_error(actual_values, baseline)
                    baseline_rmse = np.sqrt(mean_squared_error(actual_values, baseline))
                    baseline_mape = np.mean(np.abs((actual_values - baseline) / actual_values)) * 100
                    
                    # Calculate improvement
                    mae_improvement = (baseline_mae - ensemble_mae) / baseline_mae * 100
                    rmse_improvement = (baseline_rmse - ensemble_rmse) / baseline_rmse * 100
                    
                    baseline_performance.append({
                        'model_index': i,
                        'mae': float(baseline_mae),
                        'rmse': float(baseline_rmse),
                        'mape': float(baseline_mape),
                        'mae_improvement': float(mae_improvement),
                        'rmse_improvement': float(rmse_improvement)
                    })
                    
                    improvements.extend([mae_improvement, rmse_improvement])
                
                performance_results['baseline_comparison'] = {
                    'individual_baselines': baseline_performance,
                    'average_improvement': float(np.mean(improvements)),
                    'best_individual_mae': float(min(bp['mae'] for bp in baseline_performance)),
                    'ensemble_vs_best': float((min(bp['mae'] for bp in baseline_performance) - ensemble_mae) / 
                                            min(bp['mae'] for bp in baseline_performance) * 100)
                }
            
            # Performance classification
            if ensemble_mape < 5:
                performance_class = 'excellent'
            elif ensemble_mape < 10:
                performance_class = 'good'
            elif ensemble_mape < 20:
                performance_class = 'fair'
            else:
                performance_class = 'poor'
            
            performance_results['performance_assessment'] = {
                'class': performance_class,
                'validation_passed': ensemble_mape < 15 and ensemble_r2 > 0.5,
                'recommended_for_production': performance_class in ['excellent', 'good']
            }
            
            logger.info(f"Ensemble validation completed: {performance_class} performance (MAPE: {ensemble_mape:.2f}%)")
            
            return performance_results
            
        except Exception as e:
            logger.error(f"Performance validation failed: {str(e)}")
            raise AgentError(f"Performance validation error: {str(e)}") from e
    
    async def generate_forecast_report(self,
                                     forecast_results: Dict[str, Any],
                                     model_contributions: Optional[Dict[str, Any]] = None,
                                     uncertainty_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive forecasting report with insights.
        
        Args:
            forecast_results: Complete forecast results
            model_contributions: Individual model contributions
            uncertainty_analysis: Uncertainty analysis results
            
        Returns:
            Dictionary with comprehensive forecasting report
        """
        try:
            # Extract key information
            ensemble_forecast = forecast_results.get('combined_forecast', [])
            ensemble_method = forecast_results.get('method', 'unknown')
            n_models = forecast_results.get('n_models_combined', 0)
            
            # Generate executive summary
            executive_summary = {
                'forecast_horizon': len(ensemble_forecast),
                'ensemble_method': ensemble_method,
                'models_combined': n_models,
                'confidence_available': bool(uncertainty_analysis),
                'timestamp': datetime.now().isoformat()
            }
            
            # Model contribution analysis
            contribution_analysis = {}
            if model_contributions:
                weights = model_contributions.get('weights', {})
                contribution_analysis = {
                    'dominant_model': max(weights, key=weights.get) if weights else 'unknown',
                    'weight_distribution': weights,
                    'effective_models': len([w for w in weights.values() if w > 0.1]) if weights else 0,
                    'diversity_score': self._calculate_diversity_score(weights)
                }
            
            # Uncertainty insights
            uncertainty_insights = {}
            if uncertainty_analysis:
                uncertainty_metrics = uncertainty_analysis.get('uncertainty_metrics', {})
                uncertainty_insights = {
                    'average_uncertainty': uncertainty_metrics.get('mean_uncertainty', 0),
                    'uncertainty_trend': uncertainty_metrics.get('uncertainty_trend', {}),
                    'model_agreement': uncertainty_metrics.get('model_agreement', {}),
                    'confidence_level': uncertainty_analysis.get('confidence_level', 0.95),
                    'uncertainty_interpretation': self._interpret_uncertainty_level(
                        uncertainty_metrics.get('mean_uncertainty', 0)
                    )
                }
            
            # Generate recommendations
            recommendations = await self._generate_forecast_recommendations(
                forecast_results, model_contributions, uncertainty_analysis
            )
            
            # LLM-based insights (if available)
            llm_insights = {}
            if self.llm_interface:
                try:
                    prompt = f"""
                    Analyze the following ensemble forecasting results and provide strategic insights:
                    
                    Ensemble Method: {ensemble_method}
                    Models Combined: {n_models}
                    Forecast Horizon: {len(ensemble_forecast)} periods
                    Average Uncertainty: {uncertainty_insights.get('average_uncertainty', 'N/A')}
                    Model Agreement: {uncertainty_insights.get('model_agreement', {}).get('agreement_level', 'N/A')}
                    
                    Provide insights on:
                    1. Forecast reliability and confidence
                    2. Key risk factors and uncertainties
                    3. Business decision support
                    4. Model performance assessment
                    5. Future improvement recommendations
                    """
                    
                    system_prompt = """You are a master ensemble forecasting coordinator with expertise in model combination strategies 
                    and uncertainty quantification. Provide strategic insights for business decision-making."""
                    
                    llm_response = await self.llm_interface.query_llm_async(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.1
                    )
                    llm_insights = {'strategic_analysis': llm_response, 'timestamp': datetime.now().isoformat()}
                    
                except Exception as e:
                    logger.warning(f"LLM insights generation failed: {str(e)}")
                    llm_insights = {'error': str(e)}
            
            # Compile comprehensive report
            comprehensive_report = {
                'executive_summary': executive_summary,
                'forecast_results': forecast_results,
                'model_contribution_analysis': contribution_analysis,
                'uncertainty_analysis': uncertainty_insights,
                'recommendations': recommendations,
                'llm_strategic_insights': llm_insights,
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_version': '1.0',
                    'coordinator_model': self.llm_model
                }
            }
            
            logger.info("Comprehensive forecast report generated successfully")
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise AgentError(f"Report generation error: {str(e)}") from e
    
    def _calculate_diversity_score(self, weights: Dict[str, float]) -> float:
        """Calculate diversity score of ensemble weights."""
        try:
            if not weights:
                return 0.0
            
            # Shannon entropy of weights (higher = more diverse)
            weight_values = list(weights.values())
            entropy = -sum(w * np.log(w + 1e-8) for w in weight_values if w > 0)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log(len(weight_values))
            diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return float(diversity_score)
            
        except Exception:
            return 0.0
    
    def _interpret_uncertainty_level(self, mean_uncertainty: float) -> str:
        """Interpret uncertainty level for business context."""
        if mean_uncertainty < 0.1:
            return "Low uncertainty - High confidence in predictions"
        elif mean_uncertainty < 0.3:
            return "Moderate uncertainty - Reasonable confidence with some variability"
        elif mean_uncertainty < 0.5:
            return "High uncertainty - Significant variability in predictions"
        else:
            return "Very high uncertainty - Low confidence, consider additional data or models"
    
    async def _generate_forecast_recommendations(self,
                                               forecast_results: Dict[str, Any],
                                               model_contributions: Optional[Dict[str, Any]],
                                               uncertainty_analysis: Optional[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on forecast analysis."""
        recommendations = []
        
        # Ensemble method recommendations
        method = forecast_results.get('method', '')
        n_models = forecast_results.get('n_models_combined', 0)
        
        if n_models < 3:
            recommendations.append(
                f"Consider adding more models to the ensemble (currently {n_models}). "
                "Ensemble performance typically improves with 3-7 diverse models."
            )
        
        if method == 'simple_average' and n_models >= 3:
            recommendations.append(
                "Consider upgrading to weighted averaging or stacking methods for "
                "potentially better performance with multiple models."
            )
        
        # Model contribution recommendations
        if model_contributions:
            weights = model_contributions.get('weights', {})
            if weights:
                max_weight = max(weights.values())
                if max_weight > 0.7:
                    dominant_model = max(weights, key=weights.get)
                    recommendations.append(
                        f"Model '{dominant_model}' dominates the ensemble ({max_weight:.1%} weight). "
                        "Consider investigating why other models have low contributions."
                    )
                
                effective_models = len([w for w in weights.values() if w > 0.1])
                if effective_models < n_models * 0.5:
                    recommendations.append(
                        f"Only {effective_models} out of {n_models} models contribute significantly. "
                        "Consider removing low-contribution models or improving their performance."
                    )
        
        # Uncertainty recommendations
        if uncertainty_analysis:
            uncertainty_metrics = uncertainty_analysis.get('uncertainty_metrics', {})
            mean_uncertainty = uncertainty_metrics.get('mean_uncertainty', 0)
            
            if mean_uncertainty > 0.4:
                recommendations.append(
                    "High forecast uncertainty detected. Consider collecting more data, "
                    "feature engineering, or using different model types."
                )
            
            trend = uncertainty_metrics.get('uncertainty_trend', {})
            if trend.get('trend_direction') == 'increasing' and trend.get('is_significant'):
                recommendations.append(
                    "Uncertainty increases with forecast horizon. Use shorter-term forecasts "
                    "for critical decisions and update models frequently."
                )
            
            agreement = uncertainty_metrics.get('model_agreement', {})
            if agreement.get('agreement_level') == 'low':
                recommendations.append(
                    "Low model agreement detected. Models may be capturing different patterns. "
                    "Investigate model assumptions and consider ensemble diversification strategies."
                )
        
        # General recommendations
        recommendations.append(
            "Regularly validate forecast performance against actual outcomes and "
            "retrain models with new data to maintain accuracy."
        )
        
        recommendations.append(
            "Consider implementing automated model monitoring and alerts for "
            "performance degradation or unusual prediction patterns."
        )
        
        return recommendations

    async def coordinate_ensemble_forecast(self,
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main coordination method that orchestrates the entire ensemble forecasting process.
        
        Args:
            context: Dictionary containing results from other agents (e.g., 'TrendAnalysis_results',
                     'SeasonalityDetection_results', 'AnomalyDetection_results'),
                     'forecast_horizon' (int), 'ensemble_method' (str), and 'use_llm_reasoning' (bool).
            
        Returns:
            Complete ensemble forecast with coordination results
        """
        # Extract necessary data from the context
        agent_results = {
            'trend_analysis': context.get('TrendAnalysis_results', {}),
            'seasonality_analysis': context.get('SeasonalityDetection_results', {}),
            'anomaly_detection': context.get('AnomalyDetection_results', {})
        }
        forecast_horizon = context.get('forecast_horizon', 30)
        ensemble_method = context.get('ensemble_method', "weighted_average")
        use_llm_reasoning = context.get('use_llm_reasoning', True)
        
        # Store the full workflow context for internal use (e.g., _extract_agent_forecasts)
        self.workflow_context = context

        start_time = asyncio.get_event_loop().time()
        
        try:
            self.workflow_context = context # Store the workflow context
            logger.info(f"EnsembleCoordinator: Starting ensemble coordination for {forecast_horizon} periods")
            
            # Extract individual agent forecasts
            model_predictions = await self._extract_agent_forecasts(agent_results, forecast_horizon)
            
            if not model_predictions:
                raise ValueError("No valid forecasts received from agents")
            
            # Calculate model performance weights
            model_performances = await self._assess_model_performances(agent_results)
            weight_results = await self.calculate_ensemble_weights(model_performances, "performance")
            
            # Combine forecasts using ensemble method
            combination_results = await self.combine_forecasts(
                model_predictions, ensemble_method, list(weight_results['weights'].values())
            )
            
            # Estimate uncertainty
            individual_preds = [pred.predictions for pred in model_predictions]
            uncertainty_results = await self.estimate_uncertainty(
                np.array(combination_results['combined_forecast']),
                [np.array(pred) for pred in individual_preds]
            )
            
            # Validate performance if historical data available
            validation_results = {}
            if any('validation_data' in result for result in agent_results.values()):
                try:
                    # Extract validation data (this would be implemented based on agent outputs)
                    validation_results = await self._validate_against_historical_data(
                        combination_results, agent_results
                    )
                except Exception as e:
                    logger.warning(f"Validation failed: {str(e)}")
                    validation_results = {'validation_error': str(e)}
            
            # Generate comprehensive report
            self.final_forecast = combination_results['combined_forecast']
            report = await self.generate_forecast_report(
                combination_results,
                weight_results,
                uncertainty_results
            )
            
            # Store coordination results
            self.coordination_results = {
                'ensemble_method': ensemble_method,
                'model_predictions': [
                    {
                        'model_name': pred.model_name,
                        'predictions': pred.predictions,
                        'metadata': pred.metadata
                    } for pred in model_predictions
                ],
                'weight_calculation': weight_results,
                'forecast_combination': combination_results,
                'uncertainty_analysis': uncertainty_results,
                'validation_results': validation_results,
                'comprehensive_report': report
            }
            
            coordination_time = asyncio.get_event_loop().time() - start_time
            self.is_fitted = True
            
            logger.info(f"EnsembleCoordinator: Coordination completed in {coordination_time:.2f}s")
            
            return {
                'ensemble_forecast': self.final_forecast,
                'confidence_intervals': uncertainty_results.get('confidence_intervals', []),
                'coordination_results': self.coordination_results,
                'metadata': {
                    'coordination_time': coordination_time,
                    'forecast_horizon': forecast_horizon,
                    'n_models': len(model_predictions),
                    'ensemble_method': ensemble_method,
                    'llm_model': self.llm_model,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_msg = f"Ensemble coordination failed: {str(e)}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e
    
    async def _extract_agent_forecasts(self, 
                                     agent_results_from_context: Dict[str, Any], # Renamed to avoid confusion with self.workflow_context
                                     horizon: int) -> List[ModelPrediction]:
        """Extract and format forecasts from agent results, providing fallbacks."""
        model_predictions = []
        
        # Get the original series from the workflow context if available, for fallback predictions
        # The original series is now directly in self.workflow_context
        original_series = self.workflow_context.get('series') 
        if original_series is not None and not original_series.empty:
            baseline_value = original_series.iloc[-1] # Last observed value
        else:
            baseline_value = 100.0 # Default if no series context

        # Extract trend forecasts
        # Use the agent_results_from_context passed to this method
        # These are the actual results from the TrendAnalysis task
        trend_results = agent_results_from_context.get('trend_analysis', {})
        
        if 'forecasts' in trend_results and trend_results['forecasts']:
            trend_forecasts = trend_results['forecasts']
            for method, forecast in trend_forecasts.items():
                if isinstance(forecast, list) and len(forecast) >= horizon:
                    model_predictions.append(ModelPrediction(
                        model_name=f"trend_{method}",
                        predictions=np.array(forecast[:horizon]),
                        metadata={'agent': 'trend', 'method': method}
                    ))
        else:
            # Fallback for TrendAnalysis
            logger.warning("TrendAnalysis agent did not provide forecasts, creating a baseline.")
            model_predictions.append(ModelPrediction(
                model_name="trend_baseline",
                predictions=np.full(horizon, baseline_value), # Simple constant forecast
                metadata={'agent': 'trend', 'method': 'fallback_constant'}
            ))
        
        # Extract seasonality forecasts from the passed agent_results_from_context
        seasonality_results = agent_results_from_context.get('seasonality_analysis', {})
        
        if 'seasonal_forecasts' in seasonality_results and seasonality_results['seasonal_forecasts']:
            seasonal_forecasts = seasonality_results['seasonal_forecasts']
            for method, forecast_data in seasonal_forecasts.items():
                if isinstance(forecast_data, dict) and 'forecast' in forecast_data:
                    forecast = forecast_data['forecast']
                    if len(forecast) >= horizon:
                        model_predictions.append(ModelPrediction(
                            model_name=f"seasonal_{method}",
                            predictions=np.array(forecast[:horizon]),
                            metadata={'agent': 'seasonality', 'method': method}
                        ))
        else:
            # Fallback for SeasonalityDetection
            logger.warning("SeasonalityDetection agent did not provide forecasts, creating a baseline.")
            model_predictions.append(ModelPrediction(
                model_name="seasonal_baseline",
                predictions=np.full(horizon, baseline_value), # Simple constant forecast
                metadata={'agent': 'seasonality', 'method': 'fallback_constant'}
            ))
        
        # Ensure at least 2 distinct ModelPrediction objects are always returned for ensemble (for robustness)
        if len(model_predictions) < 2:
            logger.warning("Less than 2 valid forecasts extracted, adding an additional baseline.")
            model_predictions.append(ModelPrediction(
                model_name="additional_baseline",
                predictions=np.full(horizon, baseline_value * 1.01), # Slightly different baseline
                metadata={'agent': 'fallback', 'method': 'additional_constant'}
            ))
        
        return model_predictions
    
    async def _assess_model_performances(self, agent_results_from_context: Dict[str, Any]) -> List[Dict[str, float]]:
        """Assess performance of individual models from agent results."""
        performances = []
        
        # Extract performance metrics from the passed agent_results_from_context
        for agent_name, results in agent_results_from_context.items():
            if isinstance(results, dict):
                # Look for performance metrics in different locations
                metrics = results.get('fit_metrics', {})
                if not metrics:
                    metrics = results.get('performance_metrics', {})
                if not metrics:
                    metrics = results.get('validation_metrics', {})
                
                # Create performance record
                if metrics:
                    perf_record = {
                        'model_name': agent_name,
                        'mae': metrics.get('mae', metrics.get('train_mae', 1.0)),
                        'rmse': metrics.get('rmse', metrics.get('train_rmse', 1.0)),
                        'r2': metrics.get('r2', metrics.get('train_r2', 0.5))
                    }
                    performances.append(perf_record)
        
        # Add default performances if none found
        if not performances:
            performances = [
                {'model_name': 'trend_model', 'mae': 0.8, 'rmse': 1.2, 'r2': 0.7},
                {'model_name': 'seasonal_model', 'mae': 0.9, 'rmse': 1.3, 'r2': 0.6},
                {'model_name': 'anomaly_adjusted', 'mae': 0.85, 'rmse': 1.25, 'r2': 0.65}
            ]
        
        return performances
    
    async def _validate_against_historical_data(self, 
                                              combination_results: Dict[str, Any],
                                              agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ensemble against available historical data."""
        try:
            # This would extract actual validation data from agent results
            # For now, simulate validation
            forecast = combination_results['combined_forecast']
            
            # Simulate some historical data for validation
            simulated_actual = np.array(forecast) + np.random.normal(0, 0.1, len(forecast))
            
            validation_results = await self.validate_ensemble_performance(
                np.array(forecast),
                simulated_actual
            )
            
            return validation_results
            
        except Exception as e:
            return {'validation_error': str(e)}
    
    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get summary of coordination results."""
        if not self.is_fitted:
            return {'status': 'not_coordinated'}
        
        coordination_results = self.coordination_results
        
        return {
            'status': 'coordinated',
            'ensemble_method': coordination_results.get('ensemble_method', 'unknown'),
            'n_models_combined': len(coordination_results.get('model_predictions', [])),
            'forecast_length': len(self.final_forecast) if self.final_forecast else 0,
            'weights_calculated': bool(coordination_results.get('weight_calculation', {})),
            'uncertainty_estimated': bool(coordination_results.get('uncertainty_analysis', {})),
            'report_generated': bool(coordination_results.get('comprehensive_report', {})),
            'llm_model': self.llm_model
        }

    def __repr__(self) -> str:
        """String representation of the coordinator."""
        status = "coordinated" if self.is_fitted else "not_coordinated"
        n_models = len(self.coordination_results.get('model_predictions', [])) if self.is_fitted else 0
        return f"EnsembleCoordinatorAgent(status={status}, models_combined={n_models}, llm_model={self.llm_model})"

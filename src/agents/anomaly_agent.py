"""
Async Anomaly Detection Agent with CrewAI and Local LLM Integration
Handles identification of anomalies and outliers in time series data.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

# CrewAI imports
from crewai import Agent, Task
from local_llm_function_calling import Generator

# Project imports
from src.config import settings
from src.data_.preprocessor import AsyncTimeSeriesPreprocessor
from src.utils.exceptions import AgentError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AnomalyDetectionAgent:
    """
    Specialized agent for anomaly detection using multiple methods and local LLM reasoning.
    Integrates with CrewAI for multi-agent orchestration.
    """
    
    def __init__(self, 
                 agent_config: Optional[Dict[str, Any]] = None,
                 llm_model: str = settings.ANOMALY_DETECTION_MODEL,
                 timeout: Optional[int] = None,
                 **kwargs):
        """
        Initialize anomaly detection agent.
        
        Args:
            agent_config: Agent configuration from YAML
            llm_model: Local LLM model for reasoning
            timeout: Task timeout in seconds
        """
        self.agent_config = agent_config or {}
        self.llm_model = llm_model
        self.timeout = timeout or 180  # Default 3 minutes
        
        # Initialize models and preprocessor
        self.isolation_forest = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_jobs=min(4, settings.MAX_WORKERS)
        )
        self.scaler = StandardScaler()
        self.preprocessor = AsyncTimeSeriesPreprocessor()
        
        # Agent state
        self.is_fitted = False
        self.anomaly_results = {}
        self.detected_anomalies = []
        self.anomaly_scores = []
        
        # Function definitions for LLM
        self.functions = [
            {
                "name": "detect_statistical_outliers",
                "description": "Detect outliers using statistical methods (Z-score, IQR, modified Z-score)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "series": {"type": "array", "description": "Time series data"},
                        "method": {"type": "string", "enum": ["zscore", "iqr", "modified_zscore"]},
                        "threshold": {"type": "number", "description": "Threshold for outlier detection"}
                    },
                    "required": ["series", "method"]
                }
            },
            {
                "name": "identify_contextual_anomalies",
                "description": "Identify contextual anomalies considering seasonal and trend patterns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "series": {"type": "array", "description": "Time series data"},
                        "seasonal_component": {"type": "array", "description": "Seasonal component"},
                        "trend_component": {"type": "array", "description": "Trend component"},
                        "sensitivity": {"type": "number", "description": "Anomaly detection sensitivity"}
                    },
                    "required": ["series"]
                }
            },
            {
                "name": "analyze_anomaly_patterns",
                "description": "Analyze patterns in detected anomalies",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anomaly_indices": {"type": "array", "description": "Indices of detected anomalies"},
                        "anomaly_values": {"type": "array", "description": "Values of detected anomalies"},
                        "time_index": {"type": "array", "description": "Time index"},
                        "analysis_type": {"type": "string", "enum": ["temporal", "magnitude", "clustering"]}
                    },
                    "required": ["anomaly_indices", "anomaly_values"]
                }
            },
            {
                "name": "validate_anomaly_significance",
                "description": "Validate statistical significance of detected anomalies",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anomaly_scores": {"type": "array", "description": "Anomaly scores"},
                        "normal_data": {"type": "array", "description": "Normal data for comparison"},
                        "confidence_level": {"type": "number", "description": "Confidence level for validation"}
                    },
                    "required": ["anomaly_scores", "normal_data"]
                }
            }
        ]
        
        # Initialize LLM generator
        try:
            self.llm_generator = Generator.hf(self.functions, self.llm_model)
            logger.info(f"AnomalyAgent initialized with LLM: {self.llm_model}")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {str(e)}, using fallback mode")
            self.llm_generator = None
    
    def get_crewai_agent(self) -> Agent:
        """Create CrewAI agent instance for orchestration."""
        return Agent(
            role="Anomaly Detection Specialist",
            goal="Identify outliers, anomalies, and unusual patterns that could indicate data quality issues or significant events",
            backstory="""You are an expert in statistical outlier detection and anomaly identification. 
            You use multiple detection methods and can distinguish between different types of anomalies.
            Your expertise helps ensure data quality and identifies significant events in time series.""",
            tools=list(self.functions),
            llm=self.llm_model,
            verbose=True,
            allow_delegation=False, # Keep as False unless specific delegation is intended
            max_execution_time=self.timeout, # Pass the timeout to the CrewAI Agent
            memory=True
        )
    
    async def detect_statistical_outliers(self, 
                                        series: np.ndarray,
                                        method: str = "zscore",
                                        threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers using statistical methods.
        
        Args:
            series: Time series data
            method: Detection method ('zscore', 'iqr', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with detected outliers
        """
        try:
            outlier_indices = []
            outlier_values = []
            outlier_scores = []
            
            if method == "zscore":
                # Standard Z-score method
                z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
                outlier_mask = z_scores > threshold
                
                outlier_indices = np.where(outlier_mask)[0].tolist()
                outlier_values = series[outlier_mask].tolist()
                outlier_scores = z_scores[outlier_mask].tolist()
            
            elif method == "iqr":
                # Interquartile Range method
                Q1 = np.percentile(series, 25)
                Q3 = np.percentile(series, 75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (series < lower_bound) | (series > upper_bound)
                outlier_indices = np.where(outlier_mask)[0].tolist()
                outlier_values = series[outlier_mask].tolist()
                
                # Calculate IQR-based scores
                outlier_scores = []
                for val in outlier_values:
                    if val < lower_bound:
                        score = (lower_bound - val) / IQR
                    else:
                        score = (val - upper_bound) / IQR
                    outlier_scores.append(float(score))
            
            elif method == "modified_zscore":
                # Modified Z-score using median absolute deviation
                median = np.median(series)
                mad = np.median(np.abs(series - median))
                
                if mad == 0:
                    # Fallback to standard deviation
                    mad = np.std(series)
                
                modified_z_scores = 0.6745 * (series - median) / mad
                outlier_mask = np.abs(modified_z_scores) > threshold
                
                outlier_indices = np.where(outlier_mask)[0].tolist()
                outlier_values = series[outlier_mask].tolist()
                outlier_scores = np.abs(modified_z_scores[outlier_mask]).tolist()
            
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            return {
                'method': method,
                'threshold': threshold,
                'outlier_indices': outlier_indices,
                'outlier_values': outlier_values,
                'outlier_scores': outlier_scores,
                'n_outliers': len(outlier_indices),
                'outlier_percentage': float(len(outlier_indices) / len(series) * 100)
            }
            
        except Exception as e:
            logger.error(f"Statistical outlier detection failed: {str(e)}")
            raise AgentError(f"Statistical outlier detection error: {str(e)}") from e
    
    async def identify_contextual_anomalies(self, 
                                          series: np.ndarray,
                                          seasonal_component: Optional[np.ndarray] = None,
                                          trend_component: Optional[np.ndarray] = None,
                                          sensitivity: float = 2.0) -> Dict[str, Any]:
        """
        Identify contextual anomalies considering seasonal and trend patterns.
        
        Args:
            series: Time series data
            seasonal_component: Seasonal component (optional)
            trend_component: Trend component (optional)
            sensitivity: Anomaly detection sensitivity
            
        Returns:
            Dictionary with contextual anomalies
        """
        try:
            # If components not provided, perform decomposition
            if seasonal_component is None or trend_component is None:
                from src.models.statistical import AsyncSTLDecomposition

                # Determine a suitable period for STL decomposition
                # For weekly data (like the sample), a period of 52 (yearly seasonality) is common.
                # Ensure series is long enough for at least two periods.
                stl_period = None
                if len(series) >= 2 * 52: # Check for weekly yearly period
                    stl_period = 52
                elif len(series) >= 2 * 7: # Fallback for daily-like data if weekly is not enough
                    stl_period = 7 # This might not be relevant for weekly data, but as a general fallback

                stl_model = AsyncSTLDecomposition(period=stl_period) if stl_period else AsyncSTLDecomposition()
                await stl_model.fit(pd.Series(series))
                components = stl_model.get_components()
                
                if seasonal_component is None:
                    seasonal_component = components['seasonal'].values
                if trend_component is None:
                    trend_component = components['trend'].values
            
            # Calculate residuals after removing trend and seasonality
            residuals = series - trend_component - seasonal_component
            
            # Detect anomalies in residuals
            residual_outliers = await self.detect_statistical_outliers(
                residuals, method="modified_zscore", threshold=sensitivity
            )
            
            # Additional contextual checks
            contextual_anomalies = []
            
            # 1. Sudden level shifts
            diff_series = np.diff(series)
            diff_threshold = sensitivity * np.std(diff_series)
            level_shift_indices = np.where(np.abs(diff_series) > diff_threshold)[0]
            
            # 2. Seasonal pattern violations
            seasonal_anomalies = []
            if len(seasonal_component) > 0:
                seasonal_std = np.std(seasonal_component)
                seasonal_threshold = sensitivity * seasonal_std
                
                for i, (seasonal_val, residual_val) in enumerate(zip(seasonal_component, residuals)):
                    expected_seasonal = seasonal_val
                    actual_seasonal = series[i] - trend_component[i] if i < len(trend_component) else 0
                    
                    if abs(actual_seasonal - expected_seasonal) > seasonal_threshold:
                        seasonal_anomalies.append(i)
            
            # Combine all contextual anomalies
            all_contextual_indices = set(residual_outliers['outlier_indices'] + 
                                       level_shift_indices.tolist() + 
                                       seasonal_anomalies)
            
            contextual_anomalies = []
            for idx in sorted(all_contextual_indices):
                if idx < len(series):
                    anomaly_info = {
                        'index': int(idx),
                        'value': float(series[idx]),
                        'residual': float(residuals[idx]) if idx < len(residuals) else 0.0,
                        'seasonal_expected': float(seasonal_component[idx]) if idx < len(seasonal_component) else 0.0,
                        'trend_expected': float(trend_component[idx]) if idx < len(trend_component) else 0.0,
                        'anomaly_types': []
                    }
                    
                    # Classify anomaly types
                    if idx in residual_outliers['outlier_indices']:
                        anomaly_info['anomaly_types'].append('residual_outlier')
                    if idx in level_shift_indices:
                        anomaly_info['anomaly_types'].append('level_shift')
                    if idx in seasonal_anomalies:
                        anomaly_info['anomaly_types'].append('seasonal_violation')
                    
                    contextual_anomalies.append(anomaly_info)
            
            return {
                'contextual_anomalies': contextual_anomalies,
                'n_contextual_anomalies': len(contextual_anomalies),
                'residual_outliers': residual_outliers,
                'level_shifts': len(level_shift_indices),
                'seasonal_violations': len(seasonal_anomalies),
                'sensitivity': sensitivity
            }
            
        except Exception as e:
            logger.error(f"Contextual anomaly detection failed: {str(e)}")
            raise AgentError(f"Contextual anomaly detection error: {str(e)}") from e
    
    async def analyze_anomaly_patterns(self, 
                                     anomaly_indices: List[int],
                                     anomaly_values: List[float],
                                     time_index: Optional[List] = None,
                                     analysis_type: str = "temporal") -> Dict[str, Any]:
        """
        Analyze patterns in detected anomalies.
        
        Args:
            anomaly_indices: Indices of detected anomalies
            anomaly_values: Values of detected anomalies
            time_index: Time index (optional)
            analysis_type: Type of analysis ('temporal', 'magnitude', 'clustering')
            
        Returns:
            Dictionary with anomaly pattern analysis
        """
        try:
            if not anomaly_indices:
                return {
                    'analysis_type': analysis_type,
                    'patterns': {},
                    'summary': 'No anomalies to analyze'
                }
            
            patterns = {}
            
            if analysis_type == "temporal":
                # Analyze temporal distribution of anomalies
                if len(anomaly_indices) > 1:
                    # Calculate gaps between anomalies
                    gaps = np.diff(sorted(anomaly_indices))
                    
                    patterns['temporal'] = {
                        'average_gap': float(np.mean(gaps)),
                        'gap_std': float(np.std(gaps)),
                        'min_gap': int(np.min(gaps)),
                        'max_gap': int(np.max(gaps)),
                        'gap_distribution': gaps.tolist()
                    }
                    
                    # Check for clustering
                    short_gaps = gaps[gaps <= np.percentile(gaps, 25)]
                    patterns['clustering'] = {
                        'has_clusters': len(short_gaps) > len(gaps) * 0.3,
                        'cluster_threshold': float(np.percentile(gaps, 25)),
                        'clustered_anomalies': int(np.sum(gaps <= np.percentile(gaps, 25)))
                    }
                
                # Analyze periodicity
                if len(anomaly_indices) >= 3:
                    from scipy.signal import find_peaks
                    
                    # Create binary anomaly series
                    max_idx = max(anomaly_indices) if anomaly_indices else 100
                    binary_series = np.zeros(max_idx + 1)
                    binary_series[anomaly_indices] = 1
                    
                    # Autocorrelation for periodicity
                    autocorr = np.correlate(binary_series, binary_series, mode='full')
                    autocorr = autocorr[autocorr.size // 2:]
                    
                    if len(autocorr) > 10:
                        peaks, _ = find_peaks(autocorr[1:min(len(autocorr), 100)], height=0.1)
                        if len(peaks) > 0:
                            patterns['periodicity'] = {
                                'periodic_patterns': True,
                                'dominant_periods': (peaks + 1).tolist()[:5],
                                'autocorrelation_peaks': autocorr[peaks + 1].tolist()[:5]
                            }
                        else:
                            patterns['periodicity'] = {'periodic_patterns': False}
            
            elif analysis_type == "magnitude":
                # Analyze magnitude distribution of anomalies
                anomaly_array = np.array(anomaly_values)
                
                patterns['magnitude'] = {
                    'mean_magnitude': float(np.mean(anomaly_array)),
                    'std_magnitude': float(np.std(anomaly_array)),
                    'min_magnitude': float(np.min(anomaly_array)),
                    'max_magnitude': float(np.max(anomaly_array)),
                    'magnitude_range': float(np.max(anomaly_array) - np.min(anomaly_array)),
                    'skewness': float(stats.skew(anomaly_array)),
                    'kurtosis': float(stats.kurtosis(anomaly_array))
                }
                
                # Categorize by magnitude
                q25, q75 = np.percentile(anomaly_array, [25, 75])
                patterns['magnitude_categories'] = {
                    'low_magnitude': int(np.sum(anomaly_array <= q25)),
                    'medium_magnitude': int(np.sum((anomaly_array > q25) & (anomaly_array <= q75))),
                    'high_magnitude': int(np.sum(anomaly_array > q75))
                }
            
            elif analysis_type == "clustering":
                # Spatial/temporal clustering analysis
                if len(anomaly_indices) >= 2:
                    from sklearn.cluster import DBSCAN
                    
                    # Prepare data for clustering
                    if time_index is not None and len(time_index) > max(anomaly_indices):
                        # Use time and value for clustering
                        cluster_data = np.column_stack([
                            anomaly_indices,
                            anomaly_values
                        ])
                    else:
                        # Use only indices for temporal clustering
                        cluster_data = np.array(anomaly_indices).reshape(-1, 1)
                    
                    # Normalize data
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    cluster_data_norm = scaler.fit_transform(cluster_data)
                    
                    # Apply DBSCAN clustering
                    dbscan = DBSCAN(eps=0.5, min_samples=2)
                    cluster_labels = dbscan.fit_predict(cluster_data_norm)
                    
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    
                    patterns['clustering'] = {
                        'n_clusters': int(n_clusters),
                        'n_noise_points': int(n_noise),
                        'cluster_labels': cluster_labels.tolist(),
                        'silhouette_score': 0.0  # Could add silhouette analysis
                    }
                    
                    # Analyze each cluster
                    clusters_info = {}
                    for cluster_id in set(cluster_labels):
                        if cluster_id != -1:
                            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                            cluster_anomaly_indices = [anomaly_indices[i] for i in cluster_indices]
                            cluster_values = [anomaly_values[i] for i in cluster_indices]
                            
                            clusters_info[f'cluster_{cluster_id}'] = {
                                'size': len(cluster_indices),
                                'indices': cluster_anomaly_indices,
                                'mean_value': float(np.mean(cluster_values)),
                                'value_range': [float(np.min(cluster_values)), float(np.max(cluster_values))],
                                'temporal_span': int(max(cluster_anomaly_indices) - min(cluster_anomaly_indices))
                            }
                    
                    patterns['cluster_details'] = clusters_info
            
            return {
                'analysis_type': analysis_type,
                'patterns': patterns,
                'n_anomalies_analyzed': len(anomaly_indices),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Anomaly pattern analysis failed: {str(e)}")
            raise AgentError(f"Anomaly pattern analysis error: {str(e)}") from e
    
    async def validate_anomaly_significance(self, 
                                          anomaly_scores: List[float],
                                          normal_data: np.ndarray,
                                          confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Validate statistical significance of detected anomalies.
        
        Args:
            anomaly_scores: Anomaly scores
            normal_data: Normal data for comparison
            confidence_level: Confidence level for validation
            
        Returns:
            Dictionary with validation results
        """
        try:
            if not anomaly_scores:
                return {
                    'is_significant': False,
                    'reason': 'No anomaly scores provided',
                    'validation_metrics': {}
                }
            
            validation_metrics = {}
            
            # 1. Statistical significance test
            # Compare anomaly scores against normal data distribution
            normal_mean = np.mean(normal_data)
            normal_std = np.std(normal_data)
            
            # Calculate z-scores for anomaly scores relative to normal distribution
            anomaly_z_scores = [(score - normal_mean) / normal_std for score in anomaly_scores]
            
            # Determine significance threshold
            alpha = 1 - confidence_level
            z_threshold = stats.norm.ppf(1 - alpha/2)
            
            significant_anomalies = [abs(z) > z_threshold for z in anomaly_z_scores]
            n_significant = sum(significant_anomalies)
            
            validation_metrics['statistical_test'] = {
                'z_threshold': float(z_threshold),
                'n_significant': int(n_significant),
                'n_total': len(anomaly_scores),
                'significance_rate': float(n_significant / len(anomaly_scores)),
                'anomaly_z_scores': anomaly_z_scores
            }
            
            # 2. Effect size analysis
            # Cohen's d for effect size
            pooled_std = np.sqrt(((len(normal_data) - 1) * normal_std**2 + 
                                 (len(anomaly_scores) - 1) * np.var(anomaly_scores)) / 
                                (len(normal_data) + len(anomaly_scores) - 2))
            
            cohens_d = abs(np.mean(anomaly_scores) - normal_mean) / pooled_std
            
            # Effect size interpretation
            if cohens_d < 0.2:
                effect_size = 'small'
            elif cohens_d < 0.5:
                effect_size = 'medium'
            elif cohens_d < 0.8:
                effect_size = 'large'
            else:
                effect_size = 'very_large'
            
            validation_metrics['effect_size'] = {
                'cohens_d': float(cohens_d),
                'interpretation': effect_size,
                'pooled_std': float(pooled_std)
            }
            
            # 3. Distribution comparison
            # Kolmogorov-Smirnov test
            try:
                ks_statistic, ks_p_value = stats.ks_2samp(normal_data, anomaly_scores)
                validation_metrics['distribution_test'] = {
                    'ks_statistic': float(ks_statistic),
                    'ks_p_value': float(ks_p_value),
                    'distributions_different': bool(ks_p_value < alpha)
                }
            except Exception:
                validation_metrics['distribution_test'] = {'error': 'KS test failed'}
            
            # 4. Overall significance assessment
            overall_significant = (
                n_significant > 0 and
                n_significant / len(anomaly_scores) > 0.1 and  # At least 10% significant
                cohens_d > 0.2  # At least small effect size
            )
            
            return {
                'is_significant': bool(overall_significant),
                'confidence_level': confidence_level,
                'validation_metrics': validation_metrics,
                'summary': {
                    'significant_anomalies': int(n_significant),
                    'total_anomalies': len(anomaly_scores),
                    'effect_size': effect_size,
                    'overall_assessment': 'significant' if overall_significant else 'not_significant'
                }
            }
            
        except Exception as e:
            logger.error(f"Anomaly significance validation failed: {str(e)}")
            raise AgentError(f"Anomaly significance validation error: {str(e)}") from e
    
    async def analyze_comprehensive_anomalies(self, 
                                            series: pd.Series,
                                            seasonal_component: Optional[np.ndarray] = None,
                                            trend_component: Optional[np.ndarray] = None,
                                            use_llm_reasoning: bool = True) -> Dict[str, Any]:
        """
        Comprehensive anomaly analysis using multiple methods and LLM reasoning.
        
        Args:
            series: Time series data
            seasonal_component: Seasonal component (optional)
            trend_component: Trend component (optional)
            use_llm_reasoning: Whether to use LLM for analysis reasoning
            
        Returns:
            Complete anomaly analysis results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"AnomalyAgent: Starting comprehensive anomaly analysis for {len(series)} observations")
            
            series_values = series.values
            
            # 1. Statistical outlier detection
            statistical_outliers = {}
            for method in ['zscore', 'iqr', 'modified_zscore']:
                outliers = await self.detect_statistical_outliers(series_values, method=method)
                statistical_outliers[method] = outliers
            
            # 2. Isolation Forest anomaly detection
            isolation_anomalies = await self._isolation_forest_detection(series_values)
            
            # 3. Contextual anomaly detection
            contextual_anomalies = await self.identify_contextual_anomalies(
                series_values, seasonal_component, trend_component
            )
            
            # 4. Combine all detected anomalies
            all_anomaly_indices = set()
            all_anomaly_info = {}
            
            # Add statistical outliers
            for method, outliers in statistical_outliers.items():
                for idx, val, score in zip(outliers['outlier_indices'], 
                                         outliers['outlier_values'], 
                                         outliers['outlier_scores']):
                    all_anomaly_indices.add(idx)
                    if idx not in all_anomaly_info:
                        all_anomaly_info[idx] = {
                            'index': idx,
                            'value': val,
                            'detection_methods': [],
                            'scores': {}
                        }
                    all_anomaly_info[idx]['detection_methods'].append(f'statistical_{method}')
                    all_anomaly_info[idx]['scores'][f'{method}_score'] = score
            
            # Add isolation forest anomalies
            for idx, val, score in zip(isolation_anomalies['anomaly_indices'],
                                     isolation_anomalies['anomaly_values'],
                                     isolation_anomalies['anomaly_scores']):
                all_anomaly_indices.add(idx)
                if idx not in all_anomaly_info:
                    all_anomaly_info[idx] = {
                        'index': idx,
                        'value': val,
                        'detection_methods': [],
                        'scores': {}
                    }
                all_anomaly_info[idx]['detection_methods'].append('isolation_forest')
                all_anomaly_info[idx]['scores']['isolation_score'] = score
            
            # Add contextual anomalies
            for anomaly in contextual_anomalies['contextual_anomalies']:
                idx = anomaly['index']
                all_anomaly_indices.add(idx)
                if idx not in all_anomaly_info:
                    all_anomaly_info[idx] = {
                        'index': idx,
                        'value': anomaly['value'],
                        'detection_methods': [],
                        'scores': {}
                    }
                all_anomaly_info[idx]['detection_methods'].extend(['contextual_' + t for t in anomaly['anomaly_types']])
                all_anomaly_info[idx]['scores']['residual_score'] = anomaly['residual']
            
            # Convert to list and sort by index
            self.detected_anomalies = [all_anomaly_info[idx] for idx in sorted(all_anomaly_indices)]
            
            # 5. Pattern analysis
            if self.detected_anomalies:
                anomaly_indices = [a['index'] for a in self.detected_anomalies]
                anomaly_values = [a['value'] for a in self.detected_anomalies]
                
                temporal_patterns = await self.analyze_anomaly_patterns(
                    anomaly_indices, anomaly_values, analysis_type="temporal"
                )
                magnitude_patterns = await self.analyze_anomaly_patterns(
                    anomaly_indices, anomaly_values, analysis_type="magnitude"
                )
                clustering_patterns = await self.analyze_anomaly_patterns(
                    anomaly_indices, anomaly_values, analysis_type="clustering"
                )
                
                # 6. Significance validation
                normal_data = np.array([series_values[i] for i in range(len(series_values)) 
                                      if i not in all_anomaly_indices])
                anomaly_scores = [a['scores'].get('isolation_score', 0) for a in self.detected_anomalies]
                
                significance_validation = await self.validate_anomaly_significance(
                    anomaly_scores, normal_data
                )
            else:
                temporal_patterns = {}
                magnitude_patterns = {}
                clustering_patterns = {}
                significance_validation = {'is_significant': False, 'reason': 'No anomalies detected'}
            
            # 7. LLM-based reasoning (if available)
            llm_insights = {}
            if use_llm_reasoning and self.llm_generator and self.detected_anomalies:
                try:
                    n_anomalies = len(self.detected_anomalies)
                    anomaly_rate = n_anomalies / len(series) * 100
                    
                    prompt = f"""
                    Analyze the following anomaly detection results and provide insights:
                    
                    Total Anomalies Detected: {n_anomalies}
                    Anomaly Rate: {anomaly_rate:.2f}%
                    Statistical Significance: {significance_validation['is_significant']}
                    Detection Methods Used: {len(statistical_outliers) + 2}
                    
                    Provide a comprehensive analysis including:
                    1. Anomaly severity assessment
                    2. Potential causes of anomalies
                    3. Impact on forecasting
                    4. Data quality implications
                    5. Recommended actions
                    """
                    
                    llm_response = await asyncio.to_thread(self.llm_generator.generate, prompt)
                    llm_insights = {'analysis': llm_response, 'timestamp': datetime.now().isoformat()}
                    
                except Exception as e:
                    logger.warning(f"LLM reasoning failed: {str(e)}")
                    llm_insights = {'error': str(e)}
            
            # Compile final results
            self.anomaly_results = {
                'statistical_outliers': statistical_outliers,
                'isolation_forest': isolation_anomalies,
                'contextual_anomalies': contextual_anomalies,
                'detected_anomalies': self.detected_anomalies,
                'temporal_patterns': temporal_patterns,
                'magnitude_patterns': magnitude_patterns,
                'clustering_patterns': clustering_patterns,
                'significance_validation': significance_validation,
                'llm_insights': llm_insights
            }
            
            analysis_time = asyncio.get_event_loop().time() - start_time
            
            self.is_fitted = True
            
            logger.info(f"AnomalyAgent: Comprehensive analysis completed in {analysis_time:.2f}s")
            
            return {
                'anomaly_analysis': self.anomaly_results,
                'metadata': {
                    'analysis_time': analysis_time,
                    'series_length': len(series),
                    'n_anomalies': len(self.detected_anomalies),
                    'llm_model': self.llm_model,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_msg = f"Comprehensive anomaly analysis failed: {str(e)}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e

    async def _isolation_forest_detection(self, series: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary with isolation forest anomaly results
        """
        try:
            # Reshape for sklearn
            X = series.reshape(-1, 1)
            
            # Fit isolation forest
            await asyncio.to_thread(self.isolation_forest.fit, X)
            
            # Predict anomalies (-1 for anomalies, 1 for normal)
            predictions = await asyncio.to_thread(self.isolation_forest.predict, X)
            
            # Get anomaly scores (lower scores indicate anomalies)
            scores = await asyncio.to_thread(self.isolation_forest.score_samples, X)
            
            # Find anomaly indices
            anomaly_mask = predictions == -1
            anomaly_indices = np.where(anomaly_mask)[0].tolist()
            anomaly_values = series[anomaly_mask].tolist()
            
            # Convert scores to positive anomaly scores (higher = more anomalous)
            anomaly_scores = (-scores[anomaly_mask]).tolist()
            
            return {
                'method': 'isolation_forest',
                'contamination': self.isolation_forest.contamination,
                'anomaly_indices': anomaly_indices,
                'anomaly_values': anomaly_values,
                'anomaly_scores': anomaly_scores,
                'n_anomalies': len(anomaly_indices),
                'anomaly_percentage': float(len(anomaly_indices) / len(series) * 100)
            }
            
        except Exception as e:
            logger.error(f"Isolation Forest detection failed: {str(e)}")
            raise AgentError(f"Isolation Forest detection error: {str(e)}") from e

    async def get_anomaly_impact_assessment(self) -> Dict[str, Any]:
        """
        Assess the impact of detected anomalies on the time series.
        
        Returns:
            Dictionary with anomaly impact assessment
        """
        if not self.is_fitted or not self.detected_anomalies:
            return {'status': 'no_anomalies', 'impact': 'none'}
        
        try:
            anomaly_indices = [a['index'] for a in self.detected_anomalies]
            anomaly_values = [a['value'] for a in self.detected_anomalies]
            
            # Calculate impact metrics
            impact_assessment = {
                'data_quality_impact': self._assess_data_quality_impact(),
                'forecasting_impact': self._assess_forecasting_impact(),
                'business_impact': self._assess_business_impact(),
                'severity_classification': self._classify_anomaly_severity()
            }
            
            return {
                'status': 'assessed',
                'n_anomalies': len(self.detected_anomalies),
                'impact_assessment': impact_assessment,
                'recommendations': await self._generate_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Anomaly impact assessment failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def _assess_data_quality_impact(self) -> Dict[str, Any]:
        """Assess impact on data quality."""
        if not self.detected_anomalies:
            return {'quality_score': 1.0, 'impact_level': 'none'}
        
        n_anomalies = len(self.detected_anomalies)
        series_length = self.anomaly_results.get('metadata', {}).get('series_length', 1000)
        
        # Calculate anomaly rate
        anomaly_rate = n_anomalies / series_length
        
        # Assess quality impact
        if anomaly_rate < 0.01:  # < 1%
            quality_score = 0.95
            impact_level = 'low'
        elif anomaly_rate < 0.05:  # < 5%
            quality_score = 0.80
            impact_level = 'medium'
        elif anomaly_rate < 0.10:  # < 10%
            quality_score = 0.60
            impact_level = 'high'
        else:  # >= 10%
            quality_score = 0.40
            impact_level = 'severe'
        
        return {
            'quality_score': float(quality_score),
            'impact_level': impact_level,
            'anomaly_rate': float(anomaly_rate),
            'n_anomalies': n_anomalies,
            'series_length': series_length
        }

    def _assess_forecasting_impact(self) -> Dict[str, Any]:
        """Assess impact on forecasting accuracy."""
        if not self.detected_anomalies:
            return {'impact_level': 'none', 'confidence_reduction': 0.0}
        
        # Analyze anomaly distribution
        anomaly_indices = [a['index'] for a in self.detected_anomalies]
        series_length = self.anomaly_results.get('metadata', {}).get('series_length', 1000)
        
        # Check if anomalies are recent (affects forecasting more)
        recent_threshold = int(series_length * 0.2)  # Last 20% of data
        recent_anomalies = sum(1 for idx in anomaly_indices if idx >= series_length - recent_threshold)
        
        # Check anomaly clustering
        clustering_patterns = self.anomaly_results.get('clustering_patterns', {})
        has_clusters = clustering_patterns.get('patterns', {}).get('clustering', {}).get('n_clusters', 0) > 0
        
        # Assess forecasting impact
        if recent_anomalies == 0 and not has_clusters:
            impact_level = 'low'
            confidence_reduction = 0.1
        elif recent_anomalies <= 2 and not has_clusters:
            impact_level = 'medium'
            confidence_reduction = 0.25
        elif recent_anomalies > 2 or has_clusters:
            impact_level = 'high'
            confidence_reduction = 0.50
        else:
            impact_level = 'severe'
            confidence_reduction = 0.75
        
        return {
            'impact_level': impact_level,
            'confidence_reduction': float(confidence_reduction),
            'recent_anomalies': recent_anomalies,
            'has_clustering': has_clusters,
            'recommendation': 'Consider anomaly treatment before forecasting'
        }

    def _assess_business_impact(self) -> Dict[str, Any]:
        """Assess business impact of anomalies."""
        if not self.detected_anomalies:
            return {'impact_level': 'none', 'business_risk': 'low'}
        
        # Analyze anomaly magnitudes
        anomaly_values = [a['value'] for a in self.detected_anomalies]
        anomaly_scores = []
        
        for anomaly in self.detected_anomalies:
            # Get highest score from different methods
            scores = anomaly.get('scores', {})
            max_score = max(scores.values()) if scores else 0
            anomaly_scores.append(max_score)
        
        # Business impact based on anomaly severity and frequency
        avg_score = np.mean(anomaly_scores) if anomaly_scores else 0
        n_anomalies = len(self.detected_anomalies)
        
        if avg_score < 2.0 and n_anomalies < 5:
            impact_level = 'low'
            business_risk = 'low'
        elif avg_score < 3.0 and n_anomalies < 10:
            impact_level = 'medium'
            business_risk = 'medium'
        elif avg_score < 4.0 or n_anomalies < 20:
            impact_level = 'high'
            business_risk = 'high'
        else:
            impact_level = 'severe'
            business_risk = 'critical'
        
        return {
            'impact_level': impact_level,
            'business_risk': business_risk,
            'avg_anomaly_score': float(avg_score),
            'n_anomalies': n_anomalies,
            'requires_investigation': business_risk in ['high', 'critical']
        }

    def _classify_anomaly_severity(self) -> Dict[str, Any]:
        """Classify anomalies by severity."""
        if not self.detected_anomalies:
            return {'severity_distribution': {}, 'dominant_severity': 'none'}
        
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for anomaly in self.detected_anomalies:
            scores = anomaly.get('scores', {})
            detection_methods = anomaly.get('detection_methods', [])
            
            # Get maximum score across methods
            max_score = max(scores.values()) if scores else 0
            n_methods = len(detection_methods)
            
            # Classify severity based on score and detection consistency
            if max_score < 2.0 or n_methods == 1:
                severity_counts['low'] += 1
            elif max_score < 3.0 or n_methods == 2:
                severity_counts['medium'] += 1
            elif max_score < 4.0 or n_methods == 3:
                severity_counts['high'] += 1
            else:
                severity_counts['critical'] += 1
        
        # Find dominant severity
        dominant_severity = max(severity_counts, key=severity_counts.get)
        
        return {
            'severity_distribution': severity_counts,
            'dominant_severity': dominant_severity,
            'total_anomalies': len(self.detected_anomalies),
            'critical_anomalies': severity_counts['critical'],
            'high_severity_anomalies': severity_counts['high']
        }

    async def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on anomaly analysis."""
        if not self.detected_anomalies:
            return ["No anomalies detected. Data quality appears good."]
        
        recommendations = []
        
        # Data quality recommendations
        data_quality = self._assess_data_quality_impact()
        if data_quality['impact_level'] in ['high', 'severe']:
            recommendations.append(
                f"High anomaly rate ({data_quality['anomaly_rate']:.2%}) detected. "
                "Consider data cleaning and validation procedures."
            )
        
        # Forecasting recommendations
        forecasting_impact = self._assess_forecasting_impact()
        if forecasting_impact['recent_anomalies'] > 0:
            recommendations.append(
                f"{forecasting_impact['recent_anomalies']} recent anomalies may affect forecasting. "
                "Consider anomaly treatment or robust forecasting methods."
            )
        
        # Business recommendations
        business_impact = self._assess_business_impact()
        if business_impact['requires_investigation']:
            recommendations.append(
                f"Business risk level: {business_impact['business_risk']}. "
                "Investigate root causes of anomalies."
            )
        
        # Severity-based recommendations
        severity = self._classify_anomaly_severity()
        if severity['critical_anomalies'] > 0:
            recommendations.append(
                f"{severity['critical_anomalies']} critical anomalies require immediate attention."
            )
        
        # Pattern-based recommendations
        temporal_patterns = self.anomaly_results.get('temporal_patterns', {})
        if temporal_patterns.get('patterns', {}).get('periodicity', {}).get('periodic_patterns', False):
            recommendations.append(
                "Periodic anomaly patterns detected. Consider seasonal adjustments or "
                "systematic issue investigation."
            )
        
        clustering_patterns = self.anomaly_results.get('clustering_patterns', {})
        if clustering_patterns.get('patterns', {}).get('clustering', {}).get('n_clusters', 0) > 0:
            recommendations.append(
                "Clustered anomalies detected. Investigate time-specific events or "
                "systematic issues during these periods."
            )
        
        # Default recommendation if no specific issues
        if not recommendations:
            recommendations.append(
                "Low-to-moderate anomaly levels detected. Monitor trends and "
                "consider preprocessing for improved model performance."
            )
        
        return recommendations

    async def export_anomaly_report(self, 
                                format: str = 'dict',
                                include_details: bool = True) -> Union[Dict, str]:
        """
        Export comprehensive anomaly analysis report.
        
        Args:
            format: Export format ('dict', 'json', 'summary')
            include_details: Whether to include detailed analysis
            
        Returns:
            Formatted anomaly report
        """
        if not self.is_fitted:
            return {'error': 'Anomaly analysis not performed yet'}
        
        try:
            # Create comprehensive report
            report = {
                'summary': self.get_analysis_summary(),
                'anomaly_count': len(self.detected_anomalies),
                'impact_assessment': await self.get_anomaly_impact_assessment(),
            }
            
            if include_details:
                report.update({
                    'detailed_results': self.anomaly_results,
                    'detected_anomalies': self.detected_anomalies,
                    'recommendations': await self._generate_recommendations()
                })
            
            if format == 'json':
                import json
                return json.dumps(report, indent=2, default=str)
            elif format == 'summary':
                return self._create_summary_report(report)
            else:
                return report
                
        except Exception as e:
            logger.error(f"Report export failed: {str(e)}")
            return {'error': f'Report export failed: {str(e)}'}

    def _create_summary_report(self, report: Dict[str, Any]) -> str:
        """Create human-readable summary report."""
        summary = report['summary']
        impact = report['impact_assessment']
        
        report_text = f"""
        ANOMALY DETECTION SUMMARY REPORT
        ================================
        
        Analysis Status: {summary['status']}
        Total Anomalies: {summary['n_anomalies']}
        Anomaly Rate: {summary.get('anomaly_rate', 0):.2%}
        
        IMPACT ASSESSMENT
        -----------------
        Data Quality Score: {impact['impact_assessment']['data_quality_impact']['quality_score']:.2f}
        Forecasting Impact: {impact['impact_assessment']['forecasting_impact']['impact_level']}
        Business Risk: {impact['impact_assessment']['business_impact']['business_risk']}
        
        SEVERITY BREAKDOWN
        ------------------
        Critical: {impact['impact_assessment']['severity_classification']['critical_anomalies']}
        High: {impact['impact_assessment']['severity_classification']['high_severity_anomalies']}
        Total: {summary['n_anomalies']}
        
        RECOMMENDATIONS
        ---------------
        """
        
        for i, rec in enumerate(report.get('recommendations', []), 1):
            report_text += f"{i}. {rec}\n    "
        
        return report_text

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of anomaly analysis results."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        # Calculate basic statistics
        n_anomalies = len(self.detected_anomalies)
        series_length = self.anomaly_results.get('metadata', {}).get('series_length', 0)
        anomaly_rate = n_anomalies / series_length if series_length > 0 else 0
        
        # Get method statistics
        statistical_methods = len(self.anomaly_results.get('statistical_outliers', {}))
        has_isolation_forest = bool(self.anomaly_results.get('isolation_forest', {}).get('n_anomalies', 0))
        has_contextual = bool(self.anomaly_results.get('contextual_anomalies', {}).get('n_contextual_anomalies', 0))
        
        # Get significance
        significance = self.anomaly_results.get('significance_validation', {})
        is_significant = significance.get('is_significant', False)
        
        return {
            'status': 'fitted',
            'n_anomalies': n_anomalies,
            'anomaly_rate': float(anomaly_rate),
            'series_length': series_length,
            'detection_methods_used': {
                'statistical_methods': statistical_methods,
                'isolation_forest': has_isolation_forest,
                'contextual_analysis': has_contextual
            },
            'statistical_significance': is_significant,
            'analysis_timestamp': self.anomaly_results.get('metadata', {}).get('timestamp', ''),
            'llm_model': self.llm_model
        }

    async def clear_analysis(self):
        """Clear all analysis results and reset agent state."""
        self.is_fitted = False
        self.anomaly_results = {}
        self.detected_anomalies = []
        self.anomaly_scores = []
        logger.info("AnomalyAgent: Analysis results cleared")

    def __repr__(self) -> str:
        """String representation of the agent."""
        status = "fitted" if self.is_fitted else "not_fitted"
        n_anomalies = len(self.detected_anomalies) if self.is_fitted else 0
        return f"AnomalyDetectionAgent(status={status}, anomalies_detected={n_anomalies}, llm_model={self.llm_model})"

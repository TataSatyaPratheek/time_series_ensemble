"""
Enhanced CrewAI configuration for multi-agent orchestration in Time Series Forecasting Ensemble.
Supports async operations, local LLM integration, and sophisticated workflow management.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseSettings, Field, validator
from datetime import datetime, timedelta
import os

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CrewAIConfig(BaseSettings):
    """Enhanced configuration settings for CrewAI orchestration with async support."""
    
    # === CORE CREWAI SETTINGS ===
    telemetry_opt_out: bool = Field(default=True, env="CREWAI_TELEMETRY_OPT_OUT")
    log_level: str = Field(default="INFO", env="CREWAI_LOG_LEVEL")
    max_execution_time: int = Field(default=600, env="CREWAI_MAX_EXECUTION_TIME")  # 10 minutes
    memory_enabled: bool = Field(default=True, env="CREWAI_MEMORY_ENABLED")
    
    # === MULTI-AGENT COORDINATION ===
    max_concurrent_agents: int = Field(default=2, env="CREWAI_MAX_CONCURRENT_AGENTS")  # M1 Air optimized
    sequential_execution: bool = Field(default=False, env="CREWAI_SEQUENTIAL_EXECUTION")
    agent_timeout: int = Field(default=180, env="CREWAI_AGENT_TIMEOUT")  # 3 minutes per agent
    coordination_timeout: int = Field(default=300, env="CREWAI_COORDINATION_TIMEOUT")  # 5 minutes
    
    # === RETRY AND RESILIENCE ===
    retry_attempts: int = Field(default=3, env="CREWAI_RETRY_ATTEMPTS")
    backoff_factor: float = Field(default=1.5, env="CREWAI_BACKOFF_FACTOR")
    retry_delay: float = Field(default=2.0, env="CREWAI_RETRY_DELAY")
    graceful_degradation: bool = Field(default=True, env="CREWAI_GRACEFUL_DEGRADATION")
    
    # === LOCAL LLM INTEGRATION ===
    local_llm_enabled: bool = Field(default=True, env="CREWAI_LOCAL_LLM_ENABLED")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    default_llm_temperature: float = Field(default=0.1, env="CREWAI_DEFAULT_TEMPERATURE")
    llm_context_length: int = Field(default=8192, env="CREWAI_CONTEXT_LENGTH")
    
    # === WORKFLOW MANAGEMENT ===
    enable_dynamic_workflows: bool = Field(default=True, env="CREWAI_DYNAMIC_WORKFLOWS")
    workflow_checkpoints: bool = Field(default=True, env="CREWAI_WORKFLOW_CHECKPOINTS")
    intermediate_results_storage: bool = Field(default=True, env="CREWAI_STORE_INTERMEDIATE")
    
    # === PERFORMANCE OPTIMIZATION ===
    enable_caching: bool = Field(default=True, env="CREWAI_ENABLE_CACHING")
    cache_ttl: int = Field(default=3600, env="CREWAI_CACHE_TTL")  # 1 hour
    memory_limit_mb: int = Field(default=4096, env="CREWAI_MEMORY_LIMIT")  # 4GB for M1 Air
    
    # === MONITORING AND LOGGING ===
    enable_metrics: bool = Field(default=True, env="CREWAI_ENABLE_METRICS")
    metrics_interval: int = Field(default=30, env="CREWAI_METRICS_INTERVAL")  # seconds
    log_agent_communications: bool = Field(default=True, env="CREWAI_LOG_COMMUNICATIONS")
    performance_monitoring: bool = Field(default=True, env="CREWAI_PERFORMANCE_MONITORING")
    
    @validator('max_concurrent_agents')
    def validate_concurrent_agents(cls, v):
        """Ensure concurrent agents don't exceed hardware limits."""
        max_safe = min(4, settings.MAX_WORKERS)  # Safe for M1 Air
        if v > max_safe:
            logger.warning(f"Reducing concurrent agents from {v} to {max_safe} for hardware compatibility")
            return max_safe
        return v
    
    @validator('memory_limit_mb')
    def validate_memory_limit(cls, v):
        """Ensure memory limit is reasonable for M1 Air."""
        max_memory = settings.MAX_MEMORY_USAGE_GB * 1024  # Convert to MB
        if v > max_memory:
            logger.warning(f"Reducing memory limit from {v}MB to {max_memory}MB")
            return int(max_memory)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class WorkflowConfig(BaseSettings):
    """Configuration for workflow execution patterns."""
    
    # === EXECUTION PATTERNS ===
    default_execution_mode: str = Field(default="hybrid", env="WORKFLOW_EXECUTION_MODE")  # sequential, parallel, hybrid
    enable_conditional_execution: bool = Field(default=True, env="WORKFLOW_CONDITIONAL_EXECUTION")
    adaptive_resource_allocation: bool = Field(default=True, env="WORKFLOW_ADAPTIVE_RESOURCES")
    
    # === TASK DEPENDENCIES ===
    dependency_resolution: str = Field(default="smart", env="WORKFLOW_DEPENDENCY_RESOLUTION")  # strict, smart, relaxed
    circular_dependency_detection: bool = Field(default=True, env="WORKFLOW_CIRCULAR_DEPENDENCY_CHECK")
    
    # === ERROR HANDLING ===
    continue_on_agent_failure: bool = Field(default=True, env="WORKFLOW_CONTINUE_ON_FAILURE")
    minimum_successful_agents: int = Field(default=2, env="WORKFLOW_MIN_SUCCESSFUL_AGENTS")
    fallback_strategies: bool = Field(default=True, env="WORKFLOW_FALLBACK_STRATEGIES")
    
    @validator('default_execution_mode')
    def validate_execution_mode(cls, v):
        valid_modes = ['sequential', 'parallel', 'hybrid']
        if v not in valid_modes:
            raise ValueError(f"Execution mode must be one of {valid_modes}")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class AgentPoolConfig(BaseSettings):
    """Configuration for agent pool management."""
    
    # === AGENT LIFECYCLE ===
    agent_warm_start: bool = Field(default=True, env="AGENT_WARM_START")
    agent_connection_pooling: bool = Field(default=True, env="AGENT_CONNECTION_POOLING")
    agent_health_checks: bool = Field(default=True, env="AGENT_HEALTH_CHECKS")
    health_check_interval: int = Field(default=60, env="AGENT_HEALTH_CHECK_INTERVAL")  # seconds
    
    # === RESOURCE MANAGEMENT ===
    max_agent_memory_mb: int = Field(default=1024, env="AGENT_MAX_MEMORY_MB")  # 1GB per agent
    agent_cpu_limit: float = Field(default=0.8, env="AGENT_CPU_LIMIT")  # 80% CPU per agent
    resource_monitoring: bool = Field(default=True, env="AGENT_RESOURCE_MONITORING")
    
    # === SCALING ===
    auto_scaling: bool = Field(default=False, env="AGENT_AUTO_SCALING")  # Disabled for local deployment
    scale_up_threshold: float = Field(default=0.8, env="AGENT_SCALE_UP_THRESHOLD")
    scale_down_threshold: float = Field(default=0.3, env="AGENT_SCALE_DOWN_THRESHOLD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class OrchestrationMetrics:
    """Metrics collection for orchestration monitoring."""
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all metrics to initial state."""
        self.start_time = None
        self.end_time = None
        self.agent_execution_times = {}
        self.task_completion_status = {}
        self.error_counts = {}
        self.resource_usage = {}
        self.communication_stats = {}
    
    def start_orchestration(self):
        """Mark the start of orchestration."""
        self.start_time = datetime.now()
        logger.info("Orchestration metrics collection started")
    
    def end_orchestration(self):
        """Mark the end of orchestration."""
        self.end_time = datetime.now()
        total_time = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Orchestration completed in {total_time:.2f} seconds")
    
    def record_agent_execution(self, agent_name: str, execution_time: float, status: str):
        """Record agent execution metrics."""
        self.agent_execution_times[agent_name] = execution_time
        self.task_completion_status[agent_name] = status
    
    def record_error(self, error_type: str, agent_name: Optional[str] = None):
        """Record error occurrence."""
        key = f"{error_type}_{agent_name}" if agent_name else error_type
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get orchestration metrics summary."""
        total_time = 0
        if self.start_time and self.end_time:
            total_time = (self.end_time - self.start_time).total_seconds()
        
        return {
            'orchestration_duration': total_time,
            'agents_executed': len(self.agent_execution_times),
            'successful_agents': len([s for s in self.task_completion_status.values() if s == 'success']),
            'failed_agents': len([s for s in self.task_completion_status.values() if s == 'failed']),
            'total_errors': sum(self.error_counts.values()),
            'agent_execution_times': self.agent_execution_times,
            'error_breakdown': self.error_counts
        }


# Global configuration instances
crew_config = CrewAIConfig()
workflow_config = WorkflowConfig()
agent_pool_config = AgentPoolConfig()
orchestration_metrics = OrchestrationMetrics()

# Configuration validation
def validate_configuration() -> Dict[str, Any]:
    """Validate all configuration settings."""
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    try:
        # Validate CrewAI config
        if crew_config.max_concurrent_agents > 4:
            validation_results['warnings'].append(
                f"High concurrent agents ({crew_config.max_concurrent_agents}) may exceed M1 Air capacity"
            )
        
        # Validate memory settings
        total_memory_mb = crew_config.memory_limit_mb + (
            agent_pool_config.max_agent_memory_mb * crew_config.max_concurrent_agents
        )
        if total_memory_mb > 6144:  # 6GB threshold for 8GB system
            validation_results['warnings'].append(
                f"Total memory allocation ({total_memory_mb}MB) may exceed system capacity"
            )
        
        # Validate timeout settings
        if crew_config.agent_timeout > crew_config.max_execution_time:
            validation_results['errors'].append(
                "Agent timeout cannot exceed max execution time"
            )
        
        # Validate LLM connectivity
        if crew_config.local_llm_enabled:
            import httpx
            try:
                response = httpx.get(f"{crew_config.ollama_base_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    validation_results['warnings'].append(
                        f"Ollama service not responding at {crew_config.ollama_base_url}"
                    )
            except Exception as e:
                validation_results['warnings'].append(
                    f"Cannot connect to Ollama: {str(e)}"
                )
        
        if validation_results['errors']:
            validation_results['valid'] = False
        
        logger.info(f"Configuration validation: {'PASSED' if validation_results['valid'] else 'FAILED'}")
        
    except Exception as e:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Configuration validation error: {str(e)}")
    
    return validation_results

# Helper functions
def get_optimal_agent_config(agent_type: str) -> Dict[str, Any]:
    """Get optimal configuration for specific agent type."""
    base_config = {
        'timeout': crew_config.agent_timeout,
        'retry_attempts': crew_config.retry_attempts,
        'memory_enabled': crew_config.memory_enabled,
        'llm_temperature': crew_config.default_llm_temperature
    }
    
    # Agent-specific optimizations
    agent_optimizations = {
        'trend': {
            'llm_temperature': 0.1,  # Low temperature for analytical tasks
            'timeout': 120,  # Trend analysis can be complex
            'priority': 'high'
        },
        'seasonality': {
            'llm_temperature': 0.2,
            'timeout': 150,  # Seasonal analysis may take longer
            'priority': 'high'
        },
        'anomaly': {
            'llm_temperature': 0.3,  # Slightly higher for pattern recognition
            'timeout': 100,
            'priority': 'medium'
        },
        'ensemble': {
            'llm_temperature': 0.1,  # Low for coordination tasks
            'timeout': 200,  # Needs time for model combination
            'priority': 'critical'
        }
    }
    
    if agent_type in agent_optimizations:
        base_config.update(agent_optimizations[agent_type])
    
    return base_config

def setup_orchestration_environment():
    """Setup the orchestration environment with all necessary configurations."""
    try:
        # Validate configuration
        validation = validate_configuration()
        if not validation['valid']:
            logger.error(f"Configuration validation failed: {validation['errors']}")
            raise ValueError("Invalid configuration")
        
        # Log warnings if any
        for warning in validation['warnings']:
            logger.warning(warning)
        
        # Setup telemetry
        if crew_config.telemetry_opt_out:
            os.environ['CREWAI_TELEMETRY_OPT_OUT'] = 'true'
        
        # Setup logging level
        logging.getLogger('crewai').setLevel(crew_config.log_level)
        
        # Initialize metrics
        orchestration_metrics.reset_metrics()
        
        logger.info("Orchestration environment setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup orchestration environment: {str(e)}")
        return False

# Export configuration summary
def get_configuration_summary() -> Dict[str, Any]:
    """Get summary of current configuration."""
    return {
        'crew_config': {
            'max_concurrent_agents': crew_config.max_concurrent_agents,
            'max_execution_time': crew_config.max_execution_time,
            'memory_enabled': crew_config.memory_enabled,
            'local_llm_enabled': crew_config.local_llm_enabled,
            'retry_attempts': crew_config.retry_attempts
        },
        'workflow_config': {
            'execution_mode': workflow_config.default_execution_mode,
            'conditional_execution': workflow_config.enable_conditional_execution,
            'continue_on_failure': workflow_config.continue_on_agent_failure,
            'min_successful_agents': workflow_config.minimum_successful_agents
        },
        'agent_pool_config': {
            'warm_start': agent_pool_config.agent_warm_start,
            'health_checks': agent_pool_config.agent_health_checks,
            'max_memory_per_agent': agent_pool_config.max_agent_memory_mb,
            'resource_monitoring': agent_pool_config.resource_monitoring
        },
        'hardware_optimization': {
            'platform': 'M1_Air_8GB',
            'memory_limit_mb': crew_config.memory_limit_mb,
            'cpu_cores_available': settings.MAX_WORKERS,
            'ollama_integration': crew_config.local_llm_enabled
        }
    }

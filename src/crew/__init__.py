"""
Enhanced Crew module initialization.
Exports configuration, orchestration, and workflow management for multi-agent systems.
"""

import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

from .crew_config import (
    crew_config,
    workflow_config,
    agent_pool_config,
    orchestration_metrics,
    CrewAIConfig,
    WorkflowConfig,
    AgentPoolConfig,
    OrchestrationMetrics,
    validate_configuration,
    setup_orchestration_environment,
    get_optimal_agent_config,
    get_configuration_summary
)

from .tasks import (
    EnhancedTask,
    TaskStatus,
    TaskPriority,
    TaskMetadata,
    WorkflowOrchestrator,
    create_trend_analysis_task,
    create_seasonality_detection_task,
    create_anomaly_detection_task,
    create_ensemble_coordination_task,
    create_standard_workflow,
    create_fast_workflow,
    create_comprehensive_workflow,
    get_workflow_by_name,
    WORKFLOW_REGISTRY,
    TASKS,
    ORCHESTRATION_FLOW,
    get_task_by_name
)

__all__ = [
    # Configuration classes and instances
    'CrewAIConfig',
    'WorkflowConfig', 
    'AgentPoolConfig',
    'OrchestrationMetrics',
    'crew_config',
    'workflow_config',
    'agent_pool_config',
    'orchestration_metrics',
    
    # Configuration utilities
    'validate_configuration',
    'setup_orchestration_environment',
    'get_optimal_agent_config',
    'get_configuration_summary',
    
    # Task and workflow classes
    'EnhancedTask',
    'TaskStatus',
    'TaskPriority', 
    'TaskMetadata',
    'WorkflowOrchestrator',
    
    # Task creation functions
    'create_trend_analysis_task',
    'create_seasonality_detection_task',
    'create_anomaly_detection_task',
    'create_ensemble_coordination_task',
    
    # Workflow creation functions
    'create_standard_workflow',
    'create_fast_workflow',
    'create_comprehensive_workflow',
    'get_workflow_by_name',
    
    # Registries and legacy compatibility
    'WORKFLOW_REGISTRY',
    'TASKS',
    'ORCHESTRATION_FLOW',
    'get_task_by_name'
]

# Module metadata
__version__ = "0.1.0"
__author__ = "Time Series Ensemble Team"
__description__ = "Advanced multi-agent orchestration for time series forecasting with CrewAI"

# Workflow templates
WORKFLOW_TEMPLATES = {
    'standard': {
        'description': 'Standard workflow with all analysis components',
        'execution_time': '5-10 minutes',
        'resource_usage': 'moderate',
        'accuracy': 'high'
    },
    'fast': {
        'description': 'Fast workflow for quick analysis',
        'execution_time': '2-5 minutes', 
        'resource_usage': 'low',
        'accuracy': 'medium'
    },
    'comprehensive': {
        'description': 'Comprehensive analysis with extended timeouts',
        'execution_time': '10-20 minutes',
        'resource_usage': 'high',
        'accuracy': 'very high'
    }
}

# Utility functions
def list_available_workflows() -> List[str]:
    """List all available workflow templates."""
    return list(WORKFLOW_REGISTRY.keys())

def get_workflow_info(workflow_name: str) -> Dict[str, Any]:
    """Get information about a specific workflow."""
    if workflow_name not in WORKFLOW_TEMPLATES:
        raise ValueError(f"Workflow '{workflow_name}' not found")
    
    return WORKFLOW_TEMPLATES[workflow_name]

def create_custom_workflow(tasks: List[Dict[str, Any]]) -> WorkflowOrchestrator:
    """
    Create custom workflow from task specifications.
    
    Args:
        tasks: List of task specifications with name, agent_class, dependencies, etc.
        
    Returns:
        Configured WorkflowOrchestrator
    """
    orchestrator = WorkflowOrchestrator()
    
    for task_spec in tasks:
        task = EnhancedTask(
            name=task_spec['name'],
            description=task_spec.get('description', f"{task_spec['name']} task"),
            agent_class=task_spec['agent_class'],
            agent_method=task_spec.get('agent_method', 'analyze_comprehensive'),
            dependencies=task_spec.get('dependencies', []),
            priority=TaskPriority[task_spec.get('priority', 'MEDIUM')],
            timeout=task_spec.get('timeout'),
            retry_attempts=task_spec.get('retry_attempts')
        )
        orchestrator.add_task(task)
    
    return orchestrator

async def quick_forecast(series: 'pd.Series', 
                        workflow_type: str = 'standard',
                        **kwargs) -> Dict[str, Any]:
    """
    Quick forecast using specified workflow.
    
    Args:
        series: Time series data
        workflow_type: Type of workflow to use
        **kwargs: Additional workflow parameters
        
    Returns:
        Forecast results
    """
    try:
        # Setup environment
        if not setup_orchestration_environment():
            raise RuntimeError("Failed to setup orchestration environment")
        
        # Create workflow
        orchestrator = get_workflow_by_name(workflow_type)
        
        # Execute workflow
        results = await orchestrator.execute_workflow(
            {'series': series}, 
            **kwargs
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Quick forecast failed: {str(e)}")
        raise

# Auto-setup on import
try:
    setup_orchestration_environment()
    logger.info("CrewAI orchestration environment initialized successfully")
except Exception as e:
    logger.warning(f"Failed to auto-setup orchestration environment: {str(e)}")

# Export workflow creation shortcuts
def standard_workflow() -> WorkflowOrchestrator:
    """Shortcut for creating standard workflow."""
    return create_standard_workflow()

def fast_workflow() -> WorkflowOrchestrator:
    """Shortcut for creating fast workflow."""
    return create_fast_workflow()

def comprehensive_workflow() -> WorkflowOrchestrator:
    """Shortcut for creating comprehensive workflow."""
    return create_comprehensive_workflow()

# Add shortcuts to exports
__all__.extend([
    'WORKFLOW_TEMPLATES',
    'list_available_workflows',
    'get_workflow_info', 
    'create_custom_workflow',
    'quick_forecast',
    'standard_workflow',
    'fast_workflow',
    'comprehensive_workflow'
])

"""
Agent module initialization.
Exports specialized agents for time series analysis and multi-agent orchestration.
"""

from typing import Optional, List, Dict, Any

from .trend_agent import TrendAnalysisAgent
from .seasonality_agent import SeasonalityAgent
from .anomaly_agent import AnomalyDetectionAgent
from .ensemble_coordinator import EnsembleCoordinatorAgent

__all__ = [
    'TrendAnalysisAgent',
    'SeasonalityAgent', 
    'AnomalyDetectionAgent',
    'EnsembleCoordinatorAgent'
]

# Agent registry for dynamic agent loading
AGENT_REGISTRY = {
    'trend': TrendAnalysisAgent,
    'trend_analysis': TrendAnalysisAgent,
    'seasonality': SeasonalityAgent,
    'seasonal': SeasonalityAgent,
    'anomaly': AnomalyDetectionAgent,
    'anomaly_detection': AnomalyDetectionAgent,
    'ensemble': EnsembleCoordinatorAgent,
    'coordinator': EnsembleCoordinatorAgent,
    'ensemble_coordinator': EnsembleCoordinatorAgent
}

# Agent categories
AGENT_CATEGORIES = {
    'analysis': ['trend', 'seasonality', 'anomaly'],
    'coordination': ['ensemble', 'coordinator'],
    'all': list(AGENT_REGISTRY.keys())
}

def get_agent_by_name(agent_name: str, **kwargs):
    """
    Get agent instance by name from registry.
    
    Args:
        agent_name: Name of the agent
        **kwargs: Agent initialization parameters
        
    Returns:
        Agent instance
        
    Raises:
        ValueError: If agent name not found
    """
    if agent_name not in AGENT_REGISTRY:
        available_agents = list(AGENT_REGISTRY.keys())
        raise ValueError(f"Agent '{agent_name}' not found. Available agents: {available_agents}")
    
    agent_class = AGENT_REGISTRY[agent_name]
    return agent_class(**kwargs)

def list_available_agents(category: Optional[str] = None) -> List[str]:
    """
    List available agents by category.
    
    Args:
        category: Agent category ('analysis', 'coordination', or None for all)
        
    Returns:
        List of available agent names
    """
    if category is None:
        return list(AGENT_REGISTRY.keys())
    
    if category not in AGENT_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Available: {list(AGENT_CATEGORIES.keys())}")
    
    return AGENT_CATEGORIES[category]

def create_agent_ensemble(agents: List[str], **kwargs) -> Dict[str, Any]:
    """
    Create multiple agents for ensemble coordination.
    
    Args:
        agents: List of agent names to create
        **kwargs: Common initialization parameters
        
    Returns:
        Dictionary of initialized agents
    """
    agent_ensemble = {}
    
    for agent_name in agents:
        try:
            agent_ensemble[agent_name] = get_agent_by_name(agent_name, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to create agent '{agent_name}': {str(e)}")
    
    return agent_ensemble

# Setup logger
import logging
logger = logging.getLogger(__name__)

# Version and metadata
__version__ = "0.1.0"
__author__ = "Time Series Ensemble Team"
__description__ = "Specialized agents for multi-agent time series forecasting with CrewAI integration"

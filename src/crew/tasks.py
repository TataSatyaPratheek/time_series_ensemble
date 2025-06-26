"""
Enhanced task definitions and multi-agent coordination for CrewAI orchestration.
Supports async execution, dynamic workflows, and sophisticated error handling.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from crewai import Task, Crew, Agent
from crewai.task import TaskOutput

from src.agents import (
    TrendAnalysisAgent,
    SeasonalityAgent,
    AnomalyDetectionAgent,
    EnsembleCoordinatorAgent
)
from src.utils.exceptions import AgentError
from src.utils.logging import get_logger
from .crew_config import crew_config, workflow_config, orchestration_metrics, get_optimal_agent_config

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Task execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskMetadata:
    """Metadata for task execution tracking."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    retry_count: int = 0
    error_message: Optional[str] = None
    dependencies_met: bool = False
    resource_usage: Dict[str, Any] = field(default_factory=dict)


class EnhancedTask:
    """Enhanced task wrapper with advanced orchestration capabilities."""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 agent_class: type,
                 agent_method: str = "analyze_comprehensive",
                 dependencies: Optional[List[str]] = None,
                 priority: TaskPriority = TaskPriority.MEDIUM,
                 timeout: Optional[int] = None,
                 retry_attempts: Optional[int] = None,
                 conditional_execution: Optional[Callable] = None,
                 post_processing: Optional[Callable] = None):
        """
        Initialize enhanced task.
        
        Args:
            name: Task name
            description: Task description
            agent_class: Agent class to execute
            agent_method: Method to call on agent
            dependencies: List of dependency task names
            priority: Task priority
            timeout: Task timeout in seconds
            retry_attempts: Number of retry attempts
            conditional_execution: Function to determine if task should run
            post_processing: Function to process task results
        """
        self.name = name
        self.description = description
        self.agent_class = agent_class
        self.agent_method = agent_method
        self.dependencies = dependencies or []
        self.priority = priority
        self.timeout = timeout or crew_config.agent_timeout
        self.retry_attempts = retry_attempts or crew_config.retry_attempts
        self.conditional_execution = conditional_execution
        self.post_processing = post_processing
        
        # Task state
        self.metadata = TaskMetadata(priority=priority)
        self.agent_instance = None
        self.crewai_task = None
        self.results = None
        self.context_data = {}
        
        # Performance tracking
        self.execution_history = []
        self.resource_usage = {}
    
    def initialize_agent(self, **kwargs) -> Any:
        """Initialize the agent instance with optimal configuration."""
        try:
            if not self.agent_instance:
                # Get optimal config for agent type
                agent_type = self.agent_class.__name__.lower().replace('agent', '')
                optimal_config = get_optimal_agent_config(agent_type)
                
                # Merge with provided kwargs
                agent_kwargs = {**optimal_config, **kwargs}
                
                # Initialize agent
                self.agent_instance = self.agent_class(**agent_kwargs)
                logger.debug(f"Initialized agent {self.name} with config: {agent_kwargs}")
            
            return self.agent_instance
            
        except Exception as e:
            logger.error(f"Failed to initialize agent for task {self.name}: {str(e)}")
            raise AgentError(f"Agent initialization failed: {str(e)}") from e
    
    def create_crewai_task(self) -> Task:
        """Create CrewAI task instance."""
        try:
            if not self.agent_instance:
                self.initialize_agent()
            
            # Create CrewAI agent
            crewai_agent = self.agent_instance.get_crewai_agent()
            
            # Create task with enhanced configuration
            self.crewai_task = Task(
                description=f"{self.description}\n\nTask ID: {self.metadata.task_id}\nPriority: {self.priority.name}",
                agent=crewai_agent,
                expected_output=f"Comprehensive {self.name.lower()} analysis results with insights and recommendations",
                context=self.context_data
            )
            
            return self.crewai_task
            
        except Exception as e:
            logger.error(f"Failed to create CrewAI task for {self.name}: {str(e)}")
            raise AgentError(f"CrewAI task creation failed: {str(e)}") from e
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the task with full orchestration support.
        
        Args:
            context: Execution context and shared data
            
        Returns:
            Task execution results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Update metadata
            self.metadata.started_at = datetime.now()
            self.metadata.status = TaskStatus.RUNNING
            
            # Check conditional execution
            if self.conditional_execution and not self.conditional_execution(context):
                self.metadata.status = TaskStatus.SKIPPED
                logger.info(f"Task {self.name} skipped due to conditional execution")
                return {'status': 'skipped', 'reason': 'conditional_execution_failed'}
            
            logger.info(f"Executing task: {self.name}")
            
            # Update context
            self.context_data.update(context)
            
            # Execute agent method
            if not self.agent_instance:
                self.initialize_agent()
            
            # Get the method to execute
            if hasattr(self.agent_instance, self.agent_method):
                agent_method = getattr(self.agent_instance, self.agent_method)
            else:
                raise AttributeError(f"Agent method '{self.agent_method}' not found")
            
            # Execute with timeout
            try:
                if asyncio.iscoroutinefunction(agent_method):
                    self.results = await asyncio.wait_for(
                        agent_method(context), # Pass the entire context
                        timeout=self.timeout
                    )
                else:
                    # Run synchronous method in thread pool
                    self.results = await asyncio.wait_for(
                        asyncio.to_thread(agent_method, context), # Pass the entire context
                        timeout=self.timeout
                    )
            except asyncio.TimeoutError:
                raise AgentError(f"Task {self.name} timed out after {self.timeout} seconds")
            
            # Post-processing
            if self.post_processing:
                self.results = self.post_processing(self.results, context)
            
            # Update metadata
            execution_time = asyncio.get_event_loop().time() - start_time
            self.metadata.completed_at = datetime.now()
            self.metadata.duration = execution_time
            self.metadata.status = TaskStatus.COMPLETED
            
            # Record metrics
            orchestration_metrics.record_agent_execution(
                self.name, execution_time, 'success'
            )
            
            logger.info(f"Task {self.name} completed successfully in {execution_time:.2f}s")
            
            return {
                'status': 'completed',
                'results': self.results,
                'metadata': self.metadata,
                'execution_time': execution_time
            }
            
        except Exception as e:
            # Handle execution failure
            execution_time = asyncio.get_event_loop().time() - start_time
            self.metadata.status = TaskStatus.FAILED
            self.metadata.error_message = str(e)
            self.metadata.duration = execution_time
            
            # Record error
            orchestration_metrics.record_error('execution_error', self.name)
            
            logger.error(f"Task {self.name} failed after {execution_time:.2f}s: {str(e)}")
            
            # Check if retry is needed
            if self.metadata.retry_count < self.retry_attempts:
                self.metadata.retry_count += 1
                self.metadata.status = TaskStatus.RETRYING
                
                logger.info(f"Retrying task {self.name} (attempt {self.metadata.retry_count}/{self.retry_attempts})")
                
                # Exponential backoff
                delay = crew_config.retry_delay * (crew_config.backoff_factor ** (self.metadata.retry_count - 1))
                await asyncio.sleep(delay)
                
                return await self.execute(context)
            
            # Max retries exceeded
            raise AgentError(f"Task {self.name} failed after {self.retry_attempts} attempts: {str(e)}") from e
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get task status summary."""
        return {
            'name': self.name,
            'status': self.metadata.status.value,
            'priority': self.metadata.priority.name,
            'duration': self.metadata.duration,
            'retry_count': self.metadata.retry_count,
            'dependencies_met': self.metadata.dependencies_met,
            'created_at': self.metadata.created_at.isoformat(),
            'started_at': self.metadata.started_at.isoformat() if self.metadata.started_at else None,
            'completed_at': self.metadata.completed_at.isoformat() if self.metadata.completed_at else None,
            'error_message': self.metadata.error_message
        }


class WorkflowOrchestrator:
    """Advanced workflow orchestrator for multi-agent coordination."""
    
    def __init__(self):
        self.tasks: Dict[str, EnhancedTask] = {}
        self.execution_graph: Dict[str, List[str]] = {}
        self.execution_results: Dict[str, Any] = {}
        self.workflow_context: Dict[str, Any] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Workflow state
        self.workflow_id = str(uuid.uuid4())
        self.start_time = None
        self.end_time = None
        self.workflow_status = "initialized"
    
    def add_task(self, task: EnhancedTask):
        """Add task to the workflow."""
        self.tasks[task.name] = task
        
        # Update execution graph
        if task.name not in self.execution_graph:
            self.execution_graph[task.name] = []
        
        # Add dependencies
        for dep in task.dependencies:
            if dep not in self.execution_graph:
                self.execution_graph[dep] = []
            self.execution_graph[dep].append(task.name)
        
        logger.debug(f"Added task {task.name} with dependencies: {task.dependencies}")
    
    def validate_workflow(self) -> Dict[str, Any]:
        """Validate workflow for circular dependencies and other issues."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check for circular dependencies
            if self._has_circular_dependencies():
                validation_result['valid'] = False
                validation_result['errors'].append("Circular dependencies detected")
            
            # Check if all dependencies exist
            for task_name, task in self.tasks.items():
                for dep in task.dependencies:
                    if dep not in self.tasks:
                        validation_result['errors'].append(
                            f"Task {task_name} depends on non-existent task {dep}"
                        )
            
            # Check for orphaned tasks (no dependencies and no dependents)
            orphaned_tasks = []
            for task_name in self.tasks:
                has_dependencies = bool(self.tasks[task_name].dependencies)
                has_dependents = bool(self.execution_graph.get(task_name, []))
                
                if not has_dependencies and not has_dependents and len(self.tasks) > 1:
                    orphaned_tasks.append(task_name)
            
            if orphaned_tasks:
                validation_result['warnings'].append(
                    f"Orphaned tasks detected: {orphaned_tasks}"
                )
            
            if validation_result['errors']:
                validation_result['valid'] = False
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.execution_graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for task_name in self.tasks:
            if task_name not in visited:
                if dfs(task_name):
                    return True
        
        return False
    
    def _get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready_tasks = []
        
        for task_name, task in self.tasks.items():
            if task.metadata.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                dependencies_met = True
                for dep in task.dependencies:
                    dep_task = self.tasks.get(dep)
                    if not dep_task or dep_task.metadata.status != TaskStatus.COMPLETED:
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    task.metadata.dependencies_met = True
                    ready_tasks.append(task_name)
        
        # Sort by priority (critical first)
        ready_tasks.sort(key=lambda x: self.tasks[x].priority.value, reverse=True)
        
        return ready_tasks
    
    async def execute_workflow(self, 
                             input_data: Dict[str, Any],
                             execution_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the complete workflow with advanced orchestration.
        
        Args:
            input_data: Input data for the workflow
            execution_mode: Execution mode override
            
        Returns:
            Workflow execution results
        """
        self.start_time = datetime.now()
        self.workflow_status = "running"
        orchestration_metrics.start_orchestration()
        
        try:
            logger.info(f"Starting workflow execution: {self.workflow_id}")
            
            # Validate workflow
            validation = self.validate_workflow()
            if not validation['valid']:
                raise ValueError(f"Workflow validation failed: {validation['errors']}")
            
            # Log warnings
            for warning in validation['warnings']:
                logger.warning(warning)
            
            # Initialize workflow context
            self.workflow_context.update(input_data)
            
            # Determine execution mode
            mode = execution_mode or workflow_config.default_execution_mode
            
            if mode == "sequential":
                results = await self._execute_sequential()
            elif mode == "parallel":
                results = await self._execute_parallel()
            elif mode == "hybrid":
                results = await self._execute_hybrid()
            else:
                raise ValueError(f"Unknown execution mode: {mode}")
            
            # Finalize workflow
            self.end_time = datetime.now()
            self.workflow_status = "completed"
            orchestration_metrics.end_orchestration()
            
            execution_time = (self.end_time - self.start_time).total_seconds()
            logger.info(f"Workflow completed successfully in {execution_time:.2f}s")
            
            return {
                'workflow_id': self.workflow_id,
                'status': 'completed',
                'execution_time': execution_time,
                'results': results,
                'task_summaries': {name: task.get_status_summary() for name, task in self.tasks.items()},
                'metrics': orchestration_metrics.get_summary()
            }
            
        except Exception as e:
            self.workflow_status = "failed"
            self.end_time = datetime.now()
            orchestration_metrics.end_orchestration()
            
            logger.error(f"Workflow execution failed: {str(e)}")
            
            return {
                'workflow_id': self.workflow_id,
                'status': 'failed',
                'error': str(e),
                'task_summaries': {name: task.get_status_summary() for name, task in self.tasks.items()},
                'metrics': orchestration_metrics.get_summary()
            }
    
    async def _execute_sequential(self) -> Dict[str, Any]:
        """Execute tasks sequentially based on dependencies."""
        completed_tasks = set()
        
        while len(completed_tasks) < len(self.tasks):
            ready_tasks = self._get_ready_tasks()
            
            if not ready_tasks:
                # Check if we have failed tasks blocking progress
                failed_tasks = [name for name, task in self.tasks.items() 
                              if task.metadata.status == TaskStatus.FAILED]
                
                if failed_tasks:
                    if workflow_config.continue_on_agent_failure:
                        logger.warning(f"Continuing despite failed tasks: {failed_tasks}")
                        break
                    else:
                        raise AgentError(f"Workflow blocked by failed tasks: {failed_tasks}")
                else:
                    raise AgentError("No ready tasks found - possible dependency deadlock")
            
            # Execute first ready task
            task_name = ready_tasks[0]
            task = self.tasks[task_name]
            
            try:
                result = await task.execute(self.workflow_context)
                self.execution_results[task_name] = result
                completed_tasks.add(task_name)
                
                # Update context with results
                if result.get('results'):
                    self.workflow_context[f"{task_name}_results"] = result['results']
                
            except Exception as e:
                if workflow_config.continue_on_agent_failure:
                    logger.error(f"Task {task_name} failed, continuing: {str(e)}")
                    completed_tasks.add(task_name)  # Mark as processed even if failed
                else:
                    raise
        
        return self.execution_results
    
    async def _execute_parallel(self) -> Dict[str, Any]:
        """Execute tasks in parallel where possible."""
        while len(self.execution_results) < len(self.tasks):
            ready_tasks = self._get_ready_tasks()
            
            if not ready_tasks:
                # Wait for running tasks to complete
                if self.active_tasks:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    break
            
            # Start tasks up to concurrent limit
            available_slots = crew_config.max_concurrent_agents - len(self.active_tasks)
            tasks_to_start = ready_tasks[:available_slots]
            
            for task_name in tasks_to_start:
                task = self.tasks[task_name]
                
                # Create and start async task
                async_task = asyncio.create_task(
                    task.execute(self.workflow_context)
                )
                self.active_tasks[task_name] = async_task
            
            # Wait for at least one task to complete
            if self.active_tasks:
                done, pending = await asyncio.wait(
                    self.active_tasks.values(), 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for completed_task in done:
                    # Find task name
                    completed_task_name = None
                    for name, async_task in self.active_tasks.items():
                        if async_task == completed_task:
                            completed_task_name = name
                            break
                    
                    if completed_task_name:
                        try:
                            result = await completed_task
                            self.execution_results[completed_task_name] = result
                            
                            # Update context
                            if result.get('results'):
                                self.workflow_context[f"{completed_task_name}_results"] = result['results']
                            
                        except Exception as e:
                            if not workflow_config.continue_on_agent_failure:
                                # Cancel other tasks and re-raise
                                for pending_task in pending:
                                    pending_task.cancel()
                                raise
                            else:
                                logger.error(f"Task {completed_task_name} failed: {str(e)}")
                        
                        # Remove from active tasks
                        del self.active_tasks[completed_task_name]
        
        return self.execution_results
    
    async def _execute_hybrid(self) -> Dict[str, Any]:
        """Execute workflow using hybrid approach (parallel within dependency levels)."""
        # Group tasks by dependency level
        dependency_levels = self._calculate_dependency_levels()
        
        for level, task_names in dependency_levels.items():
            logger.info(f"Executing dependency level {level}: {task_names}")
            
            # Execute tasks at this level in parallel
            level_tasks = []
            for task_name in task_names:
                if task_name in self.tasks:
                    task = self.tasks[task_name]
                    level_tasks.append(task.execute(self.workflow_context))
            
            if level_tasks:
                try:
                    # Execute level tasks with concurrency limit
                    semaphore = asyncio.Semaphore(crew_config.max_concurrent_agents)
                    
                    async def execute_with_semaphore(task_coro, task_name):
                        async with semaphore:
                            return await task_coro, task_name
                    
                    level_coros = [
                        execute_with_semaphore(task_coro, task_names[i]) 
                        for i, task_coro in enumerate(level_tasks)
                    ]
                    
                    results = await asyncio.gather(*level_coros, return_exceptions=True)
                    
                    # Process results
                    for i, result in enumerate(results):
                        task_name = task_names[i]
                        
                        if isinstance(result, Exception):
                            if not workflow_config.continue_on_agent_failure:
                                raise result
                            else:
                                logger.error(f"Task {task_name} failed: {str(result)}")
                        else:
                            task_result, _ = result
                            self.execution_results[task_name] = task_result
                            
                            # Update context
                            if task_result.get('results'):
                                self.workflow_context[f"{task_name}_results"] = task_result['results']
                
                except Exception as e:
                    if not workflow_config.continue_on_agent_failure:
                        raise
                    else:
                        logger.error(f"Level {level} execution had errors: {str(e)}")
        
        return self.execution_results
    
    def _calculate_dependency_levels(self) -> Dict[int, List[str]]:
        """Calculate dependency levels for hybrid execution."""
        levels = {}
        visited = set()
        
        def calculate_level(task_name):
            if task_name in visited:
                return levels.get(task_name, 0)
            
            visited.add(task_name)
            task = self.tasks[task_name]
            
            if not task.dependencies:
                level = 0
            else:
                max_dep_level = max(calculate_level(dep) for dep in task.dependencies)
                level = max_dep_level + 1
            
            levels[task_name] = level
            return level
        
        # Calculate levels for all tasks
        for task_name in self.tasks:
            calculate_level(task_name)
        
        # Group by level
        level_groups = {}
        for task_name, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(task_name)
        
        return level_groups
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        task_statuses = {}
        for name, task in self.tasks.items():
            task_statuses[name] = task.metadata.status.value
        
        return {
            'workflow_id': self.workflow_id,
            'status': self.workflow_status,
            'tasks': task_statuses,
            'completed_tasks': len([t for t in self.tasks.values() if t.metadata.status == TaskStatus.COMPLETED]),
            'total_tasks': len(self.tasks),
            'execution_time': (
                (self.end_time or datetime.now()) - self.start_time
            ).total_seconds() if self.start_time else 0
        }


# Define enhanced tasks
def create_trend_analysis_task() -> EnhancedTask:
    """Create enhanced trend analysis task."""
    return EnhancedTask(
        name="TrendAnalysis",
        description="Analyze long-term trends, detect change points, and extrapolate trend components for forecasting",
        agent_class=TrendAnalysisAgent,
        agent_method="analyze_comprehensive_trend",
        dependencies=[],
        priority=TaskPriority.HIGH,
        timeout=180
    )

def create_seasonality_detection_task() -> EnhancedTask:
    """Create enhanced seasonality detection task."""
    return EnhancedTask(
        name="SeasonalityDetection",
        description="Detect seasonal patterns, analyze multiple seasonalities, and model holiday effects",
        agent_class=SeasonalityAgent,
        agent_method="analyze_comprehensive_seasonality",
        dependencies=[],
        priority=TaskPriority.HIGH,
        timeout=200
    )

def create_anomaly_detection_task() -> EnhancedTask:
    """Create enhanced anomaly detection task."""
    return EnhancedTask(
        name="AnomalyDetection", 
        description="Identify anomalies, analyze patterns, and assess impact on forecasting",
        agent_class=AnomalyDetectionAgent,
        agent_method="analyze_comprehensive_anomalies",
        dependencies=[],
        priority=TaskPriority.MEDIUM,
        timeout=150
    )

def create_ensemble_coordination_task() -> EnhancedTask:
    """Create enhanced ensemble coordination task."""
    return EnhancedTask(
        name="EnsembleCoordination",
        description="Coordinate multiple forecasting models, combine predictions, and generate final ensemble forecast",
        agent_class=EnsembleCoordinatorAgent,
        agent_method="coordinate_ensemble_forecast",
        dependencies=["TrendAnalysis", "SeasonalityDetection", "AnomalyDetection"],
        priority=TaskPriority.CRITICAL,
        timeout=300
    )

# Pre-configured workflows
def create_standard_workflow() -> WorkflowOrchestrator:
    """Create standard time series analysis workflow."""
    orchestrator = WorkflowOrchestrator()
    
    # Add tasks
    orchestrator.add_task(create_trend_analysis_task())
    orchestrator.add_task(create_seasonality_detection_task())
    orchestrator.add_task(create_anomaly_detection_task())
    orchestrator.add_task(create_ensemble_coordination_task())
    
    return orchestrator

def create_fast_workflow() -> WorkflowOrchestrator:
    """Create fast workflow with reduced analysis depth."""
    orchestrator = WorkflowOrchestrator()
    
    # Create tasks with reduced timeouts
    trend_task = create_trend_analysis_task()
    trend_task.timeout = 60
    
    seasonality_task = create_seasonality_detection_task()
    seasonality_task.timeout = 90
    
    ensemble_task = create_ensemble_coordination_task()
    ensemble_task.dependencies = ["TrendAnalysis", "SeasonalityDetection"]  # Skip anomaly detection
    ensemble_task.timeout = 120
    
    orchestrator.add_task(trend_task)
    orchestrator.add_task(seasonality_task)
    orchestrator.add_task(ensemble_task)
    
    return orchestrator

def create_comprehensive_workflow() -> WorkflowOrchestrator:
    """Create comprehensive workflow with extended analysis."""
    orchestrator = WorkflowOrchestrator()
    
    # Add all tasks with extended timeouts
    trend_task = create_trend_analysis_task()
    trend_task.timeout = 300
    
    seasonality_task = create_seasonality_detection_task()
    seasonality_task.timeout = 300
    
    anomaly_task = create_anomaly_detection_task()
    anomaly_task.timeout = 250
    
    ensemble_task = create_ensemble_coordination_task()
    ensemble_task.timeout = 400
    
    orchestrator.add_task(trend_task)
    orchestrator.add_task(seasonality_task)
    orchestrator.add_task(anomaly_task)
    orchestrator.add_task(ensemble_task)
    
    return orchestrator

# Task registry
WORKFLOW_REGISTRY = {
    'standard': create_standard_workflow,
    'fast': create_fast_workflow,
    'comprehensive': create_comprehensive_workflow
}

def get_workflow_by_name(workflow_name: str) -> WorkflowOrchestrator:
    """Get workflow by name from registry."""
    if workflow_name not in WORKFLOW_REGISTRY:
        available_workflows = list(WORKFLOW_REGISTRY.keys())
        raise ValueError(f"Workflow '{workflow_name}' not found. Available: {available_workflows}")
    
    return WORKFLOW_REGISTRY[workflow_name]()

# Legacy compatibility
TASKS = [
    create_trend_analysis_task(),
    create_seasonality_detection_task(), 
    create_anomaly_detection_task(),
    create_ensemble_coordination_task()
]

ORCHESTRATION_FLOW = {
    "parallel_start": ["TrendAnalysis", "SeasonalityDetection", "AnomalyDetection"],
    "sequential_end": ["EnsembleCoordination"]
}

def get_task_by_name(name: str) -> EnhancedTask:
    """Get task by name (legacy compatibility)."""
    for task in TASKS:
        if task.name == name:
            return task
    raise ValueError(f"Task '{name}' not found")

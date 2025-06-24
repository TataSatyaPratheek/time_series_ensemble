"""
Enhanced FastAPI endpoints for Time Series Forecasting Ensemble API.
Provides comprehensive REST API with async support, authentication, and monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import traceback
import time
import json

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks, Query, Path, status
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager

# Project imports
from src.config import settings
from src.crew import (
    quick_forecast, 
    get_workflow_by_name, 
    list_available_workflows,
    get_workflow_info,
    create_custom_workflow,
    setup_orchestration_environment,
    get_configuration_summary,
    orchestration_metrics
)
from src.utils.logging import get_logger
from src.utils.exceptions import AgentError

logger = get_logger(__name__)

# Global state for API
api_state = {
    'startup_time': None,
    'active_forecasts': {},
    'request_count': 0,
    'error_count': 0,
    'last_health_check': None
}

# Security
security = HTTPBearer(auto_error=False)

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with proper startup/shutdown."""
    # Startup
    logger.info("Starting Time Series Forecasting Ensemble API")
    api_state['startup_time'] = datetime.now()
    
    # Setup orchestration environment
    if not setup_orchestration_environment():
        logger.error("Failed to setup orchestration environment")
        raise RuntimeError("Orchestration setup failed")
    
    logger.info("API startup completed successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")
    # Cleanup active forecasts
    for forecast_id in list(api_state['active_forecasts'].keys()):
        api_state['active_forecasts'].pop(forecast_id, None)
    
    logger.info("API shutdown completed")

# FastAPI app initialization
app = FastAPI(
    title="Time Series Forecasting Ensemble API",
    description="""
    **Advanced Multi-Agent Time Series Forecasting API**
    
    This API provides sophisticated time series forecasting capabilities using:
    - Multi-agent orchestration with CrewAI
    - Local LLM integration (Ollama)
    - Ensemble forecasting methods
    - Comprehensive uncertainty quantification
    - Real-time monitoring and metrics
    
    **Features:**
    - Multiple workflow types (standard, fast, comprehensive)
    - Async processing with real-time status updates
    - Confidence intervals and uncertainty analysis
    - Agent-level insights and explanations
    - Production-ready error handling and monitoring
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, 'CORS_ORIGINS') else ["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "*.local"])

# Request tracking middleware
@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    """Track requests and add performance metrics."""
    start_time = time.time()
    api_state['request_count'] += 1
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = str(api_state['request_count'])
        return response
    except Exception as e:
        api_state['error_count'] += 1
        logger.error(f"Request failed: {str(e)}")
        raise

# Pydantic models
class TimeSeriesData(BaseModel):
    """Time series data input model."""
    values: List[float] = Field(..., description="Time series data points", min_items=10)
    timestamps: Optional[List[str]] = Field(None, description="Optional timestamps (ISO format)")
    frequency: Optional[str] = Field(None, description="Data frequency (D, H, M, etc.)")
    name: Optional[str] = Field(None, description="Series name for identification")
    
    @validator('values')
    def validate_values(cls, v):
        if len(v) < 10:
            raise ValueError("Time series must have at least 10 data points")
        if any(not isinstance(x, (int, float)) or np.isnan(x) for x in v):
            raise ValueError("All values must be valid numbers")
        return v
    
    @validator('timestamps')
    def validate_timestamps(cls, v, values):
        if v is not None:
            if 'values' in values and len(v) != len(values['values']):
                raise ValueError("Timestamps length must match values length")
            try:
                pd.to_datetime(v[:5])  # Validate first 5 timestamps
            except Exception:
                raise ValueError("Invalid timestamp format")
        return v

class ForecastRequest(BaseModel):
    """Forecast request model with comprehensive options."""
    data: TimeSeriesData
    workflow_type: str = Field(default="standard", description="Workflow type")
    forecast_horizon: int = Field(default=30, ge=1, le=365, description="Forecast periods")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level")
    enable_explanations: bool = Field(default=True, description="Include LLM explanations")
    async_processing: bool = Field(default=False, description="Process asynchronously")
    
    @validator('workflow_type')
    def validate_workflow_type(cls, v):
        available_workflows = list_available_workflows()
        if v not in available_workflows:
            raise ValueError(f"Workflow type must be one of: {available_workflows}")
        return v

class ForecastMetadata(BaseModel):
    """Forecast metadata model."""
    workflow_id: str
    execution_time: float
    model_count: int
    workflow_type: str
    confidence_level: float
    timestamp: str
    agent_insights: Optional[Dict[str, Any]] = None

class ForecastResponse(BaseModel):
    """Comprehensive forecast response model."""
    status: str = Field(..., description="Response status")
    forecast_id: Optional[str] = Field(None, description="Unique forecast identifier")
    forecast: Optional[List[float]] = Field(None, description="Forecast values")
    confidence_intervals: Optional[List[List[float]]] = Field(None, description="Confidence intervals")
    metadata: Optional[ForecastMetadata] = Field(None, description="Forecast metadata")
    agent_results: Optional[Dict[str, Any]] = Field(None, description="Individual agent results")
    explanations: Optional[Dict[str, str]] = Field(None, description="LLM explanations")
    warnings: Optional[List[str]] = Field(None, description="Warnings and recommendations")
    error: Optional[str] = Field(None, description="Error message if failed")

class AsyncForecastStatus(BaseModel):
    """Async forecast status model."""
    forecast_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    estimated_completion: Optional[str] = None
    current_agent: Optional[str] = None
    results: Optional[ForecastResponse] = None

class SystemHealth(BaseModel):
    """System health status model."""
    status: str
    uptime_seconds: float
    ollama_status: str
    memory_usage: Dict[str, Any]
    active_forecasts: int
    total_requests: int
    error_rate: float
    last_check: str

class WorkflowInfo(BaseModel):
    """Workflow information model."""
    name: str
    description: str
    execution_time: str
    resource_usage: str
    accuracy: str
    agents_used: List[str]

# Authentication dependency (optional)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Optional authentication for secured endpoints."""
    if credentials is None:
        return None  # Allow anonymous access for now
    
    # Add authentication logic here if needed
    # For local deployment, authentication might not be necessary
    return {"user_id": "local_user"}

# Helper functions
def create_pandas_series(data: TimeSeriesData) -> pd.Series:
    """Create pandas Series from TimeSeriesData."""
    if data.timestamps:
        index = pd.to_datetime(data.timestamps)
        series = pd.Series(data.values, index=index)
    else:
        series = pd.Series(data.values)
    
    if data.frequency and hasattr(series.index, 'freq'):
        try:
            series = series.asfreq(data.frequency)
        except Exception as e:
            logger.warning(f"Could not set frequency {data.frequency}: {str(e)}")
    
    return series

async def check_ollama_health() -> str:
    """Check Ollama service health."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
            return "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        return "unavailable"

# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with basic information."""
    return {
        "name": "Time Series Forecasting Ensemble API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=SystemHealth, tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        current_time = datetime.now()
        api_state['last_health_check'] = current_time
        
        # Calculate uptime
        uptime = (current_time - api_state['startup_time']).total_seconds()
        
        # Check Ollama status
        ollama_status = await check_ollama_health()
        
        # Get memory usage (basic)
        import psutil
        memory_info = psutil.virtual_memory()
        memory_usage = {
            "total_gb": round(memory_info.total / (1024**3), 2),
            "available_gb": round(memory_info.available / (1024**3), 2),
            "used_percent": memory_info.percent
        }
        
        # Calculate error rate
        error_rate = (api_state['error_count'] / max(api_state['request_count'], 1)) * 100
        
        return SystemHealth(
            status="healthy" if ollama_status == "healthy" else "degraded",
            uptime_seconds=uptime,
            ollama_status=ollama_status,
            memory_usage=memory_usage,
            active_forecasts=len(api_state['active_forecasts']),
            total_requests=api_state['request_count'],
            error_rate=error_rate,
            last_check=current_time.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return SystemHealth(
            status="unhealthy",
            uptime_seconds=0,
            ollama_status="unknown",
            memory_usage={},
            active_forecasts=0,
            total_requests=api_state['request_count'],
            error_rate=100.0,
            last_check=datetime.now().isoformat()
        )

@app.get("/workflows", response_model=List[WorkflowInfo], tags=["Workflows"])
async def list_workflows():
    """List available forecasting workflows."""
    try:
        workflows = []
        for workflow_name in list_available_workflows():
            info = get_workflow_info(workflow_name)
            
            # Add agent information
            agents_used = ["TrendAnalysis", "SeasonalityDetection", "EnsembleCoordination"]
            if workflow_name == "comprehensive":
                agents_used.append("AnomalyDetection")
            elif workflow_name == "fast":
                agents_used = ["TrendAnalysis", "EnsembleCoordination"]
            
            workflows.append(WorkflowInfo(
                name=workflow_name,
                description=info['description'],
                execution_time=info['execution_time'],
                resource_usage=info['resource_usage'],
                accuracy=info['accuracy'],
                agents_used=agents_used
            ))
        
        return workflows
        
    except Exception as e:
        logger.error(f"Failed to list workflows: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve workflows")

@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def generate_forecast(
    request: ForecastRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Generate time series forecast using specified workflow."""
    forecast_id = f"forecast_{int(time.time())}_{hash(str(request.data.values))}"
    
    try:
        logger.info(f"Starting forecast {forecast_id} with workflow {request.workflow_type}")
        
        # Create pandas series
        series = create_pandas_series(request.data)
        
        if request.async_processing:
            # Start async processing
            api_state['active_forecasts'][forecast_id] = {
                'status': 'pending',
                'progress': 0.0,
                'start_time': datetime.now()
            }
            
            background_tasks.add_task(
                process_forecast_async, 
                forecast_id, 
                request, 
                series
            )
            
            return ForecastResponse(
                status="accepted",
                forecast_id=forecast_id,
                metadata=ForecastMetadata(
                    workflow_id=forecast_id,
                    execution_time=0.0,
                    model_count=0,
                    workflow_type=request.workflow_type,
                    confidence_level=request.confidence_level,
                    timestamp=datetime.now().isoformat()
                )
            )
        
        else:
            # Synchronous processing
            return await process_forecast_sync(forecast_id, request, series)
            
    except HTTPException as he:
        logger.error(f"HTTP error in forecast {forecast_id}: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in forecast {forecast_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Forecast processing failed: {str(e)}"
        )

async def process_forecast_sync(forecast_id: str, request: ForecastRequest, series: pd.Series) -> ForecastResponse:
    """Process forecast synchronously."""
    try:
        # Get workflow
        workflow = get_workflow_by_name(request.workflow_type)
        
        # Execute workflow
        results = await workflow.execute_workflow(
            {"series": series, "forecast_horizon": request.forecast_horizon},
            execution_mode="hybrid"
        )
        
        if results.get('status') != 'completed':
            raise ValueError(f"Workflow execution failed: {results.get('error', 'Unknown error')}")
        
        # Extract results
        ensemble_forecast = results.get('ensemble_forecast', [])
        coordination_results = results.get('coordination_results', {})
        
        # Get confidence intervals
        confidence_intervals = None
        uncertainty_analysis = coordination_results.get('uncertainty_analysis', {})
        if uncertainty_analysis:
            confidence_intervals = uncertainty_analysis.get('confidence_intervals')
        
        # Extract agent results and explanations
        agent_results = {}
        explanations = {}
        
        for task_summary in results.get('task_summaries', {}).values():
            if task_summary.get('status') == 'completed':
                agent_name = task_summary.get('name', '')
                agent_results[agent_name] = {
                    'execution_time': task_summary.get('duration', 0),
                    'status': task_summary.get('status')
                }
        
        # Get LLM explanations if enabled
        if request.enable_explanations:
            comprehensive_report = coordination_results.get('comprehensive_report', {})
            llm_insights = comprehensive_report.get('llm_strategic_insights', {})
            if llm_insights and 'strategic_analysis' in llm_insights:
                explanations['strategic_analysis'] = llm_insights['strategic_analysis']
        
        # Create metadata
        metadata = ForecastMetadata(
            workflow_id=results.get('workflow_id', forecast_id),
            execution_time=results.get('execution_time', 0.0),
            model_count=results.get('metadata', {}).get('n_models', 0),
            workflow_type=request.workflow_type,
            confidence_level=request.confidence_level,
            timestamp=datetime.now().isoformat(),
            agent_insights=agent_results
        )
        
        # Generate warnings/recommendations
        warnings = []
        if coordination_results:
            report = coordination_results.get('comprehensive_report', {})
            recommendations = report.get('recommendations', [])
            warnings.extend(recommendations[:3])  # Top 3 recommendations
        
        logger.info(f"Forecast {forecast_id} completed successfully")
        
        return ForecastResponse(
            status="success",
            forecast_id=forecast_id,
            forecast=ensemble_forecast,
            confidence_intervals=confidence_intervals,
            metadata=metadata,
            agent_results=agent_results,
            explanations=explanations if explanations else None,
            warnings=warnings if warnings else None
        )
        
    except Exception as e:
        logger.error(f"Sync forecast processing failed for {forecast_id}: {str(e)}")
        raise

async def process_forecast_async(forecast_id: str, request: ForecastRequest, series: pd.Series):
    """Process forecast asynchronously with status updates."""
    try:
        # Update status
        api_state['active_forecasts'][forecast_id].update({
            'status': 'running',
            'progress': 0.1,
            'current_agent': 'initializing'
        })
        
        # Process forecast
        result = await process_forecast_sync(forecast_id, request, series)
        
        # Update with results
        api_state['active_forecasts'][forecast_id].update({
            'status': 'completed',
            'progress': 1.0,
            'current_agent': 'finished',
            'results': result,
            'completion_time': datetime.now()
        })
        
    except Exception as e:
        # Update with error
        api_state['active_forecasts'][forecast_id].update({
            'status': 'failed',
            'progress': 0.0,
            'error': str(e),
            'completion_time': datetime.now()
        })
        logger.error(f"Async forecast {forecast_id} failed: {str(e)}")

@app.get("/forecast/{forecast_id}/status", response_model=AsyncForecastStatus, tags=["Forecasting"])
async def get_forecast_status(forecast_id: str = Path(..., description="Forecast ID")):
    """Get status of an async forecast."""
    if forecast_id not in api_state['active_forecasts']:
        raise HTTPException(status_code=404, detail="Forecast not found")
    
    forecast_info = api_state['active_forecasts'][forecast_id]
    
    # Estimate completion time
    estimated_completion = None
    if forecast_info['status'] == 'running' and forecast_info['progress'] > 0:
        elapsed = (datetime.now() - forecast_info['start_time']).total_seconds()
        estimated_total = elapsed / forecast_info['progress']
        estimated_completion = (forecast_info['start_time'] + timedelta(seconds=estimated_total)).isoformat()
    
    return AsyncForecastStatus(
        forecast_id=forecast_id,
        status=forecast_info['status'],
        progress=forecast_info['progress'],
        estimated_completion=estimated_completion,
        current_agent=forecast_info.get('current_agent'),
        results=forecast_info.get('results')
    )

@app.delete("/forecast/{forecast_id}", tags=["Forecasting"])
async def cancel_forecast(forecast_id: str = Path(..., description="Forecast ID")):
    """Cancel an active forecast."""
    if forecast_id not in api_state['active_forecasts']:
        raise HTTPException(status_code=404, detail="Forecast not found")
    
    # Remove from active forecasts
    forecast_info = api_state['active_forecasts'].pop(forecast_id)
    
    return {
        "message": f"Forecast {forecast_id} cancelled",
        "status": forecast_info['status'],
        "progress": forecast_info['progress']
    }

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics(user=Depends(get_current_user)):
    """Get API and orchestration metrics."""
    try:
        # Get orchestration metrics
        orch_metrics = orchestration_metrics.get_summary()
        
        # Get configuration summary
        config_summary = get_configuration_summary()
        
        # API metrics
        api_metrics = {
            'total_requests': api_state['request_count'],
            'total_errors': api_state['error_count'],
            'error_rate': (api_state['error_count'] / max(api_state['request_count'], 1)) * 100,
            'active_forecasts': len(api_state['active_forecasts']),
            'uptime_seconds': (datetime.now() - api_state['startup_time']).total_seconds()
        }
        
        return {
            'api_metrics': api_metrics,
            'orchestration_metrics': orch_metrics,
            'configuration': config_summary,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.post("/admin/reset", tags=["Administration"])
async def reset_system(user=Depends(get_current_user)):
    """Reset system state (development only)."""
    try:
        # Clear active forecasts
        api_state['active_forecasts'].clear()
        
        # Reset counters
        api_state['request_count'] = 0
        api_state['error_count'] = 0
        
        # Reset orchestration metrics
        orchestration_metrics.reset_metrics()
        
        logger.info("System state reset")
        return {"message": "System state reset successfully"}
        
    except Exception as e:
        logger.error(f"System reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail="System reset failed")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with detailed logging."""
    logger.warning(f"HTTP {exc.status_code} on {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.detail, "type": "http_error"}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    logger.warning(f"Validation error on {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"status": "error", "error": str(exc), "type": "validation_error"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    logger.error(f"Unhandled exception on {request.url}: {str(exc)}")
    logger.error(traceback.format_exc())
    api_state['error_count'] += 1
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error", 
            "error": "Internal server error", 
            "type": "internal_error",
            "request_id": str(api_state['request_count'])
        }
    )

# Custom OpenAPI
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Time Series Forecasting Ensemble API",
        version="0.1.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Startup message
@app.on_event("startup")
async def startup_message():
    """Log startup message."""
    logger.info("="*50)
    logger.info("Time Series Forecasting Ensemble API Started")
    logger.info("="*50)
    logger.info(f"Docs available at: http://localhost:{settings.API_PORT}/docs")
    logger.info(f"Health check: http://localhost:{settings.API_PORT}/health")
    logger.info("="*50)

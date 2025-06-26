"""
API module initialization.
Exports FastAPI application and related components for time series forecasting ensemble.
"""

from .endpoints import (
    app,
    TimeSeriesData,
    ForecastRequest,
    ForecastResponse,
    ForecastMetadata,
    AsyncForecastStatus,
    SystemHealth,
    WorkflowInfo,
    api_state
)

__all__ = [
    'app',
    'TimeSeriesData',
    'ForecastRequest', 
    'ForecastResponse',
    'ForecastMetadata',
    'AsyncForecastStatus',
    'SystemHealth',
    'WorkflowInfo',
    'api_state'
]

# Module metadata
__version__ = "0.1.0"
__author__ = "Time Series Ensemble Team"
__description__ = "FastAPI-based REST API for multi-agent time series forecasting"

# API information
API_INFO = {
    'name': 'Time Series Forecasting Ensemble API',
    'version': __version__,
    'description': __description__,
    'features': [
        'Multi-agent orchestration with CrewAI',
        'Local LLM integration (Ollama)',
        'Ensemble forecasting methods',
        'Async processing support',
        'Real-time monitoring',
        'Comprehensive error handling',
        'OpenAPI documentation',
        'Health checks and metrics'
    ],
    'endpoints': {
        'forecast': 'POST /forecast - Generate time series forecasts',
        'health': 'GET /health - System health status', 
        'workflows': 'GET /workflows - List available workflows',
        'status': 'GET /forecast/{id}/status - Check forecast status',
        'metrics': 'GET /metrics - System metrics',
        'docs': 'GET /docs - Interactive API documentation'
    }
}

# Configuration helpers
def get_api_info():
    """Get API information dictionary."""
    return API_INFO

def get_available_endpoints():
    """Get list of available API endpoints."""
    return list(API_INFO['endpoints'].keys())

# Server utilities
def create_server_config(host: str = "127.0.0.1", port: int = 8000, **kwargs):
    """Create server configuration for uvicorn."""
    from src.config import settings
    
    return {
        'app': 'src.api:app',
        'host': host,
        'port': port,
        'reload': getattr(settings, 'API_RELOAD', False),
        'log_level': getattr(settings, 'API_LOG_LEVEL', 'info').lower(),
        'access_log': True,
        'use_colors': True,
        **kwargs
    }

async def start_server(host: str = "127.0.0.1", port: int = 8000, **kwargs):
    """Start the API server programmatically."""
    import uvicorn
    from src.config import settings
    
    # Use settings if available
    final_host = getattr(settings, 'API_HOST', host)
    final_port = getattr(settings, 'API_PORT', port)
    
    config = create_server_config(final_host, final_port, **kwargs)
    
    # Remove 'app' from config for uvicorn.run
    app_import = config.pop('app')
    
    uvicorn.run(app_import, **config)

# Health check utilities
async def check_api_health():
    """Programmatically check API health."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
    except Exception:
        return False # type: ignore

# Development utilities
def create_test_client():
    """Create FastAPI test client for development/testing."""
    from fastapi.testclient import TestClient
    return TestClient(app)

def get_openapi_spec():
    """Get OpenAPI specification as dictionary."""
    return app.openapi()

# Quick start function
async def quick_start_api(
    host: str = "127.0.0.1", 
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info"
):
    """Quick start the API server with default settings."""
    import uvicorn
    
    print("="*60)
    print("🚀 Starting Time Series Forecasting Ensemble API")
    print("="*60)
    print(f"📡 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"📖 Docs: http://{host}:{port}/docs")
    print(f"🏥 Health: http://{host}:{port}/health")
    print("="*60)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )

# Export convenience functions
__all__.extend([
    'get_api_info',
    'get_available_endpoints', 
    'create_server_config',
    'start_server',
    'check_api_health',
    'create_test_client',
    'get_openapi_spec',
    'quick_start_api',
    'API_INFO'
])

# CLI entry point
if __name__ == "__main__":
    import asyncio
    asyncio.run(quick_start_api())

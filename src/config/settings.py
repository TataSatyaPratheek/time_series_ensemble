import os, json, urllib.parse
from dotenv import load_dotenv
from typing import Optional, List, Union
from pydantic import BaseModel, Field

# Load environment variables from .env file
class Settings: # Merged class definition
    # --- Ollama Configuration ---
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_API_KEY: Optional[str] = os.getenv("OLLAMA_API_KEY")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "127.0.0.1")
    OLLAMA_PORT: int = int(os.getenv("OLLAMA_PORT", "11434"))
    DEFAULT_LLM_MODEL: str = "qwen3:1.7b"


    # --- Local Model Configuration (updated defaults to match .env) ---
    TREND_ANALYSIS_MODEL: str = "qwen3:1.7b"
    SEASONALITY_MODEL: str = "qwen3:1.7b"
    ANOMALY_DETECTION_MODEL: str = "qwen3:1.7b"
    ENSEMBLE_COORDINATOR_MODEL: str = "qwen3:1.7b"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

    # --- Model Settings ---
    TREND_AGENT_TEMPERATURE: float = float(os.getenv("TREND_AGENT_TEMPERATURE", "0.1"))
    SEASONALITY_AGENT_TEMPERATURE: float = float(os.getenv("SEASONALITY_AGENT_TEMPERATURE", "0.2"))
    ANOMALY_AGENT_TEMPERATURE: float = float(os.getenv("ANOMALY_AGENT_TEMPERATURE", "0.3"))
    COORDINATOR_TEMPERATURE: float = float(os.getenv("COORDINATOR_TEMPERATURE", "0.1"))

    # Context lengths for local models
    QWEN3_CONTEXT_LENGTH: int = int(os.getenv("QWEN3_CONTEXT_LENGTH", "32768"))
    LLAMA32_CONTEXT_LENGTH: int = int(os.getenv("LLAMA32_CONTEXT_LENGTH", "8192"))

    # --- Function Calling Configuration ---
    ENABLE_FUNCTION_CALLING: bool = os.getenv("ENABLE_FUNCTION_CALLING", "true").lower() == "true"
    FUNCTION_CALLING_TIMEOUT: int = int(os.getenv("FUNCTION_CALLING_TIMEOUT", "30"))
    MAX_FUNCTION_RETRIES: int = int(os.getenv("MAX_FUNCTION_RETRIES", "3"))

    # --- CrewAI Configuration ---
    CREWAI_TELEMETRY_OPT_OUT: bool = os.getenv("CREWAI_TELEMETRY_OPT_OUT", "true").lower() == "true"
    CREWAI_LOG_LEVEL: str = os.getenv("CREWAI_LOG_LEVEL", "INFO")
    CREWAI_MAX_EXECUTION_TIME: int = int(os.getenv("CREWAI_MAX_EXECUTION_TIME", "300"))
    CREWAI_MEMORY_ENABLED: bool = os.getenv("CREWAI_MEMORY_ENABLED", "true").lower() == "true"

    # --- Time Series Processing ---
    MAX_TIME_SERIES_LENGTH: int = int(os.getenv("MAX_TIME_SERIES_LENGTH", "10000"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    PARALLEL_PROCESSING: bool = os.getenv("PARALLEL_PROCESSING", "true").lower() == "true"
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    ASYNC_BATCH_SIZE: int = int(os.getenv("ASYNC_BATCH_SIZE", "8"))

    # --- Data Paths ---
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    RAW_DATA_PATH: str = os.getenv("RAW_DATA_PATH", "./data/raw")
    PROCESSED_DATA_PATH: str = os.getenv("PROCESSED_DATA_PATH", "./data/processed")
    EXTERNAL_DATA_PATH: str = os.getenv("EXTERNAL_DATA_PATH", "./data/external")
    MODEL_CACHE_PATH: str = os.getenv("MODEL_CACHE_PATH", "./models/cache")

    # --- API Configuration ---
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"
    API_LOG_LEVEL: str = os.getenv("API_LOG_LEVEL", "info")
    
    _cors_origins_str: str = os.getenv("CORS_ORIGINS", "[]")
    CORS_ORIGINS: List[str]
    try:
        CORS_ORIGINS = json.loads(_cors_origins_str)
    except json.JSONDecodeError:
        # Fallback for non-JSON array string, e.g., '["http://localhost:3000", "http://localhost:8080"]'
        # This is a bit fragile, assuming it's a comma-separated list within brackets
        CORS_ORIGINS = [origin.strip().strip('"') for origin in _cors_origins_str.strip("[]").split(",") if origin.strip()]

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/ensemble.log")
    ENABLE_STRUCTURED_LOGGING: bool = os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"

    # --- Performance Tuning (M1 Air 8GB optimization) ---
    MAX_MEMORY_USAGE_GB: int = int(os.getenv("MAX_MEMORY_USAGE_GB", "6"))
    ENABLE_MODEL_QUANTIZATION: bool = os.getenv("ENABLE_MODEL_QUANTIZATION", "true").lower() == "true"
    USE_METAL_ACCELERATION: bool = os.getenv("USE_METAL_ACCELERATION", "true").lower() == "true"
    TORCH_THREADS: int = int(os.getenv("TORCH_THREADS", "4"))

    # --- Development ---
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    ENABLE_PROFILING: bool = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
    SAVE_INTERMEDIATE_RESULTS: bool = os.getenv("SAVE_INTERMEDIATE_RESULTS", "true").lower() == "true"
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"

    # --- Testing ---
    TEST_DATA_SIZE: int = int(os.getenv("TEST_DATA_SIZE", "1000"))
    ENABLE_INTEGRATION_TESTS: bool = os.getenv("ENABLE_INTEGRATION_TESTS", "true").lower() == "true"
    MOCK_OLLAMA_IN_TESTS: bool = os.getenv("MOCK_OLLAMA_IN_TESTS", "false").lower() == "true"

settings = Settings()

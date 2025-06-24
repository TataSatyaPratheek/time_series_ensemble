"""
Configuration module for time series ensemble.
"""

import os
# from pathlib import Path # Removed unused import
from pydantic import BaseSettings, Field
from typing import List, Optional

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # === PATHS ===
    DATA_DIR: str = Field(default="./data", env="DATA_DIR")
    RAW_DATA_PATH: str = Field(default="./data/raw", env="RAW_DATA_PATH")
    PROCESSED_DATA_PATH: str = Field(default="./data/processed", env="PROCESSED_DATA_PATH")
    EXTERNAL_DATA_PATH: str = Field(default="./data/external", env="EXTERNAL_DATA_PATH")
    
    # === OLLAMA CONFIGURATION ===
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_API_KEY: Optional[str] = Field(default=None, env="OLLAMA_API_KEY") # Changed default to None
    
    # === MODEL SETTINGS ===
    TREND_ANALYSIS_MODEL: str = Field(default="qwen3:1.7b", env="TREND_ANALYSIS_MODEL")
    SEASONALITY_MODEL: str = Field(default="qwen3:1.7b", env="SEASONALITY_MODEL")
    ANOMALY_DETECTION_MODEL: str = Field(default="llama3.2:latest", env="ANOMALY_DETECTION_MODEL")
    ENSEMBLE_COORDINATOR_MODEL: str = Field(default="llama3.2:latest", env="ENSEMBLE_COORDINATOR_MODEL")
    EMBEDDING_MODEL: str = Field(default="nomic-embed-text:latest", env="EMBEDDING_MODEL")
    
    # === PERFORMANCE SETTINGS ===
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
    MAX_MEMORY_USAGE_GB: int = Field(default=6, env="MAX_MEMORY_USAGE_GB")
    BATCH_SIZE: int = Field(default=32, env="BATCH_SIZE")
    
    # === LOGGING ===
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

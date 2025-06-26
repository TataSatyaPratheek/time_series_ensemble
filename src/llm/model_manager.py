"""
Model management for Ollama models.
Adapted from tourism application approach.
"""

import subprocess
import sys
import ollama
from typing import List, Dict, Any, Tuple, Optional
import time

from src.utils.logging import get_logger
from src.config import settings

logger = get_logger(__name__)

# Recommended models for time series forecasting
# Updated to reflect the user's preference for 'qwen3:1.7b' for all LLM tasks.
FORECASTING_MODELS = {
    "qwen3:1.7b": {
        "description": "Optimized model for time series forecasting tasks",
        "strengths": ["Efficient inference", "Good reasoning for analytical tasks", "Low memory footprint"],
        "recommend_for": ["All time series analysis tasks", "Resource-constrained environments"],
        "size_gb": 1.7
    }
}

class OllamaModelManager:
    """Manage Ollama models for time series forecasting."""
    
    def __init__(self):
        self.available_models = []
        self.refresh_available_models()
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is installed and running."""
        try:
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            logger.info(f"Ollama available: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.error("Ollama not available")
            return False
    
    def refresh_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        models = []
        
        try:
            # Use ollama package to list models
            model_list = ollama.list()
            
            if 'models' in model_list:
                for model in model_list['models']:
                    model_name = model.get('name', '')
                    if model_name:
                        models.append(model_name)
            
            self.available_models = models
            logger.info(f"Found {len(models)} available models: {models}")
            
        except Exception as e:
            logger.error(f"Error refreshing models: {str(e)}")
            # Fallback: assume recommended models are available
            self.available_models = list(FORECASTING_MODELS.keys())
        
        return self.available_models
    
    def download_model(self, model_name: str, timeout: int = 600) -> Tuple[bool, str]:
        """Download an Ollama model."""
        try:
            logger.info(f"Downloading model: {model_name}")
            
            # Use ollama package to pull model
            start_time = time.time()
            
            # This is a blocking call
            ollama.pull(model_name)
            
            download_time = time.time() - start_time
            success_msg = f"Model '{model_name}' downloaded successfully in {download_time:.1f}s"
            logger.info(success_msg)
            
            # Refresh model list
            self.refresh_available_models()
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Failed to download model '{model_name}': {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def test_model(self, model_name: str) -> Tuple[bool, str, float]:
        """Test if a model is working correctly."""
        try:
            start_time = time.time()
            
            response = ollama.chat(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": "Analyze this simple time series: [1, 2, 3, 4, 5]. What is the trend?"
                }]
            )
            
            inference_time = time.time() - start_time
            
            if response and 'message' in response and 'content' in response['message']:
                content = response['message']['content']
                if 'trend' in content.lower() or 'increasing' in content.lower():
                    return True, "Model test successful", inference_time
                else:
                    return False, "Model response doesn't seem relevant", inference_time
            else:
                return False, "Invalid response format", inference_time
                
        except Exception as e:
            return False, f"Model test failed: {str(e)}", 0.0
    
    def get_recommended_models(self, available_memory_gb: float = 8.0) -> List[str]:
        """Get recommended models based on available memory."""
        recommended = []
        
        for model_name, info in FORECASTING_MODELS.items():
            if info['size_gb'] <= available_memory_gb:
                recommended.append(model_name)
        
        # Sort by size (smallest first for resource-constrained environments)
        recommended.sort(key=lambda x: FORECASTING_MODELS[x]['size_gb'])
        
        return recommended
    
    def ensure_models_available(self, required_models: List[str] = None) -> Dict[str, bool]:
        """Ensure required models are available, download if necessary."""
        if required_models is None:
            required_models = [
                settings.TREND_ANALYSIS_MODEL,
                settings.SEASONALITY_MODEL,
                settings.ANOMALY_DETECTION_MODEL,
                settings.ENSEMBLE_COORDINATOR_MODEL
            ]
        
        results = {}
        
        for model in required_models:
            if model in self.available_models:
                # Test the model
                success, message, _ = self.test_model(model)
                results[model] = success
                if success:
                    logger.info(f"Model {model} is available and working")
                else:
                    logger.warning(f"Model {model} available but test failed: {message}")
            else:
                # Try to download
                logger.info(f"Model {model} not available, attempting download...")
                success, message = self.download_model(model)
                results[model] = success
                if not success:
                    logger.error(f"Failed to ensure model {model}: {message}")
        
        return results

# Global model manager instance
model_manager = OllamaModelManager()

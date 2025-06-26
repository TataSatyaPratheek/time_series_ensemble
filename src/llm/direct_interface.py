"""
Direct Ollama interface for time series forecasting agents.
Adapted from tourism RAG application approach.
"""

import ollama
import asyncio
import time
from typing import Dict, Any, List, Optional
import json
import re

from src.utils.logging import get_logger
from src.utils.exceptions import AgentError
from src.config import settings

logger = get_logger(__name__)

class DirectOllamaInterface:
    """Direct interface to Ollama models without API servers."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.DEFAULT_LLM_MODEL
        self.conversation_history = []
        
    def query_llm(self, 
                  prompt: str, 
                  system_prompt: str = None,
                  temperature: float = 0.1,
                  max_tokens: int = 2048,
                  include_context: bool = True) -> str:
        """
        Query LLM directly using Ollama package.
        
        Args:
            prompt: User prompt/question
            system_prompt: System instructions
            temperature: Model temperature
            max_tokens: Maximum response tokens
            include_context: Whether to include conversation history
            
        Returns:
            Model response text
        """
        try:
            # Build conversation messages
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({
                    "role": "system", 
                    "content": system_prompt
                })
            
            # Add conversation history if enabled
            if include_context:
                messages.extend(self.conversation_history[-3:])  # Last 3 exchanges
            
            # Add current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Call Ollama directly
            start_time = time.time()
            
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_k": 40,
                    "top_p": 0.9,
                }
            )
            
            inference_time = time.time() - start_time
            
            if not response or "message" not in response or "content" not in response["message"]:
                raise ValueError("Invalid response from Ollama model")
            
            response_content = response["message"]["content"]
            
            # Update conversation history
            if include_context:
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": response_content})
                
                # Keep only last 6 messages (3 exchanges)
                if len(self.conversation_history) > 6:
                    self.conversation_history = self.conversation_history[-6:]
            
            logger.info(f"LLM response generated in {inference_time:.2f}s using {self.model_name}")
            
            return response_content
            
        except Exception as e:
            error_msg = f"Error querying LLM model {self.model_name}: {e}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e
    
    async def query_llm_async(self, 
                             prompt: str, 
                             system_prompt: str = None,
                             temperature: float = 0.1,
                             max_tokens: int = 2048) -> str:
        """Async wrapper for LLM queries."""
        return await asyncio.to_thread(
            self.query_llm, 
            prompt, 
            system_prompt, 
            temperature, 
            max_tokens
        )
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Try to parse entire response as JSON
            return json.loads(response)
            
        except json.JSONDecodeError:
            # Fallback: parse structured text
            return self._parse_structured_text(response)
    
    def _parse_structured_text(self, text: str) -> Dict[str, Any]:
        """Parse structured text response into dictionary."""
        result = {}
        
        # Look for key-value patterns
        patterns = [
            r'(\w+):\s*([^\n]+)',
            r'(\w+)\s*=\s*([^\n]+)',
            r'-\s*(\w+):\s*([^\n]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for key, value in matches:
                result[key.lower()] = value.strip()
        
        return result if result else {"response": text}
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

# Global instances for different agent types
trend_llm = DirectOllamaInterface(settings.TREND_ANALYSIS_MODEL)
seasonality_llm = DirectOllamaInterface(settings.SEASONALITY_MODEL)
anomaly_llm = DirectOllamaInterface(settings.ANOMALY_DETECTION_MODEL)
ensemble_llm = DirectOllamaInterface(settings.ENSEMBLE_COORDINATOR_MODEL)

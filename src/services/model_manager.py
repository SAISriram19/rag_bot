"""Model Manager for detecting and managing Ollama models."""

import logging
import time
from typing import List, Dict, Any, Optional
import ollama
from ollama import Client
from src.config import config

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages Ollama model detection, validation, and switching."""
    
    def __init__(self, base_url: str = None):
        """Initialize Model Manager.
        
        Args:
            base_url: Ollama server URL (defaults to config value)
        """
        self.base_url = base_url or config.ollama_base_url
        self._model_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes cache TTL
        
        try:
            self.client = Client(host=self.base_url)
            logger.info(f"Initialized Model Manager with base URL: {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise ConnectionError(f"Could not connect to Ollama at {self.base_url}: {e}")
    
    def list_ollama_models(self, refresh_cache: bool = False) -> List[str]:
        """List all available Ollama models.
        
        Args:
            refresh_cache: Force refresh of model cache
            
        Returns:
            List of available model names
            
        Raises:
            ConnectionError: If unable to connect to Ollama
        """
        current_time = time.time()
        
        # Use cache if it's still valid and not forcing refresh
        if (not refresh_cache and 
            self._model_cache and 
            current_time - self._cache_timestamp < self._cache_ttl):
            logger.debug("Using cached model list")
            return list(self._model_cache.keys())
        
        try:
            logger.debug("Fetching fresh model list from Ollama")
            models_response = self.client.list()
            models = []
            model_info = {}
            
            if 'models' in models_response:
                for model in models_response['models']:
                    if 'name' in model:
                        model_name = model['name']
                        models.append(model_name)
                        model_info[model_name] = {
                            'size': model.get('size', 'Unknown'),
                            'modified_at': model.get('modified_at'),
                            'digest': model.get('digest', 'Unknown'),
                            'details': model.get('details', {})
                        }
            
            # Update cache
            self._model_cache = model_info
            self._cache_timestamp = current_time
            
            logger.info(f"Found {len(models)} available models: {models}")
            return models
            
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            raise ConnectionError(f"Could not retrieve models from Ollama: {e}")
    
    def validate_model(self, model_name: str) -> bool:
        """Validate if a model is available and functional.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if model is valid and functional, False otherwise
        """
        try:
            # Check if model exists in the list
            available_models = self.list_ollama_models()
            if model_name not in available_models:
                logger.warning(f"Model '{model_name}' not found in available models")
                return False
            
            # Test model functionality with a simple prompt
            logger.debug(f"Testing model functionality: {model_name}")
            test_response = self.client.generate(
                model=model_name,
                prompt="Test prompt. Respond with 'OK'.",
                options={'max_tokens': 10},
                stream=False
            )
            
            if 'response' in test_response and test_response['response'].strip():
                logger.debug(f"Model {model_name} validation successful")
                return True
            else:
                logger.warning(f"Model {model_name} did not provide valid response")
                return False
                
        except ollama.ResponseError as e:
            logger.error(f"Ollama error validating model {model_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating model {model_name}: {e}")
            return False
    
    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model details
            
        Raises:
            ValueError: If model is not found
        """
        try:
            # Ensure we have fresh model info
            self.list_ollama_models(refresh_cache=True)
            
            if model_name not in self._model_cache:
                raise ValueError(f"Model '{model_name}' not found")
            
            model_info = self._model_cache[model_name].copy()
            model_info['name'] = model_name
            model_info['is_valid'] = self.validate_model(model_name)
            
            return model_info
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to get model details for {model_name}: {e}")
            raise RuntimeError(f"Could not retrieve model details: {e}")
    
    def get_recommended_models(self) -> List[str]:
        """Get a list of recommended models for RAG applications.
        
        Returns:
            List of recommended model names that are available
        """
        # Common models good for RAG applications
        recommended = [
            'llama3:latest',
            'llama3:8b',
            'llama3:70b',
            'llama2:latest',
            'llama2:7b',
            'llama2:13b',
            'mistral:latest',
            'mistral:7b',
            'codellama:latest',
            'codellama:7b'
        ]
        
        try:
            available_models = self.list_ollama_models()
            available_recommended = [model for model in recommended if model in available_models]
            
            logger.debug(f"Recommended models available: {available_recommended}")
            return available_recommended
            
        except Exception as e:
            logger.error(f"Failed to get recommended models: {e}")
            return []
    
    def find_best_model(self) -> Optional[str]:
        """Find the best available model for RAG applications.
        
        Returns:
            Name of the best available model, or None if no models available
        """
        try:
            recommended = self.get_recommended_models()
            
            if not recommended:
                # Fall back to any available model
                all_models = self.list_ollama_models()
                if all_models:
                    logger.info(f"No recommended models found, using first available: {all_models[0]}")
                    return all_models[0]
                else:
                    logger.warning("No models available")
                    return None
            
            # Return the first recommended model (they're ordered by preference)
            best_model = recommended[0]
            logger.info(f"Best available model: {best_model}")
            return best_model
            
        except Exception as e:
            logger.error(f"Failed to find best model: {e}")
            return None
    
    def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a specific model is available and ready to use.
        
        Args:
            model_name: Name of the model to ensure availability
            
        Returns:
            True if model is available and ready, False otherwise
        """
        try:
            if not self.validate_model(model_name):
                logger.warning(f"Model {model_name} is not available or not functional")
                return False
            
            logger.info(f"Model {model_name} is available and ready")
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring model availability for {model_name}: {e}")
            return False
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about available models.
        
        Returns:
            Dictionary containing model statistics
        """
        try:
            models = self.list_ollama_models(refresh_cache=True)
            recommended = self.get_recommended_models()
            
            stats = {
                'total_models': len(models),
                'recommended_available': len(recommended),
                'models': models,
                'recommended': recommended,
                'cache_age': time.time() - self._cache_timestamp if self._cache_timestamp else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get model stats: {e}")
            return {
                'total_models': 0,
                'recommended_available': 0,
                'models': [],
                'recommended': [],
                'cache_age': 0,
                'error': str(e)
            }
    
    def clear_cache(self):
        """Clear the model cache to force fresh data on next request."""
        self._model_cache = {}
        self._cache_timestamp = 0
        logger.debug("Model cache cleared")
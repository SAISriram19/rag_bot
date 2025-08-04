"""LLM Manager for handling Ollama model interactions."""

import logging
import time
from typing import List, Optional, Dict, Any
import ollama
from ollama import Client
from src.config import config
from .error_handler import (
    error_handler, retry_on_failure, log_function_call,
    LLMError, ErrorCategory, ErrorSeverity
)
from .logging_config import get_logging_manager

logger = logging.getLogger(__name__)
logging_manager = get_logging_manager()


class LLMManager:
    """Manages Ollama LLM interactions and response generation."""
    
    def __init__(self, base_url: str = None, default_model: str = None):
        """Initialize LLM Manager with Ollama client.
        
        Args:
            base_url: Ollama server URL (defaults to config value)
            default_model: Default model to use (defaults to config value)
        """
        self.base_url = base_url or config.ollama_base_url
        self.default_model = default_model or config.default_ollama_model
        self.current_model = self.default_model
        
        try:
            self.client = Client(host=self.base_url)
            logger.info(f"Initialized LLM Manager with base URL: {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise ConnectionError(f"Could not connect to Ollama at {self.base_url}: {e}")
    
    @log_function_call(include_args=False)  # Don't log prompts for privacy
    @retry_on_failure(max_retries=3, delay=2.0, exceptions=(ConnectionError, TimeoutError), error_handler=error_handler)
    def generate_response(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate a response using the specified or current model.
        
        Args:
            prompt: The input prompt for the model
            model: Model name to use (optional, uses current_model if not specified)
            **kwargs: Additional parameters for the model
            
        Returns:
            Generated response text
            
        Raises:
            LLMError: If generation fails or model is not available
        """
        model_to_use = model or self.current_model
        
        context = {
            'model': model_to_use,
            'prompt_length': len(prompt),
            'base_url': self.base_url
        }
        
        try:
            logger.debug(f"Generating response with model: {model_to_use}")
            start_time = time.time()
            
            # Default generation parameters
            generation_params = {
                'temperature': kwargs.get('temperature', 0.7),
                'top_p': kwargs.get('top_p', 0.9),
                'max_tokens': kwargs.get('max_tokens', 2000),
            }
            
            response = self.client.generate(
                model=model_to_use,
                prompt=prompt,
                options=generation_params,
                stream=False
            )
            
            generation_time = time.time() - start_time
            
            # Log performance metric
            logging_manager.log_performance_metric(
                'llm_response_time',
                generation_time,
                context={'model': model_to_use, 'prompt_length': len(prompt)}
            )
            
            logger.debug(f"Response generated in {generation_time:.2f} seconds")
            
            if 'response' in response:
                response_text = response['response'].strip()
                if not response_text:
                    raise LLMError(
                        "Empty response generated",
                        model_name=model_to_use,
                        severity=ErrorSeverity.MEDIUM,
                        context=context
                    )
                return response_text
            else:
                raise LLMError(
                    "No response field in Ollama output",
                    model_name=model_to_use,
                    severity=ErrorSeverity.HIGH,
                    context=context
                )
                
        except ollama.ResponseError as e:
            if "model not found" in str(e).lower():
                available_models = self.get_available_models()
                raise LLMError(
                    f"Model '{model_to_use}' not found",
                    model_name=model_to_use,
                    severity=ErrorSeverity.HIGH,
                    context={**context, 'available_models': available_models}
                ) from e
            else:
                raise LLMError(
                    f"Ollama response error: {str(e)}",
                    model_name=model_to_use,
                    severity=ErrorSeverity.HIGH,
                    context=context
                ) from e
        except ConnectionError as e:
            raise LLMError(
                f"Connection error: {str(e)}",
                model_name=model_to_use,
                severity=ErrorSeverity.CRITICAL,
                context=context
            ) from e
        except Exception as e:
            # Handle unexpected errors
            error_info = error_handler.handle_error(e, context=context, category=ErrorCategory.LLM_INTERACTION)
            raise LLMError(
                f"Unexpected error during response generation: {str(e)}",
                model_name=model_to_use,
                severity=ErrorSeverity.HIGH,
                context=context
            ) from e
    
    @log_function_call()
    @retry_on_failure(max_retries=2, delay=1.0, exceptions=(ConnectionError,), error_handler=error_handler)
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama.
        
        Returns:
            List of available model names
            
        Raises:
            LLMError: If unable to connect to Ollama or retrieve models
        """
        context = {'base_url': self.base_url}
        
        try:
            models_response = self.client.list()
            models = []
            
            if 'models' in models_response:
                for model in models_response['models']:
                    if 'name' in model:
                        models.append(model['name'])
            
            logger.debug(f"Available models: {models}")
            return models
            
        except ConnectionError as e:
            raise LLMError(
                f"Could not connect to Ollama: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                context=context
            ) from e
        except Exception as e:
            error_info = error_handler.handle_error(e, context=context, category=ErrorCategory.LLM_INTERACTION)
            raise LLMError(
                f"Could not retrieve models from Ollama: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=context
            ) from e
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model.
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            True if switch was successful, False otherwise
            
        Raises:
            ValueError: If model is not available
        """
        try:
            available_models = self.get_available_models()
            
            if model_name not in available_models:
                raise ValueError(f"Model '{model_name}' not available. Available models: {available_models}")
            
            # Test the model by generating a simple response
            test_prompt = "Hello, this is a test. Please respond with 'OK'."
            test_response = self.generate_response(test_prompt, model=model_name)
            
            if test_response:
                self.current_model = model_name
                logger.info(f"Successfully switched to model: {model_name}")
                return True
            else:
                logger.error(f"Model {model_name} did not respond to test prompt")
                return False
                
        except Exception as e:
            logger.error(f"Failed to switch to model {model_name}: {e}")
            raise
    
    def get_current_model(self) -> str:
        """Get the currently active model name.
        
        Returns:
            Current model name
        """
        return self.current_model
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            available_models = self.get_available_models()
            return model_name in available_models
        except Exception:
            return False
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model (uses current model if not specified)
            
        Returns:
            Dictionary containing model information
            
        Raises:
            ValueError: If model is not available
        """
        model_to_check = model_name or self.current_model
        
        try:
            models_response = self.client.list()
            
            if 'models' in models_response:
                for model in models_response['models']:
                    if model.get('name') == model_to_check:
                        return {
                            'name': model.get('name'),
                            'size': model.get('size', 'Unknown'),
                            'modified_at': model.get('modified_at'),
                            'digest': model.get('digest', 'Unknown')
                        }
            
            raise ValueError(f"Model '{model_to_check}' not found")
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_to_check}: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Ollama server.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.get_available_models()
            return True
        except Exception:
            return False
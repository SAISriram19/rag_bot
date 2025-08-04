"""Centralized error handling and logging system for the RAG bot."""

import logging
import traceback
import time
import functools
import sys
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better classification."""
    DOCUMENT_PROCESSING = "document_processing"
    VECTOR_STORE = "vector_store"
    LLM_INTERACTION = "llm_interaction"
    MEMORY_MANAGEMENT = "memory_management"
    FILE_UPLOAD = "file_upload"
    QUERY_PROCESSING = "query_processing"
    SYSTEM = "system"
    NETWORK = "network"
    CONFIGURATION = "configuration"


@dataclass
class ErrorInfo:
    """Structured error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    technical_details: str
    timestamp: float
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class RagBotError(Exception):
    """Base exception class for RAG bot errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 user_message: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}  # Set context first
        self.user_message = user_message or self._generate_user_message(message)
        self.timestamp = time.time()
    
    def _generate_user_message(self, technical_message: str) -> str:
        """Generate user-friendly message from technical message."""
        return f"An error occurred: {technical_message}"


class DocumentProcessingError(RagBotError):
    """Error during document processing."""
    
    def __init__(self, message: str, file_path: str = None, **kwargs):
        # Ensure context is initialized before calling super
        context = kwargs.get('context', {})
        if file_path:
            context['file_path'] = file_path
        kwargs['context'] = context
        super().__init__(message, ErrorCategory.DOCUMENT_PROCESSING, **kwargs)
    
    def _generate_user_message(self, technical_message: str) -> str:
        file_name = Path(self.context.get('file_path', 'unknown')).name
        return f"Failed to process document '{file_name}': {technical_message}"


class VectorStoreError(RagBotError):
    """Error during vector store operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.VECTOR_STORE, **kwargs)
    
    def _generate_user_message(self, technical_message: str) -> str:
        return "There was an issue with the document database. Please try again or contact support."


class LLMError(RagBotError):
    """Error during LLM interactions."""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        # Ensure context is initialized before calling super
        context = kwargs.get('context', {})
        if model_name:
            context['model_name'] = model_name
        kwargs['context'] = context
        super().__init__(message, ErrorCategory.LLM_INTERACTION, **kwargs)
    
    def _generate_user_message(self, technical_message: str) -> str:
        model = self.context.get('model_name', 'the AI model')
        return f"Unable to generate response using {model}. Please check if the model is available."


class QueryProcessingError(RagBotError):
    """Error during query processing."""
    
    def __init__(self, message: str, query: str = None, **kwargs):
        # Ensure context is initialized before calling super
        context = kwargs.get('context', {})
        if query:
            context['query'] = query[:100] + "..." if len(query) > 100 else query
        kwargs['context'] = context
        super().__init__(message, ErrorCategory.QUERY_PROCESSING, **kwargs)
    
    def _generate_user_message(self, technical_message: str) -> str:
        return "Unable to process your question. Please try rephrasing or check if documents are loaded."


class ErrorHandler:
    """Centralized error handling and logging system."""
    
    def __init__(self, log_file: str = "error_log.json"):
        self.logger = logging.getLogger(__name__)
        self.error_log_file = log_file
        self.error_history: List[ErrorInfo] = []
        self.max_error_history = 1000
        
        # User-friendly error messages mapping
        self.error_messages = {
            "ConnectionError": "Unable to connect to the service. Please check if all components are running.",
            "FileNotFoundError": "The requested file could not be found. Please check the file path.",
            "PermissionError": "Permission denied. Please check file permissions.",
            "MemoryError": "The system is running low on memory. Please try with smaller files.",
            "TimeoutError": "The operation timed out. Please try again.",
            "ValueError": "Invalid input provided. Please check your input and try again.",
            "KeyError": "Missing required configuration. Please check your settings.",
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None, 
                    category: ErrorCategory = ErrorCategory.SYSTEM,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> ErrorInfo:
        """Handle an error with comprehensive logging and user feedback."""
        
        # Generate unique error ID
        error_id = f"ERR_{int(time.time())}_{hash(str(error)) % 10000:04d}"
        
        # Extract error information
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Generate user-friendly message
        if isinstance(error, RagBotError):
            user_message = error.user_message
            category = error.category
            severity = error.severity
            context = {**(context or {}), **error.context}
        else:
            user_message = self._generate_user_friendly_message(error_type, error_message)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=error_id,
            category=category,
            severity=severity,
            message=error_message,
            user_message=user_message,
            technical_details=f"{error_type}: {error_message}",
            timestamp=time.time(),
            context=context or {},
            stack_trace=stack_trace
        )
        
        # Log the error
        self._log_error(error_info)
        
        # Store in history
        self._store_error(error_info)
        
        return error_info
    
    def _generate_user_friendly_message(self, error_type: str, error_message: str) -> str:
        """Generate user-friendly error message."""
        base_message = self.error_messages.get(error_type, 
                                             "An unexpected error occurred. Please try again.")
        
        # Add specific context for common errors
        if "connection" in error_message.lower():
            return "Unable to connect to required services. Please ensure all components are running."
        elif "file" in error_message.lower() and "not found" in error_message.lower():
            return "The requested file could not be found. Please check the file path and try again."
        elif "permission" in error_message.lower():
            return "Permission denied. Please check file permissions and try again."
        elif "memory" in error_message.lower():
            return "The system is running low on memory. Please try with smaller files or restart the application."
        elif "timeout" in error_message.lower():
            return "The operation timed out. Please try again or check your network connection."
        
        return base_message
    
    def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error with appropriate level."""
        log_message = f"[{error_info.error_id}] {error_info.category.value}: {error_info.message}"
        
        if error_info.context:
            log_message += f" | Context: {json.dumps(error_info.context, default=str)}"
        
        # Log based on severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            if error_info.stack_trace:
                self.logger.critical(f"Stack trace: {error_info.stack_trace}")
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
            if error_info.stack_trace:
                self.logger.error(f"Stack trace: {error_info.stack_trace}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _store_error(self, error_info: ErrorInfo) -> None:
        """Store error in history and persistent log."""
        # Add to memory history
        self.error_history.append(error_info)
        
        # Maintain history size
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
        
        # Write to persistent log
        try:
            error_data = {
                'error_id': error_info.error_id,
                'category': error_info.category.value,
                'severity': error_info.severity.value,
                'message': error_info.message,
                'user_message': error_info.user_message,
                'technical_details': error_info.technical_details,
                'timestamp': error_info.timestamp,
                'context': error_info.context,
                'retry_count': error_info.retry_count
            }
            
            # Append to JSON log file
            with open(self.error_log_file, 'a') as f:
                f.write(json.dumps(error_data) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to write error to log file: {e}")
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        
        # Count by category and severity
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            'total_errors': len(recent_errors),
            'time_period_hours': hours,
            'category_breakdown': category_counts,
            'severity_breakdown': severity_counts,
            'most_recent_errors': [
                {
                    'error_id': e.error_id,
                    'category': e.category.value,
                    'severity': e.severity.value,
                    'message': e.user_message,
                    'timestamp': e.timestamp
                }
                for e in recent_errors[-10:]  # Last 10 errors
            ]
        }


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    backoff_factor: float = 2.0,
                    exceptions: tuple = (Exception,),
                    error_handler: ErrorHandler = None):
    """Decorator for automatic retry on failure with exponential backoff."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Final attempt failed, handle error
                        if error_handler:
                            error_info = error_handler.handle_error(
                                e, 
                                context={
                                    'function': func.__name__,
                                    'attempt': attempt + 1,
                                    'max_retries': max_retries
                                }
                            )
                            # Update retry count in error info
                            error_info.retry_count = attempt + 1
                        raise e
                    
                    # Log retry attempt
                    logger = logging.getLogger(func.__module__)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}")
                    logger.info(f"Retrying in {current_delay:.1f} seconds...")
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def log_function_call(include_args: bool = False, include_result: bool = False):
    """Decorator to log function calls with optional arguments and results."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Log function entry
            log_msg = f"Calling {func_name}"
            if include_args and (args or kwargs):
                # Sanitize sensitive information
                safe_args = [str(arg)[:100] + "..." if len(str(arg)) > 100 else str(arg) for arg in args]
                safe_kwargs = {k: (str(v)[:100] + "..." if len(str(v)) > 100 else str(v)) for k, v in kwargs.items()}
                log_msg += f" with args={safe_args}, kwargs={safe_kwargs}"
            
            logger.debug(log_msg)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log successful completion
                log_msg = f"Completed {func_name} in {execution_time:.3f}s"
                if include_result and result is not None:
                    result_str = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    log_msg += f" with result={result_str}"
                
                logger.debug(log_msg)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Failed {func_name} after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


# Global error handler instance
error_handler = ErrorHandler()
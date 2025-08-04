"""Advanced logging configuration for the RAG bot."""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..config import config


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    log_level: str = "INFO"
    log_dir: str = "logs"
    max_file_size_mb: int = 10
    backup_count: int = 5
    console_logging: bool = True
    file_logging: bool = True
    json_logging: bool = False
    performance_logging: bool = True
    error_logging: bool = True
    debug_logging: bool = False


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class PerformanceFilter(logging.Filter):
    """Filter for performance-related log messages."""
    
    def filter(self, record):
        return hasattr(record, 'performance') or 'performance' in record.getMessage().lower()


class ErrorFilter(logging.Filter):
    """Filter for error-related log messages."""
    
    def filter(self, record):
        return record.levelno >= logging.WARNING


class LoggingManager:
    """Manages comprehensive logging configuration for the RAG bot."""
    
    def __init__(self, logging_config: LoggingConfig = None):
        self.config = logging_config or LoggingConfig()
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        
        # Ensure log directory exists
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        self._configure_root_logger()
        
        # Set up specialized loggers
        self._setup_specialized_loggers()
    
    def _configure_root_logger(self):
        """Configure the root logger with basic settings."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.config.console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
            root_logger.addHandler(console_handler)
            self.handlers['console'] = console_handler
        
        # File handler with rotation
        if self.config.file_logging:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=os.path.join(self.config.log_dir, 'rag_bot.log'),
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            
            if self.config.json_logging:
                file_formatter = JSONFormatter()
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
            
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(getattr(logging, self.config.log_level.upper()))
            root_logger.addHandler(file_handler)
            self.handlers['file'] = file_handler
    
    def _setup_specialized_loggers(self):
        """Set up specialized loggers for different components."""
        
        # Performance logger
        if self.config.performance_logging:
            perf_logger = logging.getLogger('performance')
            perf_handler = logging.handlers.RotatingFileHandler(
                filename=os.path.join(self.config.log_dir, 'performance.log'),
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            perf_formatter = JSONFormatter() if self.config.json_logging else logging.Formatter(
                '%(asctime)s - %(message)s'
            )
            perf_handler.setFormatter(perf_formatter)
            perf_handler.addFilter(PerformanceFilter())
            perf_logger.addHandler(perf_handler)
            perf_logger.setLevel(logging.INFO)
            perf_logger.propagate = False  # Don't propagate to root logger
            self.loggers['performance'] = perf_logger
            self.handlers['performance'] = perf_handler
        
        # Error logger
        if self.config.error_logging:
            error_logger = logging.getLogger('errors')
            error_handler = logging.handlers.RotatingFileHandler(
                filename=os.path.join(self.config.log_dir, 'errors.log'),
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            error_formatter = JSONFormatter() if self.config.json_logging else logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n%(exc_info)s'
            )
            error_handler.setFormatter(error_formatter)
            error_handler.addFilter(ErrorFilter())
            error_logger.addHandler(error_handler)
            error_logger.setLevel(logging.WARNING)
            error_logger.propagate = False
            self.loggers['errors'] = error_logger
            self.handlers['errors'] = error_handler
        
        # Debug logger (only when debug logging is enabled)
        if self.config.debug_logging:
            debug_logger = logging.getLogger('debug')
            debug_handler = logging.handlers.RotatingFileHandler(
                filename=os.path.join(self.config.log_dir, 'debug.log'),
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            debug_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            debug_handler.setFormatter(debug_formatter)
            debug_logger.addHandler(debug_handler)
            debug_logger.setLevel(logging.DEBUG)
            debug_logger.propagate = False
            self.loggers['debug'] = debug_logger
            self.handlers['debug'] = debug_handler
        
        # Component-specific loggers
        self._setup_component_loggers()
    
    def _setup_component_loggers(self):
        """Set up loggers for specific components."""
        components = [
            'document_processor',
            'vector_store_manager', 
            'llm_manager',
            'query_handler',
            'memory_manager',
            'gradio_interface'
        ]
        
        for component in components:
            logger = logging.getLogger(f'src.services.{component}')
            
            # Add component-specific file handler
            component_handler = logging.handlers.RotatingFileHandler(
                filename=os.path.join(self.config.log_dir, f'{component}.log'),
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            
            component_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            component_handler.setFormatter(component_formatter)
            logger.addHandler(component_handler)
            
            self.loggers[component] = logger
            self.handlers[f'{component}_file'] = component_handler
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger by name."""
        return logging.getLogger(name)
    
    def log_system_info(self):
        """Log system information at startup."""
        logger = self.get_logger('system')
        
        system_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'log_level': self.config.log_level,
            'log_directory': self.config.log_dir,
            'config': {
                'chroma_persist_directory': config.chroma_persist_directory,
                'embedding_model': config.embedding_model,
                'default_ollama_model': config.default_ollama_model,
                'ollama_base_url': config.ollama_base_url,
                'max_file_size_mb': config.max_file_size_mb
            }
        }
        
        logger.info("System startup information", extra={'system_info': system_info})
    
    def log_performance_metric(self, metric_name: str, value: float, 
                             context: Dict[str, Any] = None):
        """Log a performance metric."""
        perf_logger = self.loggers.get('performance')
        if perf_logger:
            metric_data = {
                'metric': metric_name,
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'context': context or {}
            }
            perf_logger.info(f"Performance metric: {metric_name}={value}", 
                           extra={'performance': True, 'metric_data': metric_data})
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with additional context."""
        error_logger = self.loggers.get('errors')
        if error_logger:
            error_data = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {},
                'timestamp': datetime.now().isoformat()
            }
            error_logger.error(f"Error occurred: {error}", 
                             extra={'error_data': error_data}, exc_info=True)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about logging activity."""
        stats = {
            'log_directory': self.config.log_dir,
            'active_loggers': len(self.loggers),
            'active_handlers': len(self.handlers),
            'log_files': []
        }
        
        # Get log file information
        log_dir = Path(self.config.log_dir)
        if log_dir.exists():
            for log_file in log_dir.glob('*.log*'):
                file_stats = log_file.stat()
                stats['log_files'].append({
                    'name': log_file.name,
                    'size_mb': file_stats.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                })
        
        return stats
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files."""
        logger = self.get_logger('system')
        log_dir = Path(self.config.log_dir)
        
        if not log_dir.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        cleaned_files = []
        
        for log_file in log_dir.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    cleaned_files.append(log_file.name)
                except Exception as e:
                    logger.warning(f"Failed to delete old log file {log_file}: {e}")
        
        if cleaned_files:
            logger.info(f"Cleaned up {len(cleaned_files)} old log files: {cleaned_files}")


def setup_logging(log_level: str = None, log_dir: str = None, 
                 json_logging: bool = False) -> LoggingManager:
    """Set up comprehensive logging for the RAG bot."""
    
    logging_config = LoggingConfig(
        log_level=log_level or config.log_level,
        log_dir=log_dir or "logs",
        json_logging=json_logging,
        performance_logging=config.enable_performance_monitoring,
        debug_logging=log_level == "DEBUG"
    )
    
    return LoggingManager(logging_config)


# Global logging manager instance
logging_manager = None


def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance."""
    global logging_manager
    if logging_manager is None:
        logging_manager = setup_logging()
    return logging_manager
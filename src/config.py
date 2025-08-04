"""Configuration management for the RAG bot."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


@dataclass
class AppConfig:
    """Application configuration with default values and environment variable support."""
    
    # Vector Store Settings
    chroma_persist_directory: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # LLM Settings
    default_ollama_model: str = "llama3.2:1b"
    ollama_base_url: str = "http://localhost:11434"
    
    # Processing Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_chunks: int = 3
    
    # Memory Settings
    max_conversation_history: int = 10
    memory_buffer_size: int = 2000
    
    # Interface Settings
    gradio_port: int = 7860
    gradio_share: bool = False
    
    # File Upload Settings
    max_file_size_mb: int = 50
    supported_file_types: tuple = (".pdf", ".txt", ".md")
    
    # Logging Settings
    log_level: str = "INFO"
    log_dir: str = "logs"
    max_log_file_size_mb: int = 10
    log_backup_count: int = 5
    enable_json_logging: bool = False
    enable_error_logging: bool = True
    enable_debug_logging: bool = False
    
    # Performance Monitoring Settings
    enable_performance_monitoring: bool = True
    performance_log_file: str = "performance.log"
    max_performance_records: int = 1000
    performance_metrics_window_minutes: int = 60
    memory_monitoring_enabled: bool = True
    detailed_performance_logging: bool = True
    
    # Performance Optimization Settings
    auto_optimize_performance: bool = True
    memory_threshold_mb: int = 1000
    response_time_threshold_s: float = 10.0
    vector_search_k_limit: int = 10
    context_length_limit: int = 2000
    
    def __post_init__(self):
        """Override defaults with environment variables if they exist."""
        self.chroma_persist_directory = os.getenv("CHROMA_PERSIST_DIR", self.chroma_persist_directory)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.default_ollama_model = os.getenv("DEFAULT_OLLAMA_MODEL", self.default_ollama_model)
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", self.ollama_base_url)
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.log_dir = os.getenv("LOG_DIR", self.log_dir)
        
        # Logging environment variables
        if os.getenv("MAX_LOG_FILE_SIZE_MB"):
            self.max_log_file_size_mb = int(os.getenv("MAX_LOG_FILE_SIZE_MB"))
        if os.getenv("LOG_BACKUP_COUNT"):
            self.log_backup_count = int(os.getenv("LOG_BACKUP_COUNT"))
        if os.getenv("ENABLE_JSON_LOGGING"):
            self.enable_json_logging = os.getenv("ENABLE_JSON_LOGGING").lower() == "true"
        if os.getenv("ENABLE_ERROR_LOGGING"):
            self.enable_error_logging = os.getenv("ENABLE_ERROR_LOGGING").lower() == "true"
        if os.getenv("ENABLE_DEBUG_LOGGING"):
            self.enable_debug_logging = os.getenv("ENABLE_DEBUG_LOGGING").lower() == "true"
        
        # Convert string environment variables to appropriate types
        if os.getenv("CHUNK_SIZE"):
            self.chunk_size = int(os.getenv("CHUNK_SIZE"))
        if os.getenv("CHUNK_OVERLAP"):
            self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
        if os.getenv("MAX_RETRIEVAL_CHUNKS"):
            self.max_retrieval_chunks = int(os.getenv("MAX_RETRIEVAL_CHUNKS"))
        if os.getenv("MAX_CONVERSATION_HISTORY"):
            self.max_conversation_history = int(os.getenv("MAX_CONVERSATION_HISTORY"))
        if os.getenv("MEMORY_BUFFER_SIZE"):
            self.memory_buffer_size = int(os.getenv("MEMORY_BUFFER_SIZE"))
        if os.getenv("GRADIO_PORT"):
            self.gradio_port = int(os.getenv("GRADIO_PORT"))
        if os.getenv("GRADIO_SHARE"):
            self.gradio_share = os.getenv("GRADIO_SHARE").lower() == "true"
        if os.getenv("MAX_FILE_SIZE_MB"):
            self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB"))
        
        # Performance monitoring environment variables
        if os.getenv("ENABLE_PERFORMANCE_MONITORING"):
            self.enable_performance_monitoring = os.getenv("ENABLE_PERFORMANCE_MONITORING").lower() == "true"
        if os.getenv("PERFORMANCE_LOG_FILE"):
            self.performance_log_file = os.getenv("PERFORMANCE_LOG_FILE")
        if os.getenv("MAX_PERFORMANCE_RECORDS"):
            self.max_performance_records = int(os.getenv("MAX_PERFORMANCE_RECORDS"))
        if os.getenv("MEMORY_MONITORING_ENABLED"):
            self.memory_monitoring_enabled = os.getenv("MEMORY_MONITORING_ENABLED").lower() == "true"
        if os.getenv("AUTO_OPTIMIZE_PERFORMANCE"):
            self.auto_optimize_performance = os.getenv("AUTO_OPTIMIZE_PERFORMANCE").lower() == "true"
        if os.getenv("MEMORY_THRESHOLD_MB"):
            self.memory_threshold_mb = int(os.getenv("MEMORY_THRESHOLD_MB"))
        if os.getenv("RESPONSE_TIME_THRESHOLD_S"):
            self.response_time_threshold_s = float(os.getenv("RESPONSE_TIME_THRESHOLD_S"))


# Global configuration instance
config = AppConfig()
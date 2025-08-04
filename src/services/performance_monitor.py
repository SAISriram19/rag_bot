"""Performance monitoring and optimization system for the RAG bot."""

import logging
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics
import json
import os

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Response time metrics
    response_times: List[float] = field(default_factory=list)
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Memory metrics
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    memory_growth_rate: float = 0.0
    
    # Query metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    success_rate: float = 0.0
    
    # Vector store metrics
    vector_search_times: List[float] = field(default_factory=list)
    avg_vector_search_time: float = 0.0
    vector_store_size: int = 0
    
    # LLM metrics
    llm_generation_times: List[float] = field(default_factory=list)
    avg_llm_generation_time: float = 0.0
    
    # Document processing metrics
    document_processing_times: List[float] = field(default_factory=list)
    avg_document_processing_time: float = 0.0
    documents_processed: int = 0
    
    # Timestamp
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'response_times': {
                'avg': self.avg_response_time,
                'min': self.min_response_time if self.min_response_time != float('inf') else 0.0,
                'max': self.max_response_time,
                'p95': self.p95_response_time,
                'p99': self.p99_response_time,
                'count': len(self.response_times)
            },
            'memory': {
                'current_mb': self.memory_usage_mb,
                'peak_mb': self.peak_memory_mb,
                'growth_rate': self.memory_growth_rate
            },
            'queries': {
                'total': self.total_queries,
                'successful': self.successful_queries,
                'failed': self.failed_queries,
                'success_rate': self.success_rate
            },
            'vector_search': {
                'avg_time': self.avg_vector_search_time,
                'count': len(self.vector_search_times),
                'store_size': self.vector_store_size
            },
            'llm_generation': {
                'avg_time': self.avg_llm_generation_time,
                'count': len(self.llm_generation_times)
            },
            'document_processing': {
                'avg_time': self.avg_document_processing_time,
                'count': len(self.document_processing_times),
                'documents_processed': self.documents_processed
            },
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class QueryPerformanceRecord:
    """Record for individual query performance."""
    
    query_id: str
    timestamp: datetime
    query_text: str
    response_time: float
    vector_search_time: float
    llm_generation_time: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None
    model_used: str = ""
    context_chunks_count: int = 0
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            'query_id': self.query_id,
            'timestamp': self.timestamp.isoformat(),
            'query_text': self.query_text[:100] + '...' if len(self.query_text) > 100 else self.query_text,
            'response_time': self.response_time,
            'vector_search_time': self.vector_search_time,
            'llm_generation_time': self.llm_generation_time,
            'memory_usage_mb': self.memory_usage_mb,
            'success': self.success,
            'error_message': self.error_message,
            'model_used': self.model_used,
            'context_chunks_count': self.context_chunks_count,
            'confidence_score': self.confidence_score
        }


class PerformanceMonitor:
    """Monitors and tracks performance metrics for the RAG bot."""
    
    def __init__(
        self,
        max_records: int = 1000,
        metrics_window_minutes: int = 60,
        enable_memory_monitoring: bool = True,
        enable_detailed_logging: bool = True
    ):
        """
        Initialize the performance monitor.
        
        Args:
            max_records: Maximum number of query records to keep
            metrics_window_minutes: Time window for metrics calculation
            enable_memory_monitoring: Whether to monitor memory usage
            enable_detailed_logging: Whether to log detailed performance data
        """
        self.max_records = max_records
        self.metrics_window = timedelta(minutes=metrics_window_minutes)
        self.enable_memory_monitoring = enable_memory_monitoring
        self.enable_detailed_logging = enable_detailed_logging
        
        # Performance data storage
        self.query_records: deque = deque(maxlen=max_records)
        self.metrics = PerformanceMetrics()
        
        # Memory monitoring
        self.memory_samples: deque = deque(maxlen=100)  # Last 100 memory samples
        self.peak_memory = 0.0
        
        # Performance optimization settings
        self.optimization_settings = {
            'vector_search_k_limit': 10,
            'context_length_limit': 4000,
            'memory_threshold_mb': 1000,
            'response_time_threshold_s': 10.0,
            'auto_optimize': True
        }
        
        # Background monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Start background monitoring if enabled
        if self.enable_memory_monitoring:
            self.start_monitoring()
        
        logger.info(f"Performance monitor initialized with max_records={max_records}, "
                   f"window={metrics_window_minutes}min, memory_monitoring={enable_memory_monitoring}")
    
    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self._monitor_thread.start()
        logger.info("Background performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Background performance monitoring stopped")
    
    def _background_monitor(self) -> None:
        """Background thread for continuous monitoring."""
        while self._monitoring_active:
            try:
                # Sample memory usage
                if self.enable_memory_monitoring:
                    self._sample_memory()
                
                # Update metrics
                self._update_metrics()
                
                # Check for optimization opportunities
                if self.optimization_settings['auto_optimize']:
                    self._check_optimization_opportunities()
                
                # Sleep for monitoring interval
                time.sleep(5.0)  # Sample every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                time.sleep(10.0)  # Wait longer on error
    
    def _sample_memory(self) -> None:
        """Sample current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            self.memory_samples.append({
                'timestamp': datetime.now(),
                'memory_mb': memory_mb
            })
            
            # Update peak memory
            if memory_mb > self.peak_memory:
                self.peak_memory = memory_mb
            
            # Update current memory in metrics
            self.metrics.memory_usage_mb = memory_mb
            self.metrics.peak_memory_mb = self.peak_memory
            
            # Calculate memory growth rate
            if len(self.memory_samples) >= 2:
                recent_samples = list(self.memory_samples)[-10:]  # Last 10 samples
                if len(recent_samples) >= 2:
                    time_diff = (recent_samples[-1]['timestamp'] - recent_samples[0]['timestamp']).total_seconds()
                    memory_diff = recent_samples[-1]['memory_mb'] - recent_samples[0]['memory_mb']
                    if time_diff > 0:
                        self.metrics.memory_growth_rate = memory_diff / time_diff  # MB per second
            
        except Exception as e:
            logger.warning(f"Failed to sample memory: {e}")
    
    def record_query_performance(
        self,
        query_id: str,
        query_text: str,
        response_time: float,
        vector_search_time: float = 0.0,
        llm_generation_time: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
        model_used: str = "",
        context_chunks_count: int = 0,
        confidence_score: float = 0.0
    ) -> None:
        """
        Record performance data for a query.
        
        Args:
            query_id: Unique identifier for the query
            query_text: The query text
            response_time: Total response time in seconds
            vector_search_time: Time spent on vector search
            llm_generation_time: Time spent on LLM generation
            success: Whether the query was successful
            error_message: Error message if query failed
            model_used: Name of the model used
            context_chunks_count: Number of context chunks retrieved
            confidence_score: Confidence score of the response
        """
        try:
            # Get current memory usage
            memory_mb = self.metrics.memory_usage_mb
            if not memory_mb:
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                except:
                    memory_mb = 0.0
            
            # Create performance record
            record = QueryPerformanceRecord(
                query_id=query_id,
                timestamp=datetime.now(),
                query_text=query_text,
                response_time=response_time,
                vector_search_time=vector_search_time,
                llm_generation_time=llm_generation_time,
                memory_usage_mb=memory_mb,
                success=success,
                error_message=error_message,
                model_used=model_used,
                context_chunks_count=context_chunks_count,
                confidence_score=confidence_score
            )
            
            # Add to records
            self.query_records.append(record)
            
            # Log detailed performance if enabled
            if self.enable_detailed_logging:
                logger.info(f"Query performance - ID: {query_id}, "
                           f"Response: {response_time:.2f}s, "
                           f"Vector: {vector_search_time:.2f}s, "
                           f"LLM: {llm_generation_time:.2f}s, "
                           f"Memory: {memory_mb:.1f}MB, "
                           f"Success: {success}")
            
            # Update metrics
            self._update_metrics()
            
        except Exception as e:
            logger.error(f"Failed to record query performance: {e}")
    
    def record_document_processing_performance(
        self,
        file_path: str,
        processing_time: float,
        chunks_created: int,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """
        Record performance data for document processing.
        
        Args:
            file_path: Path to the processed file
            processing_time: Time taken to process the document
            chunks_created: Number of chunks created
            success: Whether processing was successful
            error_message: Error message if processing failed
        """
        try:
            if success:
                self.metrics.document_processing_times.append(processing_time)
                self.metrics.documents_processed += 1
            
            if self.enable_detailed_logging:
                logger.info(f"Document processing - File: {os.path.basename(file_path)}, "
                           f"Time: {processing_time:.2f}s, "
                           f"Chunks: {chunks_created}, "
                           f"Success: {success}")
            
            # Update metrics after recording
            self._update_metrics()
            
        except Exception as e:
            logger.error(f"Failed to record document processing performance: {e}")
    
    def _update_metrics(self) -> None:
        """Update aggregated performance metrics."""
        try:
            # Filter recent records within the time window
            cutoff_time = datetime.now() - self.metrics_window
            recent_records = [r for r in self.query_records if r.timestamp >= cutoff_time]
            
            # Update query-based metrics only if we have recent records
            if recent_records:
                # Update response time metrics
                response_times = [r.response_time for r in recent_records]
                if response_times:
                    self.metrics.response_times = response_times
                    self.metrics.avg_response_time = statistics.mean(response_times)
                    self.metrics.min_response_time = min(response_times)
                    self.metrics.max_response_time = max(response_times)
                    
                    # Calculate percentiles
                    sorted_times = sorted(response_times)
                    if len(sorted_times) >= 20:  # Only calculate percentiles with sufficient data
                        self.metrics.p95_response_time = sorted_times[int(0.95 * len(sorted_times))]
                        self.metrics.p99_response_time = sorted_times[int(0.99 * len(sorted_times))]
                
                # Update query success metrics
                self.metrics.total_queries = len(recent_records)
                self.metrics.successful_queries = sum(1 for r in recent_records if r.success)
                self.metrics.failed_queries = self.metrics.total_queries - self.metrics.successful_queries
                
                if self.metrics.total_queries > 0:
                    self.metrics.success_rate = self.metrics.successful_queries / self.metrics.total_queries
                
                # Update vector search metrics
                vector_times = [r.vector_search_time for r in recent_records if r.vector_search_time > 0]
                if vector_times:
                    self.metrics.vector_search_times = vector_times
                    self.metrics.avg_vector_search_time = statistics.mean(vector_times)
                
                # Update LLM generation metrics
                llm_times = [r.llm_generation_time for r in recent_records if r.llm_generation_time > 0]
                if llm_times:
                    self.metrics.llm_generation_times = llm_times
                    self.metrics.avg_llm_generation_time = statistics.mean(llm_times)
            
            # Update document processing metrics
            if self.metrics.document_processing_times:
                self.metrics.avg_document_processing_time = statistics.mean(
                    self.metrics.document_processing_times
                )
            else:
                self.metrics.avg_document_processing_time = 0.0
            
            # Update timestamp
            self.metrics.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def _check_optimization_opportunities(self) -> None:
        """Check for performance optimization opportunities."""
        try:
            # Check memory usage
            if (self.metrics.memory_usage_mb > self.optimization_settings['memory_threshold_mb']):
                logger.warning(f"High memory usage detected: {self.metrics.memory_usage_mb:.1f}MB")
                self._suggest_memory_optimization()
            
            # Check response times
            if (self.metrics.avg_response_time > self.optimization_settings['response_time_threshold_s']):
                logger.warning(f"Slow response times detected: {self.metrics.avg_response_time:.2f}s")
                self._suggest_response_time_optimization()
            
            # Check vector search performance
            if (self.metrics.avg_vector_search_time > 2.0):  # More than 2 seconds
                logger.warning(f"Slow vector search detected: {self.metrics.avg_vector_search_time:.2f}s")
                self._suggest_vector_search_optimization()
            
        except Exception as e:
            logger.error(f"Failed to check optimization opportunities: {e}")
    
    def _suggest_memory_optimization(self) -> None:
        """Suggest memory optimization strategies."""
        suggestions = [
            "Consider reducing the number of cached embeddings",
            "Clear conversation history more frequently",
            "Reduce the maximum number of retrieved context chunks",
            "Process documents in smaller batches"
        ]
        
        for suggestion in suggestions:
            logger.info(f"Memory optimization suggestion: {suggestion}")
    
    def _suggest_response_time_optimization(self) -> None:
        """Suggest response time optimization strategies."""
        suggestions = [
            "Reduce the maximum context length",
            "Use a smaller, faster model",
            "Limit the number of retrieved chunks",
            "Optimize vector search parameters"
        ]
        
        for suggestion in suggestions:
            logger.info(f"Response time optimization suggestion: {suggestion}")
    
    def _suggest_vector_search_optimization(self) -> None:
        """Suggest vector search optimization strategies."""
        suggestions = [
            "Reduce the number of chunks retrieved (k parameter)",
            "Increase the relevance threshold to filter results",
            "Consider using approximate search methods",
            "Optimize embedding dimensions"
        ]
        
        for suggestion in suggestions:
            logger.info(f"Vector search optimization suggestion: {suggestion}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary.
        
        Returns:
            Dictionary containing performance summary
        """
        try:
            summary = {
                'overview': {
                    'monitoring_active': self._monitoring_active,
                    'total_records': len(self.query_records),
                    'metrics_window_minutes': self.metrics_window.total_seconds() / 60,
                    'last_updated': self.metrics.last_updated.isoformat()
                },
                'metrics': self.metrics.to_dict(),
                'optimization_settings': self.optimization_settings.copy(),
                'recent_performance': self._get_recent_performance_trends(),
                'recommendations': self._get_performance_recommendations()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {'error': str(e)}
    
    def _get_recent_performance_trends(self) -> Dict[str, Any]:
        """Get recent performance trends."""
        try:
            if len(self.query_records) < 10:
                return {'insufficient_data': True}
            
            # Get last 10 and previous 10 records for comparison
            recent_records = list(self.query_records)[-10:]
            previous_records = list(self.query_records)[-20:-10] if len(self.query_records) >= 20 else []
            
            trends = {}
            
            # Response time trend
            recent_avg = statistics.mean([r.response_time for r in recent_records])
            if previous_records:
                previous_avg = statistics.mean([r.response_time for r in previous_records])
                trends['response_time_trend'] = {
                    'recent_avg': recent_avg,
                    'previous_avg': previous_avg,
                    'change_percent': ((recent_avg - previous_avg) / previous_avg) * 100 if previous_avg > 0 else 0
                }
            
            # Success rate trend
            recent_success_rate = sum(1 for r in recent_records if r.success) / len(recent_records)
            if previous_records:
                previous_success_rate = sum(1 for r in previous_records if r.success) / len(previous_records)
                trends['success_rate_trend'] = {
                    'recent_rate': recent_success_rate,
                    'previous_rate': previous_success_rate,
                    'change_percent': ((recent_success_rate - previous_success_rate) / previous_success_rate) * 100 if previous_success_rate > 0 else 0
                }
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to get performance trends: {e}")
            return {'error': str(e)}
    
    def _get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        try:
            # Memory recommendations
            if self.metrics.memory_usage_mb > 500:
                recommendations.append("Consider reducing memory usage by clearing caches or reducing batch sizes")
            
            # Response time recommendations
            if self.metrics.avg_response_time > 5.0:
                recommendations.append("Response times are high - consider optimizing vector search or using a faster model")
            
            # Success rate recommendations
            if self.metrics.success_rate < 0.9:
                recommendations.append("Query success rate is low - check error logs and model availability")
            
            # Vector search recommendations
            if self.metrics.avg_vector_search_time > 1.0:
                recommendations.append("Vector search is slow - consider reducing the number of retrieved chunks")
            
            # General recommendations
            if len(self.query_records) > 500:
                recommendations.append("Consider archiving old performance records to reduce memory usage")
            
            if not recommendations:
                recommendations.append("Performance looks good! No specific optimizations needed at this time.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return ["Error generating recommendations"]
    
    def export_performance_data(self, file_path: str) -> bool:
        """
        Export performance data to a JSON file.
        
        Args:
            file_path: Path to export the data
            
        Returns:
            True if export was successful
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'summary': self.get_performance_summary(),
                'query_records': [record.to_dict() for record in self.query_records],
                'memory_samples': [
                    {
                        'timestamp': sample['timestamp'].isoformat(),
                        'memory_mb': sample['memory_mb']
                    }
                    for sample in self.memory_samples
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Performance data exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export performance data: {e}")
            return False
    
    def update_optimization_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update optimization settings.
        
        Args:
            settings: Dictionary of settings to update
        """
        try:
            for key, value in settings.items():
                if key in self.optimization_settings:
                    self.optimization_settings[key] = value
                    logger.info(f"Updated optimization setting {key} to {value}")
                else:
                    logger.warning(f"Unknown optimization setting: {key}")
            
        except Exception as e:
            logger.error(f"Failed to update optimization settings: {e}")
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get specific optimization recommendations based on current performance.
        
        Returns:
            Dictionary containing optimization recommendations
        """
        try:
            recommendations = {
                'vector_search': {},
                'memory': {},
                'response_time': {},
                'general': []
            }
            
            # Vector search optimizations
            if self.metrics.avg_vector_search_time > 1.0:
                current_k = self.optimization_settings.get('vector_search_k_limit', 5)
                recommended_k = max(3, current_k - 2)
                recommendations['vector_search']['reduce_k'] = {
                    'current': current_k,
                    'recommended': recommended_k,
                    'reason': 'Vector search is taking too long'
                }
            
            # Memory optimizations
            if self.metrics.memory_usage_mb > 800:
                recommendations['memory']['reduce_context_length'] = {
                    'current': self.optimization_settings.get('context_length_limit', 4000),
                    'recommended': 3000,
                    'reason': 'High memory usage detected'
                }
            
            # Response time optimizations
            if self.metrics.avg_response_time > 8.0:
                recommendations['response_time']['enable_caching'] = {
                    'description': 'Enable response caching for similar queries',
                    'expected_improvement': '30-50% faster responses for repeated queries'
                }
            
            # General recommendations
            if self.metrics.success_rate < 0.95:
                recommendations['general'].append({
                    'type': 'error_handling',
                    'description': 'Improve error handling and retry mechanisms',
                    'priority': 'high'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
            return {'error': str(e)}
    
    def __del__(self):
        """Cleanup when the monitor is destroyed."""
        try:
            self.stop_monitoring()
        except:
            pass
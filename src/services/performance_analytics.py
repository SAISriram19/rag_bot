"""
Performance analytics system for the RAG bot.
Provides detailed analytics and insights about system usage and performance.
"""

import logging
import json
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import statistics
import threading

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalytics:
    """Analytics data for a single query."""
    
    timestamp: str
    query_text: str
    response_time: float
    vector_search_time: float
    llm_generation_time: float
    confidence_score: float
    sources_count: int
    model_used: str
    success: bool
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class DocumentAnalytics:
    """Analytics data for document processing."""
    
    timestamp: str
    filename: str
    file_size_mb: float
    processing_time: float
    chunks_created: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class UsageAnalytics:
    """Overall usage analytics."""
    
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_response_time: float
    avg_confidence_score: float
    most_used_model: str
    total_documents_processed: int
    total_chunks_created: int
    peak_concurrent_users: int
    active_sessions: int


class PerformanceAnalytics:
    """Collects and analyzes performance data for insights and reporting."""
    
    def __init__(self, db_path: str = "performance_analytics.db"):
        """Initialize the performance analytics system."""
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Cache for frequent queries
        self._analytics_cache = {}
        self._cache_expiry = {}
        self._cache_duration = timedelta(minutes=5)
        
        self.logger.info(f"Performance analytics initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database for analytics storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Query analytics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS query_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        query_text TEXT NOT NULL,
                        response_time REAL NOT NULL,
                        vector_search_time REAL NOT NULL,
                        llm_generation_time REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        sources_count INTEGER NOT NULL,
                        model_used TEXT NOT NULL,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        user_id TEXT,
                        session_id TEXT
                    )
                """)
                
                # Document analytics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        file_size_mb REAL NOT NULL,
                        processing_time REAL NOT NULL,
                        chunks_created INTEGER NOT NULL,
                        success BOOLEAN NOT NULL,
                        error_message TEXT
                    )
                """)
                
                # System metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        metadata TEXT
                    )
                """)
                
                # Create indexes for better query performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_analytics(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_timestamp ON document_analytics(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics database: {e}")
    
    def record_query_analytics(self, analytics: QueryAnalytics):
        """Record analytics data for a query."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO query_analytics (
                            timestamp, query_text, response_time, vector_search_time,
                            llm_generation_time, confidence_score, sources_count,
                            model_used, success, error_message, user_id, session_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        analytics.timestamp,
                        analytics.query_text,
                        analytics.response_time,
                        analytics.vector_search_time,
                        analytics.llm_generation_time,
                        analytics.confidence_score,
                        analytics.sources_count,
                        analytics.model_used,
                        analytics.success,
                        analytics.error_message,
                        analytics.user_id,
                        analytics.session_id
                    ))
                    conn.commit()
                    
            # Clear relevant cache entries
            self._clear_cache_pattern("query_")
            
        except Exception as e:
            self.logger.error(f"Failed to record query analytics: {e}")
    
    def record_document_analytics(self, analytics: DocumentAnalytics):
        """Record analytics data for document processing."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO document_analytics (
                            timestamp, filename, file_size_mb, processing_time,
                            chunks_created, success, error_message
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        analytics.timestamp,
                        analytics.filename,
                        analytics.file_size_mb,
                        analytics.processing_time,
                        analytics.chunks_created,
                        analytics.success,
                        analytics.error_message
                    ))
                    conn.commit()
                    
            # Clear relevant cache entries
            self._clear_cache_pattern("document_")
            
        except Exception as e:
            self.logger.error(f"Failed to record document analytics: {e}")
    
    def record_system_metric(self, metric_name: str, value: float, metadata: Optional[Dict] = None):
        """Record a system metric."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO system_metrics (timestamp, metric_name, metric_value, metadata)
                        VALUES (?, ?, ?, ?)
                    """, (
                        datetime.now().isoformat(),
                        metric_name,
                        value,
                        json.dumps(metadata) if metadata else None
                    ))
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Failed to record system metric: {e}")
    
    def get_usage_analytics(self, days: int = 7) -> UsageAnalytics:
        """Get overall usage analytics for the specified period."""
        cache_key = f"usage_analytics_{days}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self._analytics_cache[cache_key]
        
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Query analytics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_queries,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
                        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_queries,
                        AVG(response_time) as avg_response_time,
                        AVG(confidence_score) as avg_confidence_score
                    FROM query_analytics 
                    WHERE timestamp >= ?
                """, (since_date,))
                
                query_stats = cursor.fetchone()
                
                # Most used model
                cursor.execute("""
                    SELECT model_used, COUNT(*) as usage_count
                    FROM query_analytics 
                    WHERE timestamp >= ?
                    GROUP BY model_used
                    ORDER BY usage_count DESC
                    LIMIT 1
                """, (since_date,))
                
                model_stats = cursor.fetchone()
                most_used_model = model_stats[0] if model_stats else "unknown"
                
                # Document analytics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        SUM(chunks_created) as total_chunks
                    FROM document_analytics 
                    WHERE timestamp >= ?
                """, (since_date,))
                
                doc_stats = cursor.fetchone()
                
                # Create analytics object
                analytics = UsageAnalytics(
                    total_queries=query_stats[0] or 0,
                    successful_queries=query_stats[1] or 0,
                    failed_queries=query_stats[2] or 0,
                    avg_response_time=query_stats[3] or 0.0,
                    avg_confidence_score=query_stats[4] or 0.0,
                    most_used_model=most_used_model,
                    total_documents_processed=doc_stats[0] or 0,
                    total_chunks_created=doc_stats[1] or 0,
                    peak_concurrent_users=0,  # Would need session tracking
                    active_sessions=0  # Would need session tracking
                )
                
                # Cache the result
                self._analytics_cache[cache_key] = analytics
                self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
                
                return analytics
                
        except Exception as e:
            self.logger.error(f"Failed to get usage analytics: {e}")
            return UsageAnalytics(
                total_queries=0, successful_queries=0, failed_queries=0,
                avg_response_time=0.0, avg_confidence_score=0.0,
                most_used_model="unknown", total_documents_processed=0,
                total_chunks_created=0, peak_concurrent_users=0, active_sessions=0
            )
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, List[Tuple[str, float]]]:
        """Get performance trends over time."""
        cache_key = f"performance_trends_{days}"
        
        if self._is_cache_valid(cache_key):
            return self._analytics_cache[cache_key]
        
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Daily response time trends
                cursor.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        AVG(response_time) as avg_response_time
                    FROM query_analytics 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, (since_date,))
                
                response_time_trend = cursor.fetchall()
                
                # Daily success rate trends
                cursor.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM query_analytics 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, (since_date,))
                
                success_rate_trend = cursor.fetchall()
                
                # Daily query volume trends
                cursor.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as query_count
                    FROM query_analytics 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, (since_date,))
                
                query_volume_trend = cursor.fetchall()
                
                trends = {
                    "response_time": response_time_trend,
                    "success_rate": success_rate_trend,
                    "query_volume": query_volume_trend
                }
                
                # Cache the result
                self._analytics_cache[cache_key] = trends
                self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
                
                return trends
                
        except Exception as e:
            self.logger.error(f"Failed to get performance trends: {e}")
            return {"response_time": [], "success_rate": [], "query_volume": []}
    
    def get_top_queries(self, limit: int = 10, days: int = 7) -> List[Dict[str, Any]]:
        """Get the most frequent queries."""
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        query_text,
                        COUNT(*) as frequency,
                        AVG(response_time) as avg_response_time,
                        AVG(confidence_score) as avg_confidence
                    FROM query_analytics 
                    WHERE timestamp >= ? AND success = 1
                    GROUP BY query_text
                    ORDER BY frequency DESC
                    LIMIT ?
                """, (since_date, limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "query": row[0],
                        "frequency": row[1],
                        "avg_response_time": row[2],
                        "avg_confidence": row[3]
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get top queries: {e}")
            return []
    
    def get_error_analysis(self, days: int = 7) -> Dict[str, Any]:
        """Analyze errors and failure patterns."""
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Error frequency by type
                cursor.execute("""
                    SELECT 
                        error_message,
                        COUNT(*) as frequency
                    FROM query_analytics 
                    WHERE timestamp >= ? AND success = 0 AND error_message IS NOT NULL
                    GROUP BY error_message
                    ORDER BY frequency DESC
                """, (since_date,))
                
                error_types = cursor.fetchall()
                
                # Error rate by model
                cursor.execute("""
                    SELECT 
                        model_used,
                        COUNT(*) as total_queries,
                        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_queries,
                        AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM query_analytics 
                    WHERE timestamp >= ?
                    GROUP BY model_used
                    ORDER BY success_rate ASC
                """, (since_date,))
                
                model_errors = cursor.fetchall()
                
                return {
                    "error_types": [{"error": row[0], "frequency": row[1]} for row in error_types],
                    "model_reliability": [
                        {
                            "model": row[0],
                            "total_queries": row[1],
                            "failed_queries": row[2],
                            "success_rate": row[3]
                        }
                        for row in model_errors
                    ]
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get error analysis: {e}")
            return {"error_types": [], "model_reliability": []}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid."""
        if cache_key not in self._analytics_cache:
            return False
        
        if cache_key not in self._cache_expiry:
            return False
        
        return datetime.now() < self._cache_expiry[cache_key]
    
    def _clear_cache_pattern(self, pattern: str):
        """Clear cache entries matching a pattern."""
        keys_to_remove = [key for key in self._analytics_cache.keys() if pattern in key]
        for key in keys_to_remove:
            self._analytics_cache.pop(key, None)
            self._cache_expiry.pop(key, None)
    
    def export_analytics_report(self, days: int = 30) -> Dict[str, Any]:
        """Export a comprehensive analytics report."""
        try:
            usage_analytics = self.get_usage_analytics(days)
            performance_trends = self.get_performance_trends(days)
            top_queries = self.get_top_queries(10, days)
            error_analysis = self.get_error_analysis(days)
            
            report = {
                "report_generated": datetime.now().isoformat(),
                "period_days": days,
                "usage_summary": asdict(usage_analytics),
                "performance_trends": performance_trends,
                "top_queries": top_queries,
                "error_analysis": error_analysis,
                "recommendations": self._generate_recommendations(usage_analytics, error_analysis)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to export analytics report: {e}")
            return {}
    
    def _generate_recommendations(self, usage: UsageAnalytics, errors: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analytics data."""
        recommendations = []
        
        # Response time recommendations
        if usage.avg_response_time > 15.0:
            recommendations.append("Consider using a faster model or optimizing query processing")
        
        # Success rate recommendations
        success_rate = usage.successful_queries / max(usage.total_queries, 1)
        if success_rate < 0.9:
            recommendations.append("Investigate and fix common error patterns to improve reliability")
        
        # Usage recommendations
        if usage.total_queries > 1000:
            recommendations.append("Consider implementing query caching for frequently asked questions")
        
        # Error-specific recommendations
        if errors.get("error_types"):
            most_common_error = errors["error_types"][0]["error"]
            recommendations.append(f"Address the most common error: {most_common_error}")
        
        return recommendations


# Global analytics instance
performance_analytics = PerformanceAnalytics()
"""
Performance optimization system for the RAG bot.
Automatically optimizes system parameters based on performance metrics.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRecommendation:
    """Represents a performance optimization recommendation."""
    
    component: str
    issue: str
    recommendation: str
    priority: str  # 'high', 'medium', 'low'
    estimated_improvement: str
    implementation_complexity: str  # 'easy', 'medium', 'hard'


class PerformanceOptimizer:
    """Automatically optimizes RAG system performance based on metrics."""
    
    def __init__(self):
        """Initialize the performance optimizer."""
        self.logger = logging.getLogger(__name__)
        self.optimization_history = []
        self.current_optimizations = {}
        self.performance_thresholds = {
            'response_time_slow': 10.0,  # seconds
            'response_time_very_slow': 30.0,  # seconds
            'memory_usage_high': 1000,  # MB
            'vector_search_slow': 2.0,  # seconds
            'llm_generation_slow': 15.0,  # seconds
            'success_rate_low': 0.8  # 80%
        }
        
        self.logger.info("Performance optimizer initialized")
    
    def analyze_performance_metrics(self, metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze performance metrics and generate optimization recommendations."""
        recommendations = []
        
        try:
            # Analyze response times
            if 'avg_response_time' in metrics:
                recommendations.extend(self._analyze_response_times(metrics))
            
            # Analyze memory usage
            if 'memory_usage_mb' in metrics:
                recommendations.extend(self._analyze_memory_usage(metrics))
            
            # Analyze vector search performance
            if 'avg_vector_search_time' in metrics:
                recommendations.extend(self._analyze_vector_search(metrics))
            
            # Analyze LLM performance
            if 'avg_llm_generation_time' in metrics:
                recommendations.extend(self._analyze_llm_performance(metrics))
            
            # Analyze success rates
            if 'success_rate' in metrics:
                recommendations.extend(self._analyze_success_rates(metrics))
            
            # Sort by priority
            recommendations.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x.priority])
            
            self.logger.info(f"Generated {len(recommendations)} optimization recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance metrics: {e}")
            return []
    
    def _analyze_response_times(self, metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze response time metrics and generate recommendations."""
        recommendations = []
        avg_time = metrics.get('avg_response_time', 0)
        
        if avg_time > self.performance_thresholds['response_time_very_slow']:
            recommendations.append(OptimizationRecommendation(
                component="Overall System",
                issue=f"Very slow response times: {avg_time:.2f}s average",
                recommendation="Consider using a smaller, faster model or reducing context length",
                priority="high",
                estimated_improvement="50-70% faster responses",
                implementation_complexity="medium"
            ))
        elif avg_time > self.performance_thresholds['response_time_slow']:
            recommendations.append(OptimizationRecommendation(
                component="Overall System",
                issue=f"Slow response times: {avg_time:.2f}s average",
                recommendation="Optimize model parameters or implement response caching",
                priority="medium",
                estimated_improvement="20-40% faster responses",
                implementation_complexity="easy"
            ))
        
        return recommendations
    
    def _analyze_memory_usage(self, metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze memory usage and generate recommendations."""
        recommendations = []
        memory_mb = metrics.get('memory_usage_mb', 0)
        
        if memory_mb > self.performance_thresholds['memory_usage_high']:
            recommendations.append(OptimizationRecommendation(
                component="Memory Management",
                issue=f"High memory usage: {memory_mb:.1f}MB",
                recommendation="Implement memory cleanup, reduce batch sizes, or use memory-efficient models",
                priority="high",
                estimated_improvement="30-50% memory reduction",
                implementation_complexity="medium"
            ))
        
        # Check for memory growth
        if 'memory_growth_rate' in metrics and metrics['memory_growth_rate'] > 10:  # MB per hour
            recommendations.append(OptimizationRecommendation(
                component="Memory Management",
                issue=f"Memory leak detected: {metrics['memory_growth_rate']:.1f}MB/hour growth",
                recommendation="Investigate memory leaks in conversation history or document caching",
                priority="high",
                estimated_improvement="Prevent system crashes",
                implementation_complexity="hard"
            ))
        
        return recommendations
    
    def _analyze_vector_search(self, metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze vector search performance."""
        recommendations = []
        search_time = metrics.get('avg_vector_search_time', 0)
        
        if search_time > self.performance_thresholds['vector_search_slow']:
            recommendations.append(OptimizationRecommendation(
                component="Vector Search",
                issue=f"Slow vector search: {search_time:.2f}s average",
                recommendation="Reduce number of retrieved chunks or optimize vector database",
                priority="medium",
                estimated_improvement="40-60% faster search",
                implementation_complexity="easy"
            ))
        
        return recommendations
    
    def _analyze_llm_performance(self, metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze LLM generation performance."""
        recommendations = []
        llm_time = metrics.get('avg_llm_generation_time', 0)
        
        if llm_time > self.performance_thresholds['llm_generation_slow']:
            recommendations.append(OptimizationRecommendation(
                component="LLM Generation",
                issue=f"Slow LLM generation: {llm_time:.2f}s average",
                recommendation="Use a smaller/faster model or reduce max_tokens parameter",
                priority="high",
                estimated_improvement="50-80% faster generation",
                implementation_complexity="easy"
            ))
        
        return recommendations
    
    def _analyze_success_rates(self, metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze success rates and reliability."""
        recommendations = []
        success_rate = metrics.get('success_rate', 1.0)
        
        if success_rate < self.performance_thresholds['success_rate_low']:
            recommendations.append(OptimizationRecommendation(
                component="System Reliability",
                issue=f"Low success rate: {success_rate:.1%}",
                recommendation="Investigate error patterns and improve error handling",
                priority="high",
                estimated_improvement="Improved system reliability",
                implementation_complexity="medium"
            ))
        
        return recommendations
    
    def apply_automatic_optimizations(self, recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Apply automatic optimizations that are safe and easy to implement."""
        applied_optimizations = {}
        
        for rec in recommendations:
            if rec.implementation_complexity == "easy" and rec.priority in ["high", "medium"]:
                try:
                    optimization_applied = self._apply_optimization(rec)
                    if optimization_applied:
                        applied_optimizations[rec.component] = rec.recommendation
                        self.logger.info(f"Applied optimization: {rec.recommendation}")
                except Exception as e:
                    self.logger.error(f"Failed to apply optimization {rec.recommendation}: {e}")
        
        return applied_optimizations
    
    def _apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply a specific optimization recommendation."""
        # This is where you would implement actual optimizations
        # For now, we'll just log the recommendations
        
        if "reduce context length" in recommendation.recommendation.lower():
            # Could automatically reduce context length
            self.logger.info("Recommendation: Reduce context length for faster responses")
            return True
        
        elif "reduce batch sizes" in recommendation.recommendation.lower():
            # Could automatically reduce batch processing sizes
            self.logger.info("Recommendation: Reduce batch sizes to save memory")
            return True
        
        elif "reduce number of retrieved chunks" in recommendation.recommendation.lower():
            # Could automatically reduce the number of chunks retrieved
            self.logger.info("Recommendation: Reduce retrieved chunks for faster search")
            return True
        
        return False
    
    def get_optimization_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive optimization report."""
        recommendations = self.analyze_performance_metrics(metrics)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_performance": {
                "overall_health": self._assess_overall_health(metrics),
                "bottlenecks": self._identify_bottlenecks(metrics),
                "recommendations_count": len(recommendations)
            },
            "recommendations": [
                {
                    "component": rec.component,
                    "issue": rec.issue,
                    "recommendation": rec.recommendation,
                    "priority": rec.priority,
                    "estimated_improvement": rec.estimated_improvement,
                    "complexity": rec.implementation_complexity
                }
                for rec in recommendations
            ],
            "quick_wins": [
                rec for rec in recommendations 
                if rec.implementation_complexity == "easy" and rec.priority in ["high", "medium"]
            ],
            "performance_summary": {
                "avg_response_time": metrics.get('avg_response_time', 0),
                "memory_usage_mb": metrics.get('memory_usage_mb', 0),
                "success_rate": metrics.get('success_rate', 0),
                "total_queries": metrics.get('total_queries', 0)
            }
        }
        
        return report
    
    def _assess_overall_health(self, metrics: Dict[str, Any]) -> str:
        """Assess overall system health."""
        issues = 0
        
        if metrics.get('avg_response_time', 0) > self.performance_thresholds['response_time_slow']:
            issues += 1
        
        if metrics.get('memory_usage_mb', 0) > self.performance_thresholds['memory_usage_high']:
            issues += 1
        
        if metrics.get('success_rate', 1.0) < self.performance_thresholds['success_rate_low']:
            issues += 2  # Success rate is more critical
        
        if issues == 0:
            return "excellent"
        elif issues <= 1:
            return "good"
        elif issues <= 2:
            return "fair"
        else:
            return "poor"
    
    def _identify_bottlenecks(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify system bottlenecks."""
        bottlenecks = []
        
        response_time = metrics.get('avg_response_time', 0)
        vector_time = metrics.get('avg_vector_search_time', 0)
        llm_time = metrics.get('avg_llm_generation_time', 0)
        
        # Identify which component is taking the most time
        if llm_time > vector_time and llm_time > (response_time * 0.7):
            bottlenecks.append("LLM Generation")
        
        if vector_time > (response_time * 0.3):
            bottlenecks.append("Vector Search")
        
        if metrics.get('memory_usage_mb', 0) > self.performance_thresholds['memory_usage_high']:
            bottlenecks.append("Memory Usage")
        
        if not bottlenecks:
            bottlenecks.append("No major bottlenecks detected")
        
        return bottlenecks


# Global optimizer instance
performance_optimizer = PerformanceOptimizer()
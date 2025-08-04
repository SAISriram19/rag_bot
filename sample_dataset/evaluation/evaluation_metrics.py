"""
Automated evaluation metrics for RAG bot response quality assessment.
"""

import json
import re
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EvaluationResult:
    """Results from evaluating a single query response."""
    query_id: str
    query_text: str
    response: str
    sources: List[str]
    
    # Metrics
    relevance_score: float
    completeness_score: float
    accuracy_score: float
    source_quality_score: float
    response_time: float
    
    # Detailed analysis
    expected_keywords_found: List[str]
    missing_keywords: List[str]
    unexpected_sources: List[str]
    missing_sources: List[str]
    
    # Overall scores
    overall_score: float
    grade: str  # A, B, C, D, F


@dataclass
class EvaluationSummary:
    """Summary of evaluation results across all test queries."""
    total_queries: int
    average_relevance: float
    average_completeness: float
    average_accuracy: float
    average_source_quality: float
    average_response_time: float
    average_overall_score: float
    
    grade_distribution: Dict[str, int]
    performance_by_type: Dict[str, Dict[str, float]]
    performance_by_difficulty: Dict[str, Dict[str, float]]


class RAGEvaluator:
    """Automated evaluation system for RAG bot responses."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the evaluator.
        
        Args:
            embedding_model: Name of the sentence transformer model for semantic similarity
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.test_queries = self._load_test_queries()
        
    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Load test queries from JSON file."""
        queries_path = Path(__file__).parent.parent / "test_queries" / "test_queries.json"
        with open(queries_path, 'r') as f:
            return json.load(f)
    
    def evaluate_response(self, query_id: str, response: str, sources: List[str], 
                         response_time: float) -> EvaluationResult:
        """
        Evaluate a single response against expected criteria.
        
        Args:
            query_id: ID of the test query
            response: Generated response text
            sources: List of source document names used
            response_time: Time taken to generate response (seconds)
            
        Returns:
            EvaluationResult with detailed metrics
        """
        # Find the test query
        test_query = next((q for q in self.test_queries if q["id"] == query_id), None)
        if not test_query:
            raise ValueError(f"Test query {query_id} not found")
        
        # Calculate individual metrics
        relevance_score = self._calculate_relevance_score(test_query, response)
        completeness_score = self._calculate_completeness_score(test_query, response)
        accuracy_score = self._calculate_accuracy_score(test_query, response)
        source_quality_score = self._calculate_source_quality_score(test_query, sources)
        
        # Analyze keywords and sources
        expected_keywords_found, missing_keywords = self._analyze_keywords(test_query, response)
        unexpected_sources, missing_sources = self._analyze_sources(test_query, sources)
        
        # Calculate overall score (weighted average)
        overall_score = (
            relevance_score * 0.3 +
            completeness_score * 0.25 +
            accuracy_score * 0.25 +
            source_quality_score * 0.2
        )
        
        # Assign letter grade
        grade = self._assign_grade(overall_score)
        
        return EvaluationResult(
            query_id=query_id,
            query_text=test_query["query"],
            response=response,
            sources=sources,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            source_quality_score=source_quality_score,
            response_time=response_time,
            expected_keywords_found=expected_keywords_found,
            missing_keywords=missing_keywords,
            unexpected_sources=unexpected_sources,
            missing_sources=missing_sources,
            overall_score=overall_score,
            grade=grade
        )
    
    def _calculate_relevance_score(self, test_query: Dict, response: str) -> float:
        """Calculate semantic relevance between query and response."""
        query_text = test_query["query"]
        
        # Get embeddings
        query_embedding = self.embedding_model.encode([query_text])
        response_embedding = self.embedding_model.encode([response])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
        
        # Convert to 0-100 scale
        return max(0, min(100, similarity * 100))
    
    def _calculate_completeness_score(self, test_query: Dict, response: str) -> float:
        """Calculate how completely the response addresses expected content."""
        expected_keywords = test_query.get("expected_answer_contains", [])
        if not expected_keywords:
            return 100.0  # No specific expectations
        
        response_lower = response.lower()
        found_keywords = []
        
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                found_keywords.append(keyword)
        
        # Calculate percentage of expected keywords found
        completeness = (len(found_keywords) / len(expected_keywords)) * 100
        return completeness
    
    def _calculate_accuracy_score(self, test_query: Dict, response: str) -> float:
        """Calculate accuracy based on factual correctness indicators."""
        # This is a simplified accuracy check
        # In a real system, you might use fact-checking models or manual verification
        
        response_lower = response.lower()
        
        # Check for uncertainty indicators (good for accuracy)
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "i cannot find", 
            "not available in the documents", "unclear from the context"
        ]
        
        has_uncertainty = any(phrase in response_lower for phrase in uncertainty_phrases)
        
        # Check for hallucination indicators (bad for accuracy)
        hallucination_indicators = [
            "according to my knowledge", "as far as i know", "generally speaking",
            "typically", "usually", "in most cases"
        ]
        
        has_hallucination = any(indicator in response_lower for indicator in hallucination_indicators)
        
        # Base score
        accuracy = 80.0
        
        # Adjust based on indicators
        if has_uncertainty and not has_hallucination:
            accuracy += 15.0  # Good - admits uncertainty
        elif has_hallucination:
            accuracy -= 20.0  # Bad - potential hallucination
        
        # Check if response contains expected factual content
        expected_content = test_query.get("expected_answer_contains", [])
        if expected_content:
            content_found = sum(1 for content in expected_content 
                              if content.lower() in response_lower)
            content_ratio = content_found / len(expected_content)
            accuracy = accuracy * 0.7 + (content_ratio * 100) * 0.3
        
        return max(0, min(100, accuracy))
    
    def _calculate_source_quality_score(self, test_query: Dict, sources: List[str]) -> float:
        """Calculate quality of source selection."""
        expected_sources = test_query.get("expected_sources", [])
        if not expected_sources:
            return 100.0  # No specific source expectations
        
        if not sources:
            return 0.0  # No sources provided
        
        # Calculate precision and recall for sources
        expected_set = set(expected_sources)
        actual_set = set(sources)
        
        # Precision: how many retrieved sources are relevant
        precision = len(expected_set.intersection(actual_set)) / len(actual_set) if actual_set else 0
        
        # Recall: how many relevant sources were retrieved
        recall = len(expected_set.intersection(actual_set)) / len(expected_set) if expected_set else 0
        
        # F1 score
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1 * 100
    
    def _analyze_keywords(self, test_query: Dict, response: str) -> Tuple[List[str], List[str]]:
        """Analyze which expected keywords were found or missing."""
        expected_keywords = test_query.get("expected_answer_contains", [])
        response_lower = response.lower()
        
        found = []
        missing = []
        
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                found.append(keyword)
            else:
                missing.append(keyword)
        
        return found, missing
    
    def _analyze_sources(self, test_query: Dict, sources: List[str]) -> Tuple[List[str], List[str]]:
        """Analyze source selection quality."""
        expected_sources = set(test_query.get("expected_sources", []))
        actual_sources = set(sources)
        
        unexpected = list(actual_sources - expected_sources)
        missing = list(expected_sources - actual_sources)
        
        return unexpected, missing
    
    def _assign_grade(self, score: float) -> str:
        """Assign letter grade based on overall score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def evaluate_batch(self, results: List[Dict[str, Any]]) -> EvaluationSummary:
        """
        Evaluate a batch of query results.
        
        Args:
            results: List of dicts with keys: query_id, response, sources, response_time
            
        Returns:
            EvaluationSummary with aggregate metrics
        """
        evaluations = []
        
        for result in results:
            evaluation = self.evaluate_response(
                query_id=result["query_id"],
                response=result["response"],
                sources=result["sources"],
                response_time=result["response_time"]
            )
            evaluations.append(evaluation)
        
        return self._create_summary(evaluations)
    
    def _create_summary(self, evaluations: List[EvaluationResult]) -> EvaluationSummary:
        """Create summary statistics from individual evaluations."""
        if not evaluations:
            return EvaluationSummary(0, 0, 0, 0, 0, 0, 0, {}, {}, {})
        
        # Calculate averages
        avg_relevance = np.mean([e.relevance_score for e in evaluations])
        avg_completeness = np.mean([e.completeness_score for e in evaluations])
        avg_accuracy = np.mean([e.accuracy_score for e in evaluations])
        avg_source_quality = np.mean([e.source_quality_score for e in evaluations])
        avg_response_time = np.mean([e.response_time for e in evaluations])
        avg_overall = np.mean([e.overall_score for e in evaluations])
        
        # Grade distribution
        grade_counts = {}
        for evaluation in evaluations:
            grade_counts[evaluation.grade] = grade_counts.get(evaluation.grade, 0) + 1
        
        # Performance by query type
        type_performance = {}
        difficulty_performance = {}
        
        for evaluation in evaluations:
            # Find the test query to get type and difficulty
            test_query = next((q for q in self.test_queries if q["id"] == evaluation.query_id), None)
            if test_query:
                query_type = test_query.get("type", "unknown")
                difficulty = test_query.get("difficulty", "unknown")
                
                # Group by type
                if query_type not in type_performance:
                    type_performance[query_type] = []
                type_performance[query_type].append(evaluation.overall_score)
                
                # Group by difficulty
                if difficulty not in difficulty_performance:
                    difficulty_performance[difficulty] = []
                difficulty_performance[difficulty].append(evaluation.overall_score)
        
        # Calculate averages for each group
        type_averages = {t: np.mean(scores) for t, scores in type_performance.items()}
        difficulty_averages = {d: np.mean(scores) for d, scores in difficulty_performance.items()}
        
        return EvaluationSummary(
            total_queries=len(evaluations),
            average_relevance=avg_relevance,
            average_completeness=avg_completeness,
            average_accuracy=avg_accuracy,
            average_source_quality=avg_source_quality,
            average_response_time=avg_response_time,
            average_overall_score=avg_overall,
            grade_distribution=grade_counts,
            performance_by_type=type_averages,
            performance_by_difficulty=difficulty_averages
        )
    
    def generate_report(self, summary: EvaluationSummary, output_path: str = None) -> str:
        """Generate a detailed evaluation report."""
        report = f"""
# RAG Bot Evaluation Report

## Overall Performance
- **Total Queries Evaluated**: {summary.total_queries}
- **Average Overall Score**: {summary.average_overall_score:.1f}/100
- **Average Response Time**: {summary.average_response_time:.2f}s

## Detailed Metrics
- **Relevance Score**: {summary.average_relevance:.1f}/100
- **Completeness Score**: {summary.average_completeness:.1f}/100
- **Accuracy Score**: {summary.average_accuracy:.1f}/100
- **Source Quality Score**: {summary.average_source_quality:.1f}/100

## Grade Distribution
"""
        
        for grade, count in sorted(summary.grade_distribution.items()):
            percentage = (count / summary.total_queries) * 100
            report += f"- **{grade}**: {count} queries ({percentage:.1f}%)\n"
        
        report += "\n## Performance by Query Type\n"
        for query_type, score in summary.performance_by_type.items():
            report += f"- **{query_type.title()}**: {score:.1f}/100\n"
        
        report += "\n## Performance by Difficulty\n"
        for difficulty, score in summary.performance_by_difficulty.items():
            report += f"- **{difficulty.title()}**: {score:.1f}/100\n"
        
        report += f"""
## Recommendations

### Strengths
- Query types with scores > 80: {[t for t, s in summary.performance_by_type.items() if s > 80]}
- Difficulty levels with scores > 80: {[d for d, s in summary.performance_by_difficulty.items() if s > 80]}

### Areas for Improvement
- Query types with scores < 70: {[t for t, s in summary.performance_by_type.items() if s < 70]}
- Difficulty levels with scores < 70: {[d for d, s in summary.performance_by_difficulty.items() if s < 70]}

### Action Items
1. Focus on improving {min(summary.performance_by_type.items(), key=lambda x: x[1])[0]} query handling
2. Optimize response time (current avg: {summary.average_response_time:.2f}s)
3. Improve source selection accuracy (current: {summary.average_source_quality:.1f}/100)
"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report


# Example usage and testing functions
def run_sample_evaluation():
    """Run a sample evaluation with mock data."""
    evaluator = RAGEvaluator()
    
    # Mock results for testing
    sample_results = [
        {
            "query_id": "factual_001",
            "response": "The base URL for the REST API is https://api.example.com/v1",
            "sources": ["rest_api_reference.md"],
            "response_time": 1.2
        },
        {
            "query_id": "technical_001", 
            "response": "To create a new user using the Python SDK, you can use client.users.create() method with email, name, and password parameters.",
            "sources": ["python_sdk_reference.md"],
            "response_time": 2.1
        }
    ]
    
    summary = evaluator.evaluate_batch(sample_results)
    report = evaluator.generate_report(summary)
    print(report)


if __name__ == "__main__":
    run_sample_evaluation()
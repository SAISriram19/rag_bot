#!/usr/bin/env python3
"""
Demo script showing how to use the sample technical documentation dataset.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

def list_available_documents() -> Dict[str, List[str]]:
    """List all available documents in the sample dataset."""
    dataset_path = Path(__file__).parent
    
    categories = {
        "API Documentation": [],
        "Tutorials": [],
        "Code Documentation": []
    }
    
    # API docs
    api_docs_path = dataset_path / "api_docs"
    if api_docs_path.exists():
        categories["API Documentation"] = [f.name for f in api_docs_path.glob("*.md")]
    
    # Tutorials
    tutorials_path = dataset_path / "tutorials"
    if tutorials_path.exists():
        categories["Tutorials"] = [f.name for f in tutorials_path.glob("*.md")]
    
    # Code docs
    code_docs_path = dataset_path / "code_docs"
    if code_docs_path.exists():
        categories["Code Documentation"] = [f.name for f in code_docs_path.glob("*.md")]
    
    return categories

def list_test_queries() -> List[Dict]:
    """Load and categorize test queries."""
    queries_path = Path(__file__).parent / "test_queries" / "test_queries.json"
    
    if not queries_path.exists():
        return []
    
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    
    return queries

def show_query_examples_by_type():
    """Display example queries organized by type."""
    queries = list_test_queries()
    
    if not queries:
        print("No test queries found.")
        return
    
    # Group queries by type
    by_type = {}
    for query in queries:
        query_type = query.get("type", "unknown")
        if query_type not in by_type:
            by_type[query_type] = []
        by_type[query_type].append(query)
    
    print("Test Queries by Type:")
    print("=" * 50)
    
    for query_type, type_queries in by_type.items():
        print(f"\n{query_type.upper()} ({len(type_queries)} queries)")
        print("-" * 30)
        
        for query in type_queries[:3]:  # Show first 3 examples
            print(f"• {query['query']}")
            if 'difficulty' in query:
                print(f"  Difficulty: {query['difficulty']}")
            if 'expected_sources' in query:
                print(f"  Expected sources: {', '.join(query['expected_sources'])}")
            print()

def show_evaluation_metrics_info():
    """Display information about available evaluation metrics."""
    print("Available Evaluation Metrics:")
    print("=" * 50)
    
    metrics = [
        ("Relevance Score", "Semantic similarity between query and response"),
        ("Completeness Score", "Percentage of expected content covered"),
        ("Accuracy Score", "Factual correctness and absence of hallucinations"),
        ("Source Quality Score", "Precision and recall of source selection"),
        ("Response Time", "Time taken to generate response"),
        ("Overall Score", "Weighted average of all metrics")
    ]
    
    for metric, description in metrics:
        print(f"• {metric}: {description}")
    
    print("\nGrading Scale:")
    print("• A: 90-100 (Excellent)")
    print("• B: 80-89 (Good)")
    print("• C: 70-79 (Satisfactory)")
    print("• D: 60-69 (Needs Improvement)")
    print("• F: 0-59 (Poor)")

def demonstrate_usage():
    """Demonstrate how to use the dataset for testing."""
    print("Sample Technical Documentation Dataset")
    print("=" * 60)
    
    # Show available documents
    print("\n1. AVAILABLE DOCUMENTS")
    print("-" * 30)
    documents = list_available_documents()
    
    for category, files in documents.items():
        if files:
            print(f"\n{category}:")
            for file in files:
                print(f"  • {file}")
    
    # Show test queries
    print("\n\n2. TEST QUERIES")
    print("-" * 30)
    show_query_examples_by_type()
    
    # Show evaluation info
    print("\n\n3. EVALUATION SYSTEM")
    print("-" * 30)
    show_evaluation_metrics_info()
    
    # Usage instructions
    print("\n\n4. USAGE INSTRUCTIONS")
    print("-" * 30)
    print("""
To use this dataset for testing your RAG bot:

1. Upload Documents:
   - Load all .md files from api_docs/, tutorials/, and code_docs/
   - These provide diverse technical content for testing

2. Run Test Queries:
   - Use queries from test_queries/test_queries.json
   - Each query has expected sources and answer content
   - Queries are categorized by type and difficulty

3. Evaluate Results:
   - Use evaluation/evaluation_metrics.py for automated scoring
   - Run evaluation/run_evaluation.py for complete assessment
   - Check evaluation/expected_behaviors.md for quality guidelines

4. Analyze Performance:
   - Review generated reports for strengths and weaknesses
   - Focus on query types with low scores
   - Monitor response times and accuracy trends

Example Commands:
  # Run quick evaluation (first 5 queries)
  python evaluation/run_evaluation.py --quick
  
  # Run full evaluation and save results
  python evaluation/run_evaluation.py
  
  # Run evaluation without saving files
  python evaluation/run_evaluation.py --no-save
""")

def show_sample_query_details():
    """Show detailed information about a sample query."""
    queries = list_test_queries()
    
    if not queries:
        print("No test queries available.")
        return
    
    # Find a good example query
    sample_query = next((q for q in queries if q.get("type") == "technical"), queries[0])
    
    print("\nSample Query Details:")
    print("=" * 40)
    print(f"ID: {sample_query['id']}")
    print(f"Type: {sample_query.get('type', 'unknown')}")
    print(f"Difficulty: {sample_query.get('difficulty', 'unknown')}")
    print(f"Query: {sample_query['query']}")
    
    if 'expected_sources' in sample_query:
        print(f"Expected Sources: {', '.join(sample_query['expected_sources'])}")
    
    if 'expected_answer_contains' in sample_query:
        print("Expected Answer Should Contain:")
        for item in sample_query['expected_answer_contains']:
            print(f"  • {item}")

def main():
    """Main function to run the demo."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "docs":
            documents = list_available_documents()
            print(json.dumps(documents, indent=2))
        elif command == "queries":
            queries = list_test_queries()
            print(json.dumps(queries, indent=2))
        elif command == "sample":
            show_sample_query_details()
        elif command == "help":
            print("Available commands:")
            print("  docs    - List available documents as JSON")
            print("  queries - List test queries as JSON")
            print("  sample  - Show sample query details")
            print("  help    - Show this help message")
            print("  (no args) - Show full demonstration")
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' to see available commands.")
    else:
        demonstrate_usage()

if __name__ == "__main__":
    main()
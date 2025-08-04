"""
Script to run automated evaluation of the RAG bot using the sample dataset.
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the path to import RAG bot components
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from evaluation_metrics import RAGEvaluator, EvaluationResult
from services.query_handler import QueryHandler
from services.document_processor import DocumentProcessor
from config import AppConfig


class RAGBotEvaluationRunner:
    """Runs comprehensive evaluation of the RAG bot using the sample dataset."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the evaluation runner.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = AppConfig()
        self.evaluator = RAGEvaluator()
        self.query_handler = None
        self.document_processor = None
        
    def setup_rag_bot(self):
        """Initialize the RAG bot components."""
        try:
            # Initialize document processor and query handler
            self.document_processor = DocumentProcessor(self.config)
            self.query_handler = QueryHandler(self.config)
            
            print("✓ RAG bot components initialized successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize RAG bot: {e}")
            return False
    
    def load_sample_documents(self) -> bool:
        """Load all sample documents into the RAG bot."""
        try:
            sample_docs_path = Path(__file__).parent.parent
            
            # Find all markdown files in the sample dataset
            doc_paths = []
            for category in ["api_docs", "tutorials", "code_docs"]:
                category_path = sample_docs_path / category
                if category_path.exists():
                    doc_paths.extend(list(category_path.glob("*.md")))
            
            if not doc_paths:
                print("✗ No sample documents found")
                return False
            
            print(f"Loading {len(doc_paths)} sample documents...")
            
            # Process each document
            for doc_path in doc_paths:
                try:
                    self.document_processor.process_document(str(doc_path))
                    print(f"  ✓ Loaded: {doc_path.name}")
                except Exception as e:
                    print(f"  ✗ Failed to load {doc_path.name}: {e}")
            
            print("✓ Sample documents loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load sample documents: {e}")
            return False
    
    def run_test_queries(self) -> List[Dict[str, Any]]:
        """Run all test queries and collect results."""
        test_queries = self.evaluator.test_queries
        results = []
        
        print(f"Running {len(test_queries)} test queries...")
        
        for i, test_query in enumerate(test_queries, 1):
            try:
                print(f"  [{i}/{len(test_queries)}] {test_query['id']}: {test_query['query'][:50]}...")
                
                # Measure response time
                start_time = time.time()
                
                # Get response from RAG bot
                response_data = self.query_handler.handle_query(
                    query=test_query["query"],
                    conversation_history=[]
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Extract response and sources
                response_text = response_data.answer if hasattr(response_data, 'answer') else str(response_data)
                sources = [chunk.metadata.get('source', 'unknown') for chunk in response_data.sources] if hasattr(response_data, 'sources') else []
                
                # Store result
                result = {
                    "query_id": test_query["id"],
                    "response": response_text,
                    "sources": sources,
                    "response_time": response_time
                }
                results.append(result)
                
                print(f"    ✓ Response time: {response_time:.2f}s")
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                # Add failed result
                results.append({
                    "query_id": test_query["id"],
                    "response": f"ERROR: {str(e)}",
                    "sources": [],
                    "response_time": 0.0
                })
        
        return results
    
    def run_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete evaluation process.
        
        Args:
            save_results: Whether to save detailed results to files
            
        Returns:
            Dictionary containing evaluation summary and detailed results
        """
        print("=" * 60)
        print("RAG Bot Evaluation - Sample Dataset")
        print("=" * 60)
        
        # Step 1: Setup RAG bot
        if not self.setup_rag_bot():
            return {"error": "Failed to setup RAG bot"}
        
        # Step 2: Load sample documents
        if not self.load_sample_documents():
            return {"error": "Failed to load sample documents"}
        
        # Step 3: Run test queries
        query_results = self.run_test_queries()
        
        # Step 4: Evaluate results
        print("\nEvaluating responses...")
        evaluation_summary = self.evaluator.evaluate_batch(query_results)
        
        # Step 5: Generate detailed results
        detailed_results = []
        for result in query_results:
            try:
                evaluation = self.evaluator.evaluate_response(
                    query_id=result["query_id"],
                    response=result["response"],
                    sources=result["sources"],
                    response_time=result["response_time"]
                )
                detailed_results.append(evaluation)
            except Exception as e:
                print(f"Failed to evaluate {result['query_id']}: {e}")
        
        # Step 6: Generate report
        report = self.evaluator.generate_report(evaluation_summary)
        
        # Step 7: Save results if requested
        if save_results:
            self._save_results(evaluation_summary, detailed_results, query_results, report)
        
        # Step 8: Display summary
        self._display_summary(evaluation_summary)
        
        return {
            "summary": evaluation_summary,
            "detailed_results": detailed_results,
            "query_results": query_results,
            "report": report
        }
    
    def _save_results(self, summary, detailed_results, query_results, report):
        """Save evaluation results to files."""
        try:
            results_dir = Path(__file__).parent / "results"
            results_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save summary
            summary_path = results_dir / f"evaluation_summary_{timestamp}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary.__dict__, f, indent=2, default=str)
            
            # Save detailed results
            detailed_path = results_dir / f"detailed_results_{timestamp}.json"
            with open(detailed_path, 'w') as f:
                json.dump([result.__dict__ for result in detailed_results], f, indent=2, default=str)
            
            # Save query results
            queries_path = results_dir / f"query_results_{timestamp}.json"
            with open(queries_path, 'w') as f:
                json.dump(query_results, f, indent=2)
            
            # Save report
            report_path = results_dir / f"evaluation_report_{timestamp}.md"
            with open(report_path, 'w') as f:
                f.write(report)
            
            print(f"\n✓ Results saved to {results_dir}")
            
        except Exception as e:
            print(f"✗ Failed to save results: {e}")
    
    def _display_summary(self, summary):
        """Display evaluation summary to console."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"Total Queries: {summary.total_queries}")
        print(f"Average Overall Score: {summary.average_overall_score:.1f}/100")
        print(f"Average Response Time: {summary.average_response_time:.2f}s")
        
        print("\nDetailed Metrics:")
        print(f"  Relevance: {summary.average_relevance:.1f}/100")
        print(f"  Completeness: {summary.average_completeness:.1f}/100")
        print(f"  Accuracy: {summary.average_accuracy:.1f}/100")
        print(f"  Source Quality: {summary.average_source_quality:.1f}/100")
        
        print("\nGrade Distribution:")
        for grade, count in sorted(summary.grade_distribution.items()):
            percentage = (count / summary.total_queries) * 100
            print(f"  {grade}: {count} queries ({percentage:.1f}%)")
        
        print("\nPerformance by Query Type:")
        for query_type, score in summary.performance_by_type.items():
            print(f"  {query_type.title()}: {score:.1f}/100")
        
        print("\nPerformance by Difficulty:")
        for difficulty, score in summary.performance_by_difficulty.items():
            print(f"  {difficulty.title()}: {score:.1f}/100")


def main():
    """Main entry point for evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG bot evaluation using sample dataset")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    parser.add_argument("--quick", action="store_true", help="Run only a subset of queries for quick testing")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = RAGBotEvaluationRunner(args.config)
    
    # Modify test queries for quick run
    if args.quick:
        # Only run first 5 queries for quick testing
        runner.evaluator.test_queries = runner.evaluator.test_queries[:5]
        print("Running in quick mode - evaluating first 5 queries only")
    
    # Run evaluation
    results = runner.run_evaluation(save_results=not args.no_save)
    
    if "error" in results:
        print(f"\nEvaluation failed: {results['error']}")
        sys.exit(1)
    else:
        print("\n✓ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
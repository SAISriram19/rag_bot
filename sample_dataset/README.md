# Sample Technical Documentation Dataset

This dataset contains diverse technical documents for testing the RAG bot's capabilities across different types of technical content. It includes 16+ test queries covering factual, technical, and follow-up questions, plus automated evaluation metrics.

## Dataset Structure

- `api_docs/` - API documentation examples (2 files)
- `tutorials/` - Step-by-step technical tutorials (2 files)
- `code_docs/` - Code documentation and reference materials (2 files)
- `test_queries/` - Predefined test queries with expected behaviors (16 queries)
- `evaluation/` - Automated evaluation metrics and scripts (3 files)

## Quick Start

```bash
# Demonstrate dataset usage
python demo_dataset_usage.py

# Run quick evaluation (first 5 queries)
python evaluation/run_evaluation.py --quick

# Run full evaluation
python evaluation/run_evaluation.py
```

## Document Categories

### API Documentation
- **rest_api_reference.md** - Complete REST API reference with endpoints, authentication, error handling
- **graphql_schema.md** - GraphQL schema definition with queries, mutations, and subscriptions

### Tutorials
- **getting_started.md** - Installation and basic usage guide with code examples
- **advanced_configuration.md** - Production-ready configuration patterns for databases, caching, security

### Code Documentation
- **python_sdk_reference.md** - Complete SDK reference with classes, methods, and error handling
- **architecture_overview.md** - System architecture with microservices, data models, and security

## Test Queries (16 total)

### Query Types
- **Factual** (3 queries) - Direct questions about documented facts
- **Technical** (3 queries) - Implementation and code-related questions  
- **Architecture** (2 queries) - System design and component questions
- **Tutorial** (2 queries) - Step-by-step process questions
- **Follow-up** (2 queries) - Context-dependent questions
- **Configuration** (1 query) - Setup and configuration questions
- **Code Example** (1 query) - Request for code snippets
- **Troubleshooting** (1 query) - Problem-solving questions
- **Complex** (1 query) - Multi-part technical questions

### Difficulty Levels
- **Easy** (6 queries) - Simple factual or basic procedural questions
- **Medium** (6 queries) - Technical implementation or multi-step questions
- **Hard** (4 queries) - Complex configuration or architectural questions

## Evaluation System

### Automated Metrics
- **Relevance Score** - Semantic similarity between query and response
- **Completeness Score** - Coverage of expected content keywords
- **Accuracy Score** - Factual correctness and hallucination detection
- **Source Quality Score** - Precision/recall of source document selection
- **Response Time** - Performance measurement
- **Overall Score** - Weighted combination of all metrics

### Grading Scale
- **A (90-100)** - Excellent responses with high accuracy and completeness
- **B (80-89)** - Good responses with minor gaps
- **C (70-79)** - Satisfactory responses meeting basic requirements
- **D (60-69)** - Poor responses needing significant improvement
- **F (0-59)** - Failed responses with major issues

### Expected Performance Targets
- **Simple factual queries**: >85% accuracy, <3s response time
- **Technical implementation**: >80% accuracy, <10s response time
- **Complex architectural**: >75% accuracy, <15s response time

## Usage Examples

### Load All Documents
```python
from src.services.document_processor import DocumentProcessor

processor = DocumentProcessor()

# Load all sample documents
documents = [
    "sample_dataset/api_docs/rest_api_reference.md",
    "sample_dataset/api_docs/graphql_schema.md", 
    "sample_dataset/tutorials/getting_started.md",
    "sample_dataset/tutorials/advanced_configuration.md",
    "sample_dataset/code_docs/python_sdk_reference.md",
    "sample_dataset/code_docs/architecture_overview.md"
]

for doc in documents:
    processor.process_document(doc)
```

### Run Test Queries
```python
import json
from src.services.query_handler import QueryHandler

# Load test queries
with open("sample_dataset/test_queries/test_queries.json") as f:
    test_queries = json.load(f)

query_handler = QueryHandler()

# Test a specific query
query = test_queries[0]  # First factual query
response = query_handler.handle_query(query["query"])
print(f"Query: {query['query']}")
print(f"Response: {response.answer}")
print(f"Sources: {[s.metadata['source'] for s in response.sources]}")
```

### Evaluate Performance
```python
from sample_dataset.evaluation.evaluation_metrics import RAGEvaluator

evaluator = RAGEvaluator()

# Evaluate a single response
result = evaluator.evaluate_response(
    query_id="factual_001",
    response="The base URL for the REST API is https://api.example.com/v1",
    sources=["rest_api_reference.md"],
    response_time=1.2
)

print(f"Overall Score: {result.overall_score:.1f}/100")
print(f"Grade: {result.grade}")
```

## Files Included

### Documents (6 files, ~15,000 words total)
1. `api_docs/rest_api_reference.md` - REST API documentation
2. `api_docs/graphql_schema.md` - GraphQL API schema
3. `tutorials/getting_started.md` - Installation and basic usage
4. `tutorials/advanced_configuration.md` - Advanced configuration patterns
5. `code_docs/python_sdk_reference.md` - Python SDK reference
6. `code_docs/architecture_overview.md` - System architecture overview

### Test Queries (1 file)
7. `test_queries/test_queries.json` - 16 test queries with metadata

### Evaluation System (4 files)
8. `evaluation/evaluation_metrics.py` - Automated evaluation metrics
9. `evaluation/run_evaluation.py` - Complete evaluation runner
10. `evaluation/expected_behaviors.md` - Quality guidelines and limitations
11. `demo_dataset_usage.py` - Usage demonstration script

## Quality Assurance

This dataset has been designed to test:
- ✅ Multiple document formats and structures
- ✅ Various query types and complexity levels  
- ✅ Source attribution accuracy
- ✅ Response completeness and accuracy
- ✅ Performance under different conditions
- ✅ Edge cases and error handling

## Contributing

To extend this dataset:
1. Add new documents to appropriate category folders
2. Create corresponding test queries in `test_queries.json`
3. Update expected behaviors in `evaluation/expected_behaviors.md`
4. Test with the evaluation system to ensure quality
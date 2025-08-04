"""Query handling orchestrator that coordinates retrieval and generation."""

import logging
import re
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..models.data_models import QueryResponse, ConversationExchange
from ..services.retriever import Retriever, RetrievalResult
from ..services.response_generator import ResponseGenerator
from ..services.memory_manager import MemoryManager
from ..config import config

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """Handles query preprocessing and optimization for technical terms."""
    
    # Technical term expansions and synonyms
    TECHNICAL_EXPANSIONS = {
        'api': ['API', 'application programming interface', 'endpoint', 'service'],
        'db': ['database', 'data store', 'storage'],
        'config': ['configuration', 'settings', 'parameters'],
        'auth': ['authentication', 'authorization', 'login', 'credentials'],
        'http': ['HTTP', 'HTTPS', 'web request', 'REST'],
        'json': ['JSON', 'JavaScript Object Notation', 'data format'],
        'xml': ['XML', 'markup language', 'structured data'],
        'sql': ['SQL', 'database query', 'structured query language'],
        'ui': ['user interface', 'interface', 'frontend'],
        'ux': ['user experience', 'usability'],
        'cli': ['command line interface', 'terminal', 'command prompt'],
        'sdk': ['software development kit', 'development tools'],
        'ide': ['integrated development environment', 'code editor'],
        'ci': ['continuous integration', 'build automation'],
        'cd': ['continuous deployment', 'deployment automation'],
        'devops': ['development operations', 'deployment', 'infrastructure'],
        'ml': ['machine learning', 'artificial intelligence', 'AI'],
        'ai': ['artificial intelligence', 'machine learning', 'ML'],
    }
    
    # Common technical patterns
    TECHNICAL_PATTERNS = [
        (r'\b(\w+)\.(\w+)\(\)', r'\1 \2 method'),  # method calls
        (r'\b(\w+)\.(\w+)', r'\1 \2 property'),   # property access
        (r'@(\w+)', r'\1 decorator'),             # decorators
        (r'#(\w+)', r'\1 hashtag'),               # hashtags/comments
        (r'\$(\w+)', r'\1 variable'),             # shell variables
    ]
    
    @classmethod
    def preprocess_query(cls, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Preprocess query for better technical term matching.
        
        Args:
            query: Original user query
            
        Returns:
            Tuple of (processed_query, preprocessing_metadata)
        """
        if not query or not query.strip():
            return query, {}
        
        original_query = query.strip()
        processed_query = original_query.lower()
        metadata = {
            'original_query': original_query,
            'expansions_applied': [],
            'patterns_matched': [],
            'technical_terms_found': []
        }
        
        # Expand technical abbreviations
        for abbrev, expansions in cls.TECHNICAL_EXPANSIONS.items():
            if abbrev in processed_query:
                # Add expanded terms to improve matching
                expanded_terms = ' '.join(expansions)
                processed_query += f" {expanded_terms}"
                metadata['expansions_applied'].append({
                    'abbreviation': abbrev,
                    'expansions': expansions
                })
        
        # Handle technical patterns
        for pattern, replacement in cls.TECHNICAL_PATTERNS:
            matches = re.findall(pattern, processed_query)
            if matches:
                processed_query = re.sub(pattern, replacement, processed_query)
                metadata['patterns_matched'].append({
                    'pattern': pattern,
                    'matches': matches,
                    'replacement': replacement
                })
        
        # Identify technical terms
        technical_terms = cls._identify_technical_terms(original_query)
        metadata['technical_terms_found'] = technical_terms
        
        # Clean up extra spaces
        processed_query = ' '.join(processed_query.split())
        
        logger.debug(f"Query preprocessing: '{original_query}' -> '{processed_query}'")
        return processed_query, metadata
    
    @classmethod
    def _identify_technical_terms(cls, query: str) -> List[str]:
        """Identify technical terms in the query."""
        technical_indicators = [
            # Programming languages
            'python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 'php',
            # Frameworks and libraries
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express',
            # Technologies
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'github',
            # File formats
            'json', 'xml', 'yaml', 'csv', 'pdf', 'html', 'css',
            # Protocols
            'http', 'https', 'tcp', 'udp', 'smtp', 'ftp', 'ssh',
            # Database terms
            'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
            # Development terms
            'api', 'rest', 'graphql', 'microservice', 'webhook', 'oauth'
        ]
        
        query_lower = query.lower()
        found_terms = []
        
        for term in technical_indicators:
            if term in query_lower:
                found_terms.append(term)
        
        return found_terms


class ResponseValidator:
    """Validates and post-processes generated responses."""
    
    @classmethod
    def validate_response(cls, response: QueryResponse, query: str) -> Tuple[bool, List[str]]:
        """
        Validate a generated response for quality and accuracy.
        
        Args:
            response: Generated QueryResponse object
            query: Original user query
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if response is empty or too short
        if not response.answer or len(response.answer.strip()) < 10:
            issues.append("Response is too short or empty")
        
        # Check for hallucination indicators
        hallucination_indicators = [
            "i don't have access to",
            "i cannot access",
            "as an ai",
            "i'm not able to browse",
            "i don't have real-time"
        ]
        
        response_lower = response.answer.lower()
        for indicator in hallucination_indicators:
            if indicator in response_lower:
                issues.append(f"Potential hallucination detected: '{indicator}'")
        
        # Check if response addresses the query
        if not cls._response_addresses_query(response.answer, query):
            issues.append("Response may not address the original query")
        
        # Check source citation quality
        if response.sources:
            citation_issues = cls._validate_citations(response.answer, response.sources)
            issues.extend(citation_issues)
        else:
            # If no sources but response is confident, flag as potential issue
            if response.confidence_score > 0.7:
                issues.append("High confidence response without source citations")
        
        # Check response length vs context
        if len(response.answer) > 5000:  # Very long response
            issues.append("Response may be excessively long")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @classmethod
    def _response_addresses_query(cls, response: str, query: str) -> bool:
        """Check if response addresses the original query."""
        # Extract key terms from query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why', 'who'}
        
        query_keywords = query_words - stop_words
        response_keywords = response_words - stop_words
        
        if not query_keywords:
            return True  # Can't determine, assume valid
        
        # Check overlap between query and response keywords
        overlap = query_keywords.intersection(response_keywords)
        overlap_ratio = len(overlap) / len(query_keywords)
        
        return overlap_ratio >= 0.3  # At least 30% keyword overlap
    
    @classmethod
    def _validate_citations(cls, response: str, sources: List) -> List[str]:
        """Validate citation quality in response."""
        issues = []
        
        # Check if response mentions sources when sources are available
        if sources and len(sources) > 0:
            citation_indicators = ['[1]', '[2]', '[3]', 'source:', 'sources:', 'according to', 'based on']
            has_citations = any(indicator in response.lower() for indicator in citation_indicators)
            
            if not has_citations:
                issues.append("Response lacks proper source citations despite available sources")
        
        return issues
    
    @classmethod
    def post_process_response(cls, response: QueryResponse) -> QueryResponse:
        """
        Post-process response to improve quality and formatting.
        
        Args:
            response: Original QueryResponse object
            
        Returns:
            Post-processed QueryResponse object
        """
        # Clean up response text
        cleaned_answer = cls._clean_response_text(response.answer)
        
        # Ensure proper citation formatting
        formatted_answer = cls._format_citations(cleaned_answer, response.sources)
        
        # Create new response with cleaned content
        processed_response = QueryResponse(
            answer=formatted_answer,
            sources=response.sources,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            model_used=response.model_used,
            timestamp=response.timestamp
        )
        
        return processed_response
    
    @classmethod
    def _clean_response_text(cls, text: str) -> str:
        """Clean up response text formatting."""
        if not text:
            return text
        
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Fix common formatting issues
        cleaned = re.sub(r'\s+([.!?])', r'\1', cleaned)  # Remove space before punctuation
        cleaned = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', cleaned)  # Ensure space after punctuation
        
        return cleaned.strip()
    
    @classmethod
    def _format_citations(cls, text: str, sources: List) -> str:
        """Ensure proper citation formatting."""
        if not sources:
            return text
        
        # If text doesn't end with sources section, add it
        if 'sources:' not in text.lower() and '[1]' not in text:
            citations = []
            for i, source in enumerate(sources, 1):
                source_name = getattr(source, 'source', 'Unknown Source')
                citations.append(f"[{i}] {source_name}")
            
            if citations:
                text += f"\n\nSources:\n" + "\n".join(citations)
        
        return text


class QueryHandler:
    """Main query processing orchestrator that coordinates retrieval and generation."""
    
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        response_generator: Optional[ResponseGenerator] = None,
        memory_manager: Optional[MemoryManager] = None,
        enable_preprocessing: bool = True,
        enable_validation: bool = True,
        max_retry_attempts: int = 2,
        performance_monitor: Optional['PerformanceMonitor'] = None
    ):
        """
        Initialize the QueryHandler.
        
        Args:
            retriever: Retriever instance for context retrieval
            response_generator: ResponseGenerator for answer generation
            memory_manager: MemoryManager for conversation history
            enable_preprocessing: Whether to enable query preprocessing
            enable_validation: Whether to enable response validation
            max_retry_attempts: Maximum number of retry attempts for failed queries
            performance_monitor: PerformanceMonitor instance for tracking performance
        """
        self.retriever = retriever or Retriever()
        self.response_generator = response_generator or ResponseGenerator()
        self.memory_manager = memory_manager or MemoryManager()
        self.enable_preprocessing = enable_preprocessing
        self.enable_validation = enable_validation
        self.max_retry_attempts = max_retry_attempts
        
        # Performance monitoring
        self.performance_monitor = performance_monitor
        if config.enable_performance_monitoring and not self.performance_monitor:
            try:
                from ..services.performance_monitor import PerformanceMonitor
                self.performance_monitor = PerformanceMonitor(
                    max_records=config.max_performance_records,
                    metrics_window_minutes=config.performance_metrics_window_minutes,
                    enable_memory_monitoring=config.memory_monitoring_enabled,
                    enable_detailed_logging=config.detailed_performance_logging
                )
            except ImportError:
                logger.warning("Performance monitoring not available")
                self.performance_monitor = None
        
        # Statistics tracking
        self.query_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_processing_time': 0.0,
            'preprocessing_enabled_count': 0,
            'validation_failed_count': 0
        }
        
        logger.info(f"QueryHandler initialized with preprocessing={enable_preprocessing}, "
                   f"validation={enable_validation}, max_retries={max_retry_attempts}")
    
    def handle_query(
        self,
        query: str,
        model: Optional[str] = None,
        max_context_chunks: Optional[int] = None,
        include_conversation_history: bool = True
    ) -> QueryResponse:
        """
        Handle a complete query processing workflow.
        
        Args:
            query: User's question
            model: Specific model to use for generation
            max_context_chunks: Maximum number of context chunks to retrieve
            include_conversation_history: Whether to include conversation history
            
        Returns:
            QueryResponse with generated answer and metadata
            
        Raises:
            ValueError: If query is invalid
            RuntimeError: If query processing fails after retries
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        query_id = str(uuid.uuid4())
        self.query_stats['total_queries'] += 1
        
        # Initialize performance tracking variables
        vector_search_time = 0.0
        llm_generation_time = 0.0
        
        try:
            logger.info(f"Processing query: '{query[:100]}...' (ID: {query_id})")
            
            # Step 1: Preprocess query
            processed_query, preprocessing_metadata = self._preprocess_query(query)
            
            # Step 2: Retrieve conversation history if needed
            conversation_history = self._get_conversation_context(include_conversation_history)
            
            # Step 3: Retrieve relevant context with timing
            context_start = time.time()
            context_chunks = self._retrieve_context(processed_query, max_context_chunks)
            vector_search_time = time.time() - context_start
            
            # Step 4: Generate response with retry logic and timing
            llm_start = time.time()
            response = self._generate_response_with_retry(
                original_query=query,
                processed_query=processed_query,
                context_chunks=context_chunks,
                conversation_history=conversation_history,
                model=model
            )
            llm_generation_time = time.time() - llm_start
            
            # Step 5: Validate and post-process response
            final_response = self._validate_and_post_process(response, query)
            
            # Step 6: Update conversation memory
            self._update_memory(query, final_response)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_statistics(processing_time, success=True)
            
            # Record performance metrics
            if self.performance_monitor:
                self.performance_monitor.record_query_performance(
                    query_id=query_id,
                    query_text=query,
                    response_time=processing_time,
                    vector_search_time=vector_search_time,
                    llm_generation_time=llm_generation_time,
                    success=True,
                    model_used=model or "unknown",
                    context_chunks_count=len(context_chunks),
                    confidence_score=final_response.confidence_score
                )
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s (Vector: {vector_search_time:.2f}s, LLM: {llm_generation_time:.2f}s)")
            return final_response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_statistics(processing_time, success=False)
            
            # Record failed performance metrics
            if self.performance_monitor:
                self.performance_monitor.record_query_performance(
                    query_id=query_id,
                    query_text=query,
                    response_time=processing_time,
                    vector_search_time=vector_search_time,
                    llm_generation_time=llm_generation_time,
                    success=False,
                    error_message=str(e),
                    model_used=model or "unknown",
                    context_chunks_count=0,
                    confidence_score=0.0
                )
            
            logger.error(f"Query processing failed: {e}")
            raise RuntimeError(f"Query processing failed: {e}")
    
    def _preprocess_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Preprocess query if preprocessing is enabled."""
        if self.enable_preprocessing:
            self.query_stats['preprocessing_enabled_count'] += 1
            return QueryPreprocessor.preprocess_query(query)
        else:
            return query, {}
    
    def _get_conversation_context(self, include_history: bool) -> Optional[List[Dict[str, str]]]:
        """Get conversation context if requested."""
        if not include_history:
            return None
        
        try:
            recent_exchanges = self.memory_manager.get_recent_exchanges(count=3)
            return [
                {
                    'query': exchange.query,
                    'response': exchange.response
                }
                for exchange in recent_exchanges
            ]
        except Exception as e:
            logger.warning(f"Failed to retrieve conversation history: {e}")
            return None
    
    def _retrieve_context(
        self,
        query: str,
        max_chunks: Optional[int]
    ) -> List[RetrievalResult]:
        """Retrieve relevant context chunks."""
        try:
            k = max_chunks or config.max_retrieval_chunks
            context_chunks = self.retriever.retrieve_context(
                query=query,
                k=k,
                apply_diversity_filter=True,
                include_metadata=True
            )
            
            logger.debug(f"Retrieved {len(context_chunks)} context chunks")
            return context_chunks
            
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            return []  # Continue without context
    
    def _generate_response_with_retry(
        self,
        original_query: str,
        processed_query: str,
        context_chunks: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]],
        model: Optional[str]
    ) -> QueryResponse:
        """Generate response with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retry_attempts + 1):
            try:
                # Use processed query for generation but keep original for response
                response = self.response_generator.generate_response(
                    query=original_query,  # Use original query for user-facing response
                    context_chunks=context_chunks,
                    conversation_history=conversation_history,
                    model=model
                )
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"Response generation attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retry_attempts:
                    # Try with simpler approach on retry
                    if context_chunks and attempt == 0:
                        # First retry: reduce context
                        context_chunks = context_chunks[:max(1, len(context_chunks) // 2)]
                        logger.debug("Retrying with reduced context")
                    elif attempt == 1:
                        # Second retry: try without context
                        context_chunks = []
                        logger.debug("Retrying without context")
                else:
                    break
        
        # If all retries failed, try simple response generation
        try:
            logger.warning("Falling back to simple response generation")
            simple_response = self.response_generator.generate_simple_response(
                query=original_query,
                model=model
            )
            
            # Create minimal QueryResponse
            return QueryResponse(
                answer=simple_response,
                sources=[],
                confidence_score=0.3,  # Low confidence for fallback
                processing_time=0.0,
                model_used=model or "unknown",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Even simple response generation failed: {e}")
            raise RuntimeError(f"All response generation attempts failed. Last error: {last_error}")
    
    def _validate_and_post_process(self, response: QueryResponse, query: str) -> QueryResponse:
        """Validate and post-process response if validation is enabled."""
        if not self.enable_validation:
            return response
        
        try:
            # Validate response
            is_valid, issues = ResponseValidator.validate_response(response, query)
            
            if not is_valid:
                self.query_stats['validation_failed_count'] += 1
                logger.warning(f"Response validation issues: {issues}")
                
                # Adjust confidence score based on validation issues
                penalty = min(0.3, len(issues) * 0.1)
                response.confidence_score = max(0.1, response.confidence_score - penalty)
            
            # Post-process response
            processed_response = ResponseValidator.post_process_response(response)
            
            return processed_response
            
        except Exception as e:
            logger.warning(f"Response validation/post-processing failed: {e}")
            return response  # Return original response if validation fails
    
    def _update_memory(self, query: str, response: QueryResponse) -> None:
        """Update conversation memory with the query and response."""
        try:
            source_names = [chunk.source for chunk in response.sources]
            
            self.memory_manager.add_exchange(
                query=query,
                response=response.answer,
                sources=source_names,
                model_used=response.model_used,
                processing_time=response.processing_time
            )
            
        except Exception as e:
            logger.warning(f"Failed to update conversation memory: {e}")
    
    def _update_statistics(self, processing_time: float, success: bool) -> None:
        """Update query processing statistics."""
        if success:
            self.query_stats['successful_queries'] += 1
        else:
            self.query_stats['failed_queries'] += 1
        
        # Update average processing time
        total_successful = self.query_stats['successful_queries']
        if total_successful > 0:
            current_avg = self.query_stats['average_processing_time']
            self.query_stats['average_processing_time'] = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )
    
    def get_query_stats(self) -> Dict[str, Any]:
        """
        Get query processing statistics.
        
        Returns:
            Dictionary containing query processing statistics
        """
        stats = self.query_stats.copy()
        
        # Add component statistics
        stats['retriever_stats'] = self.retriever.get_retrieval_stats()
        stats['generator_stats'] = self.response_generator.get_generation_stats()
        stats['memory_stats'] = self.memory_manager.get_memory_summary()
        
        # Calculate success rate
        total_queries = stats['total_queries']
        if total_queries > 0:
            stats['success_rate'] = stats['successful_queries'] / total_queries
            stats['failure_rate'] = stats['failed_queries'] / total_queries
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        return stats
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        try:
            self.memory_manager.clear_memory()
            logger.info("Conversation history cleared")
        except Exception as e:
            logger.error(f"Failed to clear conversation history: {e}")
            raise
    
    def update_settings(
        self,
        enable_preprocessing: Optional[bool] = None,
        enable_validation: Optional[bool] = None,
        max_retry_attempts: Optional[int] = None
    ) -> None:
        """
        Update QueryHandler settings.
        
        Args:
            enable_preprocessing: Whether to enable query preprocessing
            enable_validation: Whether to enable response validation
            max_retry_attempts: Maximum number of retry attempts
        """
        if enable_preprocessing is not None:
            self.enable_preprocessing = enable_preprocessing
            logger.info(f"Updated enable_preprocessing to {enable_preprocessing}")
        
        if enable_validation is not None:
            self.enable_validation = enable_validation
            logger.info(f"Updated enable_validation to {enable_validation}")
        
        if max_retry_attempts is not None:
            self.max_retry_attempts = max_retry_attempts
            logger.info(f"Updated max_retry_attempts to {max_retry_attempts}")
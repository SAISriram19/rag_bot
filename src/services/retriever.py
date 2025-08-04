"""Context retrieval system for semantic similarity search and ranking."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from ..models.data_models import DocumentChunk, QueryResponse
from ..services.vector_store_manager import VectorStoreManager
from ..services.embedding_generator import EmbeddingGenerator
from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieval result with relevance scoring."""
    
    chunk: DocumentChunk
    relevance_score: float
    rank: int
    source_citation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'chunk': self.chunk.to_dict(),
            'relevance_score': self.relevance_score,
            'rank': self.rank,
            'source_citation': self.source_citation
        }


class Retriever:
    """Handles semantic similarity search and context retrieval with ranking and filtering."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        max_chunks: Optional[int] = None,
        min_relevance_threshold: float = 0.0,
        diversity_threshold: float = 0.8
    ):
        """
        Initialize the Retriever.
        
        Args:
            vector_store: VectorStoreManager instance for similarity search
            embedding_generator: EmbeddingGenerator for query embeddings
            max_chunks: Maximum number of chunks to retrieve (uses config default if None)
            min_relevance_threshold: Minimum relevance score to include results
            diversity_threshold: Similarity threshold for diversity filtering
        """
        self.vector_store = vector_store or VectorStoreManager()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.max_chunks = max_chunks or config.max_retrieval_chunks
        self.min_relevance_threshold = min_relevance_threshold
        self.diversity_threshold = diversity_threshold
        
        logger.info(f"Initialized Retriever with max_chunks={self.max_chunks}, "
                   f"min_threshold={min_relevance_threshold}, diversity_threshold={diversity_threshold}")
    
    def retrieve_context(
        self,
        query: str,
        k: Optional[int] = None,
        apply_diversity_filter: bool = True,
        include_metadata: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant context for a query with ranking and filtering.
        
        Args:
            query: User query string
            k: Number of chunks to retrieve (uses instance default if None)
            apply_diversity_filter: Whether to apply diversity filtering
            include_metadata: Whether to include detailed metadata in results
            
        Returns:
            List of RetrievalResult objects ordered by relevance
            
        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If retrieval operation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        k = k or self.max_chunks
        
        try:
            logger.debug(f"Retrieving context for query: '{query[:100]}...' (k={k})")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query.strip())
            
            # Perform similarity search with extra results for filtering
            search_k = min(k * 2, 20)  # Get more results for better filtering
            similar_chunks = self.vector_store.similarity_search(query_embedding, k=search_k)
            
            if not similar_chunks:
                logger.info("No similar chunks found for query")
                return []
            
            # Calculate relevance scores and create retrieval results
            retrieval_results = self._calculate_relevance_scores(
                query_embedding, similar_chunks, query
            )
            
            # Apply relevance threshold filtering
            filtered_results = self._apply_relevance_filter(retrieval_results)
            
            # Apply diversity filtering if requested
            if apply_diversity_filter:
                filtered_results = self._apply_diversity_filter(filtered_results)
            
            # Limit to requested number of results
            final_results = filtered_results[:k]
            
            # Update ranks after filtering
            for i, result in enumerate(final_results):
                result.rank = i + 1
            
            logger.info(f"Retrieved {len(final_results)} relevant chunks for query")
            return final_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            raise RuntimeError(f"Context retrieval failed: {e}")
    
    def _calculate_relevance_scores(
        self,
        query_embedding: np.ndarray,
        chunks: List[DocumentChunk],
        query: str
    ) -> List[RetrievalResult]:
        """
        Calculate relevance scores for retrieved chunks.
        
        Args:
            query_embedding: Query embedding vector
            chunks: List of retrieved document chunks
            query: Original query string for citation generation
            
        Returns:
            List of RetrievalResult objects with calculated scores
        """
        results = []
        
        for i, chunk in enumerate(chunks):
            # Calculate cosine similarity as base relevance score
            if chunk.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            else:
                # Fallback: use position-based scoring if embedding is missing
                similarity = 1.0 - (i * 0.1)  # Decreasing score based on position
                logger.warning(f"Missing embedding for chunk {chunk.id}, using position-based score")
            
            # Apply additional scoring factors
            relevance_score = self._enhance_relevance_score(similarity, chunk, query)
            
            # Generate source citation
            source_citation = self._generate_source_citation(chunk, i + 1)
            
            result = RetrievalResult(
                chunk=chunk,
                relevance_score=relevance_score,
                rank=i + 1,
                source_citation=source_citation
            )
            results.append(result)
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    def _enhance_relevance_score(
        self,
        base_similarity: float,
        chunk: DocumentChunk,
        query: str
    ) -> float:
        """
        Enhance relevance score with additional factors.
        
        Args:
            base_similarity: Base cosine similarity score
            chunk: Document chunk being scored
            query: Original query string
            
        Returns:
            Enhanced relevance score
        """
        enhanced_score = base_similarity
        
        # Boost score for exact keyword matches
        query_lower = query.lower()
        content_lower = chunk.content.lower()
        
        # Count exact word matches
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        word_matches = len(query_words.intersection(content_words))
        
        if word_matches > 0:
            keyword_boost = min(0.1 * word_matches / len(query_words), 0.2)
            enhanced_score += keyword_boost
        
        # Boost score for technical content (code blocks, API references, etc.)
        if self._is_technical_content(chunk.content):
            technical_boost = 0.05
            enhanced_score += technical_boost
        
        # Boost score based on content length (prefer substantial content)
        content_length = len(chunk.content)
        if 200 <= content_length <= 2000:  # Sweet spot for technical content
            length_boost = 0.03
            enhanced_score += length_boost
        
        # Ensure score stays within reasonable bounds
        return min(enhanced_score, 1.0)
    
    def _is_technical_content(self, content: str) -> bool:
        """
        Check if content appears to be technical documentation.
        
        Args:
            content: Text content to analyze
            
        Returns:
            True if content appears technical
        """
        technical_indicators = [
            '```', 'def ', 'class ', 'function', 'import ', 'from ',
            'API', 'HTTP', 'JSON', 'XML', 'SQL', 'URL', 'URI',
            'parameter', 'argument', 'return', 'exception',
            'configuration', 'installation', 'deployment'
        ]
        
        content_lower = content.lower()
        technical_count = sum(1 for indicator in technical_indicators 
                            if indicator.lower() in content_lower)
        
        return technical_count >= 2
    
    def _apply_relevance_filter(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Filter results based on minimum relevance threshold.
        
        Args:
            results: List of retrieval results to filter
            
        Returns:
            Filtered list of results
        """
        if self.min_relevance_threshold <= 0:
            return results
        
        filtered = [r for r in results if r.relevance_score >= self.min_relevance_threshold]
        
        if len(filtered) < len(results):
            logger.debug(f"Filtered {len(results) - len(filtered)} results below threshold "
                        f"{self.min_relevance_threshold}")
        
        return filtered
    
    def _apply_diversity_filter(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Apply diversity filtering to avoid too similar results.
        
        Args:
            results: List of retrieval results to filter
            
        Returns:
            Filtered list with diverse results
        """
        if len(results) <= 1:
            return results
        
        diverse_results = [results[0]]  # Always include the top result
        
        for result in results[1:]:
            # Check if this result is sufficiently different from already selected ones
            is_diverse = True
            
            for selected in diverse_results:
                if (result.chunk.embedding is not None and 
                    selected.chunk.embedding is not None):
                    
                    similarity = self._cosine_similarity(
                        result.chunk.embedding, 
                        selected.chunk.embedding
                    )
                    
                    if similarity > self.diversity_threshold:
                        is_diverse = False
                        break
                else:
                    # Fallback: check content similarity for chunks without embeddings
                    content_similarity = self._simple_content_similarity(
                        result.chunk.content, selected.chunk.content
                    )
                    if content_similarity > self.diversity_threshold:
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_results.append(result)
        
        if len(diverse_results) < len(results):
            logger.debug(f"Diversity filtering reduced results from {len(results)} to {len(diverse_results)}")
        
        return diverse_results
    
    def _simple_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate simple content similarity based on word overlap.
        
        Args:
            content1: First content string
            content2: Second content string
            
        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            
            # Ensure result is within valid range
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _generate_source_citation(self, chunk: DocumentChunk, rank: int) -> str:
        """
        Generate a source citation for a document chunk.
        
        Args:
            chunk: Document chunk to cite
            rank: Rank of this chunk in results
            
        Returns:
            Formatted source citation string
        """
        source_name = chunk.source
        
        # Extract additional citation info from metadata
        page_info = ""
        if chunk.metadata.get('page'):
            page_info = f", page {chunk.metadata['page']}"
        elif chunk.metadata.get('section'):
            page_info = f", section {chunk.metadata['section']}"
        
        # Create citation
        citation = f"[{rank}] {source_name}{page_info}"
        
        return citation
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval system.
        
        Returns:
            Dictionary containing retrieval system statistics
        """
        vector_store_info = self.vector_store.get_collection_info()
        embedding_stats = self.embedding_generator.get_cache_stats()
        
        return {
            'max_chunks': self.max_chunks,
            'min_relevance_threshold': self.min_relevance_threshold,
            'diversity_threshold': self.diversity_threshold,
            'vector_store': vector_store_info,
            'embedding_generator': embedding_stats
        }
    
    def update_settings(
        self,
        max_chunks: Optional[int] = None,
        min_relevance_threshold: Optional[float] = None,
        diversity_threshold: Optional[float] = None
    ) -> None:
        """
        Update retrieval settings.
        
        Args:
            max_chunks: New maximum number of chunks to retrieve
            min_relevance_threshold: New minimum relevance threshold
            diversity_threshold: New diversity threshold
        """
        if max_chunks is not None:
            self.max_chunks = max_chunks
            logger.info(f"Updated max_chunks to {max_chunks}")
        
        if min_relevance_threshold is not None:
            self.min_relevance_threshold = min_relevance_threshold
            logger.info(f"Updated min_relevance_threshold to {min_relevance_threshold}")
        
        if diversity_threshold is not None:
            self.diversity_threshold = diversity_threshold
            logger.info(f"Updated diversity_threshold to {diversity_threshold}")
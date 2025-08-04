"""Response generation system that combines context with LLM generation."""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..models.data_models import DocumentChunk, QueryResponse
from ..services.llm_manager import LLMManager
from ..services.retriever import RetrievalResult
from ..config import config

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Template for generating prompts optimized for technical documentation."""
    
    SYSTEM_PROMPT = """You are a helpful AI assistant specialized in technical documentation. Your role is to answer questions based on the provided context from technical documents.

Guidelines:
- Answer questions accurately based only on the provided context
- If the context doesn't contain enough information, clearly state this limitation
- Maintain technical accuracy and use appropriate terminology
- Provide specific examples from the context when available
- Format code snippets and technical content properly
- Be concise but comprehensive in your explanations
- Always cite your sources using the provided source citations"""

    CONTEXT_TEMPLATE = """Based on the following context from technical documents, please answer the user's question.

Context:
{context}

Question: {question}

Please provide a detailed answer based on the context above. If the context doesn't contain sufficient information to answer the question, please state this clearly. Always include source citations for the information you use."""

    FOLLOW_UP_TEMPLATE = """Based on the following context from technical documents and our previous conversation, please answer the user's follow-up question.

Previous conversation:
{conversation_history}

Current context:
{context}

Follow-up question: {question}

Please provide a detailed answer that takes into account both the current context and our previous conversation. Always include source citations for new information you use."""

    CITATION_FORMAT = """

Sources:
{citations}"""

    @classmethod
    def format_context_prompt(
        cls,
        question: str,
        context_chunks: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format a complete prompt with context and question.
        
        Args:
            question: User's question
            context_chunks: Retrieved context chunks with citations
            conversation_history: Previous conversation exchanges
            
        Returns:
            Formatted prompt string
        """
        # Format context with citations
        context_text = cls._format_context_chunks(context_chunks)
        
        # Choose template based on whether this is a follow-up
        if conversation_history and len(conversation_history) > 0:
            history_text = cls._format_conversation_history(conversation_history)
            prompt = cls.FOLLOW_UP_TEMPLATE.format(
                conversation_history=history_text,
                context=context_text,
                question=question
            )
        else:
            prompt = cls.CONTEXT_TEMPLATE.format(
                context=context_text,
                question=question
            )
        
        return prompt
    
    @classmethod
    def _format_context_chunks(cls, context_chunks: List[RetrievalResult]) -> str:
        """Format context chunks with proper citations."""
        if not context_chunks:
            return "No relevant context found."
        
        formatted_chunks = []
        for result in context_chunks:
            chunk_text = f"{result.source_citation}: {result.chunk.content.strip()}"
            formatted_chunks.append(chunk_text)
        
        return "\n\n".join(formatted_chunks)
    
    @classmethod
    def _format_conversation_history(cls, history: List[Dict[str, str]]) -> str:
        """Format conversation history for context."""
        if not history:
            return "No previous conversation."
        
        formatted_history = []
        for exchange in history[-3:]:  # Only include last 3 exchanges
            if 'query' in exchange and 'response' in exchange:
                formatted_history.append(f"User: {exchange['query']}")
                formatted_history.append(f"Assistant: {exchange['response']}")
        
        return "\n".join(formatted_history)


class ResponseGenerator:
    """Generates responses by combining retrieved context with LLM generation."""
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        max_context_length: int = 4000,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize the ResponseGenerator.
        
        Args:
            llm_manager: LLMManager instance for text generation
            max_context_length: Maximum length of context to include in prompt
            temperature: Temperature for LLM generation (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        self.llm_manager = llm_manager or LLMManager()
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initialized ResponseGenerator with max_context_length={max_context_length}, "
                   f"temperature={temperature}, max_tokens={max_tokens}")
    
    def generate_response(
        self,
        query: str,
        context_chunks: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None
    ) -> QueryResponse:
        """
        Generate a response combining context with LLM generation.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks with relevance scores
            conversation_history: Previous conversation exchanges
            model: Specific model to use (optional)
            
        Returns:
            QueryResponse with generated answer and metadata
            
        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If response generation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        
        try:
            logger.debug(f"Generating response for query: '{query[:100]}...'")
            
            # Truncate context if too long
            truncated_context = self._truncate_context(context_chunks)
            
            # Generate prompt using template
            prompt = PromptTemplate.format_context_prompt(
                question=query.strip(),
                context_chunks=truncated_context,
                conversation_history=conversation_history
            )
            
            # Generate response using LLM
            raw_response = self.llm_manager.generate_response(
                prompt=prompt,
                model=model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Format response with proper citations
            formatted_response = self._format_response_with_citations(
                raw_response, truncated_context
            )
            
            # Calculate confidence score based on context quality
            confidence_score = self._calculate_confidence_score(
                truncated_context, raw_response
            )
            
            processing_time = time.time() - start_time
            model_used = model or self.llm_manager.get_current_model()
            
            # Create QueryResponse object
            response = QueryResponse(
                answer=formatted_response,
                sources=[result.chunk for result in truncated_context],
                confidence_score=confidence_score,
                processing_time=processing_time,
                model_used=model_used,
                timestamp=datetime.now()
            )
            
            logger.info(f"Generated response in {processing_time:.2f}s with confidence {confidence_score:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise RuntimeError(f"Response generation failed: {e}")
    
    def _truncate_context(self, context_chunks: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Truncate context to fit within maximum length while preserving quality.
        
        Args:
            context_chunks: Original context chunks
            
        Returns:
            Truncated list of context chunks
        """
        if not context_chunks:
            return []
        
        # Calculate total context length
        total_length = sum(len(result.chunk.content) for result in context_chunks)
        
        if total_length <= self.max_context_length:
            return context_chunks
        
        # Truncate by removing lowest-scoring chunks first
        sorted_chunks = sorted(context_chunks, key=lambda x: x.relevance_score, reverse=True)
        truncated_chunks = []
        current_length = 0
        
        for chunk in sorted_chunks:
            chunk_length = len(chunk.chunk.content)
            if current_length + chunk_length <= self.max_context_length:
                truncated_chunks.append(chunk)
                current_length += chunk_length
            else:
                # Try to include partial content if there's space
                remaining_space = self.max_context_length - current_length
                if remaining_space > 200:  # Only include if meaningful space remains
                    # Create truncated chunk
                    truncated_content = chunk.chunk.content[:remaining_space-3] + "..."
                    truncated_chunk = RetrievalResult(
                        chunk=DocumentChunk(
                            id=chunk.chunk.id,
                            content=truncated_content,
                            metadata=chunk.chunk.metadata,
                            source=chunk.chunk.source,
                            embedding=chunk.chunk.embedding
                        ),
                        relevance_score=chunk.relevance_score,
                        rank=chunk.rank,
                        source_citation=chunk.source_citation
                    )
                    truncated_chunks.append(truncated_chunk)
                break
        
        # Re-sort by original rank
        truncated_chunks.sort(key=lambda x: x.rank)
        
        if len(truncated_chunks) < len(context_chunks):
            logger.debug(f"Truncated context from {len(context_chunks)} to {len(truncated_chunks)} chunks")
        
        return truncated_chunks
    
    def _format_response_with_citations(
        self,
        raw_response: str,
        context_chunks: List[RetrievalResult]
    ) -> str:
        """
        Format response with proper source citations.
        
        Args:
            raw_response: Raw response from LLM
            context_chunks: Context chunks used for generation
            
        Returns:
            Formatted response with citations
        """
        if not context_chunks:
            return raw_response
        
        # Clean up the raw response
        formatted_response = raw_response.strip()
        
        # Generate citations list
        citations = []
        for result in context_chunks:
            citation = f"{result.source_citation} - {result.chunk.source}"
            if result.chunk.metadata.get('page'):
                citation += f" (page {result.chunk.metadata['page']})"
            elif result.chunk.metadata.get('section'):
                citation += f" (section {result.chunk.metadata['section']})"
            citations.append(citation)
        
        # Add citations to response if not already present
        if not self._has_citations(formatted_response):
            citations_text = PromptTemplate.CITATION_FORMAT.format(
                citations="\n".join(citations)
            )
            formatted_response += citations_text
        
        return formatted_response
    
    def _has_citations(self, response: str) -> bool:
        """Check if response already contains citations."""
        citation_indicators = ["[1]", "[2]", "[3]", "Sources:", "Source:"]
        return any(indicator in response for indicator in citation_indicators)
    
    def _calculate_confidence_score(
        self,
        context_chunks: List[RetrievalResult],
        response: str
    ) -> float:
        """
        Calculate confidence score based on context quality and response characteristics.
        
        Args:
            context_chunks: Context chunks used for generation
            response: Generated response text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not context_chunks:
            return 0.1  # Low confidence without context
        
        # Base confidence from average relevance scores
        avg_relevance = sum(result.relevance_score for result in context_chunks) / len(context_chunks)
        base_confidence = min(avg_relevance, 0.8)  # Cap at 0.8
        
        # Adjust based on number of sources
        source_bonus = min(len(context_chunks) * 0.05, 0.15)
        
        # Adjust based on response characteristics
        response_lower = response.lower()
        
        # Penalty for uncertainty indicators
        uncertainty_indicators = [
            "i don't know", "not sure", "unclear", "cannot determine",
            "insufficient information", "not enough context"
        ]
        uncertainty_penalty = 0.0
        for indicator in uncertainty_indicators:
            if indicator in response_lower:
                uncertainty_penalty += 0.1
        
        # Bonus for technical content in response
        technical_bonus = 0.0
        if any(indicator in response_lower for indicator in [
            "function", "method", "class", "api", "parameter", "configuration"
        ]):
            technical_bonus = 0.05
        
        # Calculate final confidence
        confidence = base_confidence + source_bonus + technical_bonus - uncertainty_penalty
        
        # Ensure confidence is within valid range
        return max(0.0, min(1.0, confidence))
    
    def generate_simple_response(
        self,
        query: str,
        model: Optional[str] = None
    ) -> str:
        """
        Generate a simple response without context (fallback method).
        
        Args:
            query: User's question
            model: Specific model to use (optional)
            
        Returns:
            Generated response text
        """
        try:
            prompt = f"""You are a helpful AI assistant. Please answer the following question to the best of your ability:

Question: {query}

Please provide a helpful and accurate response."""
            
            return self.llm_manager.generate_response(
                prompt=prompt,
                model=model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
        except Exception as e:
            logger.error(f"Failed to generate simple response: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def update_settings(
        self,
        max_context_length: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> None:
        """
        Update response generation settings.
        
        Args:
            max_context_length: New maximum context length
            temperature: New temperature setting
            max_tokens: New maximum tokens setting
        """
        if max_context_length is not None:
            self.max_context_length = max_context_length
            logger.info(f"Updated max_context_length to {max_context_length}")
        
        if temperature is not None:
            self.temperature = temperature
            logger.info(f"Updated temperature to {temperature}")
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
            logger.info(f"Updated max_tokens to {max_tokens}")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the response generation system.
        
        Returns:
            Dictionary containing generation system statistics
        """
        return {
            'max_context_length': self.max_context_length,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'current_model': self.llm_manager.get_current_model(),
            'available_models': self.llm_manager.get_available_models()
        }
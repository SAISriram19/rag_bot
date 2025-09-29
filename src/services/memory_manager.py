"""Memory management for conversation history using LangChain."""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from ..models.data_models import ConversationExchange
from ..config import config

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages conversation memory using LangChain's ConversationBufferMemory."""
    
    def __init__(self, max_token_limit: Optional[int] = None):
        """
        Initialize the MemoryManager.
        
        Args:
            max_token_limit: Maximum number of tokens to keep in memory buffer.
                           If None, uses config.memory_buffer_size.
        """
        self.max_token_limit = max_token_limit or config.memory_buffer_size
        self.max_exchanges = config.max_conversation_history
        
        # Initialize LangChain ConversationBufferMemory
        self.memory = ConversationBufferMemory(
            max_token_limit=self.max_token_limit,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Store detailed conversation history
        self.conversation_history: List[ConversationExchange] = []
        
        logger.info(f"MemoryManager initialized with max_token_limit={self.max_token_limit}, "
                   f"max_exchanges={self.max_exchanges}")
    
    def add_exchange(self, query: str, response: str, sources: List[str] = None, 
                    model_used: str = "unknown", processing_time: Optional[float] = None) -> None:
        """
        Add a conversation exchange to memory.
        
        Args:
            query: User's question
            response: AI's response
            sources: List of source document names used
            model_used: Name of the model that generated the response
            processing_time: Time taken to process the query in seconds
        """
        try:
            # Add to LangChain memory
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response)
            
            # Create detailed exchange record
            exchange = ConversationExchange(
                timestamp=datetime.now(),
                query=query,
                response=response,
                sources=sources or [],
                model_used=model_used,
                processing_time=processing_time
            )
            
            # Add to detailed history
            self.conversation_history.append(exchange)
            
            # Manage memory buffer to prevent performance issues
            self._manage_memory_buffer()
            
            logger.debug(f"Added exchange to memory: query_length={len(query)}, "
                        f"response_length={len(response)}, sources={len(sources or [])}")
            
        except Exception as e:
            logger.error(f"Error adding exchange to memory: {e}")
            raise
    
    def get_conversation_context(self, include_sources: bool = False) -> str:
        """
        Get conversation context as a formatted string.
        
        Args:
            include_sources: Whether to include source information in context
            
        Returns:
            Formatted conversation context string
        """
        try:
            # Get messages from LangChain memory
            messages = self.memory.chat_memory.messages
            
            if not messages:
                return ""
            
            context_parts = []
            
            # Format messages for context
            for message in messages:
                if isinstance(message, HumanMessage):
                    context_parts.append(f"Human: {message.content}")
                elif isinstance(message, AIMessage):
                    context_parts.append(f"Assistant: {message.content}")
            
            # Add source information if requested
            if include_sources and self.conversation_history:
                context_parts.append("\n--- Recent Sources ---")
                recent_sources = set()
                for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
                    recent_sources.update(exchange.sources)
                
                if recent_sources:
                    context_parts.append("Sources: " + ", ".join(recent_sources))
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return ""
    
    def get_recent_exchanges(self, count: int = 5) -> List[ConversationExchange]:
        """
        Get the most recent conversation exchanges.
        
        Args:
            count: Number of recent exchanges to return
            
        Returns:
            List of recent ConversationExchange objects
        """
        return self.conversation_history[-count:] if self.conversation_history else []
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary information about current memory state.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            messages = self.memory.chat_memory.messages
            total_tokens = sum(len(msg.content.split()) for msg in messages)
            
            return {
                "total_exchanges": len(self.conversation_history),
                "total_messages": len(messages),
                "estimated_tokens": total_tokens,
                "memory_buffer_limit": self.max_token_limit,
                "max_exchanges_limit": self.max_exchanges,
                "oldest_exchange": (
                    self.conversation_history[0].timestamp.isoformat() 
                    if self.conversation_history else None
                ),
                "newest_exchange": (
                    self.conversation_history[-1].timestamp.isoformat() 
                    if self.conversation_history else None
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            return {"error": str(e)}
    
    def clear_memory(self) -> None:
        """Clear all conversation memory."""
        try:
            self.memory.clear()
            self.conversation_history.clear()
            logger.info("Memory cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            raise
    
    def _manage_memory_buffer(self) -> None:
        """
        Manage memory buffer to prevent performance issues.
        Removes oldest exchanges if limits are exceeded.
        """
        try:
            # Check if we exceed the maximum number of exchanges
            if len(self.conversation_history) > self.max_exchanges:
                # Remove oldest exchanges from detailed history
                exchanges_to_remove = len(self.conversation_history) - self.max_exchanges
                self.conversation_history = self.conversation_history[exchanges_to_remove:]
                
                logger.debug(f"Removed {exchanges_to_remove} old exchanges from detailed history")
            
            # Let LangChain memory handle token-based pruning automatically
            # ConversationBufferMemory will prune based on max_token_limit
            
        except Exception as e:
            logger.error(f"Error managing memory buffer: {e}")
    
    def export_conversation(self) -> List[Dict[str, Any]]:
        """
        Export conversation history as a list of dictionaries.
        
        Returns:
            List of conversation exchanges as dictionaries
        """
        try:
            return [exchange.to_dict() for exchange in self.conversation_history]
            
        except Exception as e:
            logger.error(f"Error exporting conversation: {e}")
            return []
    
    def import_conversation(self, conversation_data: List[Dict[str, Any]]) -> None:
        """
        Import conversation history from a list of dictionaries.
        
        Args:
            conversation_data: List of conversation exchange dictionaries
        """
        try:
            # Clear existing memory
            self.clear_memory()
            
            # Import exchanges
            for exchange_data in conversation_data:
                exchange = ConversationExchange.from_dict(exchange_data)
                
                # Add to LangChain memory
                self.memory.chat_memory.add_user_message(exchange.query)
                self.memory.chat_memory.add_ai_message(exchange.response)
                
                # Add to detailed history
                self.conversation_history.append(exchange)
            
            # Manage buffer after import
            self._manage_memory_buffer()
            
            logger.info(f"Imported {len(conversation_data)} conversation exchanges")
            
        except Exception as e:
            logger.error(f"Error importing conversation: {e}")
            raise
    
    def get_conversation_summary(self, max_exchanges: int = 5) -> str:
        """
        Get a summary of the conversation for display in long chats.
        
        Args:
            max_exchanges: Maximum number of recent exchanges to include in summary
            
        Returns:
            Formatted conversation summary string
        """
        try:
            if not self.conversation_history:
                return "No conversation history available."
            
            total_exchanges = len(self.conversation_history)
            recent_exchanges = self.conversation_history[-max_exchanges:]
            
            summary_parts = []
            summary_parts.append(f"**Conversation Summary** ({total_exchanges} total exchanges)")
            summary_parts.append("")
            
            # Add memory usage info
            memory_info = self.get_memory_summary()
            summary_parts.append(f"**Memory Usage:** {memory_info.get('estimated_tokens', 0)} tokens")
            summary_parts.append(f"**Buffer Limit:** {memory_info.get('memory_buffer_limit', 0)} tokens")
            summary_parts.append("")
            
            # Add recent exchanges summary
            if total_exchanges > max_exchanges:
                summary_parts.append(f"**Recent {max_exchanges} exchanges:**")
            else:
                summary_parts.append("**All exchanges:**")
            
            for i, exchange in enumerate(recent_exchanges, 1):
                # Truncate long queries/responses for summary
                query_preview = exchange.query[:100] + "..." if len(exchange.query) > 100 else exchange.query
                response_preview = exchange.response[:150] + "..." if len(exchange.response) > 150 else exchange.response
                
                summary_parts.append(f"**{i}.** {exchange.timestamp.strftime('%H:%M:%S')}")
                summary_parts.append(f"   Q: {query_preview}")
                summary_parts.append(f"   A: {response_preview}")
                if exchange.sources:
                    summary_parts.append(f"   📚 Sources: {len(exchange.sources)}")
                summary_parts.append("")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def get_memory_usage_warning(self) -> Optional[str]:
        """
        Check if memory usage is approaching limits and return warning message.
        
        Returns:
            Warning message if memory usage is high, None otherwise
        """
        try:
            memory_info = self.get_memory_summary()
            estimated_tokens = memory_info.get('estimated_tokens', 0)
            buffer_limit = memory_info.get('memory_buffer_limit', self.max_token_limit)
            total_exchanges = memory_info.get('total_exchanges', 0)
            
            # Check token usage
            if buffer_limit > 0:
                token_usage_ratio = estimated_tokens / buffer_limit
                if token_usage_ratio > 0.9:
                    return f"⚠️ Memory usage is very high ({token_usage_ratio:.1%}). Consider clearing conversation history."
                elif token_usage_ratio > 0.7:
                    return f"⚠️ Memory usage is getting high ({token_usage_ratio:.1%}). Monitor performance."
            
            # Check exchange count
            if total_exchanges > self.max_exchanges * 0.8:
                return f"⚠️ Conversation is getting long ({total_exchanges} exchanges). Consider clearing for better performance."
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return f"Error checking memory usage: {str(e)}"
    
    def export_conversation_json(self) -> str:
        """
        Export conversation history as JSON string.
        
        Returns:
            JSON string representation of conversation history
        """
        try:
            import json
            conversation_data = self.export_conversation()
            return json.dumps(conversation_data, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error exporting conversation as JSON: {e}")
            raise
    
    def import_conversation_json(self, json_string: str) -> None:
        """
        Import conversation history from JSON string.
        
        Args:
            json_string: JSON string containing conversation data
        """
        try:
            import json
            conversation_data = json.loads(json_string)
            self.import_conversation(conversation_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            logger.error(f"Error importing conversation from JSON: {e}")
            raise
    
    def get_langchain_memory(self) -> ConversationBufferMemory:
        """
        Get the underlying LangChain memory object for integration with other components.
        
        Returns:
            ConversationBufferMemory instance
        """
        return self.memory
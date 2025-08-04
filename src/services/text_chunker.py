"""Text chunking service for processing documents into manageable chunks."""

import re
import logging
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.models.data_models import TextChunk
from src.config import config

logger = logging.getLogger(__name__)


class TextChunkingError(Exception):
    """Custom exception for text chunking errors."""
    pass


class TextChunker:
    """Handles text chunking with special handling for technical content."""
    
    def __init__(self, 
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None):
        """
        Initialize the TextChunker.
        
        Args:
            chunk_size: Size of each chunk in characters (defaults to config value)
            chunk_overlap: Overlap between chunks in characters (defaults to config value)
        """
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        
        # Custom separators for technical content
        # Order matters - more specific separators first
        self.separators = [
            # Code block separators
            "\n```\n",  # End of code blocks
            "```\n",    # Start of code blocks
            "\n```",    # Code block boundaries
            
            # Section separators
            "\n\n# ",   # Main headers
            "\n\n## ",  # Sub headers
            "\n\n### ", # Sub-sub headers
            "\n\n#### ", # Further sub headers
            
            # List and structure separators
            "\n\n- ",   # List items
            "\n\n* ",   # Alternative list items
            "\n\n1. ",  # Numbered lists
            
            # Paragraph separators
            "\n\n",     # Double newlines (paragraphs)
            "\n",       # Single newlines
            
            # Sentence separators
            ". ",       # Sentence endings
            "! ",       # Exclamation endings
            "? ",       # Question endings
            
            # Word separators (last resort)
            " ",        # Spaces
            "",         # Character level (absolute last resort)
        ]
        
        # Initialize the base text splitter
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=True,
            length_function=len,
        )
        
        # Patterns for identifying technical content
        self.code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        self.inline_code_pattern = re.compile(r'`[^`\n]+`')
        self.function_pattern = re.compile(r'\b\w+\s*\([^)]*\)', re.MULTILINE)  # Simplified function pattern
        self.api_endpoint_pattern = re.compile(r'\b(GET|POST|PUT|DELETE|PATCH)\s+/\S+', re.IGNORECASE)
        self.config_pattern = re.compile(r'^\s*[\w\-\.]+\s*[:=]\s*.+$', re.MULTILINE)
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Chunk text into manageable pieces with technical content preservation.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of TextChunk objects
            
        Raises:
            TextChunkingError: If chunking fails
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided for chunking")
            return []
        
        if metadata is None:
            metadata = {}
        
        try:
            # Preprocess text to identify and preserve technical content
            preprocessed_text = self._preprocess_technical_content(text)
            
            # Use the base splitter to create initial chunks
            raw_chunks = self.base_splitter.split_text(preprocessed_text)
            
            # Post-process chunks to ensure technical content integrity
            processed_chunks = self._post_process_chunks(raw_chunks, text)
            
            # Create TextChunk objects with metadata
            text_chunks = []
            current_position = 0
            
            for i, chunk_content in enumerate(processed_chunks):
                # Find the position of this chunk in the original text
                start_index = text.find(chunk_content.strip(), current_position)
                if start_index == -1:
                    # Fallback: use approximate position
                    start_index = current_position
                
                end_index = start_index + len(chunk_content)
                current_position = max(0, end_index - self.chunk_overlap)
                
                # Create chunk metadata
                chunk_metadata = {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(processed_chunks),
                    'chunk_size': len(chunk_content),
                    'has_code': self._contains_code(chunk_content),
                    'has_api_endpoints': self._contains_api_endpoints(chunk_content),
                    'has_config': self._contains_config(chunk_content),
                    'technical_score': self._calculate_technical_score(chunk_content)
                }
                
                text_chunk = TextChunk(
                    content=chunk_content.strip(),
                    metadata=chunk_metadata,
                    start_index=start_index,
                    end_index=end_index
                )
                
                text_chunks.append(text_chunk)
            
            logger.info(f"Successfully chunked text into {len(text_chunks)} chunks")
            return text_chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {str(e)}")
            raise TextChunkingError(f"Text chunking failed: {str(e)}")
    
    def _preprocess_technical_content(self, text: str) -> str:
        """
        Preprocess text to better handle technical content during chunking.
        
        Args:
            text: Original text
            
        Returns:
            Preprocessed text with markers for technical content
        """
        processed_text = text
        
        # Add markers around code blocks to prevent splitting
        def protect_code_block(match):
            code_content = match.group(0)
            # Add special markers that won't be split
            return f"\n\n__CODE_BLOCK_START__\n{code_content}\n__CODE_BLOCK_END__\n\n"
        
        processed_text = self.code_block_pattern.sub(protect_code_block, processed_text)
        
        # Protect function definitions and API endpoints
        lines = processed_text.split('\n')
        protected_lines = []
        
        for line in lines:
            # Protect lines with function definitions
            if self.function_pattern.search(line) or self.api_endpoint_pattern.search(line):
                protected_lines.append(f"__TECH_LINE__{line}")
            else:
                protected_lines.append(line)
        
        return '\n'.join(protected_lines)
    
    def _post_process_chunks(self, raw_chunks: List[str], original_text: str) -> List[str]:
        """
        Post-process chunks to ensure technical content integrity.
        
        Args:
            raw_chunks: Raw chunks from the base splitter
            original_text: Original text for reference
            
        Returns:
            Processed chunks with technical content preserved
        """
        processed_chunks = []
        
        for chunk in raw_chunks:
            # Remove protection markers
            cleaned_chunk = chunk.replace("__CODE_BLOCK_START__", "")
            cleaned_chunk = cleaned_chunk.replace("__CODE_BLOCK_END__", "")
            cleaned_chunk = cleaned_chunk.replace("__TECH_LINE__", "")
            
            # Ensure code blocks are complete
            cleaned_chunk = self._ensure_code_block_integrity(cleaned_chunk)
            
            # Clean up extra whitespace while preserving structure
            cleaned_chunk = self._clean_whitespace(cleaned_chunk)
            
            if cleaned_chunk.strip():
                processed_chunks.append(cleaned_chunk)
        
        return processed_chunks
    
    def _ensure_code_block_integrity(self, chunk: str) -> str:
        """
        Ensure code blocks in chunks are complete and properly formatted.
        
        Args:
            chunk: Text chunk to check
            
        Returns:
            Chunk with complete code blocks
        """
        # Count code block markers
        code_block_starts = chunk.count('```')
        
        # If odd number of markers, we have an incomplete code block
        if code_block_starts % 2 == 1:
            # Try to complete the code block
            if chunk.rstrip().endswith('```'):
                # Already ends with closing marker
                pass
            elif '```' in chunk:
                # Has opening marker but no closing
                chunk = chunk.rstrip() + '\n```'
            else:
                # Shouldn't happen, but handle gracefully
                pass
        
        return chunk
    
    def _clean_whitespace(self, text: str) -> str:
        """
        Clean up whitespace while preserving technical content structure.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Split into lines for processing
        lines = text.split('\n')
        cleaned_lines = []
        
        prev_empty = False
        for line in lines:
            # Preserve indentation for code-like content
            if line.strip():
                cleaned_lines.append(line.rstrip())
                prev_empty = False
            else:
                # Only keep one empty line between content
                if not prev_empty:
                    cleaned_lines.append('')
                prev_empty = True
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code blocks or inline code."""
        return bool(self.code_block_pattern.search(text) or 
                   self.inline_code_pattern.search(text) or
                   self.function_pattern.search(text))
    
    def _contains_api_endpoints(self, text: str) -> bool:
        """Check if text contains API endpoint definitions."""
        return bool(self.api_endpoint_pattern.search(text))
    
    def _contains_config(self, text: str) -> bool:
        """Check if text contains configuration settings."""
        return bool(self.config_pattern.search(text))
    
    def _calculate_technical_score(self, text: str) -> float:
        """
        Calculate a technical content score for the chunk.
        
        Args:
            text: Text to analyze
            
        Returns:
            Technical score between 0.0 and 1.0
        """
        score = 0.0
        text_length = len(text)
        
        if text_length == 0:
            return 0.0
        
        # Code blocks and inline code (high weight)
        code_matches = len(self.code_block_pattern.findall(text)) + len(self.inline_code_pattern.findall(text))
        score += min(code_matches * 0.3, 0.5)
        
        # Function definitions (medium weight)
        function_matches = len(self.function_pattern.findall(text))
        score += min(function_matches * 0.15, 0.3)
        
        # API endpoints (medium weight)
        api_matches = len(self.api_endpoint_pattern.findall(text))
        score += min(api_matches * 0.2, 0.4)
        
        # Configuration patterns (low weight)
        config_matches = len(self.config_pattern.findall(text))
        score += min(config_matches * 0.1, 0.2)
        
        # Technical keywords (low weight but cumulative)
        technical_keywords = [
            'function', 'class', 'method', 'variable', 'parameter', 'return',
            'import', 'export', 'module', 'package', 'library', 'framework',
            'API', 'endpoint', 'request', 'response', 'HTTP', 'JSON', 'XML',
            'database', 'query', 'schema', 'table', 'index', 'primary key',
            'algorithm', 'data structure', 'array', 'list', 'dictionary', 'hash',
            'exception', 'error', 'debug', 'log', 'trace', 'stack',
            'configuration', 'settings', 'environment', 'deployment'
        ]
        
        keyword_count = sum(1 for keyword in technical_keywords 
                          if keyword.lower() in text.lower())
        score += min(keyword_count * 0.02, 0.15)
        
        return min(score, 1.0)
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """
        Get statistics about the chunked text.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'average_chunk_size': 0,
                'chunks_with_code': 0,
                'chunks_with_api_endpoints': 0,
                'chunks_with_config': 0,
                'average_technical_score': 0.0
            }
        
        total_chars = sum(len(chunk.content) for chunk in chunks)
        chunks_with_code = sum(1 for chunk in chunks if chunk.metadata.get('has_code', False))
        chunks_with_api = sum(1 for chunk in chunks if chunk.metadata.get('has_api_endpoints', False))
        chunks_with_config = sum(1 for chunk in chunks if chunk.metadata.get('has_config', False))
        avg_technical_score = sum(chunk.metadata.get('technical_score', 0.0) for chunk in chunks) / len(chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'average_chunk_size': total_chars // len(chunks),
            'min_chunk_size': min(len(chunk.content) for chunk in chunks),
            'max_chunk_size': max(len(chunk.content) for chunk in chunks),
            'chunks_with_code': chunks_with_code,
            'chunks_with_api_endpoints': chunks_with_api,
            'chunks_with_config': chunks_with_config,
            'average_technical_score': round(avg_technical_score, 3),
            'chunk_overlap': self.chunk_overlap,
            'target_chunk_size': self.chunk_size
        }
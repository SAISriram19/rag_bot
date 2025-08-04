"""Text extraction service for various document formats."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import PyPDF2
import markdown
from io import StringIO

logger = logging.getLogger(__name__)


class TextExtractionError(Exception):
    """Custom exception for text extraction errors."""
    pass


class UnsupportedFileTypeError(TextExtractionError):
    """Exception raised when file type is not supported."""
    pass


class CorruptedFileError(TextExtractionError):
    """Exception raised when file is corrupted or cannot be read."""
    pass


class TextExtractor:
    """Handles text extraction from various document formats."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}
    
    def __init__(self):
        """Initialize the TextExtractor."""
        self.extraction_methods = {
            '.pdf': self._extract_from_pdf,
            '.txt': self._extract_from_txt,
            '.md': self._extract_from_md
        }
    
    def extract_text(self, file_path: str, file_type: Optional[str] = None) -> str:
        """
        Extract text from a file.
        
        Args:
            file_path: Path to the file to extract text from
            file_type: Optional file type override (e.g., '.pdf', '.txt', '.md')
        
        Returns:
            Extracted text content as string
            
        Raises:
            UnsupportedFileTypeError: If file type is not supported
            CorruptedFileError: If file is corrupted or cannot be read
            TextExtractionError: For other extraction errors
        """
        if not os.path.exists(file_path):
            raise TextExtractionError(f"File not found: {file_path}")
        
        # Determine file type
        if file_type is None:
            file_type = Path(file_path).suffix.lower()
        
        if file_type not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file_type}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        try:
            extraction_method = self.extraction_methods[file_type]
            text = extraction_method(file_path)
            
            if not text or not text.strip():
                logger.warning(f"No text extracted from file: {file_path}")
                return ""
            
            logger.info(f"Successfully extracted {len(text)} characters from {file_path}")
            return text.strip()
            
        except (UnsupportedFileTypeError, CorruptedFileError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error extracting text from {file_path}: {str(e)}")
            raise TextExtractionError(f"Failed to extract text from {file_path}: {str(e)}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text_content = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    raise CorruptedFileError(f"PDF file is encrypted: {file_path}")
                
                # Check if PDF has pages
                if len(pdf_reader.pages) == 0:
                    raise CorruptedFileError(f"PDF file has no pages: {file_path}")
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1} in {file_path}: {str(e)}")
                        continue
                
                if not text_content:
                    logger.warning(f"No text could be extracted from PDF: {file_path}")
                    return ""
                
                return "\n\n".join(text_content)
                
        except PyPDF2.errors.PdfReadError as e:
            raise CorruptedFileError(f"Corrupted PDF file: {file_path} - {str(e)}")
        except FileNotFoundError:
            raise TextExtractionError(f"PDF file not found: {file_path}")
        except PermissionError:
            raise TextExtractionError(f"Permission denied reading PDF file: {file_path}")
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from plain text file."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                        return content
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with error handling
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    content = file.read()
                    logger.warning(f"Used error replacement for file: {file_path}")
                    return content
            except Exception as e:
                raise CorruptedFileError(f"Could not decode text file with any encoding: {file_path} - {str(e)}")
                
        except FileNotFoundError:
            raise TextExtractionError(f"Text file not found: {file_path}")
        except PermissionError:
            raise TextExtractionError(f"Permission denied reading text file: {file_path}")
        except Exception as e:
            raise TextExtractionError(f"Unexpected error reading text file {file_path}: {str(e)}")
    
    def _extract_from_md(self, file_path: str) -> str:
        """Extract text from Markdown file."""
        try:
            # First read the raw markdown content
            raw_content = self._extract_from_txt(file_path)
            
            # Convert markdown to plain text
            # This preserves code blocks and technical content better than HTML conversion
            md = markdown.Markdown(extensions=['codehilite', 'fenced_code', 'tables'])
            
            # Convert to HTML first, then extract text
            html_content = md.convert(raw_content)
            
            # Simple HTML tag removal for plain text
            import re
            # Remove HTML tags but preserve content
            text_content = re.sub(r'<[^>]+>', '', html_content)
            
            # Clean up extra whitespace while preserving structure
            lines = text_content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                cleaned_line = line.strip()
                if cleaned_line or (cleaned_lines and cleaned_lines[-1]):  # Preserve single empty lines
                    cleaned_lines.append(cleaned_line)
            
            # Remove multiple consecutive empty lines
            final_lines = []
            prev_empty = False
            
            for line in cleaned_lines:
                if not line:
                    if not prev_empty:
                        final_lines.append(line)
                    prev_empty = True
                else:
                    final_lines.append(line)
                    prev_empty = False
            
            return '\n'.join(final_lines)
            
        except TextExtractionError:
            # Re-raise text extraction errors from _extract_from_txt
            raise
        except Exception as e:
            raise TextExtractionError(f"Failed to process Markdown file {file_path}: {str(e)}")
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata
        """
        try:
            file_stat = os.stat(file_path)
            path_obj = Path(file_path)
            
            return {
                'filename': path_obj.name,
                'file_type': path_obj.suffix.lower(),
                'file_size': file_stat.st_size,
                'modified_time': file_stat.st_mtime,
                'absolute_path': os.path.abspath(file_path)
            }
        except Exception as e:
            logger.error(f"Failed to get metadata for {file_path}: {str(e)}")
            return {
                'filename': Path(file_path).name,
                'file_type': Path(file_path).suffix.lower(),
                'error': str(e)
            }
    
    def validate_file(self, file_path: str, max_size_mb: Optional[int] = None) -> bool:
        """
        Validate if a file can be processed.
        
        Args:
            file_path: Path to the file to validate
            max_size_mb: Maximum file size in MB (optional)
            
        Returns:
            True if file is valid for processing
            
        Raises:
            TextExtractionError: If file is invalid
        """
        if not os.path.exists(file_path):
            raise TextExtractionError(f"File not found: {file_path}")
        
        if not os.path.isfile(file_path):
            raise TextExtractionError(f"Path is not a file: {file_path}")
        
        # Check file extension
        file_type = Path(file_path).suffix.lower()
        if file_type not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file_type}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        # Check file size if limit is specified
        if max_size_mb is not None:
            file_size = os.path.getsize(file_path)
            max_size_bytes = max_size_mb * 1024 * 1024
            
            if file_size > max_size_bytes:
                raise TextExtractionError(
                    f"File too large: {file_size / (1024*1024):.1f}MB. "
                    f"Maximum allowed: {max_size_mb}MB"
                )
        
        return True
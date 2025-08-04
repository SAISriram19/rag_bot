"""Document processing pipeline that orchestrates the full document ingestion workflow."""

import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

from .text_extractor import TextExtractor, TextExtractionError, UnsupportedFileTypeError, CorruptedFileError
from .text_chunker import TextChunker, TextChunkingError
from .embedding_generator import EmbeddingGenerator
from .vector_store_manager import VectorStoreManager
from .error_handler import (
    error_handler, retry_on_failure, log_function_call,
    DocumentProcessingError, ErrorCategory, ErrorSeverity
)
from .logging_config import get_logging_manager
from ..models.data_models import DocumentChunk, TextChunk
from ..config import config

logger = logging.getLogger(__name__)
logging_manager = get_logging_manager()


class ProcessingStatus(Enum):
    """Status of document processing operations."""
    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingResult:
    """Result of processing a single document."""
    file_path: str
    status: ProcessingStatus
    chunks_created: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchProcessingResult:
    """Result of processing multiple documents."""
    total_files: int
    successful_files: int
    failed_files: int
    total_chunks: int
    total_processing_time: float
    results: List[ProcessingResult]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100


class DocumentProcessor:
    """Orchestrates the full document processing pipeline."""
    
    def __init__(self,
                 text_extractor: Optional[TextExtractor] = None,
                 text_chunker: Optional[TextChunker] = None,
                 embedding_generator: Optional[EmbeddingGenerator] = None,
                 vector_store_manager: Optional[VectorStoreManager] = None):
        """
        Initialize the DocumentProcessor.
        
        Args:
            text_extractor: Text extraction service (creates default if None)
            text_chunker: Text chunking service (creates default if None)
            embedding_generator: Embedding generation service (creates default if None)
            vector_store_manager: Vector store management service (creates default if None)
        """
        self.text_extractor = text_extractor or TextExtractor()
        self.text_chunker = text_chunker or TextChunker()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        
        # Progress tracking
        self._progress_callback: Optional[Callable[[str, float, str], None]] = None
        self._status_callback: Optional[Callable[[str, ProcessingStatus, str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[str, float, str], None]) -> None:
        """
        Set callback function for progress updates.
        
        Args:
            callback: Function that receives (file_path, progress_percent, status_message)
        """
        self._progress_callback = callback
    
    def set_status_callback(self, callback: Callable[[str, ProcessingStatus, str], None]) -> None:
        """
        Set callback function for status updates.
        
        Args:
            callback: Function that receives (file_path, status, message)
        """
        self._status_callback = callback
    
    def _report_progress(self, file_path: str, progress: float, message: str) -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(file_path, progress, message)
        logger.debug(f"Progress for {file_path}: {progress:.1f}% - {message}")
    
    def _report_status(self, file_path: str, status: ProcessingStatus, message: str) -> None:
        """Report status to callback if set."""
        if self._status_callback:
            self._status_callback(file_path, status, message)
        logger.info(f"Status for {file_path}: {status.value} - {message}")
    
    @log_function_call(include_args=True)
    @retry_on_failure(max_retries=2, delay=1.0, exceptions=(ConnectionError, TimeoutError), error_handler=error_handler)
    def process_document(self, file_path: str, 
                        source_name: Optional[str] = None,
                        additional_metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process a single document through the complete pipeline.
        
        Args:
            file_path: Path to the document to process
            source_name: Optional custom source name (defaults to filename)
            additional_metadata: Optional additional metadata to include with chunks
            
        Returns:
            ProcessingResult with processing details and status
        """
        start_time = time.time()
        file_path = str(file_path)  # Ensure string type
        source_name = source_name or Path(file_path).name
        
        result = ProcessingResult(
            file_path=file_path,
            status=ProcessingStatus.PENDING,
            metadata=additional_metadata or {}
        )
        
        context = {
            'file_path': file_path,
            'source_name': source_name,
            'file_size': Path(file_path).stat().st_size if Path(file_path).exists() else 0
        }
        
        try:
            logger.info(f"Starting document processing for {file_path}")
            self._report_status(file_path, ProcessingStatus.PENDING, "Starting document processing")
            self._report_progress(file_path, 0.0, "Initializing")
            
            # Step 1: Extract text
            self._report_status(file_path, ProcessingStatus.EXTRACTING, "Extracting text from document")
            self._report_progress(file_path, 10.0, "Extracting text")
            
            try:
                extracted_text = self.text_extractor.extract_text(file_path)
                if not extracted_text.strip():
                    raise DocumentProcessingError(
                        "No text content extracted from document",
                        file_path=file_path,
                        severity=ErrorSeverity.MEDIUM,
                        context=context
                    )
            except (TextExtractionError, UnsupportedFileTypeError, CorruptedFileError) as e:
                raise DocumentProcessingError(
                    f"Text extraction failed: {str(e)}",
                    file_path=file_path,
                    severity=ErrorSeverity.HIGH,
                    context=context
                ) from e
            
            # Get file metadata
            file_metadata = self.text_extractor.get_file_metadata(file_path)
            
            self._report_progress(file_path, 25.0, f"Extracted {len(extracted_text)} characters")
            
            # Step 2: Chunk text
            self._report_status(file_path, ProcessingStatus.CHUNKING, "Chunking text into segments")
            self._report_progress(file_path, 40.0, "Creating text chunks")
            
            # Prepare metadata for chunks
            chunk_metadata = {
                'source': source_name,
                'file_path': file_path,
                'file_type': file_metadata.get('file_type', ''),
                'file_size': file_metadata.get('file_size', 0),
                'processed_at': time.time(),
                **file_metadata,
                **(additional_metadata or {})
            }
            
            text_chunks = self.text_chunker.chunk_text(extracted_text, chunk_metadata)
            if not text_chunks:
                raise TextChunkingError("No chunks created from document")
            
            self._report_progress(file_path, 55.0, f"Created {len(text_chunks)} chunks")
            
            # Step 3: Generate embeddings
            self._report_status(file_path, ProcessingStatus.EMBEDDING, "Generating embeddings")
            self._report_progress(file_path, 70.0, "Generating embeddings")
            
            document_chunks = self.embedding_generator.generate_embeddings_for_chunks(text_chunks)
            
            self._report_progress(file_path, 85.0, f"Generated embeddings for {len(document_chunks)} chunks")
            
            # Step 4: Store in vector database
            self._report_status(file_path, ProcessingStatus.STORING, "Storing in vector database")
            self._report_progress(file_path, 95.0, "Storing in database")
            
            self.vector_store_manager.add_documents(document_chunks)
            
            # Complete
            processing_time = time.time() - start_time
            result.status = ProcessingStatus.COMPLETED
            result.chunks_created = len(document_chunks)
            result.processing_time = processing_time
            
            self._report_status(file_path, ProcessingStatus.COMPLETED, 
                              f"Successfully processed {len(document_chunks)} chunks in {processing_time:.2f}s")
            self._report_progress(file_path, 100.0, "Processing completed")
            
            # Record performance metrics if monitoring is enabled
            if config.enable_performance_monitoring:
                try:
                    from ..services.performance_monitor import PerformanceMonitor
                    # This would ideally be injected, but for now we'll create a temporary instance
                    temp_monitor = PerformanceMonitor()
                    temp_monitor.record_document_processing_performance(
                        file_path=file_path,
                        processing_time=processing_time,
                        chunks_created=len(document_chunks),
                        success=True
                    )
                except ImportError:
                    pass  # Performance monitoring not available
            
            logger.info(f"Successfully processed document {file_path}: {len(document_chunks)} chunks in {processing_time:.2f}s")
            
        except DocumentProcessingError as e:
            processing_time = time.time() - start_time
            result.status = ProcessingStatus.FAILED
            result.error_message = e.user_message
            result.processing_time = processing_time
            
            # Handle error through centralized system
            error_info = error_handler.handle_error(e, context=context)
            
            self._report_status(file_path, ProcessingStatus.FAILED, result.error_message)
            logging_manager.log_error_with_context(e, context)
            
        except Exception as e:
            processing_time = time.time() - start_time
            result.status = ProcessingStatus.FAILED
            
            # Create structured error
            doc_error = DocumentProcessingError(
                f"Unexpected processing error: {str(e)}",
                file_path=file_path,
                severity=ErrorSeverity.HIGH,
                context=context
            )
            
            result.error_message = doc_error.user_message
            result.processing_time = processing_time
            
            # Handle error through centralized system
            error_info = error_handler.handle_error(doc_error, context=context)
            
            self._report_status(file_path, ProcessingStatus.FAILED, result.error_message)
            logging_manager.log_error_with_context(doc_error, context)
            
            # Record failed performance metrics if monitoring is enabled
            if config.enable_performance_monitoring:
                try:
                    from ..services.performance_monitor import PerformanceMonitor
                    temp_monitor = PerformanceMonitor()
                    temp_monitor.record_document_processing_performance(
                        file_path=file_path,
                        processing_time=processing_time,
                        chunks_created=0,
                        success=False,
                        error_message=str(e)
                    )
                except ImportError:
                    pass  # Performance monitoring not available
            
            self._report_status(file_path, ProcessingStatus.FAILED, result.error_message)
            logger.error(f"Document processing failed for {file_path}: {str(e)}")
        
        return result
    
    def process_multiple_documents(self, 
                                 file_paths: List[Union[str, Path]],
                                 source_names: Optional[List[str]] = None,
                                 additional_metadata: Optional[Dict[str, Any]] = None,
                                 continue_on_error: bool = True) -> BatchProcessingResult:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            source_names: Optional list of custom source names (must match file_paths length)
            additional_metadata: Optional additional metadata to include with all chunks
            continue_on_error: Whether to continue processing other files if one fails
            
        Returns:
            BatchProcessingResult with overall processing statistics
        """
        if not file_paths:
            return BatchProcessingResult(
                total_files=0,
                successful_files=0,
                failed_files=0,
                total_chunks=0,
                total_processing_time=0.0,
                results=[]
            )
        
        if source_names and len(source_names) != len(file_paths):
            raise ValueError("source_names length must match file_paths length")
        
        start_time = time.time()
        results = []
        successful_files = 0
        failed_files = 0
        total_chunks = 0
        
        logger.info(f"Starting batch processing of {len(file_paths)} documents")
        
        for i, file_path in enumerate(file_paths):
            try:
                # Determine source name
                source_name = source_names[i] if source_names else None
                
                # Process individual document
                result = self.process_document(
                    file_path=file_path,
                    source_name=source_name,
                    additional_metadata=additional_metadata
                )
                
                results.append(result)
                
                if result.status == ProcessingStatus.COMPLETED:
                    successful_files += 1
                    total_chunks += result.chunks_created
                else:
                    failed_files += 1
                    if not continue_on_error:
                        logger.warning(f"Stopping batch processing due to failure: {result.error_message}")
                        break
                
            except Exception as e:
                # Create failed result for unexpected errors
                failed_result = ProcessingResult(
                    file_path=str(file_path),
                    status=ProcessingStatus.FAILED,
                    error_message=f"Unexpected error: {str(e)}",
                    processing_time=0.0
                )
                results.append(failed_result)
                failed_files += 1
                
                logger.error(f"Unexpected error processing {file_path}: {str(e)}")
                
                if not continue_on_error:
                    logger.warning("Stopping batch processing due to unexpected error")
                    break
        
        total_processing_time = time.time() - start_time
        
        batch_result = BatchProcessingResult(
            total_files=len(file_paths),
            successful_files=successful_files,
            failed_files=failed_files,
            total_chunks=total_chunks,
            total_processing_time=total_processing_time,
            results=results
        )
        
        logger.info(f"Batch processing completed: {successful_files}/{len(file_paths)} files successful, "
                   f"{total_chunks} total chunks, {total_processing_time:.2f}s total time")
        
        return batch_result
    
    def validate_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Validate multiple files before processing.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid_files': [],
            'invalid_files': [],
            'total_size_mb': 0.0,
            'supported_types': {},
            'errors': []
        }
        
        for file_path in file_paths:
            file_path_str = str(file_path)
            
            try:
                # Check if file exists
                if not os.path.exists(file_path_str):
                    validation_results['invalid_files'].append(file_path_str)
                    validation_results['errors'].append(f"{file_path_str}: File not found")
                    continue
                
                # Validate with text extractor
                self.text_extractor.validate_file(file_path_str, config.max_file_size_mb)
                
                # Get file info
                file_metadata = self.text_extractor.get_file_metadata(file_path_str)
                file_size_mb = file_metadata.get('file_size', 0) / (1024 * 1024)
                file_type = file_metadata.get('file_type', '')
                
                validation_results['valid_files'].append(file_path_str)
                validation_results['total_size_mb'] += file_size_mb
                
                # Track file types
                if file_type in validation_results['supported_types']:
                    validation_results['supported_types'][file_type] += 1
                else:
                    validation_results['supported_types'][file_type] = 1
                
            except (TextExtractionError, UnsupportedFileTypeError) as e:
                validation_results['invalid_files'].append(file_path_str)
                validation_results['errors'].append(f"{file_path_str}: {str(e)}")
            except Exception as e:
                validation_results['invalid_files'].append(file_path_str)
                validation_results['errors'].append(f"{file_path_str}: Unexpected error - {str(e)}")
        
        return validation_results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current state of the processing system.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            # Get vector store info
            collection_info = self.vector_store_manager.get_collection_info()
            
            # Get embedding generator stats
            embedding_stats = self.embedding_generator.get_cache_stats()
            
            # Get chunker stats (if available)
            chunker_stats = {
                'chunk_size': self.text_chunker.chunk_size,
                'chunk_overlap': self.text_chunker.chunk_overlap,
                'separators_count': len(self.text_chunker.separators)
            }
            
            return {
                'vector_store': collection_info,
                'embedding_generator': embedding_stats,
                'text_chunker': chunker_stats,
                'supported_file_types': list(self.text_extractor.SUPPORTED_EXTENSIONS),
                'max_file_size_mb': config.max_file_size_mb
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing statistics: {str(e)}")
            return {
                'error': str(e),
                'supported_file_types': list(self.text_extractor.SUPPORTED_EXTENSIONS),
                'max_file_size_mb': config.max_file_size_mb
            }
    
    def clear_all_data(self) -> bool:
        """
        Clear all processed data from the system.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Reset vector store
            self.vector_store_manager.reset_collection()
            
            # Clear embedding cache
            self.embedding_generator.clear_cache()
            
            logger.info("Successfully cleared all processed data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear all data: {str(e)}")
            return False
"""Vector store management using ChromaDB for document storage and retrieval."""

import logging
import time
import uuid
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

from ..models.data_models import DocumentChunk
from ..config import config
from .error_handler import (
    error_handler, retry_on_failure, log_function_call,
    VectorStoreError, ErrorCategory, ErrorSeverity
)
from .logging_config import get_logging_manager

logger = logging.getLogger(__name__)
logging_manager = get_logging_manager()


class VectorStoreManager:
    """Manages ChromaDB operations for storing and retrieving document embeddings."""
    
    def __init__(self, persist_directory_or_config = None, collection_name: str = "documents"):
        """
        Initialize the VectorStoreManager.
        
        Args:
            persist_directory_or_config: Directory to persist ChromaDB data, or AppConfig object. Uses config default if None.
            collection_name: Name of the ChromaDB collection to use.
        """
        # Handle both string path and AppConfig object
        if hasattr(persist_directory_or_config, 'chroma_persist_directory'):
            # It's an AppConfig object
            self.persist_directory = persist_directory_or_config.chroma_persist_directory
        elif isinstance(persist_directory_or_config, str):
            # It's a string path
            self.persist_directory = persist_directory_or_config
        else:
            # Use default from config
            self.persist_directory = config.chroma_persist_directory
            
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_function = None
        
        # Initialize the database
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize ChromaDB client and collection."""
        context = {
            'persist_directory': self.persist_directory,
            'collection_name': self.collection_name
        }
        
        try:
            # Create ChromaDB client with persistence
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Technical documentation chunks for RAG"}
            )
            
            logger.info(f"Initialized ChromaDB with collection '{self.collection_name}' at {self.persist_directory}")
            
        except Exception as e:
            error_info = error_handler.handle_error(
                e, 
                context=context, 
                category=ErrorCategory.VECTOR_STORE,
                severity=ErrorSeverity.CRITICAL
            )
            raise VectorStoreError(
                f"Failed to initialize ChromaDB: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                context=context
            ) from e
    
    @log_function_call(include_args=True)
    @retry_on_failure(max_retries=3, delay=1.0, exceptions=(ConnectionError, TimeoutError), error_handler=error_handler)
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects to add to the store.
            
        Raises:
            VectorStoreError: If chunks list is empty, contains invalid data, or database operation fails.
        """
        if not chunks:
            raise VectorStoreError(
                "Cannot add empty list of chunks",
                severity=ErrorSeverity.MEDIUM,
                context={'chunk_count': 0}
            )
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for chunk in chunks:
                if not chunk.id:
                    chunk.id = str(uuid.uuid4())
                
                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.id} missing embedding")
                
                ids.append(chunk.id)
                documents.append(chunk.content)
                
                # Prepare metadata (ChromaDB supports basic types)
                metadata = {
                    "source": chunk.source,
                    **{k: v if isinstance(v, (str, int, float, bool)) else str(v) 
                       for k, v in chunk.metadata.items()}
                }
                metadatas.append(metadata)
                embeddings.append(chunk.embedding.tolist())
            
            # Add to collection
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"Added {len(chunks)} document chunks to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise RuntimeError(f"Database operation failed: {e}")
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[DocumentChunk]:
        """
        Perform similarity search using query embedding.
        
        Args:
            query_embedding: Query embedding vector.
            k: Number of similar chunks to retrieve.
            
        Returns:
            List of DocumentChunk objects ordered by similarity.
            
        Raises:
            ValueError: If k is invalid or query_embedding is malformed.
            RuntimeError: If database operation fails.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        
        if query_embedding is None or len(query_embedding) == 0:
            raise ValueError("Query embedding cannot be empty")
        
        search_start_time = time.time()
        
        try:
            # Check if collection is empty
            collection_size = self.get_collection_size()
            if collection_size == 0:
                logger.debug("Collection is empty, returning empty results")
                return []
            
            # Optimize k based on collection size and performance settings
            optimized_k = min(k, collection_size, config.vector_search_k_limit)
            if optimized_k != k:
                logger.debug(f"Optimized k from {k} to {optimized_k} for better performance")
            
            # Query the collection
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=optimized_k
            )
            
            search_time = time.time() - search_start_time
            
            # Convert results to DocumentChunk objects
            chunks = []
            if results['ids'] and results['ids'][0]:  # Check if we have results
                for i in range(len(results['ids'][0])):
                    chunk = DocumentChunk(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] or {},
                        source=results['metadatas'][0][i].get('source', 'unknown') if results['metadatas'][0][i] else 'unknown',
                        embedding=np.array(results['embeddings'][0][i]) if results['embeddings'] and results['embeddings'][0] else None
                    )
                    chunks.append(chunk)
            
            logger.debug(f"Retrieved {len(chunks)} similar chunks for query in {search_time:.3f}s")
            
            # Log performance warning if search is slow
            if search_time > 2.0:
                logger.warning(f"Slow vector search detected: {search_time:.2f}s for k={optimized_k}, collection_size={collection_size}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise RuntimeError(f"Database operation failed: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary containing collection metadata and statistics.
        """
        try:
            collection_count = self._collection.count()
            collection_metadata = self._collection.metadata
            
            return {
                "name": self.collection_name,
                "count": collection_count,
                "metadata": collection_metadata,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "name": self.collection_name,
                "count": 0,
                "metadata": {},
                "persist_directory": self.persist_directory,
                "error": str(e)
            }
    
    def get_collection_size(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents in the collection.
        """
        try:
            return self._collection.count()
        except Exception as e:
            logger.error(f"Failed to get collection size: {e}")
            return 0
    
    def delete_collection(self) -> None:
        """
        Delete the current collection and all its data.
        
        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            self._client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
            
            # Reinitialize with empty collection
            self._initialize_database()
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise RuntimeError(f"Collection deletion failed: {e}")
    
    def get_document_by_id(self, doc_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a specific document by its ID.
        
        Args:
            doc_id: The document ID to retrieve.
            
        Returns:
            DocumentChunk if found, None otherwise.
        """
        try:
            results = self._collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if results['ids'] and len(results['ids']) > 0:
                return DocumentChunk(
                    id=results['ids'][0],
                    content=results['documents'][0],
                    metadata=results['metadatas'][0] or {},
                    source=results['metadatas'][0].get('source', 'unknown') if results['metadatas'][0] else 'unknown',
                    embedding=np.array(results['embeddings'][0]) if results['embeddings'] and results['embeddings'][0] else None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document by ID {doc_id}: {e}")
            return None
    
    def update_document(self, chunk: DocumentChunk) -> bool:
        """
        Update an existing document in the collection.
        
        Args:
            chunk: DocumentChunk with updated data.
            
        Returns:
            True if update was successful, False otherwise.
        """
        try:
            if not chunk.id:
                raise ValueError("Document chunk must have an ID for updates")
            
            if chunk.embedding is None:
                raise ValueError("Document chunk must have an embedding for updates")
            
            # Prepare metadata (ChromaDB supports basic types)
            metadata = {
                "source": chunk.source,
                **{k: v if isinstance(v, (str, int, float, bool)) else str(v) 
                   for k, v in chunk.metadata.items()}
            }
            
            # Update the document
            self._collection.update(
                ids=[chunk.id],
                documents=[chunk.content],
                metadatas=[metadata],
                embeddings=[chunk.embedding.tolist()]
            )
            
            logger.debug(f"Updated document {chunk.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {chunk.id}: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the collection.
        
        Args:
            doc_id: ID of the document to delete.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            self._collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def reset_collection(self) -> None:
        """
        Reset the collection by deleting and recreating it.
        
        Raises:
            RuntimeError: If reset operation fails.
        """
        try:
            # Delete existing collection
            try:
                self._client.delete_collection(name=self.collection_name)
            except Exception:
                # Collection might not exist, which is fine
                pass
            
            # Recreate collection
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"description": "Technical documentation chunks for RAG"}
            )
            
            logger.info(f"Reset collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise RuntimeError(f"Collection reset failed: {e}")
    
    def close(self) -> None:
        """Close the database connection and cleanup resources."""
        try:
            # ChromaDB client doesn't need explicit closing, but we can clear references
            self._collection = None
            self._client = None
            logger.debug("Closed vector store manager")
            
        except Exception as e:
            logger.error(f"Error during vector store cleanup: {e}")
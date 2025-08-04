"""Embedding generation service using sentence-transformers."""

import hashlib
import logging
import os
import pickle
from typing import List, Dict, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from ..models.data_models import TextChunk, DocumentChunk
from ..config import config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text chunks using sentence-transformers with caching."""
    
    def __init__(self, model_name_or_config = None, cache_dir: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name_or_config: Name of the sentence-transformer model to use, or AppConfig object
            cache_dir: Directory to store embedding cache files
        """
        # Handle both string model name and AppConfig object
        if hasattr(model_name_or_config, 'embedding_model'):
            # It's an AppConfig object
            self.model_name = model_name_or_config.embedding_model
            if cache_dir is None:
                cache_dir = os.path.join(model_name_or_config.chroma_persist_directory, "embedding_cache")
        elif isinstance(model_name_or_config, str):
            # It's a string model name
            self.model_name = model_name_or_config
        else:
            # Use default from config
            self.model_name = config.embedding_model
            
        self.cache_dir = cache_dir or os.path.join(config.chroma_persist_directory, "embedding_cache")
        self.model: Optional[SentenceTransformer] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
    
    def _get_model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Successfully loaded model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
        return self.model
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text."""
        # Use SHA-256 hash of the text as cache key
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        cache_file = os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_cache.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self._embedding_cache = {}
        else:
            logger.info("No existing embedding cache found")
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        cache_file = os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_cache.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
            logger.debug(f"Saved {len(self._embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text if it exists."""
        cache_key = self._get_cache_key(text)
        return self._embedding_cache.get(cache_key)
    
    def _cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Cache an embedding for the given text."""
        cache_key = self._get_cache_key(text)
        self._embedding_cache[cache_key] = embedding
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Numpy array containing the embedding vector
        """
        # Check cache first
        cached_embedding = self._get_cached_embedding(text)
        if cached_embedding is not None:
            logger.debug("Using cached embedding")
            return cached_embedding
        
        # Generate new embedding
        model = self._get_model()
        try:
            embedding = model.encode(text, convert_to_numpy=True)
            
            # Cache the embedding
            self._cache_embedding(text, embedding)
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with batch processing.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of numpy arrays containing embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts in batches
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} uncached texts")
            model = self._get_model()
            
            try:
                # Process in batches for memory efficiency
                for i in range(0, len(uncached_texts), batch_size):
                    batch_texts = uncached_texts[i:i + batch_size]
                    batch_indices = uncached_indices[i:i + batch_size]
                    
                    logger.debug(f"Processing batch {i//batch_size + 1}/{(len(uncached_texts) + batch_size - 1)//batch_size}")
                    
                    # Generate embeddings for this batch
                    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, batch_size=batch_size)
                    
                    # Store embeddings and cache them
                    for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                        original_index = batch_indices[j]
                        embeddings[original_index] = embedding
                        self._cache_embedding(text, embedding)
                
                # Save cache after batch processing
                self._save_cache()
                
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                raise
        
        return embeddings
    
    def generate_embeddings_for_chunks(self, chunks: List[TextChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for text chunks and convert to DocumentChunks.
        
        Args:
            chunks: List of TextChunk objects to process
            
        Returns:
            List of DocumentChunk objects with embeddings
        """
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract texts from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.generate_embeddings_batch(texts)
        
        # Convert to DocumentChunks
        document_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate unique ID for the chunk
            chunk_id = hashlib.md5(f"{chunk.content[:100]}{i}".encode()).hexdigest()
            
            # Extract source from metadata or use default
            source = chunk.metadata.get('source', 'unknown')
            
            document_chunk = DocumentChunk(
                id=chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                source=source,
                embedding=embedding
            )
            document_chunks.append(document_chunk)
        
        logger.info(f"Successfully generated {len(document_chunks)} document chunks with embeddings")
        return document_chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        model = self._get_model()
        return model.get_sentence_embedding_dimension()
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        cache_file = os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_cache.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache."""
        return {
            'cache_size': len(self._embedding_cache),
            'model_name': self.model_name,
            'cache_dir': self.cache_dir,
            'embedding_dimension': self.get_embedding_dimension() if self.model else None
        }
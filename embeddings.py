import os
import time
import hashlib
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Wrapper for embedding models with caching and optimization"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "models"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.dimension = None
        self.max_seq_length = 512
        self._lock = threading.RLock()
        self._loaded = False
        
        logger.info(f"Initialized embedding model: {model_name}")
    
    def _load_model(self):
        """Load the model (lazy loading)"""
        with self._lock:
            if self._loaded:
                return
            
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                start_time = time.time()
                
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(self.cache_dir)
                )
                
                # Set device
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model.to(device)
                
                # Get model info
                self.dimension = self.model.get_sentence_embedding_dimension()
                self.max_seq_length = self.model.max_seq_length
                
                # Load tokenizer for text preprocessing
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        cache_dir=str(self.cache_dir)
                    )
                except:
                    self.tokenizer = None
                
                load_time = time.time() - start_time
                logger.info(f"Model loaded successfully in {load_time:.2f}s - Device: {device}, Dimension: {self.dimension}")
                
                self._loaded = True
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise RuntimeError(f"Could not load embedding model: {e}")
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        if not self._loaded:
            self._load_model()
        
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            if not texts:
                return np.array([])
            
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding"""
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Handle very long texts
        if self.tokenizer and len(text) > self.max_seq_length * 4:
            # Rough estimation: 4 chars per token
            max_chars = self.max_seq_length * 4
            text = text[:max_chars]
            logger.debug(f"Truncated long text to {max_chars} characters")
        
        return text
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        embeddings = self.encode([text1, text2])
        if len(embeddings) != 2:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self._loaded:
            self._load_model()
        
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'max_seq_length': self.max_seq_length,
            'device': str(self.model.device) if self.model else 'not_loaded',
            'loaded': self._loaded
        }

class EmbeddingCache:
    """Cache for embeddings to avoid recomputation"""
    
    def __init__(self, cache_dir: str = "data/embeddings_cache", max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._cache = {}
        self._access_times = {}
        self._lock = threading.RLock()
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model"""
        combined = f"{model_name}:{text}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        with self._lock:
            cache_key = self._get_cache_key(text, model_name)
            
            if cache_key in self._cache:
                self._access_times[cache_key] = time.time()
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return self._cache[cache_key]
            
            return None
    
    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """Cache an embedding"""
        with self._lock:
            cache_key = self._get_cache_key(text, model_name)
            
            # Remove oldest items if cache is full
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            self._cache[cache_key] = embedding.copy()
            self._access_times[cache_key] = time.time()
            logger.debug(f"Cached embedding for text: {text[:50]}...")
    
    def _evict_oldest(self):
        """Remove oldest cached items"""
        if not self._access_times:
            return
        
        # Remove 20% of oldest items
        items_to_remove = max(1, len(self._access_times) // 5)
        oldest_keys = sorted(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        for key in oldest_keys[:items_to_remove]:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
        
        logger.debug(f"Evicted {items_to_remove} old cache entries")
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            logger.info("Cleared embedding cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': 'N/A',  # Would need more tracking for this
                'memory_usage_mb': sum(arr.nbytes for arr in self._cache.values()) / 1024 / 1024
            }

class EmbeddingService:
    """Main service for generating and managing embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_cache: bool = True):
        self.model_name = model_name
        self.model = EmbeddingModel(model_name)
        self.cache = EmbeddingCache() if use_cache else None
        self.use_cache = use_cache
        
        logger.info(f"Initialized embedding service with model: {model_name}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        if not text or not text.strip():
            # Return zero vector for empty text
            if not self.model._loaded:
                self.model._load_model()
            return np.zeros(self.model.dimension, dtype=np.float32)
        
        # Check cache first
        if self.use_cache and self.cache:
            cached_embedding = self.cache.get(text, self.model_name)
            if cached_embedding is not None:
                return cached_embedding
        
        # Generate embedding
        embedding = self.model.encode(text)
        
        # Cache the result
        if self.use_cache and self.cache:
            self.cache.set(text, self.model_name, embedding)
        
        return embedding.flatten()
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode multiple texts to embeddings"""
        if not texts:
            return []
        
        # Check cache for all texts first
        cached_results = {}
        uncached_texts = []
        uncached_indices = []
        
        if self.use_cache and self.cache:
            for i, text in enumerate(texts):
                cached_embedding = self.cache.get(text, self.model_name)
                if cached_embedding is not None:
                    cached_results[i] = cached_embedding
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} texts")
            new_embeddings = self.model.encode(uncached_texts, batch_size=batch_size)
            
            # Cache new embeddings
            if self.use_cache and self.cache:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.cache.set(text, self.model_name, embedding)
        
        # Combine cached and new results
        results = [None] * len(texts)
        
        # Fill in cached results
        for i, embedding in cached_results.items():
            results[i] = embedding
        
        # Fill in new results
        for i, new_idx in enumerate(uncached_indices):
            if i < len(new_embeddings):
                results[new_idx] = new_embeddings[i]
        
        # Handle any remaining None values (empty texts)
        if not self.model._loaded:
            self.model._load_model()
        
        for i, result in enumerate(results):
            if result is None:
                results[i] = np.zeros(self.model.dimension, dtype=np.float32)
        
        return results
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return self.model.get_similarity(text1, text2)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if not self.model._loaded:
            self.model._load_model()
        return self.model.dimension
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        info = {
            'model_info': self.model.get_info(),
            'cache_enabled': self.use_cache,
            'service_initialized': True
        }
        
        if self.cache:
            info['cache_stats'] = self.cache.get_stats()
        
        return info
    
    def clear_cache(self):
        """Clear embedding cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")

# Global embedding service
embedding_service = None

def get_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    """Get or create global embedding service"""
    global embedding_service
    
    if embedding_service is None or embedding_service.model_name != model_name:
        embedding_service = EmbeddingService(model_name)
        logger.info(f"Created new embedding service with model: {model_name}")
    
    return embedding_service

def encode_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Convenience function to encode text"""
    service = get_embedding_service(model_name)
    return service.encode_text(text)

def encode_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32) -> List[np.ndarray]:
    """Convenience function to encode multiple texts"""
    service = get_embedding_service(model_name)
    return service.encode_texts(texts, batch_size)

def calculate_similarity(text1: str, text2: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Convenience function to calculate text similarity"""
    service = get_embedding_service(model_name)
    return service.calculate_similarity(text1, text2)
import os
import json
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class VectorChunk:
    def __init__(self, id: str, text: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        self.id = id
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()

class SimpleVectorDatabase:
    """Simple vector database using sklearn for Windows compatibility"""
    
    def __init__(self, bot_id: str, dimension: int = 384, storage_path: str = "data/vectors"):
        self.bot_id = bot_id
        self.dimension = dimension
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.metadata_path = self.storage_path / f"{bot_id}_metadata.json"
        self.chunks_path = self.storage_path / f"{bot_id}_chunks.pkl"
        self.vectors_path = self.storage_path / f"{bot_id}_vectors.npy"
        
        # Data storage
        self.chunks = {}
        self.embeddings = []
        self.chunk_ids = []
        self.nearest_neighbors = None
        
        self._load_data()
    
    def _load_data(self):
        """Load existing data"""
        try:
            # Load chunks
            if self.chunks_path.exists():
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
            
            # Load vectors
            if self.vectors_path.exists():
                self.embeddings = np.load(str(self.vectors_path))
                if len(self.embeddings) > 0:
                    self.chunk_ids = list(self.chunks.keys())
                    self._build_index()
            
            logger.info(f"Loaded vector database for bot {self.bot_id}: {len(self.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            self.chunks = {}
            self.embeddings = []
            self.chunk_ids = []
    
    def _save_data(self):
        """Save data to disk"""
        try:
            # Save chunks
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Save vectors
            if len(self.embeddings) > 0:
                np.save(str(self.vectors_path), self.embeddings)
            
            # Save metadata
            metadata = {
                'bot_id': self.bot_id,
                'dimension': self.dimension,
                'total_vectors': len(self.embeddings),
                'chunk_count': len(self.chunks),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
    
    def _build_index(self):
        """Build the search index"""
        try:
            if len(self.embeddings) == 0:
                return
            
            # Use cosine similarity for better semantic search
            self.nearest_neighbors = NearestNeighbors(
                n_neighbors=min(10, len(self.embeddings)),
                metric='cosine',
                algorithm='brute'  # More reliable for small datasets
            )
            
            self.nearest_neighbors.fit(self.embeddings)
            logger.debug(f"Built search index with {len(self.embeddings)} vectors")
            
        except Exception as e:
            logger.error(f"Error building search index: {e}")
    
    def add_vectors(self, chunks: List[VectorChunk]) -> bool:
        """Add multiple vectors to the database"""
        try:
            if not chunks:
                return True
            
            new_embeddings = []
            new_chunk_ids = []
            
            for chunk in chunks:
                if chunk.embedding.shape[0] != self.dimension:
                    logger.warning(f"Embedding dimension mismatch: {chunk.embedding.shape[0]} vs {self.dimension}")
                    continue
                
                # Store chunk data
                self.chunks[chunk.id] = {
                    'text': chunk.text,
                    'metadata': chunk.metadata,
                    'created_at': chunk.created_at.isoformat()
                }
                
                new_embeddings.append(chunk.embedding)
                new_chunk_ids.append(chunk.id)
            
            if new_embeddings:
                # Add to existing embeddings
                if len(self.embeddings) == 0:
                    self.embeddings = np.array(new_embeddings)
                    self.chunk_ids = new_chunk_ids
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])
                    self.chunk_ids.extend(new_chunk_ids)
                
                # Rebuild index
                self._build_index()
                
                # Save to disk
                self._save_data()
                
                logger.info(f"Added {len(new_embeddings)} vectors to bot {self.bot_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int = 5, threshold: float = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            if len(self.embeddings) == 0:
                return []
            
            if self.nearest_neighbors is None:
                self._build_index()
            
            if self.nearest_neighbors is None:
                return []
            
            # Ensure query vector has correct shape
            query_vector = query_vector.reshape(1, -1)
            
            # Search for nearest neighbors
            k = min(k, len(self.embeddings))
            distances, indices = self.nearest_neighbors.kneighbors(query_vector, n_neighbors=k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= len(self.chunk_ids):
                    continue
                
                chunk_id = self.chunk_ids[idx]
                if chunk_id not in self.chunks:
                    continue
                
                # Convert cosine distance to similarity
                similarity = 1.0 - distance
                
                # Apply threshold if specified
                if threshold and similarity < threshold:
                    continue
                
                chunk_data = self.chunks[chunk_id]
                results.append({
                    'id': chunk_id,
                    'text': chunk_data['text'],
                    'metadata': chunk_data['metadata'],
                    'distance': float(distance),
                    'similarity': float(similarity),
                    'rank': i + 1
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID"""
        return self.chunks.get(chunk_id)
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk"""
        try:
            if chunk_id in self.chunks:
                # Remove from chunks
                del self.chunks[chunk_id]
                
                # Find and remove from embeddings
                if chunk_id in self.chunk_ids:
                    idx = self.chunk_ids.index(chunk_id)
                    self.embeddings = np.delete(self.embeddings, idx, axis=0)
                    self.chunk_ids.remove(chunk_id)
                    
                    # Rebuild index
                    if len(self.embeddings) > 0:
                        self._build_index()
                    else:
                        self.nearest_neighbors = None
                
                self._save_data()
                logger.info(f"Deleted chunk {chunk_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {e}")
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'bot_id': self.bot_id,
            'total_vectors': len(self.embeddings),
            'dimension': self.dimension,
            'total_chunks': len(self.chunks),
            'storage_size_mb': self._get_storage_size(),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def _get_storage_size(self) -> float:
        """Calculate storage size in MB"""
        total_size = 0
        for path in [self.metadata_path, self.chunks_path, self.vectors_path]:
            if path.exists():
                total_size += path.stat().st_size
        return round(total_size / (1024 * 1024), 2)
    
    def clear(self) -> bool:
        """Clear all data"""
        try:
            self.chunks = {}
            self.embeddings = []
            self.chunk_ids = []
            self.nearest_neighbors = None
            
            # Remove files
            for path in [self.metadata_path, self.chunks_path, self.vectors_path]:
                if path.exists():
                    path.unlink()
            
            logger.info(f"Cleared all data for bot {self.bot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

class VectorDatabaseManager:
    """Manages multiple vector databases for different bots"""
    
    def __init__(self, storage_path: str = "data/vectors"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.databases = {}
    
    def get_database(self, bot_id: str, dimension: int = 384) -> SimpleVectorDatabase:
        """Get or create vector database for a bot"""
        if bot_id not in self.databases:
            self.databases[bot_id] = SimpleVectorDatabase(
                bot_id=bot_id,
                dimension=dimension,
                storage_path=str(self.storage_path)
            )
        return self.databases[bot_id]
    
    def delete_database(self, bot_id: str) -> bool:
        """Delete vector database for a bot"""
        try:
            if bot_id in self.databases:
                self.databases[bot_id].clear()
                del self.databases[bot_id]
            
            # Remove files
            for pattern in [f"{bot_id}_*"]:
                for file_path in self.storage_path.glob(pattern):
                    file_path.unlink()
            
            logger.info(f"Deleted vector database for bot {bot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting database for bot {bot_id}: {e}")
            return False
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all databases"""
        stats = {}
        for bot_id, db in self.databases.items():
            stats[bot_id] = db.get_stats()
        return stats

# Global manager instance
vector_manager = VectorDatabaseManager()

def get_vector_database(bot_id: str, dimension: int = 384) -> SimpleVectorDatabase:
    """Get vector database for a bot"""
    return vector_manager.get_database(bot_id, dimension)

def create_text_hash(text: str) -> str:
    """Create a unique hash for text content"""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
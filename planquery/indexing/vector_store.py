"""
Vector store implementation for semantic search using embeddings.
Supports multiple backends: FAISS, pgvector, and Qdrant.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant not available")

from .database import DatabaseManager, Region


class VectorStore:
    """Vector store for semantic search capabilities."""
    
    def __init__(self, 
                 backend: str = "faiss",
                 model_name: str = "all-MiniLM-L6-v2",
                 dimension: int = 384,
                 index_path: Optional[str] = None,
                 database_manager: Optional[DatabaseManager] = None):
        """
        Initialize vector store.
        
        Args:
            backend: Vector store backend ("faiss", "pgvector", "qdrant")
            model_name: Sentence transformer model name
            dimension: Vector dimension
            index_path: Path to save/load index
            database_manager: Database manager for pgvector backend
        """
        self.backend = backend
        self.model_name = model_name
        self.dimension = dimension
        self.index_path = index_path
        self.database_manager = database_manager
        
        # Initialize components
        self.encoder = None
        self.index = None
        self.id_mapping = {}  # Maps vector index to region IDs
        
        self._initialize_encoder()
        self._initialize_backend()
    
    def _initialize_encoder(self):
        """Initialize sentence transformer encoder."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformers not available")
            return
        
        try:
            self.encoder = SentenceTransformer(self.model_name)
            logger.info(f"Initialized encoder: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize encoder: {e}")
    
    def _initialize_backend(self):
        """Initialize vector store backend."""
        if self.backend == "faiss" and FAISS_AVAILABLE:
            self._initialize_faiss()
        elif self.backend == "pgvector":
            self._initialize_pgvector()
        elif self.backend == "qdrant" and QDRANT_AVAILABLE:
            self._initialize_qdrant()
        else:
            logger.warning(f"Backend {self.backend} not available, using in-memory fallback")
            self.backend = "memory"
            self._initialize_memory()
    
    def _initialize_faiss(self):
        """Initialize FAISS index."""
        try:
            # Use IndexFlatIP for cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
            
            # Load existing index if available
            if self.index_path and Path(self.index_path).exists():
                self.load_index()
            
            logger.info("FAISS vector store initialized")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self.backend = "memory"
            self._initialize_memory()
    
    def _initialize_pgvector(self):
        """Initialize pgvector backend."""
        if not self.database_manager:
            logger.error("Database manager required for pgvector backend")
            self.backend = "memory"
            self._initialize_memory()
            return
        
        logger.info("pgvector backend initialized")
    
    def _initialize_qdrant(self):
        """Initialize Qdrant client."""
        try:
            self.client = QdrantClient(":memory:")  # In-memory for demo
            
            # Create collection
            self.client.recreate_collection(
                collection_name="planquery",
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE
                )
            )
            
            logger.info("Qdrant vector store initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            self.backend = "memory"
            self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize in-memory fallback."""
        self.vectors = []
        self.metadata = []
        logger.info("In-memory vector store initialized")
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to vector."""
        if not self.encoder:
            return None
        
        try:
            embedding = self.encoder.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to encode text: {e}")
            return None
    
    def encode_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode batch of texts to vectors."""
        if not self.encoder:
            return None
        
        try:
            embeddings = self.encoder.encode(texts, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to encode batch: {e}")
            return None
    
    def add_text(self, text: str, region_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add text to vector store.
        
        Args:
            text: Text to encode and store
            region_id: Unique identifier for the text
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        if not text.strip():
            return False
        
        # Encode text
        vector = self.encode_text(text)
        if vector is None:
            return False
        
        # Store based on backend
        if self.backend == "faiss":
            return self._add_to_faiss(vector, region_id, metadata)
        elif self.backend == "pgvector":
            return self._add_to_pgvector(vector, region_id, metadata)
        elif self.backend == "qdrant":
            return self._add_to_qdrant(vector, region_id, metadata)
        else:  # memory
            return self._add_to_memory(vector, region_id, metadata)
    
    def _add_to_faiss(self, vector: np.ndarray, region_id: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Add vector to FAISS index."""
        try:
            # Normalize for cosine similarity
            faiss.normalize_L2(vector.reshape(1, -1))
            
            # Add to index
            current_count = self.index.ntotal
            self.index.add(vector.reshape(1, -1))
            
            # Update mapping
            self.id_mapping[current_count] = {
                'region_id': region_id,
                'metadata': metadata or {}
            }
            
            return True
        except Exception as e:
            logger.error(f"Failed to add to FAISS: {e}")
            return False
    
    def _add_to_pgvector(self, vector: np.ndarray, region_id: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Add vector to pgvector database."""
        try:
            # Update region embedding in database
            self.database_manager.update_region_embedding(region_id, vector.tolist())
            return True
        except Exception as e:
            logger.error(f"Failed to add to pgvector: {e}")
            return False
    
    def _add_to_qdrant(self, vector: np.ndarray, region_id: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Add vector to Qdrant."""
        try:
            self.client.upsert(
                collection_name="planquery",
                points=[
                    models.PointStruct(
                        id=region_id,
                        vector=vector.tolist(),
                        payload=metadata or {}
                    )
                ]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add to Qdrant: {e}")
            return False
    
    def _add_to_memory(self, vector: np.ndarray, region_id: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Add vector to memory store."""
        try:
            self.vectors.append(vector)
            self.metadata.append({
                'region_id': region_id,
                'metadata': metadata or {}
            })
            return True
        except Exception as e:
            logger.error(f"Failed to add to memory: {e}")
            return False
    
    def search(self, query: str, k: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar texts.
        
        Args:
            query: Query text
            k: Number of results to return
            filters: Optional filters (floor_id, discipline, etc.)
            
        Returns:
            List of (region_id, similarity_score, metadata) tuples
        """
        # Encode query
        query_vector = self.encode_text(query)
        if query_vector is None:
            return []
        
        # Search based on backend
        if self.backend == "faiss":
            return self._search_faiss(query_vector, k, filters)
        elif self.backend == "pgvector":
            return self._search_pgvector(query_vector, k, filters)
        elif self.backend == "qdrant":
            return self._search_qdrant(query_vector, k, filters)
        else:  # memory
            return self._search_memory(query_vector, k, filters)
    
    def _search_faiss(self, query_vector: np.ndarray, k: int, 
                     filters: Optional[Dict[str, Any]]) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search FAISS index."""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query vector
            faiss.normalize_L2(query_vector.reshape(1, -1))
            
            # Search
            scores, indices = self.index.search(query_vector.reshape(1, -1), min(k, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # No more results
                    break
                
                mapping = self.id_mapping.get(idx, {})
                region_id = mapping.get('region_id', str(idx))
                metadata = mapping.get('metadata', {})
                
                # Apply filters if provided
                if filters and not self._matches_filters(metadata, filters):
                    continue
                
                results.append((region_id, float(score), metadata))
            
            return results
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _search_pgvector(self, query_vector: np.ndarray, k: int, 
                        filters: Optional[Dict[str, Any]]) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search using pgvector."""
        try:
            # Use database manager for vector search
            floor_id = filters.get('floor_id') if filters else None
            discipline = filters.get('discipline') if filters else None
            
            regions = self.database_manager.vector_search(
                query_vector.tolist(), 
                limit=k,
                floor_id=floor_id,
                discipline=discipline
            )
            
            results = []
            for region in regions:
                # Calculate similarity score (placeholder - actual score would come from database)
                score = 0.8  # Placeholder
                
                metadata = {
                    'floor_id': region.page.floor_id,
                    'discipline': region.page.document.discipline,
                    'text_type': region.text_type,
                    'region_type': region.region_type
                }
                
                results.append((str(region.id), score, metadata))
            
            return results
        except Exception as e:
            logger.error(f"pgvector search failed: {e}")
            return []
    
    def _search_qdrant(self, query_vector: np.ndarray, k: int, 
                      filters: Optional[Dict[str, Any]]) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search Qdrant collection."""
        try:
            # Build filter conditions
            filter_conditions = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                
                if conditions:
                    filter_conditions = models.Filter(must=conditions)
            
            # Search
            search_result = self.client.search(
                collection_name="planquery",
                query_vector=query_vector.tolist(),
                query_filter=filter_conditions,
                limit=k
            )
            
            results = []
            for hit in search_result:
                results.append((
                    str(hit.id),
                    hit.score,
                    hit.payload or {}
                ))
            
            return results
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
    def _search_memory(self, query_vector: np.ndarray, k: int, 
                      filters: Optional[Dict[str, Any]]) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search in-memory vectors."""
        if not self.vectors:
            return []
        
        try:
            # Calculate similarities
            vectors_array = np.array(self.vectors)
            
            # Normalize vectors for cosine similarity
            query_norm = query_vector / np.linalg.norm(query_vector)
            vectors_norm = vectors_array / np.linalg.norm(vectors_array, axis=1, keepdims=True)
            
            # Calculate cosine similarities
            similarities = np.dot(vectors_norm, query_norm)
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                metadata_entry = self.metadata[idx]
                region_id = metadata_entry['region_id']
                metadata = metadata_entry['metadata']
                score = float(similarities[idx])
                
                # Apply filters if provided
                if filters and not self._matches_filters(metadata, filters):
                    continue
                
                results.append((region_id, score, metadata))
            
            return results
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def save_index(self, path: Optional[str] = None):
        """Save vector index to disk."""
        save_path = path or self.index_path
        if not save_path:
            logger.warning("No save path specified")
            return
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.backend == "faiss":
            try:
                faiss.write_index(self.index, str(save_path))
                
                # Save ID mapping
                mapping_path = save_path.with_suffix('.mapping.pkl')
                with open(mapping_path, 'wb') as f:
                    pickle.dump(self.id_mapping, f)
                
                logger.info(f"FAISS index saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")
        
        elif self.backend == "memory":
            try:
                data = {
                    'vectors': self.vectors,
                    'metadata': self.metadata
                }
                with open(save_path, 'wb') as f:
                    pickle.dump(data, f)
                
                logger.info(f"Memory index saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save memory index: {e}")
    
    def load_index(self, path: Optional[str] = None):
        """Load vector index from disk."""
        load_path = path or self.index_path
        if not load_path or not Path(load_path).exists():
            logger.warning(f"Index file not found: {load_path}")
            return
        
        load_path = Path(load_path)
        
        if self.backend == "faiss":
            try:
                self.index = faiss.read_index(str(load_path))
                
                # Load ID mapping
                mapping_path = load_path.with_suffix('.mapping.pkl')
                if mapping_path.exists():
                    with open(mapping_path, 'rb') as f:
                        self.id_mapping = pickle.load(f)
                
                logger.info(f"FAISS index loaded from {load_path}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
        
        elif self.backend == "memory":
            try:
                with open(load_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.vectors = data['vectors']
                self.metadata = data['metadata']
                
                logger.info(f"Memory index loaded from {load_path}")
            except Exception as e:
                logger.error(f"Failed to load memory index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = {
            'backend': self.backend,
            'model_name': self.model_name,
            'dimension': self.dimension,
        }
        
        if self.backend == "faiss":
            stats['total_vectors'] = self.index.ntotal if self.index else 0
        elif self.backend == "memory":
            stats['total_vectors'] = len(self.vectors)
        elif self.backend == "qdrant":
            try:
                collection_info = self.client.get_collection("planquery")
                stats['total_vectors'] = collection_info.points_count
            except:
                stats['total_vectors'] = 0
        else:
            stats['total_vectors'] = 0
        
        return stats
    
    def clear(self):
        """Clear all vectors from the store."""
        if self.backend == "faiss":
            self.index.reset()
            self.id_mapping.clear()
        elif self.backend == "memory":
            self.vectors.clear()
            self.metadata.clear()
        elif self.backend == "qdrant":
            try:
                self.client.delete_collection("planquery")
                self._initialize_qdrant()
            except Exception as e:
                logger.error(f"Failed to clear Qdrant collection: {e}")
        
        logger.info("Vector store cleared")

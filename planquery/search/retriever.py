"""
Hybrid retrieval system combining vector search, keyword search, and metadata filtering.
Implements the core search logic for the RAG pipeline.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger

from .query_parser import ParsedQuery, QueryType, SearchScope
from ..indexing.vector_store import VectorStore
from ..indexing.keyword_index import KeywordIndex, SearchResult
from ..indexing.database import DatabaseManager


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval with multiple scoring components."""
    region_id: str
    text: str
    vector_score: float
    keyword_score: float
    combined_score: float
    metadata: Dict[str, Any]
    source: str  # "vector", "keyword", or "hybrid"
    highlights: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "text": self.text,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "combined_score": self.combined_score,
            "metadata": self.metadata,
            "source": self.source,
            "highlights": self.highlights
        }


class HybridRetriever:
    """Hybrid retrieval system combining multiple search approaches."""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 keyword_index: KeywordIndex,
                 database_manager: DatabaseManager,
                 vector_weight: float = 0.6,
                 keyword_weight: float = 0.4):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store for semantic search
            keyword_index: Keyword index for BM25 search
            database_manager: Database for metadata filtering
            vector_weight: Weight for vector search scores
            keyword_weight: Weight for keyword search scores
        """
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.database_manager = database_manager
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        
        # Ensure weights sum to 1
        total_weight = vector_weight + keyword_weight
        self.vector_weight = vector_weight / total_weight
        self.keyword_weight = keyword_weight / total_weight
    
    def retrieve(self, parsed_query: ParsedQuery, 
                limit: int = 20,
                min_score: float = 0.1) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval based on parsed query.
        
        Args:
            parsed_query: Parsed query with intent and filters
            limit: Maximum number of results
            min_score: Minimum combined score threshold
            
        Returns:
            List of retrieval results sorted by combined score
        """
        logger.info(f"Retrieving for query: {parsed_query.original_query}")
        
        # Choose retrieval strategy based on query type
        if parsed_query.query_type == QueryType.COUNT:
            return self._handle_count_query(parsed_query)
        elif parsed_query.query_type == QueryType.LIST:
            return self._handle_list_query(parsed_query, limit)
        elif parsed_query.query_type == QueryType.LOCATION:
            return self._handle_location_query(parsed_query, limit)
        else:
            return self._handle_search_query(parsed_query, limit, min_score)
    
    def _handle_search_query(self, parsed_query: ParsedQuery, 
                           limit: int, min_score: float) -> List[RetrievalResult]:
        """Handle general search queries with hybrid approach."""
        # Prepare search query
        search_text = " ".join(parsed_query.search_terms)
        
        if not search_text.strip():
            # If no search terms, fall back to filtered browsing
            return self._handle_filtered_browse(parsed_query, limit)
        
        # Perform vector search
        vector_results = []
        if self.vector_store:
            try:
                vector_hits = self.vector_store.search(
                    search_text, 
                    k=limit * 2,  # Get more candidates for reranking
                    filters=parsed_query.filters
                )
                vector_results = self._convert_vector_results(vector_hits)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # Perform keyword search
        keyword_results = []
        if self.keyword_index:
            try:
                keyword_hits = self.keyword_index.search(
                    search_text,
                    filters=parsed_query.filters,
                    limit=limit * 2
                )
                keyword_results = self._convert_keyword_results(keyword_hits)
            except Exception as e:
                logger.warning(f"Keyword search failed: {e}")
        
        # Combine and score results
        combined_results = self._combine_results(vector_results, keyword_results)
        
        # Filter by minimum score and limit
        filtered_results = [r for r in combined_results if r.combined_score >= min_score]
        
        return filtered_results[:limit]
    
    def _handle_count_query(self, parsed_query: ParsedQuery) -> List[RetrievalResult]:
        """Handle count queries by returning database statistics."""
        try:
            # Get regions matching filters
            regions = self.database_manager.get_regions_with_filters(
                floor_id=parsed_query.filters.get('floor_id'),
                discipline=parsed_query.filters.get('discipline'),
                text_type=parsed_query.filters.get('text_type'),
                region_type=parsed_query.filters.get('region_type'),
                limit=10000  # High limit to get accurate count
            )
            
            count = len(regions)
            
            # Create a synthetic result with the count
            count_text = f"Found {count} items matching your criteria"
            if parsed_query.filters:
                filter_desc = []
                for key, value in parsed_query.filters.items():
                    filter_desc.append(f"{key}: {value}")
                count_text += f" ({', '.join(filter_desc)})"
            
            return [RetrievalResult(
                region_id="count_result",
                text=count_text,
                vector_score=0.0,
                keyword_score=1.0,
                combined_score=1.0,
                metadata={"count": count, "query_type": "count"},
                source="database",
                highlights=[]
            )]
            
        except Exception as e:
            logger.error(f"Count query failed: {e}")
            return []
    
    def _handle_list_query(self, parsed_query: ParsedQuery, limit: int) -> List[RetrievalResult]:
        """Handle list queries by returning filtered database results."""
        try:
            regions = self.database_manager.get_regions_with_filters(
                floor_id=parsed_query.filters.get('floor_id'),
                discipline=parsed_query.filters.get('discipline'),
                text_type=parsed_query.filters.get('text_type'),
                region_type=parsed_query.filters.get('region_type'),
                limit=limit
            )
            
            results = []
            for region in regions:
                if region.text_content:
                    metadata = {
                        'floor_id': region.page.floor_id,
                        'discipline': region.page.document.discipline,
                        'text_type': region.text_type,
                        'region_type': region.region_type,
                        'page_number': region.page.page_number,
                        'bbox': region.bbox,
                        'confidence': region.text_confidence
                    }
                    
                    results.append(RetrievalResult(
                        region_id=str(region.id),
                        text=region.text_content,
                        vector_score=0.0,
                        keyword_score=0.8,  # Default relevance for list results
                        combined_score=0.8,
                        metadata=metadata,
                        source="database",
                        highlights=[]
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"List query failed: {e}")
            return []
    
    def _handle_location_query(self, parsed_query: ParsedQuery, limit: int) -> List[RetrievalResult]:
        """Handle location queries with emphasis on spatial information."""
        # For location queries, prioritize results with good spatial metadata
        search_text = " ".join(parsed_query.search_terms)
        
        # Perform hybrid search but boost results with location information
        results = self._handle_search_query(parsed_query, limit * 2, 0.05)
        
        # Boost scores for results that likely contain location information
        location_keywords = ['room', 'floor', 'level', 'area', 'zone', 'section', 'grid', 'elevation']
        
        for result in results:
            text_lower = result.text.lower()
            location_boost = 0.0
            
            for keyword in location_keywords:
                if keyword in text_lower:
                    location_boost += 0.1
            
            # Apply boost
            result.combined_score = min(1.0, result.combined_score + location_boost)
        
        # Re-sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results[:limit]
    
    def _handle_filtered_browse(self, parsed_query: ParsedQuery, limit: int) -> List[RetrievalResult]:
        """Handle queries with filters but no search terms."""
        return self._handle_list_query(parsed_query, limit)
    
    def _convert_vector_results(self, vector_hits: List[Tuple[str, float, Dict[str, Any]]]) -> List[RetrievalResult]:
        """Convert vector search results to RetrievalResult objects."""
        results = []
        
        for region_id, score, metadata in vector_hits:
            # Get text content from database
            try:
                with self.database_manager.get_session() as session:
                    region = session.query(self.database_manager.Region).filter(
                        self.database_manager.Region.id == region_id
                    ).first()
                    
                    if region and region.text_content:
                        result_metadata = {
                            'floor_id': region.page.floor_id,
                            'discipline': region.page.document.discipline,
                            'text_type': region.text_type,
                            'region_type': region.region_type,
                            'page_number': region.page.page_number,
                            'bbox': region.bbox,
                            'confidence': region.text_confidence
                        }
                        result_metadata.update(metadata)
                        
                        results.append(RetrievalResult(
                            region_id=region_id,
                            text=region.text_content,
                            vector_score=score,
                            keyword_score=0.0,
                            combined_score=score * self.vector_weight,
                            metadata=result_metadata,
                            source="vector",
                            highlights=[]
                        ))
            except Exception as e:
                logger.warning(f"Failed to get region {region_id}: {e}")
                continue
        
        return results
    
    def _convert_keyword_results(self, keyword_hits: List[SearchResult]) -> List[RetrievalResult]:
        """Convert keyword search results to RetrievalResult objects."""
        results = []
        
        for hit in keyword_hits:
            results.append(RetrievalResult(
                region_id=hit.region_id,
                text=hit.text,
                vector_score=0.0,
                keyword_score=hit.score,
                combined_score=hit.score * self.keyword_weight,
                metadata=hit.metadata,
                source="keyword",
                highlights=hit.highlights
            ))
        
        return results
    
    def _combine_results(self, vector_results: List[RetrievalResult], 
                        keyword_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Combine vector and keyword results with score fusion."""
        # Create a map of region_id to results
        result_map = {}
        
        # Add vector results
        for result in vector_results:
            result_map[result.region_id] = result
        
        # Merge keyword results
        for result in keyword_results:
            if result.region_id in result_map:
                # Combine scores
                existing = result_map[result.region_id]
                existing.keyword_score = result.keyword_score
                existing.combined_score = (
                    existing.vector_score * self.vector_weight +
                    result.keyword_score * self.keyword_weight
                )
                existing.source = "hybrid"
                
                # Merge highlights
                if result.highlights:
                    existing.highlights.extend(result.highlights)
            else:
                # Add new keyword-only result
                result_map[result.region_id] = result
        
        # Convert back to list and sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return combined_results
    
    def search_patterns(self, pattern_type: str, 
                       filters: Optional[Dict[str, Any]] = None,
                       limit: int = 50) -> List[RetrievalResult]:
        """
        Search for specific AEC patterns (CSI codes, dimensions, etc.).
        
        Args:
            pattern_type: Type of pattern to search for
            filters: Optional metadata filters
            limit: Maximum number of results
            
        Returns:
            List of retrieval results
        """
        try:
            keyword_hits = self.keyword_index.search_patterns(
                pattern_type, filters, limit
            )
            
            results = []
            for hit in keyword_hits:
                results.append(RetrievalResult(
                    region_id=hit.region_id,
                    text=hit.text,
                    vector_score=0.0,
                    keyword_score=hit.score,
                    combined_score=hit.score,
                    metadata=hit.metadata,
                    source="pattern",
                    highlights=hit.highlights
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Pattern search failed: {e}")
            return []
    
    def get_similar_regions(self, region_id: str, limit: int = 10) -> List[RetrievalResult]:
        """
        Find regions similar to a given region.
        
        Args:
            region_id: ID of the reference region
            limit: Maximum number of results
            
        Returns:
            List of similar regions
        """
        try:
            # Get the reference region
            with self.database_manager.get_session() as session:
                region = session.query(self.database_manager.Region).filter(
                    self.database_manager.Region.id == region_id
                ).first()
                
                if not region or not region.text_content:
                    return []
                
                # Use the region's text for similarity search
                vector_hits = self.vector_store.search(
                    region.text_content,
                    k=limit + 1,  # +1 to exclude the original
                    filters={}
                )
                
                # Convert and filter out the original region
                results = self._convert_vector_results(vector_hits)
                results = [r for r in results if r.region_id != region_id]
                
                return results[:limit]
                
        except Exception as e:
            logger.error(f"Similar regions search failed: {e}")
            return []
    
    def get_context_around_region(self, region_id: str, 
                                 context_radius: int = 2) -> List[RetrievalResult]:
        """
        Get regions around a specific region for context.
        
        Args:
            region_id: ID of the center region
            context_radius: Number of nearby regions to include
            
        Returns:
            List of contextual regions
        """
        try:
            with self.database_manager.get_session() as session:
                # Get the center region
                center_region = session.query(self.database_manager.Region).filter(
                    self.database_manager.Region.id == region_id
                ).first()
                
                if not center_region:
                    return []
                
                # Get regions on the same page
                page_regions = session.query(self.database_manager.Region).filter(
                    self.database_manager.Region.page_id == center_region.page_id,
                    self.database_manager.Region.text_content.isnot(None)
                ).all()
                
                # Calculate spatial distances and sort
                context_regions = []
                center_x = (center_region.bbox_x1 + center_region.bbox_x2) / 2
                center_y = (center_region.bbox_y1 + center_region.bbox_y2) / 2
                
                for region in page_regions:
                    if region.id == center_region.id:
                        continue
                    
                    region_x = (region.bbox_x1 + region.bbox_x2) / 2
                    region_y = (region.bbox_y1 + region.bbox_y2) / 2
                    
                    distance = ((region_x - center_x) ** 2 + (region_y - center_y) ** 2) ** 0.5
                    
                    metadata = {
                        'floor_id': region.page.floor_id,
                        'discipline': region.page.document.discipline,
                        'text_type': region.text_type,
                        'region_type': region.region_type,
                        'page_number': region.page.page_number,
                        'bbox': region.bbox,
                        'distance': distance
                    }
                    
                    context_regions.append((distance, RetrievalResult(
                        region_id=str(region.id),
                        text=region.text_content,
                        vector_score=0.0,
                        keyword_score=1.0 / (1.0 + distance / 1000),  # Distance-based score
                        combined_score=1.0 / (1.0 + distance / 1000),
                        metadata=metadata,
                        source="context",
                        highlights=[]
                    )))
                
                # Sort by distance and return closest regions
                context_regions.sort(key=lambda x: x[0])
                
                return [result for _, result in context_regions[:context_radius * 2]]
                
        except Exception as e:
            logger.error(f"Context search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            'vector_store_stats': self.vector_store.get_stats() if self.vector_store else {},
            'keyword_index_stats': self.keyword_index.get_stats() if self.keyword_index else {},
            'database_stats': self.database_manager.get_database_stats(),
            'weights': {
                'vector_weight': self.vector_weight,
                'keyword_weight': self.keyword_weight
            }
        }

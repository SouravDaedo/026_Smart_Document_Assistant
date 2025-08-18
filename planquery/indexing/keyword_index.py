"""
Keyword search index using BM25 and regex for fast text-based queries.
Handles CSI codes, dimensions, and other structured text patterns.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import pickle
from collections import defaultdict, Counter
import math
from dataclasses import dataclass
from loguru import logger

try:
    from whoosh import index
    from whoosh.fields import Schema, TEXT, ID, KEYWORD, NUMERIC
    from whoosh.qparser import QueryParser, MultifieldParser
    from whoosh.analysis import StandardAnalyzer, RegexTokenizer
    from whoosh.query import And, Or, Term, Regex
    WHOOSH_AVAILABLE = True
except ImportError:
    WHOOSH_AVAILABLE = False
    logger.warning("Whoosh not available, using simple keyword search")


@dataclass
class SearchResult:
    """Search result with relevance scoring."""
    region_id: str
    text: str
    score: float
    highlights: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "text": self.text,
            "score": self.score,
            "highlights": self.highlights,
            "metadata": self.metadata
        }


class KeywordIndex:
    """Keyword search index with BM25 scoring and regex support."""
    
    def __init__(self, index_dir: Optional[str] = None):
        """
        Initialize keyword index.
        
        Args:
            index_dir: Directory to store Whoosh index
        """
        self.index_dir = Path(index_dir) if index_dir else None
        self.use_whoosh = WHOOSH_AVAILABLE and index_dir
        
        # Whoosh components
        self.schema = None
        self.index = None
        self.writer = None
        
        # Fallback components
        self.documents = {}  # region_id -> document data
        self.inverted_index = defaultdict(set)  # term -> set of region_ids
        self.term_frequencies = defaultdict(Counter)  # region_id -> Counter of terms
        self.document_lengths = {}  # region_id -> document length
        
        # Common AEC patterns for regex search
        self._compile_aec_patterns()
        
        if self.use_whoosh:
            self._initialize_whoosh()
        else:
            logger.info("Using fallback keyword search")
    
    def _compile_aec_patterns(self):
        """Compile regex patterns for AEC-specific searches."""
        self.aec_patterns = {
            'csi_codes': re.compile(r'\b\d{2}\s*\d{2}\s*\d{2}\b|\b\d{2}-\d{2}-\d{2}\b'),
            'dimensions': re.compile(r'\d+[\'"]\s*-?\s*\d*[\'"]*|\d+\.\d+\s*(?:mm|cm|m|ft|in)|\d+\s*x\s*\d+'),
            'room_numbers': re.compile(r'\b(?:ROOM|RM)\s*\d+\b', re.IGNORECASE),
            'equipment_tags': re.compile(r'\b[A-Z]{2,4}-\d+\b'),
            'detail_refs': re.compile(r'\b\d+/[A-Z]\d+\b'),
            'grid_lines': re.compile(r'\b[A-Z]\.\d+\b|\b[A-Z]\d*\b'),
            'elevations': re.compile(r'\bEL\.?\s*\d+[\'"]\s*-?\s*\d*[\'"]*\b', re.IGNORECASE),
            'levels': re.compile(r'\b(?:LEVEL|LVL|L)\s*\d+\b', re.IGNORECASE),
        }
    
    def _initialize_whoosh(self):
        """Initialize Whoosh search index."""
        try:
            # Define schema
            self.schema = Schema(
                region_id=ID(stored=True, unique=True),
                text=TEXT(stored=True, analyzer=StandardAnalyzer()),
                text_type=KEYWORD(stored=True),
                region_type=KEYWORD(stored=True),
                floor_id=KEYWORD(stored=True),
                discipline=KEYWORD(stored=True),
                page_number=NUMERIC(stored=True),
                confidence=NUMERIC(stored=True),
                bbox_x1=NUMERIC(stored=True),
                bbox_y1=NUMERIC(stored=True),
                bbox_x2=NUMERIC(stored=True),
                bbox_y2=NUMERIC(stored=True),
            )
            
            # Create or open index
            if self.index_dir.exists():
                self.index = index.open_dir(str(self.index_dir))
                logger.info(f"Opened existing Whoosh index: {self.index_dir}")
            else:
                self.index_dir.mkdir(parents=True, exist_ok=True)
                self.index = index.create_in(str(self.index_dir), self.schema)
                logger.info(f"Created new Whoosh index: {self.index_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whoosh: {e}")
            self.use_whoosh = False
    
    def add_document(self, region_id: str, text: str, 
                    text_type: Optional[str] = None,
                    region_type: Optional[str] = None,
                    floor_id: Optional[str] = None,
                    discipline: Optional[str] = None,
                    page_number: Optional[int] = None,
                    confidence: Optional[float] = None,
                    bbox: Optional[Dict[str, int]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add document to keyword index.
        
        Args:
            region_id: Unique identifier for the region
            text: Text content to index
            text_type: AEC text type classification
            region_type: Layout region type
            floor_id: Floor identifier
            discipline: Document discipline (A/M/E/S/C)
            page_number: Page number
            confidence: OCR confidence
            bbox: Bounding box coordinates
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        if not text.strip():
            return False
        
        if self.use_whoosh:
            return self._add_to_whoosh(
                region_id, text, text_type, region_type, 
                floor_id, discipline, page_number, confidence, bbox, metadata
            )
        else:
            return self._add_to_fallback(
                region_id, text, text_type, region_type,
                floor_id, discipline, page_number, confidence, bbox, metadata
            )
    
    def _add_to_whoosh(self, region_id: str, text: str, 
                      text_type: Optional[str] = None,
                      region_type: Optional[str] = None,
                      floor_id: Optional[str] = None,
                      discipline: Optional[str] = None,
                      page_number: Optional[int] = None,
                      confidence: Optional[float] = None,
                      bbox: Optional[Dict[str, int]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add document to Whoosh index."""
        try:
            writer = self.index.writer()
            
            doc_data = {
                'region_id': region_id,
                'text': text,
                'text_type': text_type or '',
                'region_type': region_type or '',
                'floor_id': floor_id or '',
                'discipline': discipline or '',
                'page_number': page_number or 0,
                'confidence': confidence or 0.0,
            }
            
            if bbox:
                doc_data.update({
                    'bbox_x1': bbox.get('x1', 0),
                    'bbox_y1': bbox.get('y1', 0),
                    'bbox_x2': bbox.get('x2', 0),
                    'bbox_y2': bbox.get('y2', 0),
                })
            
            writer.add_document(**doc_data)
            writer.commit()
            
            return True
        except Exception as e:
            logger.error(f"Failed to add document to Whoosh: {e}")
            return False
    
    def _add_to_fallback(self, region_id: str, text: str,
                        text_type: Optional[str] = None,
                        region_type: Optional[str] = None,
                        floor_id: Optional[str] = None,
                        discipline: Optional[str] = None,
                        page_number: Optional[int] = None,
                        confidence: Optional[float] = None,
                        bbox: Optional[Dict[str, int]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add document to fallback index."""
        try:
            # Store document data
            self.documents[region_id] = {
                'text': text,
                'text_type': text_type,
                'region_type': region_type,
                'floor_id': floor_id,
                'discipline': discipline,
                'page_number': page_number,
                'confidence': confidence,
                'bbox': bbox,
                'metadata': metadata or {}
            }
            
            # Tokenize and build inverted index
            terms = self._tokenize(text)
            self.term_frequencies[region_id] = Counter(terms)
            self.document_lengths[region_id] = len(terms)
            
            # Update inverted index
            for term in set(terms):
                self.inverted_index[term].add(region_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add document to fallback index: {e}")
            return False
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for indexing."""
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        # Add special handling for AEC-specific patterns
        # Keep original case for certain patterns
        original_tokens = []
        for pattern_name, pattern in self.aec_patterns.items():
            matches = pattern.findall(text)
            original_tokens.extend(matches)
        
        return tokens + original_tokens
    
    def search(self, query: str, 
              filters: Optional[Dict[str, Any]] = None,
              limit: int = 50,
              use_regex: bool = False) -> List[SearchResult]:
        """
        Search the keyword index.
        
        Args:
            query: Search query
            filters: Optional filters (floor_id, discipline, text_type, etc.)
            limit: Maximum number of results
            use_regex: Whether to use regex search
            
        Returns:
            List of search results
        """
        if self.use_whoosh:
            return self._search_whoosh(query, filters, limit, use_regex)
        else:
            return self._search_fallback(query, filters, limit, use_regex)
    
    def _search_whoosh(self, query: str, filters: Optional[Dict[str, Any]], 
                      limit: int, use_regex: bool) -> List[SearchResult]:
        """Search using Whoosh."""
        try:
            with self.index.searcher() as searcher:
                # Build query
                if use_regex:
                    # Use regex query
                    whoosh_query = Regex("text", query)
                else:
                    # Use text query with multiple fields
                    parser = MultifieldParser(
                        ["text", "text_type", "region_type"], 
                        self.index.schema
                    )
                    whoosh_query = parser.parse(query)
                
                # Add filters
                if filters:
                    filter_queries = []
                    for field, value in filters.items():
                        if field in self.schema:
                            filter_queries.append(Term(field, value))
                    
                    if filter_queries:
                        if len(filter_queries) == 1:
                            whoosh_query = And([whoosh_query, filter_queries[0]])
                        else:
                            whoosh_query = And([whoosh_query] + filter_queries)
                
                # Search
                results = searcher.search(whoosh_query, limit=limit)
                
                # Convert to SearchResult objects
                search_results = []
                for hit in results:
                    highlights = []
                    if hasattr(hit, 'highlights'):
                        highlights = [hit.highlights("text")]
                    
                    metadata = {
                        'text_type': hit.get('text_type', ''),
                        'region_type': hit.get('region_type', ''),
                        'floor_id': hit.get('floor_id', ''),
                        'discipline': hit.get('discipline', ''),
                        'page_number': hit.get('page_number', 0),
                        'confidence': hit.get('confidence', 0.0),
                        'bbox': {
                            'x1': hit.get('bbox_x1', 0),
                            'y1': hit.get('bbox_y1', 0),
                            'x2': hit.get('bbox_x2', 0),
                            'y2': hit.get('bbox_y2', 0),
                        }
                    }
                    
                    search_results.append(SearchResult(
                        region_id=hit['region_id'],
                        text=hit['text'],
                        score=hit.score,
                        highlights=highlights,
                        metadata=metadata
                    ))
                
                return search_results
                
        except Exception as e:
            logger.error(f"Whoosh search failed: {e}")
            return []
    
    def _search_fallback(self, query: str, filters: Optional[Dict[str, Any]], 
                        limit: int, use_regex: bool) -> List[SearchResult]:
        """Search using fallback implementation."""
        try:
            if use_regex:
                return self._regex_search(query, filters, limit)
            else:
                return self._bm25_search(query, filters, limit)
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    def _regex_search(self, pattern: str, filters: Optional[Dict[str, Any]], 
                     limit: int) -> List[SearchResult]:
        """Perform regex search."""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            results = []
            
            for region_id, doc_data in self.documents.items():
                # Apply filters
                if filters and not self._matches_filters(doc_data, filters):
                    continue
                
                text = doc_data['text']
                matches = list(regex.finditer(text))
                
                if matches:
                    # Calculate score based on number of matches
                    score = len(matches) / len(text.split())
                    
                    # Extract highlights
                    highlights = [match.group() for match in matches[:5]]  # Limit highlights
                    
                    results.append(SearchResult(
                        region_id=region_id,
                        text=text,
                        score=score,
                        highlights=highlights,
                        metadata=self._extract_metadata(doc_data)
                    ))
            
            # Sort by score and limit
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Regex search failed: {e}")
            return []
    
    def _bm25_search(self, query: str, filters: Optional[Dict[str, Any]], 
                    limit: int) -> List[SearchResult]:
        """Perform BM25 search."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        # BM25 parameters
        k1 = 1.2
        b = 0.75
        
        # Calculate average document length
        if not self.document_lengths:
            return []
        
        avg_doc_length = sum(self.document_lengths.values()) / len(self.document_lengths)
        
        # Calculate BM25 scores
        scores = {}
        
        for region_id, doc_data in self.documents.items():
            # Apply filters
            if filters and not self._matches_filters(doc_data, filters):
                continue
            
            if region_id not in self.term_frequencies:
                continue
            
            doc_length = self.document_lengths[region_id]
            tf_counter = self.term_frequencies[region_id]
            
            score = 0.0
            matched_terms = []
            
            for term in query_terms:
                if term in tf_counter:
                    # Term frequency in document
                    tf = tf_counter[term]
                    
                    # Document frequency (number of documents containing term)
                    df = len(self.inverted_index[term])
                    
                    # Inverse document frequency
                    idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5))
                    
                    # BM25 score component
                    score_component = idf * (tf * (k1 + 1)) / (
                        tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                    )
                    
                    score += score_component
                    matched_terms.append(term)
            
            if score > 0:
                scores[region_id] = {
                    'score': score,
                    'matched_terms': matched_terms
                }
        
        # Convert to SearchResult objects and sort
        results = []
        for region_id, score_data in scores.items():
            doc_data = self.documents[region_id]
            
            results.append(SearchResult(
                region_id=region_id,
                text=doc_data['text'],
                score=score_data['score'],
                highlights=score_data['matched_terms'],
                metadata=self._extract_metadata(doc_data)
            ))
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def _matches_filters(self, doc_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document matches filters."""
        for key, value in filters.items():
            doc_value = doc_data.get(key)
            if doc_value != value:
                return False
        return True
    
    def _extract_metadata(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from document data."""
        return {
            'text_type': doc_data.get('text_type', ''),
            'region_type': doc_data.get('region_type', ''),
            'floor_id': doc_data.get('floor_id', ''),
            'discipline': doc_data.get('discipline', ''),
            'page_number': doc_data.get('page_number', 0),
            'confidence': doc_data.get('confidence', 0.0),
            'bbox': doc_data.get('bbox', {}),
        }
    
    def search_patterns(self, pattern_type: str, 
                       filters: Optional[Dict[str, Any]] = None,
                       limit: int = 50) -> List[SearchResult]:
        """
        Search for specific AEC patterns.
        
        Args:
            pattern_type: Type of pattern to search for
            filters: Optional filters
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        if pattern_type not in self.aec_patterns:
            logger.warning(f"Unknown pattern type: {pattern_type}")
            return []
        
        pattern = self.aec_patterns[pattern_type]
        return self._regex_search(pattern.pattern, filters, limit)
    
    def get_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """Get search suggestions based on partial query."""
        if not partial_query.strip():
            return []
        
        partial_lower = partial_query.lower()
        suggestions = set()
        
        # Find terms that start with the partial query
        for term in self.inverted_index.keys():
            if term.lower().startswith(partial_lower):
                suggestions.add(term)
                if len(suggestions) >= limit:
                    break
        
        return sorted(list(suggestions))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.use_whoosh:
            try:
                with self.index.searcher() as searcher:
                    doc_count = searcher.doc_count()
                    return {
                        'backend': 'whoosh',
                        'total_documents': doc_count,
                        'index_dir': str(self.index_dir)
                    }
            except:
                return {'backend': 'whoosh', 'total_documents': 0}
        else:
            return {
                'backend': 'fallback',
                'total_documents': len(self.documents),
                'total_terms': len(self.inverted_index),
                'avg_doc_length': sum(self.document_lengths.values()) / len(self.document_lengths) if self.document_lengths else 0
            }
    
    def optimize(self):
        """Optimize the index."""
        if self.use_whoosh:
            try:
                writer = self.index.writer()
                writer.commit(optimize=True)
                logger.info("Whoosh index optimized")
            except Exception as e:
                logger.error(f"Failed to optimize Whoosh index: {e}")
    
    def clear(self):
        """Clear the index."""
        if self.use_whoosh:
            try:
                writer = self.index.writer()
                writer.commit(mergetype=index.CLEAR)
                logger.info("Whoosh index cleared")
            except Exception as e:
                logger.error(f"Failed to clear Whoosh index: {e}")
        else:
            self.documents.clear()
            self.inverted_index.clear()
            self.term_frequencies.clear()
            self.document_lengths.clear()
            logger.info("Fallback index cleared")
    
    def close(self):
        """Close the index."""
        if self.use_whoosh and self.index:
            self.index.close()
            logger.info("Whoosh index closed")

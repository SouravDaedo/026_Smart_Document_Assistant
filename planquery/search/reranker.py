"""
Result reranking module for improving search result quality.
Uses multiple signals to reorder results for better relevance.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math
from loguru import logger

from .retriever import RetrievalResult
from .query_parser import ParsedQuery, QueryType


@dataclass
class RerankingSignal:
    """Individual reranking signal with weight."""
    name: str
    score: float
    weight: float
    explanation: str


class ResultReranker:
    """Reranks search results using multiple relevance signals."""
    
    def __init__(self):
        """Initialize reranker with default weights."""
        self.signal_weights = {
            'query_match': 0.3,      # How well text matches query terms
            'text_type_match': 0.2,  # Relevance of text type to query
            'floor_relevance': 0.15, # Floor-specific relevance
            'confidence_boost': 0.1, # OCR/detection confidence
            'spatial_context': 0.1,  # Spatial relationship signals
            'freshness': 0.05,       # Recency of information
            'completeness': 0.1      # Completeness of metadata
        }
    
    def rerank(self, results: List[RetrievalResult], 
              parsed_query: ParsedQuery,
              max_results: Optional[int] = None) -> List[RetrievalResult]:
        """
        Rerank search results based on multiple signals.
        
        Args:
            results: List of retrieval results to rerank
            parsed_query: Original parsed query for context
            max_results: Maximum number of results to return
            
        Returns:
            Reranked list of results
        """
        if not results:
            return results
        
        logger.info(f"Reranking {len(results)} results")
        
        # Calculate reranking signals for each result
        reranked_results = []
        
        for result in results:
            signals = self._calculate_signals(result, parsed_query)
            new_score = self._combine_signals(signals, result.combined_score)
            
            # Create new result with updated score
            reranked_result = RetrievalResult(
                region_id=result.region_id,
                text=result.text,
                vector_score=result.vector_score,
                keyword_score=result.keyword_score,
                combined_score=new_score,
                metadata=result.metadata.copy(),
                source=result.source,
                highlights=result.highlights.copy()
            )
            
            # Add reranking signals to metadata
            reranked_result.metadata['reranking_signals'] = [
                {'name': s.name, 'score': s.score, 'weight': s.weight, 'explanation': s.explanation}
                for s in signals
            ]
            
            reranked_results.append(reranked_result)
        
        # Sort by new combined score
        reranked_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Apply result limit
        if max_results:
            reranked_results = reranked_results[:max_results]
        
        logger.info(f"Reranking complete, returning {len(reranked_results)} results")
        return reranked_results
    
    def _calculate_signals(self, result: RetrievalResult, 
                          parsed_query: ParsedQuery) -> List[RerankingSignal]:
        """Calculate all reranking signals for a result."""
        signals = []
        
        # Query match signal
        query_match_score = self._calculate_query_match(result, parsed_query)
        signals.append(RerankingSignal(
            name='query_match',
            score=query_match_score,
            weight=self.signal_weights['query_match'],
            explanation=f"Query term overlap: {query_match_score:.2f}"
        ))
        
        # Text type relevance
        text_type_score = self._calculate_text_type_relevance(result, parsed_query)
        signals.append(RerankingSignal(
            name='text_type_match',
            score=text_type_score,
            weight=self.signal_weights['text_type_match'],
            explanation=f"Text type relevance: {text_type_score:.2f}"
        ))
        
        # Floor relevance
        floor_score = self._calculate_floor_relevance(result, parsed_query)
        signals.append(RerankingSignal(
            name='floor_relevance',
            score=floor_score,
            weight=self.signal_weights['floor_relevance'],
            explanation=f"Floor match: {floor_score:.2f}"
        ))
        
        # Confidence boost
        confidence_score = self._calculate_confidence_boost(result)
        signals.append(RerankingSignal(
            name='confidence_boost',
            score=confidence_score,
            weight=self.signal_weights['confidence_boost'],
            explanation=f"OCR confidence: {confidence_score:.2f}"
        ))
        
        # Spatial context
        spatial_score = self._calculate_spatial_context(result, parsed_query)
        signals.append(RerankingSignal(
            name='spatial_context',
            score=spatial_score,
            weight=self.signal_weights['spatial_context'],
            explanation=f"Spatial relevance: {spatial_score:.2f}"
        ))
        
        # Freshness (placeholder - would need revision dates)
        freshness_score = self._calculate_freshness(result)
        signals.append(RerankingSignal(
            name='freshness',
            score=freshness_score,
            weight=self.signal_weights['freshness'],
            explanation=f"Information freshness: {freshness_score:.2f}"
        ))
        
        # Completeness
        completeness_score = self._calculate_completeness(result)
        signals.append(RerankingSignal(
            name='completeness',
            score=completeness_score,
            weight=self.signal_weights['completeness'],
            explanation=f"Metadata completeness: {completeness_score:.2f}"
        ))
        
        return signals
    
    def _calculate_query_match(self, result: RetrievalResult, 
                              parsed_query: ParsedQuery) -> float:
        """Calculate how well the result matches query terms."""
        if not parsed_query.search_terms:
            return 0.5  # Neutral score for queries without search terms
        
        text_lower = result.text.lower()
        matched_terms = 0
        
        for term in parsed_query.search_terms:
            if term.lower() in text_lower:
                matched_terms += 1
        
        # Calculate match ratio
        match_ratio = matched_terms / len(parsed_query.search_terms)
        
        # Boost for exact phrase matches
        full_query = " ".join(parsed_query.search_terms)
        if full_query.lower() in text_lower:
            match_ratio = min(1.0, match_ratio + 0.3)
        
        return match_ratio
    
    def _calculate_text_type_relevance(self, result: RetrievalResult, 
                                      parsed_query: ParsedQuery) -> float:
        """Calculate relevance based on text type matching."""
        result_text_type = result.metadata.get('text_type', '')
        
        # If query specifies text types, check for match
        if parsed_query.text_types:
            if result_text_type in parsed_query.text_types:
                return 1.0
            else:
                return 0.3  # Partial penalty for type mismatch
        
        # Query type based scoring
        if parsed_query.query_type == QueryType.LOCATION:
            # Boost location-relevant text types
            location_types = ['RoomLabel', 'Elevation', 'GridLine', 'DrawingTitle']
            if result_text_type in location_types:
                return 0.9
        elif parsed_query.query_type == QueryType.COUNT:
            # Boost structured data types for counting
            count_types = ['ScheduleText', 'Equipment', 'RoomLabel']
            if result_text_type in count_types:
                return 0.8
        
        return 0.5  # Neutral score
    
    def _calculate_floor_relevance(self, result: RetrievalResult, 
                                  parsed_query: ParsedQuery) -> float:
        """Calculate relevance based on floor matching."""
        result_floor = result.metadata.get('floor_id', '')
        
        if not parsed_query.floor_ids:
            return 0.5  # Neutral if no floor specified
        
        if result_floor in parsed_query.floor_ids:
            return 1.0  # Perfect match
        
        # Check for related floors (e.g., adjacent levels)
        for query_floor in parsed_query.floor_ids:
            if self._are_floors_related(result_floor, query_floor):
                return 0.7  # Partial match for related floors
        
        return 0.2  # Penalty for unrelated floors
    
    def _are_floors_related(self, floor1: str, floor2: str) -> bool:
        """Check if two floors are related (adjacent levels)."""
        try:
            # Extract numeric parts for comparison
            num1 = self._extract_floor_number(floor1)
            num2 = self._extract_floor_number(floor2)
            
            if num1 is not None and num2 is not None:
                return abs(num1 - num2) <= 1  # Adjacent floors
            
        except:
            pass
        
        return False
    
    def _extract_floor_number(self, floor_id: str) -> Optional[int]:
        """Extract numeric floor number from floor ID."""
        if not floor_id:
            return None
        
        # Handle formats like L1, L2, LB1, etc.
        if floor_id.startswith('L'):
            if floor_id.startswith('LB'):
                # Basement floors (negative)
                try:
                    return -int(floor_id[2:])
                except:
                    return -1
            elif floor_id[1:].isdigit():
                return int(floor_id[1:])
        
        return None
    
    def _calculate_confidence_boost(self, result: RetrievalResult) -> float:
        """Calculate boost based on OCR and detection confidence."""
        ocr_confidence = result.metadata.get('confidence', 0.0)
        
        # Normalize confidence to 0-1 range and apply sigmoid
        normalized_confidence = max(0.0, min(1.0, ocr_confidence))
        
        # Apply sigmoid function to make confidence differences more pronounced
        sigmoid_confidence = 1 / (1 + math.exp(-5 * (normalized_confidence - 0.5)))
        
        return sigmoid_confidence
    
    def _calculate_spatial_context(self, result: RetrievalResult, 
                                  parsed_query: ParsedQuery) -> float:
        """Calculate spatial context relevance."""
        bbox = result.metadata.get('bbox', {})
        
        if not bbox:
            return 0.3  # Penalty for missing spatial info
        
        # Calculate region size (larger regions might be more important)
        width = bbox.get('x2', 0) - bbox.get('x1', 0)
        height = bbox.get('y2', 0) - bbox.get('y1', 0)
        area = width * height
        
        if area <= 0:
            return 0.3
        
        # Normalize area (assuming typical page size)
        normalized_area = min(1.0, area / 100000)  # Adjust based on typical areas
        
        # Position-based scoring (title blocks often in corners, important text in center)
        x_center = (bbox.get('x1', 0) + bbox.get('x2', 0)) / 2
        y_center = (bbox.get('y1', 0) + bbox.get('y2', 0)) / 2
        
        # This is a simplified scoring - in practice, you'd want page dimensions
        position_score = 0.5  # Neutral position score
        
        # Combine area and position
        spatial_score = (normalized_area * 0.7 + position_score * 0.3)
        
        return min(1.0, spatial_score)
    
    def _calculate_freshness(self, result: RetrievalResult) -> float:
        """Calculate freshness score (placeholder implementation)."""
        # In a real implementation, this would check revision dates, modification times, etc.
        # For now, return neutral score
        return 0.5
    
    def _calculate_completeness(self, result: RetrievalResult) -> float:
        """Calculate completeness based on available metadata."""
        metadata = result.metadata
        
        # Check for presence of key metadata fields
        key_fields = ['floor_id', 'discipline', 'text_type', 'bbox', 'page_number']
        present_fields = sum(1 for field in key_fields if metadata.get(field))
        
        completeness = present_fields / len(key_fields)
        
        # Boost for additional useful metadata
        bonus_fields = ['confidence', 'region_type', 'sheet_number']
        bonus_score = sum(0.1 for field in bonus_fields if metadata.get(field))
        
        return min(1.0, completeness + bonus_score)
    
    def _combine_signals(self, signals: List[RerankingSignal], 
                        original_score: float) -> float:
        """Combine reranking signals with original score."""
        # Calculate weighted sum of signals
        signal_sum = sum(signal.score * signal.weight for signal in signals)
        total_weight = sum(signal.weight for signal in signals)
        
        if total_weight == 0:
            return original_score
        
        # Normalize signal score
        normalized_signal_score = signal_sum / total_weight
        
        # Combine with original score (70% original, 30% signals)
        combined_score = 0.7 * original_score + 0.3 * normalized_signal_score
        
        return min(1.0, max(0.0, combined_score))
    
    def explain_ranking(self, result: RetrievalResult) -> Dict[str, Any]:
        """Provide explanation for why a result was ranked as it was."""
        explanation = {
            'region_id': result.region_id,
            'final_score': result.combined_score,
            'original_scores': {
                'vector_score': result.vector_score,
                'keyword_score': result.keyword_score
            },
            'reranking_signals': result.metadata.get('reranking_signals', []),
            'summary': self._generate_ranking_summary(result)
        }
        
        return explanation
    
    def _generate_ranking_summary(self, result: RetrievalResult) -> str:
        """Generate human-readable ranking summary."""
        signals = result.metadata.get('reranking_signals', [])
        
        if not signals:
            return "Ranked based on original search scores only."
        
        # Find top contributing signals
        top_signals = sorted(signals, key=lambda x: x['score'] * x['weight'], reverse=True)[:3]
        
        summary_parts = []
        for signal in top_signals:
            contribution = signal['score'] * signal['weight']
            if contribution > 0.1:  # Only mention significant contributors
                summary_parts.append(f"{signal['name']} ({signal['explanation']})")
        
        if summary_parts:
            return f"Ranked highly due to: {', '.join(summary_parts)}"
        else:
            return "Standard ranking based on search relevance."
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update signal weights for tuning."""
        for signal_name, weight in new_weights.items():
            if signal_name in self.signal_weights:
                self.signal_weights[signal_name] = weight
        
        logger.info(f"Updated reranking weights: {self.signal_weights}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            'signal_weights': self.signal_weights.copy(),
            'total_signals': len(self.signal_weights)
        }

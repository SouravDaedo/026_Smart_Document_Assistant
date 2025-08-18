"""
Query parser for interpreting user intent and extracting search filters.
Handles natural language queries and converts them to structured search parameters.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class QueryType(Enum):
    """Types of queries the system can handle."""
    SEARCH = "search"           # General text search
    LOCATION = "location"       # "Where is..." queries
    FILTER = "filter"          # Filtered search with specific criteria
    COMPARISON = "comparison"   # Compare items across floors/disciplines
    LIST = "list"              # List all items of a type
    COUNT = "count"            # Count items


class SearchScope(Enum):
    """Scope of search."""
    GLOBAL = "global"          # Search across all documents
    FLOOR = "floor"            # Search within specific floor(s)
    DISCIPLINE = "discipline"  # Search within specific discipline(s)
    DOCUMENT = "document"      # Search within specific document(s)


@dataclass
class ParsedQuery:
    """Parsed query with extracted intent and filters."""
    original_query: str
    query_type: QueryType
    search_scope: SearchScope
    search_terms: List[str]
    filters: Dict[str, Any]
    floor_ids: List[str]
    disciplines: List[str]
    text_types: List[str]
    region_types: List[str]
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "query_type": self.query_type.value,
            "search_scope": self.search_scope.value,
            "search_terms": self.search_terms,
            "filters": self.filters,
            "floor_ids": self.floor_ids,
            "disciplines": self.disciplines,
            "text_types": self.text_types,
            "region_types": self.region_types,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class QueryParser:
    """Parses natural language queries into structured search parameters."""
    
    def __init__(self):
        """Initialize query parser with patterns and mappings."""
        self._compile_patterns()
        self._setup_mappings()
    
    def _compile_patterns(self):
        """Compile regex patterns for query parsing."""
        self.patterns = {
            # Query type patterns
            'location_queries': [
                re.compile(r'\b(?:where|locate|find|show me)\b.*\b(?:is|are|can I find)\b', re.IGNORECASE),
                re.compile(r'\b(?:location of|position of)\b', re.IGNORECASE),
                re.compile(r'\b(?:which sheet|what sheet|what page)\b', re.IGNORECASE),
            ],
            
            'list_queries': [
                re.compile(r'\b(?:list|show all|give me all|what are all)\b', re.IGNORECASE),
                re.compile(r'\b(?:all the|every)\b.*\b(?:on|in)\b', re.IGNORECASE),
            ],
            
            'count_queries': [
                re.compile(r'\b(?:how many|count|number of)\b', re.IGNORECASE),
                re.compile(r'\b(?:total|sum)\b.*\b(?:of|are)\b', re.IGNORECASE),
            ],
            
            # Floor patterns
            'floor_patterns': [
                re.compile(r'\b(?:level|floor|lvl|l)\s*(\d+)\b', re.IGNORECASE),
                re.compile(r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s*(?:floor|level)\b', re.IGNORECASE),
                re.compile(r'\b(ground|basement|roof|penthouse|mezzanine|mechanical)\s*(?:floor|level)?\b', re.IGNORECASE),
                re.compile(r'\b(l\d+|lb\d+|lr|lp|lm|lmz)\b', re.IGNORECASE),
            ],
            
            # Discipline patterns
            'discipline_patterns': [
                re.compile(r'\b(architectural|arch)\b', re.IGNORECASE),
                re.compile(r'\b(mechanical|mech|hvac|plumbing|mep)\b', re.IGNORECASE),
                re.compile(r'\b(electrical|elec|power|lighting)\b', re.IGNORECASE),
                re.compile(r'\b(structural|struct|foundation|framing)\b', re.IGNORECASE),
                re.compile(r'\b(civil|site|grading|utility)\b', re.IGNORECASE),
            ],
            
            # Text type patterns
            'text_type_patterns': [
                re.compile(r'\b(notes|general notes|specifications|specs)\b', re.IGNORECASE),
                re.compile(r'\b(dimensions|measurements|sizes)\b', re.IGNORECASE),
                re.compile(r'\b(callouts|callout|references|refs)\b', re.IGNORECASE),
                re.compile(r'\b(schedules|schedule|tables|table)\b', re.IGNORECASE),
                re.compile(r'\b(equipment|tags|labels)\b', re.IGNORECASE),
                re.compile(r'\b(room\s*(?:names|labels|numbers))\b', re.IGNORECASE),
                re.compile(r'\b(title\s*block|drawing\s*title)\b', re.IGNORECASE),
                re.compile(r'\b(elevations|elevation\s*markers)\b', re.IGNORECASE),
                re.compile(r'\b(materials|material\s*labels)\b', re.IGNORECASE),
            ],
            
            # Specific AEC terms
            'aec_terms': [
                re.compile(r'\b(rcp|reflected\s*ceiling\s*plan)\b', re.IGNORECASE),
                re.compile(r'\b(floor\s*plan|plan\s*view)\b', re.IGNORECASE),
                re.compile(r'\b(section|elevation|detail)\b', re.IGNORECASE),
                re.compile(r'\b(stair|stairs|stairway|staircase)\b', re.IGNORECASE),
                re.compile(r'\b(elevator|lift)\b', re.IGNORECASE),
                re.compile(r'\b(toilet|restroom|bathroom)\b', re.IGNORECASE),
                re.compile(r'\b(office|conference|meeting)\b', re.IGNORECASE),
                re.compile(r'\b(corridor|hallway|lobby)\b', re.IGNORECASE),
                re.compile(r'\b(storage|closet|utility)\b', re.IGNORECASE),
            ],
            
            # CSI codes and specifications
            'csi_patterns': [
                re.compile(r'\b(\d{2}\s*\d{2}\s*\d{2}|\d{2}-\d{2}-\d{2})\b'),
                re.compile(r'\b(spec\s*(?:section)?\s*\d+)\b', re.IGNORECASE),
            ],
            
            # Equipment patterns
            'equipment_patterns': [
                re.compile(r'\b([A-Z]{2,4}-\d+)\b'),  # Equipment tags like AHU-1
                re.compile(r'\b(ahu|air\s*handling\s*unit)\b', re.IGNORECASE),
                re.compile(r'\b(fcu|fan\s*coil\s*unit)\b', re.IGNORECASE),
                re.compile(r'\b(vav|variable\s*air\s*volume)\b', re.IGNORECASE),
                re.compile(r'\b(pump|pumps)\b', re.IGNORECASE),
                re.compile(r'\b(panel|electrical\s*panel)\b', re.IGNORECASE),
            ],
        }
    
    def _setup_mappings(self):
        """Setup mappings for text normalization."""
        self.floor_word_to_number = {
            'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
            'sixth': '6', 'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10',
            'ground': '1', 'basement': 'B1', 'roof': 'R', 'penthouse': 'P',
            'mezzanine': 'MZ', 'mechanical': 'M'
        }
        
        self.discipline_mapping = {
            'architectural': 'A', 'arch': 'A',
            'mechanical': 'M', 'mech': 'M', 'hvac': 'M', 'plumbing': 'M', 'mep': 'M',
            'electrical': 'E', 'elec': 'E', 'power': 'E', 'lighting': 'E',
            'structural': 'S', 'struct': 'S', 'foundation': 'S', 'framing': 'S',
            'civil': 'C', 'site': 'C', 'grading': 'C', 'utility': 'C'
        }
        
        self.text_type_mapping = {
            'notes': 'GeneralNotes', 'general notes': 'GeneralNotes', 'specifications': 'GeneralNotes', 'specs': 'GeneralNotes',
            'dimensions': 'Dimensions', 'measurements': 'Dimensions', 'sizes': 'Dimensions',
            'callouts': 'Callout', 'callout': 'Callout', 'references': 'Callout', 'refs': 'Callout',
            'schedules': 'ScheduleText', 'schedule': 'ScheduleText', 'tables': 'ScheduleText', 'table': 'ScheduleText',
            'equipment': 'Equipment', 'tags': 'Equipment', 'labels': 'Equipment',
            'room names': 'RoomLabel', 'room labels': 'RoomLabel', 'room numbers': 'RoomLabel',
            'title block': 'TitleBlock', 'drawing title': 'DrawingTitle',
            'elevations': 'Elevation', 'elevation markers': 'Elevation',
            'materials': 'MaterialLabel', 'material labels': 'MaterialLabel'
        }
    
    def parse_query(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query into structured parameters.
        
        Args:
            query: Natural language query string
            
        Returns:
            ParsedQuery object with extracted intent and filters
        """
        query = query.strip()
        if not query:
            return self._create_empty_query(query)
        
        # Determine query type
        query_type = self._determine_query_type(query)
        
        # Extract search terms (remove common words and filters)
        search_terms = self._extract_search_terms(query)
        
        # Extract filters
        floor_ids = self._extract_floors(query)
        disciplines = self._extract_disciplines(query)
        text_types = self._extract_text_types(query)
        region_types = self._extract_region_types(query)
        
        # Determine search scope
        search_scope = self._determine_search_scope(floor_ids, disciplines)
        
        # Build filters dictionary
        filters = {}
        if floor_ids:
            filters['floor_id'] = floor_ids[0] if len(floor_ids) == 1 else floor_ids
        if disciplines:
            filters['discipline'] = disciplines[0] if len(disciplines) == 1 else disciplines
        if text_types:
            filters['text_type'] = text_types[0] if len(text_types) == 1 else text_types
        if region_types:
            filters['region_type'] = region_types[0] if len(region_types) == 1 else region_types
        
        # Calculate confidence based on how well we parsed the query
        confidence = self._calculate_confidence(query, search_terms, filters)
        
        # Extract additional metadata
        metadata = self._extract_metadata(query)
        
        return ParsedQuery(
            original_query=query,
            query_type=query_type,
            search_scope=search_scope,
            search_terms=search_terms,
            filters=filters,
            floor_ids=floor_ids,
            disciplines=disciplines,
            text_types=text_types,
            region_types=region_types,
            confidence=confidence,
            metadata=metadata
        )
    
    def _determine_query_type(self, query: str) -> QueryType:
        """Determine the type of query based on patterns."""
        query_lower = query.lower()
        
        # Check for location queries
        for pattern in self.patterns['location_queries']:
            if pattern.search(query):
                return QueryType.LOCATION
        
        # Check for list queries
        for pattern in self.patterns['list_queries']:
            if pattern.search(query):
                return QueryType.LIST
        
        # Check for count queries
        for pattern in self.patterns['count_queries']:
            if pattern.search(query):
                return QueryType.COUNT
        
        # Check for comparison indicators
        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'between']):
            return QueryType.COMPARISON
        
        # Default to search
        return QueryType.SEARCH
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract meaningful search terms from query."""
        # Remove common question words and prepositions
        stop_words = {
            'where', 'what', 'how', 'when', 'why', 'who', 'which', 'is', 'are', 'can', 'could',
            'would', 'should', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'find', 'show', 'me',
            'all', 'any', 'some', 'level', 'floor', 'sheet', 'page', 'plan', 'drawing'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        search_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add back important AEC terms that might have been filtered
        for pattern_group in ['aec_terms', 'equipment_patterns']:
            for pattern in self.patterns[pattern_group]:
                matches = pattern.findall(query)
                search_terms.extend([match.lower() if isinstance(match, str) else match for match in matches])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms
    
    def _extract_floors(self, query: str) -> List[str]:
        """Extract floor identifiers from query."""
        floors = []
        
        for pattern in self.patterns['floor_patterns']:
            matches = pattern.findall(query)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # Get first group from tuple
                
                match = match.lower().strip()
                
                # Convert word to number if needed
                if match in self.floor_word_to_number:
                    floor_id = f"L{self.floor_word_to_number[match]}"
                elif match.isdigit():
                    floor_id = f"L{match}"
                elif match.startswith('l') and match[1:].isdigit():
                    floor_id = f"L{match[1:]}"
                elif match in ['lb', 'lr', 'lp', 'lm', 'lmz']:
                    floor_id = match.upper()
                else:
                    # Try to extract number from match
                    numbers = re.findall(r'\d+', match)
                    if numbers:
                        floor_id = f"L{numbers[0]}"
                    else:
                        continue
                
                if floor_id not in floors:
                    floors.append(floor_id)
        
        return floors
    
    def _extract_disciplines(self, query: str) -> List[str]:
        """Extract discipline codes from query."""
        disciplines = []
        
        for pattern in self.patterns['discipline_patterns']:
            matches = pattern.findall(query)
            for match in matches:
                match_lower = match.lower()
                if match_lower in self.discipline_mapping:
                    discipline = self.discipline_mapping[match_lower]
                    if discipline not in disciplines:
                        disciplines.append(discipline)
        
        return disciplines
    
    def _extract_text_types(self, query: str) -> List[str]:
        """Extract text type classifications from query."""
        text_types = []
        
        for pattern in self.patterns['text_type_patterns']:
            matches = pattern.findall(query)
            for match in matches:
                match_lower = match.lower().strip()
                if match_lower in self.text_type_mapping:
                    text_type = self.text_type_mapping[match_lower]
                    if text_type not in text_types:
                        text_types.append(text_type)
        
        return text_types
    
    def _extract_region_types(self, query: str) -> List[str]:
        """Extract region types from query."""
        region_types = []
        
        # Map certain terms to region types
        region_mapping = {
            'title': 'title',
            'table': 'table',
            'schedule': 'table',
            'drawing': 'drawing',
            'image': 'figure',
            'figure': 'figure'
        }
        
        query_lower = query.lower()
        for term, region_type in region_mapping.items():
            if term in query_lower and region_type not in region_types:
                region_types.append(region_type)
        
        return region_types
    
    def _determine_search_scope(self, floor_ids: List[str], disciplines: List[str]) -> SearchScope:
        """Determine the scope of the search."""
        if floor_ids and disciplines:
            return SearchScope.FLOOR  # Most specific
        elif floor_ids:
            return SearchScope.FLOOR
        elif disciplines:
            return SearchScope.DISCIPLINE
        else:
            return SearchScope.GLOBAL
    
    def _calculate_confidence(self, query: str, search_terms: List[str], filters: Dict[str, Any]) -> float:
        """Calculate confidence in query parsing."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for recognized patterns
        if search_terms:
            confidence += 0.2
        
        if filters:
            confidence += 0.2 * len(filters)
        
        # Increase confidence for specific AEC terms
        aec_term_count = 0
        for pattern_group in ['aec_terms', 'equipment_patterns', 'csi_patterns']:
            for pattern in self.patterns[pattern_group]:
                if pattern.search(query):
                    aec_term_count += 1
        
        confidence += min(0.2, aec_term_count * 0.05)
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _extract_metadata(self, query: str) -> Dict[str, Any]:
        """Extract additional metadata from query."""
        metadata = {}
        
        # Check for CSI codes
        csi_codes = []
        for pattern in self.patterns['csi_patterns']:
            matches = pattern.findall(query)
            csi_codes.extend(matches)
        
        if csi_codes:
            metadata['csi_codes'] = csi_codes
        
        # Check for equipment tags
        equipment_tags = []
        for pattern in self.patterns['equipment_patterns']:
            if isinstance(pattern.pattern, str) and 'A-Z' in pattern.pattern:  # Equipment tag pattern
                matches = pattern.findall(query)
                equipment_tags.extend(matches)
        
        if equipment_tags:
            metadata['equipment_tags'] = equipment_tags
        
        # Check for plan types
        plan_types = []
        plan_type_patterns = {
            'floor plan': 'floor_plan',
            'rcp': 'reflected_ceiling_plan',
            'reflected ceiling plan': 'reflected_ceiling_plan',
            'section': 'section',
            'elevation': 'elevation',
            'detail': 'detail'
        }
        
        query_lower = query.lower()
        for term, plan_type in plan_type_patterns.items():
            if term in query_lower:
                plan_types.append(plan_type)
        
        if plan_types:
            metadata['plan_types'] = plan_types
        
        return metadata
    
    def _create_empty_query(self, query: str) -> ParsedQuery:
        """Create an empty parsed query for invalid input."""
        return ParsedQuery(
            original_query=query,
            query_type=QueryType.SEARCH,
            search_scope=SearchScope.GLOBAL,
            search_terms=[],
            filters={},
            floor_ids=[],
            disciplines=[],
            text_types=[],
            region_types=[],
            confidence=0.0,
            metadata={}
        )
    
    def suggest_corrections(self, query: str) -> List[str]:
        """Suggest corrections or improvements to the query."""
        suggestions = []
        
        # Check for common misspellings
        corrections = {
            'lvl': 'level',
            'flr': 'floor',
            'elev': 'elevation',
            'mech': 'mechanical',
            'elec': 'electrical',
            'struct': 'structural',
            'arch': 'architectural'
        }
        
        query_words = query.lower().split()
        corrected_words = []
        has_corrections = False
        
        for word in query_words:
            if word in corrections:
                corrected_words.append(corrections[word])
                has_corrections = True
            else:
                corrected_words.append(word)
        
        if has_corrections:
            suggestions.append(' '.join(corrected_words))
        
        # Suggest adding floor context if missing
        if not self._extract_floors(query) and any(term in query.lower() for term in ['where', 'find', 'show']):
            suggestions.append(f"{query} on level 1")
            suggestions.append(f"{query} on level 2")
        
        # Suggest adding discipline context
        if not self._extract_disciplines(query):
            suggestions.append(f"{query} architectural")
            suggestions.append(f"{query} mechanical")
        
        return suggestions[:3]  # Limit to top 3 suggestions

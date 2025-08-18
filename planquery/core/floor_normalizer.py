"""
Floor normalization module for standardizing floor labels across different naming conventions.
Maps various floor representations (LEVEL 02, L02, Second Floor) to canonical format (L2).
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class FloorInfo:
    """Normalized floor information."""
    canonical_id: str  # L1, L2, LB1, LR, etc.
    original_text: str
    confidence: float
    floor_number: Optional[int] = None
    floor_type: str = "standard"  # standard, basement, roof, mezzanine
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "original_text": self.original_text,
            "confidence": self.confidence,
            "floor_number": self.floor_number,
            "floor_type": self.floor_type,
            "metadata": self.metadata or {}
        }


class FloorNormalizer:
    """Normalizes floor labels to canonical format."""
    
    def __init__(self):
        """Initialize floor normalizer with pattern matching rules."""
        self._compile_patterns()
        
        # Common floor synonyms and mappings
        self.floor_synonyms = {
            'ground': 1,
            'first': 1,
            'second': 2,
            'third': 3,
            'fourth': 4,
            'fifth': 5,
            'sixth': 6,
            'seventh': 7,
            'eighth': 8,
            'ninth': 9,
            'tenth': 10,
            'eleventh': 11,
            'twelfth': 12,
        }
        
        # Special floor types
        self.special_floors = {
            'basement': 'B',
            'cellar': 'B',
            'sub': 'B',
            'lower': 'B',
            'roof': 'R',
            'penthouse': 'P',
            'mechanical': 'M',
            'mezzanine': 'MZ',
            'mezz': 'MZ',
            'parking': 'P',
            'garage': 'G',
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for floor detection."""
        self.patterns = {
            # Standard level patterns
            'level_numeric': [
                re.compile(r'LEVEL\s*(\d+)', re.IGNORECASE),
                re.compile(r'LVL\s*(\d+)', re.IGNORECASE),
                re.compile(r'L\s*(\d+)', re.IGNORECASE),
                re.compile(r'FLOOR\s*(\d+)', re.IGNORECASE),
                re.compile(r'FLR\s*(\d+)', re.IGNORECASE),
                re.compile(r'(\d+)(?:ST|ND|RD|TH)?\s*FLOOR', re.IGNORECASE),
            ],
            
            # Zero-padded levels
            'level_padded': [
                re.compile(r'LEVEL\s*0*(\d+)', re.IGNORECASE),
                re.compile(r'L0*(\d+)', re.IGNORECASE),
                re.compile(r'FLOOR\s*0*(\d+)', re.IGNORECASE),
            ],
            
            # Word-based floors
            'level_words': [
                re.compile(r'(GROUND|FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)\s*FLOOR', re.IGNORECASE),
                re.compile(r'(GROUND|FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)\s*LEVEL', re.IGNORECASE),
            ],
            
            # Basement patterns
            'basement': [
                re.compile(r'BASEMENT\s*(\d*)', re.IGNORECASE),
                re.compile(r'CELLAR\s*(\d*)', re.IGNORECASE),
                re.compile(r'SUB\s*LEVEL\s*(\d*)', re.IGNORECASE),
                re.compile(r'LOWER\s*LEVEL\s*(\d*)', re.IGNORECASE),
                re.compile(r'B\s*(\d+)', re.IGNORECASE),
                re.compile(r'LB\s*(\d+)', re.IGNORECASE),
                re.compile(r'LEVEL\s*B\s*(\d*)', re.IGNORECASE),
            ],
            
            # Roof patterns
            'roof': [
                re.compile(r'ROOF\s*LEVEL', re.IGNORECASE),
                re.compile(r'ROOF\s*PLAN', re.IGNORECASE),
                re.compile(r'ROOF', re.IGNORECASE),
                re.compile(r'R\s*LEVEL', re.IGNORECASE),
                re.compile(r'LR', re.IGNORECASE),
            ],
            
            # Special floors
            'special': [
                re.compile(r'PENTHOUSE\s*(\d*)', re.IGNORECASE),
                re.compile(r'MECHANICAL\s*(\d*)', re.IGNORECASE),
                re.compile(r'MEZZANINE\s*(\d*)', re.IGNORECASE),
                re.compile(r'MEZZ\s*(\d*)', re.IGNORECASE),
                re.compile(r'PARKING\s*(\d*)', re.IGNORECASE),
                re.compile(r'GARAGE\s*(\d*)', re.IGNORECASE),
            ],
            
            # Plan type indicators (help with context)
            'plan_types': [
                re.compile(r'FLOOR\s*PLAN', re.IGNORECASE),
                re.compile(r'REFLECTED\s*CEILING\s*PLAN', re.IGNORECASE),
                re.compile(r'RCP', re.IGNORECASE),
                re.compile(r'CEILING\s*PLAN', re.IGNORECASE),
                re.compile(r'ELECTRICAL\s*PLAN', re.IGNORECASE),
                re.compile(r'LIGHTING\s*PLAN', re.IGNORECASE),
                re.compile(r'HVAC\s*PLAN', re.IGNORECASE),
                re.compile(r'PLUMBING\s*PLAN', re.IGNORECASE),
            ]
        }
    
    def normalize_floor_from_text(self, text: str) -> Optional[FloorInfo]:
        """
        Extract and normalize floor information from text.
        
        Args:
            text: Text that may contain floor information
            
        Returns:
            FloorInfo object or None if no floor found
        """
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        # Try different pattern categories
        for category, patterns in self.patterns.items():
            if category == 'plan_types':  # Skip plan type patterns for direct matching
                continue
                
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    return self._process_match(text, category, match)
        
        # Try extracting from title block or sheet name context
        return self._extract_from_context(text)
    
    def _process_match(self, original_text: str, category: str, match: re.Match) -> FloorInfo:
        """Process a regex match and create FloorInfo."""
        if category == 'level_numeric' or category == 'level_padded':
            floor_num = int(match.group(1))
            return FloorInfo(
                canonical_id=f"L{floor_num}",
                original_text=original_text,
                confidence=0.9,
                floor_number=floor_num,
                floor_type="standard",
                metadata={'pattern_category': category, 'match': match.group(0)}
            )
        
        elif category == 'level_words':
            word = match.group(1).lower()
            floor_num = self.floor_synonyms.get(word, 1)
            return FloorInfo(
                canonical_id=f"L{floor_num}",
                original_text=original_text,
                confidence=0.8,
                floor_number=floor_num,
                floor_type="standard",
                metadata={'pattern_category': category, 'word': word}
            )
        
        elif category == 'basement':
            basement_num = match.group(1) if match.group(1) else "1"
            basement_num = int(basement_num) if basement_num.isdigit() else 1
            return FloorInfo(
                canonical_id=f"LB{basement_num}",
                original_text=original_text,
                confidence=0.9,
                floor_number=-basement_num,  # Negative for basement
                floor_type="basement",
                metadata={'pattern_category': category, 'basement_level': basement_num}
            )
        
        elif category == 'roof':
            return FloorInfo(
                canonical_id="LR",
                original_text=original_text,
                confidence=0.9,
                floor_number=None,
                floor_type="roof",
                metadata={'pattern_category': category}
            )
        
        elif category == 'special':
            # Determine special floor type
            match_text = match.group(0).lower()
            for special_type, code in self.special_floors.items():
                if special_type in match_text:
                    num_suffix = match.group(1) if match.group(1) else ""
                    canonical_id = f"L{code}{num_suffix}" if num_suffix else f"L{code}"
                    
                    return FloorInfo(
                        canonical_id=canonical_id,
                        original_text=original_text,
                        confidence=0.8,
                        floor_number=None,
                        floor_type=special_type,
                        metadata={'pattern_category': category, 'special_type': special_type}
                    )
        
        # Fallback
        return FloorInfo(
            canonical_id="L1",
            original_text=original_text,
            confidence=0.3,
            floor_number=1,
            floor_type="standard",
            metadata={'pattern_category': category, 'fallback': True}
        )
    
    def _extract_from_context(self, text: str) -> Optional[FloorInfo]:
        """Try to extract floor info from broader context clues."""
        text_lower = text.lower()
        
        # Look for numeric patterns that might be floors
        numeric_matches = re.findall(r'\b(\d+)\b', text)
        
        if numeric_matches:
            # Heuristics for determining if a number is a floor
            for num_str in numeric_matches:
                num = int(num_str)
                
                # Reasonable floor range
                if 1 <= num <= 50:
                    # Check context for floor-related words
                    context_words = ['plan', 'level', 'floor', 'sheet', 'drawing']
                    has_context = any(word in text_lower for word in context_words)
                    
                    if has_context:
                        return FloorInfo(
                            canonical_id=f"L{num}",
                            original_text=text,
                            confidence=0.5,  # Lower confidence for context-based
                            floor_number=num,
                            floor_type="standard",
                            metadata={'extraction_method': 'context', 'inferred': True}
                        )
        
        return None
    
    def normalize_floor_list(self, texts: List[str]) -> List[FloorInfo]:
        """
        Normalize multiple text strings and return floor information.
        
        Args:
            texts: List of text strings to process
            
        Returns:
            List of FloorInfo objects (may be shorter than input if some texts don't contain floors)
        """
        floors = []
        
        for text in texts:
            floor_info = self.normalize_floor_from_text(text)
            if floor_info:
                floors.append(floor_info)
        
        return floors
    
    def get_unique_floors(self, floor_infos: List[FloorInfo]) -> List[FloorInfo]:
        """
        Get unique floors, keeping the highest confidence version of each.
        
        Args:
            floor_infos: List of FloorInfo objects
            
        Returns:
            List of unique floors (by canonical_id)
        """
        floor_map = {}
        
        for floor_info in floor_infos:
            canonical_id = floor_info.canonical_id
            
            if canonical_id not in floor_map or floor_info.confidence > floor_map[canonical_id].confidence:
                floor_map[canonical_id] = floor_info
        
        return list(floor_map.values())
    
    def sort_floors(self, floor_infos: List[FloorInfo]) -> List[FloorInfo]:
        """
        Sort floors in logical order (basement, then standard floors, then special).
        
        Args:
            floor_infos: List of FloorInfo objects
            
        Returns:
            Sorted list of floors
        """
        def sort_key(floor_info: FloorInfo) -> Tuple[int, int, str]:
            """Generate sort key for floor."""
            floor_type = floor_info.floor_type
            floor_num = floor_info.floor_number or 0
            canonical_id = floor_info.canonical_id
            
            # Sort order: basement (negative), standard (positive), special (high positive)
            if floor_type == "basement":
                return (0, floor_num, canonical_id)  # Basement floors (negative numbers)
            elif floor_type == "standard":
                return (1, floor_num, canonical_id)  # Standard floors
            elif floor_type == "roof":
                return (3, 999, canonical_id)  # Roof at top
            else:
                return (2, floor_num or 100, canonical_id)  # Special floors
        
        return sorted(floor_infos, key=sort_key)
    
    def create_floor_mapping(self, floor_infos: List[FloorInfo]) -> Dict[str, str]:
        """
        Create mapping from original text variations to canonical IDs.
        
        Args:
            floor_infos: List of FloorInfo objects
            
        Returns:
            Dictionary mapping original text to canonical ID
        """
        mapping = {}
        
        for floor_info in floor_infos:
            mapping[floor_info.original_text] = floor_info.canonical_id
            
            # Add common variations
            original_lower = floor_info.original_text.lower()
            mapping[original_lower] = floor_info.canonical_id
            
            # Add without spaces
            no_spaces = original_lower.replace(' ', '')
            mapping[no_spaces] = floor_info.canonical_id
        
        return mapping
    
    def validate_floor_sequence(self, floor_infos: List[FloorInfo]) -> Dict[str, Any]:
        """
        Validate that floor sequence makes sense and identify potential issues.
        
        Args:
            floor_infos: List of FloorInfo objects
            
        Returns:
            Validation results with warnings and suggestions
        """
        sorted_floors = self.sort_floors(floor_infos)
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'suggestions': [],
            'floor_count': len(sorted_floors),
            'floor_range': None
        }
        
        # Check for standard floors
        standard_floors = [f for f in sorted_floors if f.floor_type == "standard"]
        
        if standard_floors:
            floor_numbers = [f.floor_number for f in standard_floors if f.floor_number is not None]
            
            if floor_numbers:
                min_floor = min(floor_numbers)
                max_floor = max(floor_numbers)
                validation_results['floor_range'] = (min_floor, max_floor)
                
                # Check for gaps in sequence
                expected_floors = set(range(min_floor, max_floor + 1))
                actual_floors = set(floor_numbers)
                missing_floors = expected_floors - actual_floors
                
                if missing_floors:
                    validation_results['warnings'].append(
                        f"Missing floors in sequence: {sorted(missing_floors)}"
                    )
                
                # Check for unusual patterns
                if max_floor - min_floor > 20:
                    validation_results['warnings'].append(
                        f"Large floor range ({min_floor}-{max_floor}) - verify floor numbers"
                    )
        
        # Check for duplicate canonical IDs with different original text
        canonical_counts = {}
        for floor_info in floor_infos:
            canonical_id = floor_info.canonical_id
            if canonical_id not in canonical_counts:
                canonical_counts[canonical_id] = []
            canonical_counts[canonical_id].append(floor_info.original_text)
        
        for canonical_id, original_texts in canonical_counts.items():
            if len(set(original_texts)) > 1:
                validation_results['warnings'].append(
                    f"Multiple representations for {canonical_id}: {set(original_texts)}"
                )
        
        return validation_results
    
    def get_floor_statistics(self, floor_infos: List[FloorInfo]) -> Dict[str, Any]:
        """Get statistics about detected floors."""
        if not floor_infos:
            return {'total_floors': 0}
        
        stats = {
            'total_floors': len(floor_infos),
            'by_type': {},
            'confidence_stats': {
                'avg': sum(f.confidence for f in floor_infos) / len(floor_infos),
                'min': min(f.confidence for f in floor_infos),
                'max': max(f.confidence for f in floor_infos)
            },
            'canonical_ids': [f.canonical_id for f in floor_infos]
        }
        
        # Count by floor type
        for floor_info in floor_infos:
            floor_type = floor_info.floor_type
            stats['by_type'][floor_type] = stats['by_type'].get(floor_type, 0) + 1
        
        return stats

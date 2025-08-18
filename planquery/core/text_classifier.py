"""
Text classification module for categorizing extracted text into AEC-specific types.
Classifies regions into actionable categories like GeneralNotes, Dimensions, Callouts, etc.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path
from loguru import logger

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, using rule-based classification")

from .ocr_engine import OCRResult


@dataclass
class ClassificationResult:
    """Result from text classification."""
    text_type: str
    confidence: float
    original_text: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_type": self.text_type,
            "confidence": self.confidence,
            "original_text": self.original_text,
            "metadata": self.metadata or {}
        }


class TextClassifier:
    """Classifies extracted text into AEC-specific categories."""
    
    # AEC-specific text types
    TEXT_TYPES = {
        'GeneralNotes': 'General notes and specifications',
        'Dimensions': 'Dimensional annotations and measurements',
        'Callout': 'Callout bubbles and references',
        'SpecReference': 'Specification section references (CSI codes)',
        'TitleBlock': 'Title block information',
        'ScheduleText': 'Schedule and table content',
        'RoomLabel': 'Room names and numbers',
        'MaterialLabel': 'Material specifications',
        'Equipment': 'Equipment tags and labels',
        'Elevation': 'Elevation markers and levels',
        'GridLine': 'Grid line labels',
        'DetailReference': 'Detail and section references',
        'RevisionText': 'Revision clouds and notes',
        'DrawingTitle': 'Drawing titles and sheet names',
        'Other': 'Unclassified text'
    }
    
    def __init__(self, model_type: str = "rules", model_path: Optional[str] = None):
        """
        Initialize text classifier.
        
        Args:
            model_type: Type of classifier ("rules", "sklearn", "transformer")
            model_path: Path to saved model (for sklearn/transformer)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.classifier = None
        self.vectorizer = None
        
        # Compile regex patterns for rule-based classification
        self._compile_patterns()
        
        if model_type == "sklearn" and model_path:
            self._load_sklearn_model()
        elif model_type == "transformer" and TRANSFORMERS_AVAILABLE:
            self._load_transformer_model()
        else:
            logger.info("Using rule-based text classification")
            self.model_type = "rules"
    
    def _compile_patterns(self):
        """Compile regex patterns for rule-based classification."""
        self.patterns = {
            'Dimensions': [
                r'\d+[\'"]\s*-?\s*\d*[\'"]*',  # Feet and inches: 10'-6"
                r'\d+\.\d+\s*mm',              # Millimeters: 150.5 mm
                r'\d+\s*mm',                   # Millimeters: 150 mm
                r'\d+\.\d+\s*m',               # Meters: 1.5 m
                r'\d+\s*cm',                   # Centimeters: 15 cm
                r'R\d+',                       # Radius: R10
                r'Ø\d+',                       # Diameter: Ø20
                r'\d+\s*x\s*\d+',             # Dimensions: 10 x 20
            ],
            'SpecReference': [
                r'\d{2}\s*\d{2}\s*\d{2}',     # CSI codes: 07 21 00
                r'\d{2}-\d{2}-\d{2}',         # CSI codes: 07-21-00
                r'SPEC\s*\d+',                # Spec references
                r'Section\s*\d+',             # Section references
            ],
            'Callout': [
                r'^[A-Z]\d*$',                # Single letter+number: A1, B2
                r'^\d+$',                     # Single number: 1, 2, 3
                r'^[A-Z]$',                   # Single letter: A, B, C
                r'DETAIL\s*[A-Z0-9]+',        # Detail callouts
                r'TYP\.?$',                   # Typical callouts
            ],
            'RoomLabel': [
                r'ROOM\s*\d+',                # Room numbers
                r'OFFICE\s*\d*',              # Office labels
                r'CONFERENCE\s*ROOM',         # Conference room
                r'STORAGE\s*\d*',             # Storage rooms
                r'TOILET\s*\d*',              # Toilets
                r'KITCHEN',                   # Kitchen
                r'LOBBY',                     # Lobby
                r'CORRIDOR',                  # Corridor
                r'STAIR\s*\d*',              # Stairs
                r'ELEVATOR\s*\d*',           # Elevators
            ],
            'Equipment': [
                r'[A-Z]{2,3}-\d+',           # Equipment tags: AHU-1, FCU-2
                r'AHU\s*\d*',                # Air handling units
                r'FCU\s*\d*',                # Fan coil units
                r'VAV\s*\d*',                # VAV boxes
                r'PUMP\s*\d*',               # Pumps
                r'PANEL\s*[A-Z0-9]*',        # Electrical panels
            ],
            'Elevation': [
                r'EL\.?\s*\d+[\'"]\s*-?\s*\d*[\'"]*',  # Elevation: EL. 100'-0"
                r'LEVEL\s*\d+',              # Level markers
                r'FLOOR\s*\d+',              # Floor markers
                r'ROOF',                     # Roof level
                r'BASEMENT',                 # Basement
                r'MEZZANINE',                # Mezzanine
            ],
            'GridLine': [
                r'^[A-Z]\.\d+$',             # Grid lines: A.1, B.2
                r'^[A-Z]$',                  # Grid letters: A, B, C
                r'^\d+$',                    # Grid numbers: 1, 2, 3
            ],
            'DetailReference': [
                r'\d+/[A-Z]\d+',             # Detail references: 1/A101
                r'SECTION\s*[A-Z0-9]+',      # Section references
                r'DETAIL\s*[A-Z0-9]+',       # Detail references
                r'ELEV\s*[A-Z0-9]+',         # Elevation references
            ],
            'MaterialLabel': [
                r'CONCRETE',                 # Concrete
                r'STEEL',                    # Steel
                r'WOOD',                     # Wood
                r'MASONRY',                  # Masonry
                r'GYPSUM',                   # Gypsum
                r'INSULATION',               # Insulation
                r'MEMBRANE',                 # Membrane
                r'SEALANT',                  # Sealant
            ],
        }
        
        # Compile all patterns
        for text_type, patterns in self.patterns.items():
            self.patterns[text_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _load_sklearn_model(self):
        """Load trained sklearn model."""
        try:
            if self.model_path and Path(self.model_path).exists():
                with open(self.model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                logger.info(f"Loaded sklearn model from {self.model_path}")
            else:
                logger.warning("Sklearn model path not found, falling back to rules")
                self.model_type = "rules"
        except Exception as e:
            logger.error(f"Failed to load sklearn model: {e}")
            self.model_type = "rules"
    
    def _load_transformer_model(self):
        """Load transformer model for classification."""
        try:
            # Use a general text classification model (can be fine-tuned later)
            model_name = "distilbert-base-uncased"
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                return_all_scores=True
            )
            logger.info(f"Loaded transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            self.model_type = "rules"
    
    def classify_text(self, text: str, region_type: Optional[str] = None) -> ClassificationResult:
        """
        Classify a single text string.
        
        Args:
            text: Text to classify
            region_type: Optional region type hint from layout detection
            
        Returns:
            Classification result
        """
        if self.model_type == "sklearn":
            return self._classify_with_sklearn(text, region_type)
        elif self.model_type == "transformer":
            return self._classify_with_transformer(text, region_type)
        else:
            return self._classify_with_rules(text, region_type)
    
    def classify_ocr_results(self, ocr_results: List[OCRResult]) -> List[ClassificationResult]:
        """
        Classify multiple OCR results.
        
        Args:
            ocr_results: List of OCR results to classify
            
        Returns:
            List of classification results
        """
        results = []
        
        for ocr_result in ocr_results:
            classification = self.classify_text(
                ocr_result.text,
                ocr_result.region_type
            )
            results.append(classification)
        
        return results
    
    def _classify_with_rules(self, text: str, region_type: Optional[str] = None) -> ClassificationResult:
        """Classify text using rule-based patterns."""
        text = text.strip()
        
        if not text:
            return ClassificationResult(
                text_type="Other",
                confidence=0.0,
                original_text=text
            )
        
        # Check patterns in order of specificity
        scores = {}
        
        for text_type, patterns in self.patterns.items():
            score = 0
            matches = []
            
            for pattern in patterns:
                if pattern.search(text):
                    score += 1
                    matches.append(pattern.pattern)
            
            if score > 0:
                # Normalize score by text length and pattern specificity
                normalized_score = min(score / len(patterns), 1.0)
                scores[text_type] = {
                    'score': normalized_score,
                    'matches': matches
                }
        
        # Use region type as hint if no strong pattern matches
        if not scores and region_type:
            region_to_text_type = {
                'title': 'DrawingTitle',
                'title_block': 'TitleBlock',
                'table': 'ScheduleText',
                'callout': 'Callout',
                'dimension': 'Dimensions'
            }
            
            if region_type in region_to_text_type:
                return ClassificationResult(
                    text_type=region_to_text_type[region_type],
                    confidence=0.6,  # Medium confidence from region hint
                    original_text=text,
                    metadata={'source': 'region_hint', 'region_type': region_type}
                )
        
        # Additional heuristics based on text characteristics
        if not scores:
            scores.update(self._apply_heuristics(text))
        
        if scores:
            # Get best match
            best_type = max(scores.keys(), key=lambda k: scores[k]['score'])
            best_score = scores[best_type]['score']
            
            return ClassificationResult(
                text_type=best_type,
                confidence=best_score,
                original_text=text,
                metadata={
                    'matches': scores[best_type].get('matches', []),
                    'all_scores': {k: v['score'] for k, v in scores.items()}
                }
            )
        
        # Default classification
        return ClassificationResult(
            text_type="Other",
            confidence=0.3,
            original_text=text,
            metadata={'reason': 'no_pattern_match'}
        )
    
    def _apply_heuristics(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Apply additional heuristics for classification."""
        scores = {}
        
        # Length-based heuristics
        if len(text) <= 5 and text.isalnum():
            scores['Callout'] = {'score': 0.7, 'matches': ['short_alphanumeric']}
        
        # All caps heuristic (often labels or callouts)
        if text.isupper() and len(text) > 1:
            if len(text) <= 10:
                scores['Callout'] = {'score': 0.6, 'matches': ['all_caps_short']}
            else:
                scores['RoomLabel'] = {'score': 0.5, 'matches': ['all_caps_long']}
        
        # Number-only heuristic
        if text.isdigit():
            if len(text) <= 3:
                scores['Callout'] = {'score': 0.8, 'matches': ['number_only']}
            else:
                scores['RoomLabel'] = {'score': 0.6, 'matches': ['long_number']}
        
        # Mixed case with spaces (likely general notes)
        if ' ' in text and not text.isupper() and len(text) > 20:
            scores['GeneralNotes'] = {'score': 0.5, 'matches': ['mixed_case_long']}
        
        return scores
    
    def _classify_with_sklearn(self, text: str, region_type: Optional[str] = None) -> ClassificationResult:
        """Classify text using sklearn model."""
        try:
            if self.classifier is None:
                return self._classify_with_rules(text, region_type)
            
            # Get prediction and probabilities
            prediction = self.classifier.predict([text])[0]
            probabilities = self.classifier.predict_proba([text])[0]
            
            # Get confidence (max probability)
            confidence = max(probabilities)
            
            return ClassificationResult(
                text_type=prediction,
                confidence=confidence,
                original_text=text,
                metadata={'model': 'sklearn', 'all_probabilities': probabilities.tolist()}
            )
            
        except Exception as e:
            logger.warning(f"Sklearn classification failed: {e}")
            return self._classify_with_rules(text, region_type)
    
    def _classify_with_transformer(self, text: str, region_type: Optional[str] = None) -> ClassificationResult:
        """Classify text using transformer model."""
        try:
            if self.classifier is None:
                return self._classify_with_rules(text, region_type)
            
            # Get predictions
            results = self.classifier(text)
            
            # Find best match (this is a simplified approach)
            # In practice, you'd fine-tune the model on AEC-specific data
            best_result = max(results, key=lambda x: x['score'])
            
            # Map generic labels to AEC-specific types (simplified)
            label_mapping = {
                'POSITIVE': 'GeneralNotes',
                'NEGATIVE': 'Other',
                # Add more mappings based on your fine-tuned model
            }
            
            text_type = label_mapping.get(best_result['label'], 'Other')
            
            return ClassificationResult(
                text_type=text_type,
                confidence=best_result['score'],
                original_text=text,
                metadata={'model': 'transformer', 'original_label': best_result['label']}
            )
            
        except Exception as e:
            logger.warning(f"Transformer classification failed: {e}")
            return self._classify_with_rules(text, region_type)
    
    def train_sklearn_model(self, training_data: List[Tuple[str, str]], 
                           save_path: Optional[str] = None) -> float:
        """
        Train sklearn model on labeled data.
        
        Args:
            training_data: List of (text, label) tuples
            save_path: Path to save trained model
            
        Returns:
            Training accuracy
        """
        if not training_data:
            raise ValueError("No training data provided")
        
        texts, labels = zip(*training_data)
        
        # Create pipeline with TF-IDF and Logistic Regression
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                class_weight='balanced'
            ))
        ])
        
        # Train model
        self.classifier.fit(texts, labels)
        
        # Calculate training accuracy
        accuracy = self.classifier.score(texts, labels)
        
        # Save model if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            logger.info(f"Model saved to {save_path}")
        
        self.model_type = "sklearn"
        logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def get_text_type_stats(self, classifications: List[ClassificationResult]) -> Dict[str, Any]:
        """Get statistics about classified text types."""
        if not classifications:
            return {}
        
        type_counts = {}
        confidence_sums = {}
        
        for result in classifications:
            text_type = result.text_type
            type_counts[text_type] = type_counts.get(text_type, 0) + 1
            confidence_sums[text_type] = confidence_sums.get(text_type, 0) + result.confidence
        
        stats = {}
        total_count = len(classifications)
        
        for text_type in type_counts:
            count = type_counts[text_type]
            avg_confidence = confidence_sums[text_type] / count
            
            stats[text_type] = {
                'count': count,
                'percentage': (count / total_count) * 100,
                'avg_confidence': avg_confidence
            }
        
        return stats
    
    def filter_by_confidence(self, classifications: List[ClassificationResult], 
                           min_confidence: float = 0.5) -> List[ClassificationResult]:
        """Filter classifications by minimum confidence threshold."""
        return [c for c in classifications if c.confidence >= min_confidence]
    
    def group_by_type(self, classifications: List[ClassificationResult]) -> Dict[str, List[ClassificationResult]]:
        """Group classifications by text type."""
        groups = {}
        
        for classification in classifications:
            text_type = classification.text_type
            if text_type not in groups:
                groups[text_type] = []
            groups[text_type].append(classification)
        
        return groups

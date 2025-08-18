"""
OCR engine for extracting text from detected layout regions.
Supports both Tesseract (baseline) and TrOCR (transformer-based) approaches.
"""

import pytesseract
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, using Tesseract only")

from .layout_detector import DetectedRegion, BoundingBox


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float
    bbox: BoundingBox
    region_type: str
    word_boxes: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "region_type": self.region_type,
            "word_boxes": self.word_boxes,
            "metadata": self.metadata or {}
        }


class OCREngine:
    """OCR engine with multiple backend support."""
    
    def __init__(self, engine: str = "tesseract", model_name: str = None):
        """
        Initialize OCR engine.
        
        Args:
            engine: OCR engine to use ("tesseract" or "trocr")
            model_name: Model name for TrOCR (if using TrOCR)
        """
        self.engine = engine
        self.model_name = model_name or "microsoft/trocr-base-printed"
        self.processor = None
        self.model = None
        
        if engine == "trocr" and TRANSFORMERS_AVAILABLE:
            self._initialize_trocr()
        elif engine == "tesseract":
            self._initialize_tesseract()
        else:
            logger.warning("Falling back to Tesseract OCR")
            self.engine = "tesseract"
            self._initialize_tesseract()
    
    def _initialize_tesseract(self):
        """Initialize Tesseract OCR."""
        try:
            # Test Tesseract installation
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized")
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")
            raise
    
    def _initialize_trocr(self):
        """Initialize TrOCR model."""
        try:
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            logger.info(f"TrOCR initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize TrOCR: {e}")
            raise
    
    def extract_text_from_regions(self, image: np.ndarray, 
                                 regions: List[DetectedRegion]) -> List[OCRResult]:
        """
        Extract text from detected regions in an image.
        
        Args:
            image: Source image as numpy array
            regions: List of detected layout regions
            
        Returns:
            List of OCR results for each region
        """
        results = []
        
        for region in regions:
            try:
                # Extract region from image
                region_image = self._extract_region_image(image, region.bbox)
                
                # Preprocess region for better OCR
                processed_image = self._preprocess_region(region_image, region.region_type)
                
                # Perform OCR
                if self.engine == "trocr":
                    ocr_result = self._ocr_with_trocr(processed_image, region)
                else:
                    ocr_result = self._ocr_with_tesseract(processed_image, region)
                
                if ocr_result and ocr_result.text.strip():
                    results.append(ocr_result)
                    
            except Exception as e:
                logger.warning(f"OCR failed for region {region.region_type}: {e}")
                continue
        
        return results
    
    def _extract_region_image(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Extract a region from the image using bounding box."""
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(bbox.x1, w))
        y1 = max(0, min(bbox.y1, h))
        x2 = max(x1, min(bbox.x2, w))
        y2 = max(y1, min(bbox.y2, h))
        
        return image[y1:y2, x1:x2]
    
    def _preprocess_region(self, region_image: np.ndarray, region_type: str) -> np.ndarray:
        """
        Preprocess region image for better OCR results.
        
        Args:
            region_image: Region image to preprocess
            region_type: Type of region (affects preprocessing strategy)
            
        Returns:
            Preprocessed image
        """
        if len(region_image.shape) == 3:
            gray = cv2.cvtColor(region_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = region_image.copy()
        
        # Skip if image is too small
        if gray.shape[0] < 10 or gray.shape[1] < 10:
            return gray
        
        # Apply different preprocessing based on region type
        if region_type in ["title", "title_block"]:
            # For titles, enhance contrast and reduce noise
            processed = self._enhance_text_image(gray, aggressive=True)
        elif region_type in ["table", "schedule"]:
            # For tables, preserve structure
            processed = self._enhance_text_image(gray, aggressive=False)
        elif region_type == "dimension":
            # For dimensions, enhance thin lines
            processed = self._enhance_dimension_text(gray)
        else:
            # Standard text preprocessing
            processed = self._enhance_text_image(gray, aggressive=False)
        
        return processed
    
    def _enhance_text_image(self, gray_image: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """Enhance text image for better OCR."""
        # Resize if too small (OCR works better on larger images)
        h, w = gray_image.shape
        if h < 32 or w < 32:
            scale_factor = max(32 / h, 32 / w, 2.0)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            gray_image = cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_image)
        
        if aggressive:
            # More aggressive preprocessing for titles
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
        else:
            # Light denoising
            denoised = cv2.fastNlMeansDenoising(enhanced)
            return denoised
    
    def _enhance_dimension_text(self, gray_image: np.ndarray) -> np.ndarray:
        """Special preprocessing for dimension text (often small and thin)."""
        # Upscale more aggressively for dimensions
        h, w = gray_image.shape
        scale_factor = max(48 / h, 48 / w, 3.0)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        upscaled = cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def _ocr_with_tesseract(self, image: np.ndarray, region: DetectedRegion) -> Optional[OCRResult]:
        """Perform OCR using Tesseract."""
        try:
            # Configure Tesseract based on region type
            config = self._get_tesseract_config(region.region_type)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            text_parts = []
            confidences = []
            word_boxes = []
            
            for i in range(len(data['text'])):
                word = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if word and conf > 0:  # Filter out empty words and low confidence
                    text_parts.append(word)
                    confidences.append(conf)
                    
                    # Store word-level bounding box
                    word_boxes.append({
                        'text': word,
                        'confidence': conf,
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'w': data['width'][i],
                            'h': data['height'][i]
                        }
                    })
            
            if not text_parts:
                return None
            
            # Combine text and calculate average confidence
            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence / 100.0,  # Normalize to 0-1
                bbox=region.bbox,
                region_type=region.region_type,
                word_boxes=word_boxes,
                metadata={'ocr_engine': 'tesseract', 'config': config}
            )
            
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")
            return None
    
    def _ocr_with_trocr(self, image: np.ndarray, region: DetectedRegion) -> Optional[OCRResult]:
        """Perform OCR using TrOCR."""
        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image, mode='L')
            else:
                pil_image = Image.fromarray(image)
            
            # Process with TrOCR
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # TrOCR doesn't provide confidence scores, so we use a default
            confidence = 0.8  # Default confidence for TrOCR
            
            return OCRResult(
                text=generated_text,
                confidence=confidence,
                bbox=region.bbox,
                region_type=region.region_type,
                metadata={'ocr_engine': 'trocr', 'model': self.model_name}
            )
            
        except Exception as e:
            logger.warning(f"TrOCR failed: {e}")
            return None
    
    def _get_tesseract_config(self, region_type: str) -> str:
        """Get Tesseract configuration based on region type."""
        base_config = "--oem 3 --psm"
        
        # Page Segmentation Mode (PSM) based on region type
        psm_configs = {
            "title": "8",      # Single word
            "text": "6",       # Single uniform block
            "table": "6",      # Single uniform block
            "title_block": "6", # Single uniform block
            "dimension": "8",   # Single word
            "callout": "8",     # Single word
            "schedule": "6",    # Single uniform block
        }
        
        psm = psm_configs.get(region_type, "6")  # Default to single block
        
        # Additional configurations for better accuracy
        additional_config = "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-/: "
        
        return f"{base_config} {psm} {additional_config}"
    
    def extract_text_from_image(self, image: np.ndarray, 
                               bbox: Optional[BoundingBox] = None) -> Optional[OCRResult]:
        """
        Extract text from a full image or specific region.
        
        Args:
            image: Source image
            bbox: Optional bounding box to extract from
            
        Returns:
            OCR result or None if extraction failed
        """
        if bbox:
            region_image = self._extract_region_image(image, bbox)
        else:
            region_image = image
            bbox = BoundingBox(0, 0, image.shape[1], image.shape[0])
        
        # Create a dummy region for processing
        dummy_region = DetectedRegion(
            bbox=bbox,
            region_type="text",
            confidence=1.0
        )
        
        if self.engine == "trocr":
            return self._ocr_with_trocr(region_image, dummy_region)
        else:
            return self._ocr_with_tesseract(region_image, dummy_region)
    
    def batch_ocr(self, images: List[np.ndarray], 
                  regions_list: List[List[DetectedRegion]]) -> List[List[OCRResult]]:
        """
        Perform OCR on multiple images in batch.
        
        Args:
            images: List of images
            regions_list: List of regions for each image
            
        Returns:
            List of OCR results for each image
        """
        results = []
        
        for image, regions in zip(images, regions_list):
            image_results = self.extract_text_from_regions(image, regions)
            results.append(image_results)
        
        return results

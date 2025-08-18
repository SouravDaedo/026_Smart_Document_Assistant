"""
Layout detection module for identifying regions in architectural plan sheets.
Uses computer vision models to detect text blocks, titles, tables, and other elements.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import torch
from PIL import Image
from loguru import logger

try:
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    logger.warning("Detectron2 not available, using fallback layout detection")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available")


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_dict(self) -> Dict[str, int]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class DetectedRegion:
    """Represents a detected layout region."""
    bbox: BoundingBox
    region_type: str  # text, title, table, image, drawing, etc.
    confidence: float
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox.to_dict(),
            "region_type": self.region_type,
            "confidence": self.confidence,
            "text_content": self.text_content,
            "metadata": self.metadata or {}
        }


class LayoutDetector:
    """Detects layout regions in architectural plan images."""
    
    # Standard region types for architectural plans
    REGION_TYPES = {
        0: "text",
        1: "title", 
        2: "list",
        3: "table",
        4: "figure",
        5: "title_block",
        6: "drawing",
        7: "dimension",
        8: "callout",
        9: "schedule"
    }
    
    def __init__(self, model_type: str = "detectron2", confidence_threshold: float = 0.5):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.predictor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the layout detection model."""
        if self.model_type == "detectron2" and DETECTRON2_AVAILABLE:
            self._init_detectron2()
        elif self.model_type == "yolo" and YOLO_AVAILABLE:
            self._init_yolo()
        else:
            logger.warning("Using fallback rule-based layout detection")
            self.model_type = "fallback"
    
    def _init_detectron2(self):
        """Initialize Detectron2 model for layout detection."""
        try:
            cfg = get_cfg()
            # Use a pre-trained model (can be fine-tuned on DocLayNet later)
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.REGION_TYPES)
            
            self.predictor = DefaultPredictor(cfg)
            logger.info("Detectron2 layout detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Detectron2: {e}")
            self.model_type = "fallback"
    
    def _init_yolo(self):
        """Initialize YOLO model for layout detection."""
        try:
            # Use YOLOv8 (can be fine-tuned on document layout data)
            self.predictor = YOLO('yolov8n.pt')  # Start with nano model
            logger.info("YOLO layout detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            self.model_type = "fallback"
    
    def detect_regions(self, image: np.ndarray) -> List[DetectedRegion]:
        """
        Detect layout regions in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected regions with bounding boxes and types
        """
        if self.model_type == "detectron2":
            return self._detect_with_detectron2(image)
        elif self.model_type == "yolo":
            return self._detect_with_yolo(image)
        else:
            return self._detect_with_fallback(image)
    
    def _detect_with_detectron2(self, image: np.ndarray) -> List[DetectedRegion]:
        """Detect regions using Detectron2."""
        try:
            outputs = self.predictor(image)
            
            predictions = outputs["instances"].to("cpu")
            boxes = predictions.pred_boxes.tensor.numpy()
            scores = predictions.scores.numpy()
            classes = predictions.pred_classes.numpy()
            
            regions = []
            for box, score, cls in zip(boxes, scores, classes):
                if score >= self.confidence_threshold:
                    bbox = BoundingBox(
                        x1=int(box[0]), y1=int(box[1]),
                        x2=int(box[2]), y2=int(box[3])
                    )
                    
                    region_type = self.REGION_TYPES.get(cls, "unknown")
                    
                    region = DetectedRegion(
                        bbox=bbox,
                        region_type=region_type,
                        confidence=float(score)
                    )
                    regions.append(region)
            
            return regions
            
        except Exception as e:
            logger.error(f"Detectron2 detection failed: {e}")
            return self._detect_with_fallback(image)
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[DetectedRegion]:
        """Detect regions using YOLO."""
        try:
            results = self.predictor(image)
            
            regions = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        if confidence >= self.confidence_threshold:
                            coords = box.xyxy[0].cpu().numpy()
                            bbox = BoundingBox(
                                x1=int(coords[0]), y1=int(coords[1]),
                                x2=int(coords[2]), y2=int(coords[3])
                            )
                            
                            # Map YOLO class to region type
                            cls = int(box.cls[0])
                            region_type = self.REGION_TYPES.get(cls, "text")
                            
                            region = DetectedRegion(
                                bbox=bbox,
                                region_type=region_type,
                                confidence=confidence
                            )
                            regions.append(region)
            
            return regions
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return self._detect_with_fallback(image)
    
    def _detect_with_fallback(self, image: np.ndarray) -> List[DetectedRegion]:
        """
        Fallback rule-based layout detection using OpenCV.
        Detects basic text regions and potential title blocks.
        """
        logger.info("Using fallback layout detection")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        regions = []
        
        # Detect text regions using contours
        text_regions = self._detect_text_regions_cv(gray)
        regions.extend(text_regions)
        
        # Detect potential title block (usually in bottom right)
        title_block = self._detect_title_block_cv(gray)
        if title_block:
            regions.append(title_block)
        
        # Detect large text areas (potential titles)
        title_regions = self._detect_title_regions_cv(gray)
        regions.extend(title_regions)
        
        return regions
    
    def _detect_text_regions_cv(self, gray_image: np.ndarray) -> List[DetectedRegion]:
        """Detect text regions using OpenCV morphological operations."""
        # Apply morphological operations to connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        morph = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        min_area = 500  # Minimum area for text regions
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (text regions are usually wider than tall)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 20:
                    bbox = BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h)
                    
                    region = DetectedRegion(
                        bbox=bbox,
                        region_type="text",
                        confidence=0.7  # Default confidence for fallback
                    )
                    regions.append(region)
        
        return regions
    
    def _detect_title_block_cv(self, gray_image: np.ndarray) -> Optional[DetectedRegion]:
        """Detect title block (usually in bottom-right corner)."""
        h, w = gray_image.shape
        
        # Look in bottom-right quadrant
        roi_x = w * 3 // 4
        roi_y = h * 3 // 4
        roi = gray_image[roi_y:, roi_x:]
        
        # Find rectangular regions that might be title blocks
        edges = cv2.Canny(roi, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 50000:  # Reasonable size for title block
                x, y, w_roi, h_roi = cv2.boundingRect(contour)
                
                # Adjust coordinates to full image
                bbox = BoundingBox(
                    x1=roi_x + x, y1=roi_y + y,
                    x2=roi_x + x + w_roi, y2=roi_y + y + h_roi
                )
                
                return DetectedRegion(
                    bbox=bbox,
                    region_type="title_block",
                    confidence=0.6
                )
        
        return None
    
    def _detect_title_regions_cv(self, gray_image: np.ndarray) -> List[DetectedRegion]:
        """Detect potential title regions (large text areas)."""
        # Use larger kernel to detect bigger text blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        morph = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        h, w = gray_image.shape
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2000 < area < 20000:  # Size range for titles
                x, y, w_cont, h_cont = cv2.boundingRect(contour)
                
                # Titles are usually in upper portion and have good aspect ratio
                if y < h // 2 and 2 < w_cont / h_cont < 15:
                    bbox = BoundingBox(x1=x, y1=y, x2=x+w_cont, y2=y+h_cont)
                    
                    region = DetectedRegion(
                        bbox=bbox,
                        region_type="title",
                        confidence=0.6
                    )
                    regions.append(region)
        
        return regions
    
    def visualize_regions(self, image: np.ndarray, regions: List[DetectedRegion]) -> np.ndarray:
        """
        Visualize detected regions on the image.
        
        Args:
            image: Original image
            regions: List of detected regions
            
        Returns:
            Image with bounding boxes drawn
        """
        vis_image = image.copy()
        
        # Color map for different region types
        colors = {
            "text": (0, 255, 0),      # Green
            "title": (255, 0, 0),     # Red  
            "table": (0, 0, 255),     # Blue
            "title_block": (255, 255, 0),  # Yellow
            "drawing": (255, 0, 255), # Magenta
            "callout": (0, 255, 255), # Cyan
            "unknown": (128, 128, 128) # Gray
        }
        
        for region in regions:
            color = colors.get(region.region_type, colors["unknown"])
            bbox = region.bbox
            
            # Draw bounding box
            cv2.rectangle(vis_image, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
            
            # Add label
            label = f"{region.region_type} ({region.confidence:.2f})"
            cv2.putText(vis_image, label, (bbox.x1, bbox.y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image
    
    def filter_regions(self, regions: List[DetectedRegion], 
                      min_area: int = 100, 
                      max_overlap: float = 0.5) -> List[DetectedRegion]:
        """
        Filter and clean up detected regions.
        
        Args:
            regions: List of detected regions
            min_area: Minimum area threshold
            max_overlap: Maximum allowed overlap ratio
            
        Returns:
            Filtered list of regions
        """
        # Filter by minimum area
        filtered = [r for r in regions if r.bbox.area >= min_area]
        
        # Remove overlapping regions (keep higher confidence)
        final_regions = []
        for region in sorted(filtered, key=lambda x: x.confidence, reverse=True):
            overlap_found = False
            
            for existing in final_regions:
                if self._calculate_overlap(region.bbox, existing.bbox) > max_overlap:
                    overlap_found = True
                    break
            
            if not overlap_found:
                final_regions.append(region)
        
        return final_regions
    
    def _calculate_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        # Calculate intersection
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = bbox1.area + bbox2.area - intersection
        
        return intersection / union if union > 0 else 0.0

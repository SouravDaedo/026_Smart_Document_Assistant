"""
CNN Model Configuration for Layout Detection.
Provides configurable CNN architectures for document layout detection training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
from loguru import logger

try:
    from detectron2.config import get_cfg
    from detectron2.model_zoo import model_zoo
    from detectron2.engine import DefaultTrainer, DefaultPredictor
    from detectron2.data import DatasetCatalog, MetadataCatalog
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class CNNModelConfig:
    """Configuration for CNN layout detection models."""
    
    # Model Architecture
    model_type: str = "detectron2"  # detectron2, yolo, custom
    backbone: str = "resnet50"  # resnet50, resnet101, efficientnet, swin
    
    # Input Configuration
    input_size: Tuple[int, int] = (1024, 1024)
    num_classes: int = 11  # Background + 10 layout classes
    
    # Training Configuration
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9
    epochs: int = 100
    
    # Model Specific Settings
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    
    # Paths
    model_dir: str = "models"
    weights_path: Optional[str] = None
    config_path: Optional[str] = None
    
    # Device Configuration
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    
    # Data Augmentation
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "rotation": 10,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.1,
        "horizontal_flip": True,
        "vertical_flip": False,
        "scale_jitter": 0.1
    })
    
    # Categories for architectural documents
    categories: Dict[int, str] = field(default_factory=lambda: {
        0: "background",
        1: "text",
        2: "title", 
        3: "dimension",
        4: "symbol",
        5: "table",
        6: "drawing",
        7: "annotation",
        8: "legend",
        9: "title_block",
        10: "other"
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_type": self.model_type,
            "backbone": self.backbone,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "epochs": self.epochs,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "max_detections": self.max_detections,
            "model_dir": self.model_dir,
            "weights_path": self.weights_path,
            "config_path": self.config_path,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "augmentation": self.augmentation,
            "categories": self.categories
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CNNModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to file."""
        config_dict = self.to_dict()
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'CNNModelConfig':
        """Load configuration from file."""
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


class CNNModelFactory:
    """Factory for creating and configuring CNN models."""
    
    def __init__(self, config: CNNModelConfig):
        self.config = config
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        """Determine the device to use."""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    def create_detectron2_config(self) -> Any:
        """Create Detectron2 configuration."""
        if not DETECTRON2_AVAILABLE:
            raise ImportError("Detectron2 not available")
        
        cfg = get_cfg()
        
        # Model configuration
        if self.config.backbone == "resnet50":
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        elif self.config.backbone == "resnet101":
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        
        # Override with custom weights if provided
        if self.config.weights_path:
            cfg.MODEL.WEIGHTS = self.config.weights_path
        
        # Device configuration
        cfg.MODEL.DEVICE = self.device
        
        # Model parameters
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.config.num_classes - 1  # Exclude background
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.confidence_threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.config.nms_threshold
        
        # Input configuration
        cfg.INPUT.MIN_SIZE_TRAIN = (self.config.input_size[0],)
        cfg.INPUT.MAX_SIZE_TRAIN = self.config.input_size[1]
        cfg.INPUT.MIN_SIZE_TEST = self.config.input_size[0]
        cfg.INPUT.MAX_SIZE_TEST = self.config.input_size[1]
        
        # Training configuration
        cfg.SOLVER.IMS_PER_BATCH = self.config.batch_size
        cfg.SOLVER.BASE_LR = self.config.learning_rate
        cfg.SOLVER.WEIGHT_DECAY = self.config.weight_decay
        cfg.SOLVER.MOMENTUM = self.config.momentum
        cfg.SOLVER.MAX_ITER = self.config.epochs * 1000  # Approximate
        
        # Data augmentation
        cfg.INPUT.RANDOM_FLIP = "horizontal" if self.config.augmentation.get("horizontal_flip") else "none"
        
        # Output directory
        cfg.OUTPUT_DIR = str(Path(self.config.model_dir) / "detectron2_output")
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        return cfg
    
    def create_yolo_config(self) -> Dict[str, Any]:
        """Create YOLO configuration."""
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available")
        
        config = {
            # Model
            "model": f"yolov8{self.config.backbone[-1] if self.config.backbone.startswith('yolo') else 'n'}.pt",
            
            # Training
            "epochs": self.config.epochs,
            "batch": self.config.batch_size,
            "lr0": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            
            # Input
            "imgsz": self.config.input_size[0],
            
            # Validation
            "conf": self.config.confidence_threshold,
            "iou": self.config.nms_threshold,
            
            # Device
            "device": self.device,
            
            # Augmentation
            "degrees": self.config.augmentation.get("rotation", 0),
            "brightness": self.config.augmentation.get("brightness", 0),
            "contrast": self.config.augmentation.get("contrast", 0),
            "saturation": self.config.augmentation.get("saturation", 0),
            "flipud": self.config.augmentation.get("vertical_flip", 0),
            "fliplr": self.config.augmentation.get("horizontal_flip", 0),
            
            # Output
            "project": str(Path(self.config.model_dir) / "yolo_output"),
            "name": "layout_detection"
        }
        
        return config
    
    def create_custom_cnn(self) -> nn.Module:
        """Create custom CNN architecture for layout detection."""
        
        class LayoutDetectionCNN(nn.Module):
            def __init__(self, num_classes: int, input_size: Tuple[int, int]):
                super().__init__()
                
                # Feature extraction backbone
                self.backbone = nn.Sequential(
                    # Conv Block 1
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    
                    # Conv Block 2
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # Conv Block 3
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # Conv Block 4
                    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                
                # Feature Pyramid Network (FPN) for multi-scale detection
                self.fpn = nn.ModuleDict({
                    'lateral_conv1': nn.Conv2d(512, 256, 1),
                    'lateral_conv2': nn.Conv2d(256, 256, 1),
                    'lateral_conv3': nn.Conv2d(128, 256, 1),
                    'output_conv1': nn.Conv2d(256, 256, 3, padding=1),
                    'output_conv2': nn.Conv2d(256, 256, 3, padding=1),
                    'output_conv3': nn.Conv2d(256, 256, 3, padding=1),
                })
                
                # Detection heads
                self.classification_head = nn.Conv2d(256, num_classes, 1)
                self.regression_head = nn.Conv2d(256, 4, 1)  # x, y, w, h
                
                # Initialize weights
                self._initialize_weights()
            
            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                # Extract features
                features = self.backbone(x)
                
                # Apply FPN (simplified version)
                fpn_features = self.fpn['lateral_conv1'](features)
                fpn_features = self.fpn['output_conv1'](fpn_features)
                
                # Detection outputs
                classification = self.classification_head(fpn_features)
                regression = self.regression_head(fpn_features)
                
                return {
                    'classification': classification,
                    'regression': regression,
                    'features': fpn_features
                }
        
        return LayoutDetectionCNN(self.config.num_classes, self.config.input_size)


def create_default_configs() -> Dict[str, CNNModelConfig]:
    """Create default configurations for different use cases."""
    
    configs = {}
    
    # Fast inference configuration
    configs['fast'] = CNNModelConfig(
        model_type="yolo",
        backbone="yolov8n",
        input_size=(640, 640),
        batch_size=16,
        learning_rate=0.01,
        epochs=50,
        confidence_threshold=0.4
    )
    
    # Balanced configuration
    configs['balanced'] = CNNModelConfig(
        model_type="detectron2",
        backbone="resnet50",
        input_size=(1024, 1024),
        batch_size=8,
        learning_rate=0.001,
        epochs=100,
        confidence_threshold=0.5
    )
    
    # High accuracy configuration
    configs['accurate'] = CNNModelConfig(
        model_type="detectron2",
        backbone="resnet101",
        input_size=(1333, 1333),
        batch_size=4,
        learning_rate=0.0005,
        epochs=200,
        confidence_threshold=0.6
    )
    
    # Custom CNN configuration
    configs['custom'] = CNNModelConfig(
        model_type="custom",
        backbone="custom_cnn",
        input_size=(1024, 1024),
        batch_size=8,
        learning_rate=0.001,
        epochs=150,
        confidence_threshold=0.5
    )
    
    return configs


def save_default_configs(output_dir: str = "configs"):
    """Save default configurations to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    configs = create_default_configs()
    
    for name, config in configs.items():
        config_file = output_path / f"{name}_config.yaml"
        config.save(str(config_file))
        logger.info(f"Saved {name} configuration to {config_file}")


if __name__ == "__main__":
    # Create and save default configurations
    save_default_configs()
    
    # Example usage
    config = CNNModelConfig()
    factory = CNNModelFactory(config)
    
    print("Available model types:")
    print("- detectron2" if DETECTRON2_AVAILABLE else "- detectron2 (not available)")
    print("- yolo" if YOLO_AVAILABLE else "- yolo (not available)")
    print("- custom (always available)")

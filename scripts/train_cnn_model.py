"""
CNN Model Training Script for Layout Detection.
Train custom CNN models on architectural document datasets.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import yaml
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from planquery.models.cnn_config import CNNModelConfig, CNNModelFactory, create_default_configs
from planquery.core.layout_detector import LayoutDetector

try:
    from detectron2.engine import DefaultTrainer
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.evaluation import COCOEvaluator
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class DocumentLayoutDataset(Dataset):
    """Custom dataset for document layout detection."""
    
    def __init__(self, data_dir: str, annotations_file: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load COCO format annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_id = list(self.images.keys())[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.data_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        annotations = self.image_annotations.get(img_id, [])
        
        # Convert to tensors
        boxes = []
        labels = []
        
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            # Convert to [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


class CNNTrainer:
    """CNN model trainer for layout detection."""
    
    def __init__(self, config: CNNModelConfig):
        self.config = config
        self.device = torch.device(config.device if config.device != "auto" else 
                                 ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_factory = CNNModelFactory(config)
        
        logger.info(f"Using device: {self.device}")
    
    def train_detectron2(self, train_data_dir: str, train_annotations: str, 
                        val_data_dir: str = None, val_annotations: str = None):
        """Train using Detectron2."""
        if not DETECTRON2_AVAILABLE:
            raise ImportError("Detectron2 not available for training")
        
        # Register datasets
        register_coco_instances("layout_train", {}, train_annotations, train_data_dir)
        if val_annotations:
            register_coco_instances("layout_val", {}, val_annotations, val_data_dir or train_data_dir)
        
        # Update metadata
        MetadataCatalog.get("layout_train").thing_classes = list(self.config.categories.values())[1:]  # Exclude background
        if val_annotations:
            MetadataCatalog.get("layout_val").thing_classes = list(self.config.categories.values())[1:]
        
        # Create Detectron2 config
        cfg = self.model_factory.create_detectron2_config()
        cfg.DATASETS.TRAIN = ("layout_train",)
        cfg.DATASETS.TEST = ("layout_val",) if val_annotations else ()
        
        # Create trainer
        class LayoutTrainer(DefaultTrainer):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                if output_folder is None:
                    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
                return COCOEvaluator(dataset_name, cfg, True, output_folder)
        
        trainer = LayoutTrainer(cfg)
        trainer.resume_or_load(resume=False)
        
        logger.info("Starting Detectron2 training...")
        trainer.train()
        
        # Save final model
        model_path = Path(cfg.OUTPUT_DIR) / "final_model.pth"
        torch.save(trainer.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        return str(model_path)
    
    def train_yolo(self, data_yaml_path: str):
        """Train using YOLO."""
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available for training")
        
        # Create YOLO model
        model_name = f"yolov8{self.config.backbone[-1] if 'yolo' in self.config.backbone else 'n'}.pt"
        model = YOLO(model_name)
        
        # Get YOLO config
        yolo_config = self.model_factory.create_yolo_config()
        
        logger.info("Starting YOLO training...")
        results = model.train(
            data=data_yaml_path,
            **yolo_config
        )
        
        # Save model
        model_path = Path(yolo_config['project']) / yolo_config['name'] / 'weights' / 'best.pt'
        logger.info(f"Best model saved to {model_path}")
        
        return str(model_path)
    
    def train_custom_cnn(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train custom CNN model."""
        model = self.model_factory.create_custom_cnn()
        model = model.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        logger.info("Starting custom CNN training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}") as pbar:
                for batch_idx, (images, targets) in enumerate(pbar):
                    images = images.to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(images)
                    
                    # Calculate loss (simplified for classification)
                    # In practice, you'd need more complex loss for object detection
                    loss = criterion(outputs['classification'].view(-1, self.config.num_classes), 
                                   torch.zeros(outputs['classification'].numel() // self.config.num_classes, 
                                             dtype=torch.long).to(self.device))
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Loss': f"{loss.item():.4f}",
                        'Avg Loss': f"{train_loss/(batch_idx+1):.4f}"
                    })
            
            # Validation phase
            if val_loader:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for images, targets in val_loader:
                        images = images.to(self.device)
                        outputs = model(images)
                        
                        loss = criterion(outputs['classification'].view(-1, self.config.num_classes),
                                       torch.zeros(outputs['classification'].numel() // self.config.num_classes,
                                                 dtype=torch.long).to(self.device))
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                           f"Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = Path(self.config.model_dir) / "best_custom_model.pth"
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"New best model saved: {model_path}")
            
            scheduler.step()
        
        # Save final model
        final_model_path = Path(self.config.model_dir) / "final_custom_model.pth"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved: {final_model_path}")
        
        return str(final_model_path)


def create_yolo_data_yaml(data_dir: str, output_path: str, categories: Dict[int, str]):
    """Create YOLO data.yaml file."""
    data_config = {
        'path': str(Path(data_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(categories) - 1,  # Exclude background
        'names': list(categories.values())[1:]  # Exclude background
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    logger.info(f"YOLO data config saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train CNN model for layout detection")
    parser.add_argument("--config", required=True, help="Model configuration file")
    parser.add_argument("--data-dir", required=True, help="Training data directory")
    parser.add_argument("--train-annotations", required=True, help="Training annotations file")
    parser.add_argument("--val-annotations", help="Validation annotations file")
    parser.add_argument("--val-data-dir", help="Validation data directory")
    parser.add_argument("--model-type", choices=["detectron2", "yolo", "custom"], 
                       help="Override model type from config")
    
    args = parser.parse_args()
    
    # Load configuration
    config = CNNModelConfig.load(args.config)
    
    # Override model type if specified
    if args.model_type:
        config.model_type = args.model_type
    
    # Create trainer
    trainer = CNNTrainer(config)
    
    # Train based on model type
    if config.model_type == "detectron2":
        if not DETECTRON2_AVAILABLE:
            logger.error("Detectron2 not available. Install with: pip install detectron2")
            return
        
        model_path = trainer.train_detectron2(
            args.data_dir,
            args.train_annotations,
            args.val_data_dir,
            args.val_annotations
        )
        
    elif config.model_type == "yolo":
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available. Install with: pip install ultralytics")
            return
        
        # Create YOLO data.yaml
        data_yaml = create_yolo_data_yaml(
            args.data_dir,
            Path(args.data_dir) / "data.yaml",
            config.categories
        )
        
        model_path = trainer.train_yolo(data_yaml)
        
    elif config.model_type == "custom":
        # Create data loaders for custom training
        train_dataset = DocumentLayoutDataset(args.data_dir, args.train_annotations)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        val_loader = None
        if args.val_annotations:
            val_dataset = DocumentLayoutDataset(
                args.val_data_dir or args.data_dir,
                args.val_annotations
            )
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        model_path = trainer.train_custom_cnn(train_loader, val_loader)
    
    else:
        logger.error(f"Unknown model type: {config.model_type}")
        return
    
    logger.success(f"Training completed! Model saved to: {model_path}")
    
    # Update layout detector configuration to use trained model
    config.weights_path = model_path
    config.save(args.config.replace('.yaml', '_trained.yaml'))
    logger.info(f"Updated config saved with trained model path")


if __name__ == "__main__":
    main()

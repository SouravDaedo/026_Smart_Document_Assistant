"""
Hugging Face Integration for Layout Detection Training.
Uses pre-trained models and datasets from Hugging Face Hub.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from transformers import (
        LayoutLMv3ForTokenClassification,
        LayoutLMv3Processor,
        AutoProcessor,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback
    )
    from datasets import load_dataset, Dataset, DatasetDict
    import evaluate
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.error("Hugging Face transformers not available. Install with: pip install transformers datasets")

from PIL import Image
import numpy as np


class HuggingFaceLayoutTrainer:
    """Trainer for layout detection using Hugging Face models."""
    
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base"):
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face transformers required")
        
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DocLayNet categories
        self.id2label = {
            0: "Caption",
            1: "Footnote", 
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title"
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        
    def load_model(self):
        """Load pre-trained model and processor."""
        logger.info(f"Loading model: {self.model_name}")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)
        
    def load_dataset_from_hub(self, dataset_name: str = "nielsr/docLayNet-base") -> DatasetDict:
        """Load dataset from Hugging Face Hub."""
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            dataset = load_dataset(dataset_name)
            return dataset
        except Exception as e:
            logger.warning(f"Could not load {dataset_name} from Hub: {e}")
            return self.load_local_dataset()
    
    def load_local_dataset(self) -> DatasetDict:
        """Load local DocLayNet dataset."""
        data_dir = Path("data/doclaynet")
        
        if not (data_dir / "COCO").exists():
            raise FileNotFoundError(
                "DocLayNet not found. Extract data/doclaynet/DocLayNet_core.zip first"
            )
        
        # Convert COCO format to Hugging Face dataset
        train_dataset = self._coco_to_hf_dataset(
            str(data_dir / "PNG/train"),
            str(data_dir / "COCO/train.json")
        )
        
        val_dataset = self._coco_to_hf_dataset(
            str(data_dir / "PNG/val"),
            str(data_dir / "COCO/val.json")
        )
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
    
    def _coco_to_hf_dataset(self, image_dir: str, annotations_file: str) -> Dataset:
        """Convert COCO format to Hugging Face dataset."""
        import json
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image index
        images = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Convert to HF format
        examples = []
        for img_id, img_info in images.items():
            image_path = Path(image_dir) / img_info['file_name']
            if not image_path.exists():
                continue
                
            annotations = image_annotations.get(img_id, [])
            
            # Convert bounding boxes and labels
            bboxes = []
            labels = []
            for ann in annotations:
                x, y, w, h = ann['bbox']
                # Normalize to [0, 1]
                bbox = [
                    x / img_info['width'],
                    y / img_info['height'],
                    (x + w) / img_info['width'],
                    (y + h) / img_info['height']
                ]
                bboxes.append(bbox)
                labels.append(ann['category_id'])
            
            examples.append({
                'image': str(image_path),
                'bboxes': bboxes,
                'labels': labels
            })
        
        return Dataset.from_list(examples)
    
    def preprocess_data(self, examples):
        """Preprocess data for training."""
        images = [Image.open(path).convert('RGB') for path in examples['image']]
        
        # Process with LayoutLM processor
        encoding = self.processor(
            images,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return encoding
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        predictions = predictions[labels != -100].flatten()
        labels = labels[labels != -100].flatten()
        
        # Load metrics
        metric = evaluate.load("seqeval")
        
        # Convert to label names
        true_predictions = [
            [self.id2label[p] for p in prediction]
            for prediction in predictions
        ]
        true_labels = [
            [self.id2label[l] for l in label]
            for label in labels
        ]
        
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def train(self, 
              output_dir: str = "models/hf_layout_detection",
              num_epochs: int = 10,
              batch_size: int = 4,
              learning_rate: float = 2e-5,
              dataset_name: Optional[str] = None):
        """Train the model."""
        
        # Load model and data
        self.load_model()
        
        if dataset_name:
            dataset = self.load_dataset_from_hub(dataset_name)
        else:
            dataset = self.load_local_dataset()
        
        # Preprocess datasets
        train_dataset = dataset["train"].map(
            self.preprocess_data,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        eval_dataset = dataset["validation"].map(
            self.preprocess_data,
            batched=True,
            remove_columns=dataset["validation"].column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_steps=500,
            save_steps=1000,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_num_workers=2,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        
        logger.success(f"Training completed! Model saved to: {output_dir}")
        return output_dir
    
    def push_to_hub(self, model_path: str, repo_name: str, token: str):
        """Push trained model to Hugging Face Hub."""
        from huggingface_hub import HfApi
        
        api = HfApi()
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            token=token,
            repo_type="model"
        )
        
        logger.success(f"Model pushed to Hub: https://huggingface.co/{repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Train layout detection with Hugging Face")
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base", 
                       help="Pre-trained model name")
    parser.add_argument("--dataset-name", help="Dataset name from Hub (optional)")
    parser.add_argument("--output-dir", default="models/hf_layout_detection",
                       help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--push-to-hub", help="Push to Hub repo name")
    parser.add_argument("--hf-token", help="Hugging Face token for Hub")
    
    args = parser.parse_args()
    
    if not HF_AVAILABLE:
        logger.error("Install Hugging Face: pip install transformers datasets evaluate")
        return
    
    # Initialize trainer
    trainer = HuggingFaceLayoutTrainer(args.model_name)
    
    # Train model
    model_path = trainer.train(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dataset_name=args.dataset_name
    )
    
    # Push to Hub if requested
    if args.push_to_hub and args.hf_token:
        trainer.push_to_hub(model_path, args.push_to_hub, args.hf_token)


if __name__ == "__main__":
    main()

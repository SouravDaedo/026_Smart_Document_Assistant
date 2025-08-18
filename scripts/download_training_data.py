"""
Download and prepare training datasets for CNN layout detection.
Supports multiple document layout datasets including PubLayNet, DocLayNet, and architectural-specific datasets.
"""

import os
import requests
import zipfile
import tarfile
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm
from loguru import logger
import hashlib


class DatasetDownloader:
    """Download and manage training datasets for document layout detection."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Dataset configurations
        self.datasets = {
            "publaynet": {
                "name": "PubLayNet",
                "description": "Large-scale document layout detection dataset with 360k+ images",
                "url": "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz",
                "size": "96GB",
                "categories": ["text", "title", "list", "table", "figure"],
                "format": "COCO",
                "paper": "https://arxiv.org/abs/1908.07836"
            },
            "doclaynet": {
                "name": "DocLayNet",
                "description": "Human-annotated document layout segmentation dataset",
                "url": "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip",
                "size": "7GB",
                "categories": ["Caption", "Footnote", "Formula", "List-item", "Page-footer", 
                             "Page-header", "Picture", "Section-header", "Table", "Text", "Title"],
                "format": "COCO",
                "paper": "https://arxiv.org/abs/2206.01062"
            },
            "prima": {
                "name": "PRImA Layout Analysis Dataset",
                "description": "Page layout analysis dataset with ground truth",
                "url": "https://www.primaresearch.org/datasets/Layout_Analysis",
                "size": "500MB",
                "categories": ["text-region", "image-region", "line-drawing", "graphic", "table"],
                "format": "PAGE XML",
                "paper": "https://www.primaresearch.org/publications"
            },
            "rvl_cdip": {
                "name": "RVL-CDIP",
                "description": "Document image classification dataset (400k images, 16 classes)",
                "url": "https://www.cs.cmu.edu/~aharley/rvl-cdip/",
                "size": "45GB",
                "categories": ["letter", "form", "email", "handwritten", "advertisement", 
                             "scientific report", "scientific publication", "specification", 
                             "file folder", "news article", "budget", "invoice", 
                             "presentation", "questionnaire", "resume", "memo"],
                "format": "Images + Labels",
                "paper": "https://arxiv.org/abs/1502.07058"
            },
            "tablebank": {
                "name": "TableBank",
                "description": "Table detection and recognition dataset",
                "url": "https://github.com/doc-analysis/TableBank",
                "size": "20GB",
                "categories": ["table"],
                "format": "COCO + LaTeX",
                "paper": "https://arxiv.org/abs/1903.01949"
            },
            "funsd": {
                "name": "FUNSD",
                "description": "Form Understanding in Noisy Scanned Documents",
                "url": "https://guillaumejaume.github.io/FUNSD/download/",
                "size": "20MB",
                "categories": ["question", "answer", "header", "other"],
                "format": "JSON + Images",
                "paper": "https://arxiv.org/abs/1905.13538"
            }
        }
    
    def list_datasets(self):
        """List all available datasets with details."""
        print("\n📊 Available Training Datasets for CNN Layout Detection:\n")
        print("=" * 80)
        
        for key, dataset in self.datasets.items():
            print(f"\n🔹 {dataset['name']} ({key})")
            print(f"   Description: {dataset['description']}")
            print(f"   Size: {dataset['size']}")
            print(f"   Categories: {', '.join(dataset['categories'])}")
            print(f"   Format: {dataset['format']}")
            print(f"   Paper: {dataset['paper']}")
            print(f"   Status: {' Downloaded' if self._is_downloaded(key) else ' Not downloaded'}")
    
    def _is_downloaded(self, dataset_key: str) -> bool:
        """Check if dataset is already downloaded."""
        dataset_path = self.data_dir / dataset_key
        return dataset_path.exists() and any(dataset_path.iterdir())
    
    def _download_file(self, url: str, filepath: Path, chunk_size: int = 8192):
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def _extract_archive(self, archive_path: Path, extract_to: Path):
        """Extract archive file."""
        logger.info(f"Extracting {archive_path} to {extract_to}")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz'] or '.tar.' in archive_path.name:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    def download_publaynet(self):
        """Download PubLayNet dataset - best for general document layout."""
        dataset_key = "publaynet"
        dataset_info = self.datasets[dataset_key]
        dataset_dir = self.data_dir / dataset_key
        dataset_dir.mkdir(exist_ok=True)
        
        if self._is_downloaded(dataset_key):
            logger.info(f"PubLayNet already downloaded at {dataset_dir}")
            return dataset_dir
        
        logger.info("Downloading PubLayNet dataset (96GB) - this will take a while...")
        
        # Download main dataset
        archive_path = dataset_dir / "publaynet.tar.gz"
        self._download_file(dataset_info["url"], archive_path)
        
        # Extract
        self._extract_archive(archive_path, dataset_dir)
        
        # Clean up archive
        archive_path.unlink()
        
        logger.info(f"PubLayNet downloaded and extracted to {dataset_dir}")
        return dataset_dir
    
    def download_doclaynet(self):
        """Download DocLayNet dataset - good for diverse document types."""
        dataset_key = "doclaynet"
        dataset_info = self.datasets[dataset_key]
        dataset_dir = self.data_dir / dataset_key
        dataset_dir.mkdir(exist_ok=True)
        
        if self._is_downloaded(dataset_key):
            logger.info(f"DocLayNet already downloaded at {dataset_dir}")
            return dataset_dir
        
        logger.info("Downloading DocLayNet dataset (7GB)...")
        
        # Download main dataset
        archive_path = dataset_dir / "DocLayNet_core.zip"
        self._download_file(dataset_info["url"], archive_path)
        
        # Extract
        self._extract_archive(archive_path, dataset_dir)
        
        # Clean up archive
        archive_path.unlink()
        
        logger.info(f"DocLayNet downloaded and extracted to {dataset_dir}")
        return dataset_dir
    
    def download_funsd(self):
        """Download FUNSD dataset - good for form understanding."""
        dataset_key = "funsd"
        dataset_dir = self.data_dir / dataset_key
        dataset_dir.mkdir(exist_ok=True)
        
        if self._is_downloaded(dataset_key):
            logger.info(f"FUNSD already downloaded at {dataset_dir}")
            return dataset_dir
        
        logger.info("Downloading FUNSD dataset...")
        
        # FUNSD has separate train/test downloads
        base_url = "https://guillaumejaume.github.io/FUNSD/dataset/"
        files = [
            "training_data.zip",
            "testing_data.zip"
        ]
        
        for filename in files:
            file_path = dataset_dir / filename
            self._download_file(f"{base_url}{filename}", file_path)
            self._extract_archive(file_path, dataset_dir)
            file_path.unlink()
        
        logger.info(f"FUNSD downloaded and extracted to {dataset_dir}")
        return dataset_dir
    
    def download_tablebank(self):
        """Download TableBank dataset - specialized for table detection."""
        dataset_key = "tablebank"
        dataset_dir = self.data_dir / dataset_key
        dataset_dir.mkdir(exist_ok=True)
        
        if self._is_downloaded(dataset_key):
            logger.info(f"TableBank already downloaded at {dataset_dir}")
            return dataset_dir
        
        logger.info("Downloading TableBank dataset...")
        logger.warning("TableBank requires manual download from GitHub. Please visit:")
        logger.warning("https://github.com/doc-analysis/TableBank")
        
        return dataset_dir
    
    def download_architectural_samples(self):
        """Download sample architectural drawings for specialized training."""
        dataset_key = "architectural_samples"
        dataset_dir = self.data_dir / dataset_key
        dataset_dir.mkdir(exist_ok=True)
        
        if self._is_downloaded(dataset_key):
            logger.info(f"Architectural samples already downloaded at {dataset_dir}")
            return dataset_dir
        
        logger.info("Creating architectural sample dataset...")
        
        # Create sample architectural document structure
        categories = {
            "title_block": "Title blocks and drawing headers",
            "dimensions": "Dimension lines and measurements", 
            "symbols": "Architectural symbols and legends",
            "text_annotations": "Text labels and notes",
            "floor_plans": "Floor plan drawings",
            "elevations": "Building elevations",
            "sections": "Cross-sections and details"
        }
        
        # Create directory structure
        for category in categories.keys():
            (dataset_dir / "images" / category).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "annotations" / category).mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        metadata = {
            "name": "Architectural Document Samples",
            "description": "Sample architectural drawings for layout detection training",
            "categories": categories,
            "format": "COCO",
            "instructions": "Add your architectural PDF samples to the images folders and create COCO annotations"
        }
        
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Architectural sample structure created at {dataset_dir}")
        logger.info("Please add your architectural PDF samples to train the model on domain-specific data")
        
        return dataset_dir
    
    def download_dataset(self, dataset_key: str):
        """Download a specific dataset."""
        if dataset_key not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        download_methods = {
            "publaynet": self.download_publaynet,
            "doclaynet": self.download_doclaynet,
            "funsd": self.download_funsd,
            "tablebank": self.download_tablebank,
            "architectural_samples": self.download_architectural_samples
        }
        
        if dataset_key in download_methods:
            return download_methods[dataset_key]()
        else:
            logger.warning(f"No automatic download available for {dataset_key}")
            logger.info(f"Please visit: {self.datasets[dataset_key]['url']}")
    
    def download_recommended_for_architectural(self):
        """Download recommended datasets for architectural document analysis."""
        logger.info("🏗️ Downloading recommended datasets for architectural document analysis...")
        
        recommended = [
            ("doclaynet", "Best overall document layout dataset"),
            ("funsd", "Good for form-like architectural drawings"),
            ("architectural_samples", "Domain-specific architectural samples")
        ]
        
        downloaded_paths = []
        
        for dataset_key, reason in recommended:
            logger.info(f"\n📥 Downloading {dataset_key}: {reason}")
            try:
                path = self.download_dataset(dataset_key)
                downloaded_paths.append(path)
                logger.success(f"✅ {dataset_key} downloaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to download {dataset_key}: {e}")
        
        return downloaded_paths
    
    def create_training_config(self, datasets: List[str]):
        """Create training configuration file for selected datasets."""
        config = {
            "training": {
                "datasets": datasets,
                "data_dir": str(self.data_dir),
                "batch_size": 8,
                "learning_rate": 0.001,
                "epochs": 100,
                "validation_split": 0.2,
                "augmentation": {
                    "rotation": 5,
                    "brightness": 0.1,
                    "contrast": 0.1,
                    "flip": True
                }
            },
            "model": {
                "architecture": "detectron2",
                "backbone": "resnet50",
                "num_classes": 11,  # Adjust based on combined categories
                "input_size": [1024, 1024],
                "anchor_sizes": [32, 64, 128, 256, 512]
            },
            "categories": {
                "0": "background",
                "1": "text",
                "2": "title", 
                "3": "list",
                "4": "table",
                "5": "figure",
                "6": "dimension",
                "7": "symbol",
                "8": "drawing",
                "9": "annotation",
                "10": "other"
            }
        }
        
        config_path = self.data_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training configuration saved to {config_path}")
        return config_path


def main():
    parser = argparse.ArgumentParser(description="Download training datasets for CNN layout detection")
    parser.add_argument("--data-dir", default="data", help="Directory to store datasets")
    parser.add_argument("--dataset", help="Specific dataset to download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--recommended", action="store_true", help="Download recommended datasets for architectural documents")
    parser.add_argument("--all", action="store_true", help="Download all available datasets")
    parser.add_argument("--cloud-storage", choices=["s3", "gcs", "azure"], help="Upload to cloud storage instead of local")
    parser.add_argument("--bucket-name", help="Cloud storage bucket name")
    parser.add_argument("--streaming", action="store_true", help="Generate streaming dataset configuration")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.list:
        downloader.list_datasets()
    elif args.recommended:
        paths = downloader.download_recommended_for_architectural()
        config_path = downloader.create_training_config(["doclaynet", "funsd", "architectural_samples"])
        print(f"\n✅ Recommended datasets downloaded. Training config: {config_path}")
    elif args.dataset:
        downloader.download_dataset(args.dataset)
    elif args.all:
        for dataset_key in downloader.datasets.keys():
            try:
                downloader.download_dataset(dataset_key)
            except Exception as e:
                logger.error(f"Failed to download {dataset_key}: {e}")
    else:
        downloader.list_datasets()
        print("\nUse --recommended to download datasets suitable for architectural documents")
        print("Use --dataset <name> to download a specific dataset")


if __name__ == "__main__":
    main()

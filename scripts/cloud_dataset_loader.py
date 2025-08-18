"""
Cloud-based dataset loaders for training without local storage.
Supports streaming from AWS S3, Google Cloud Storage, and Azure Blob Storage.
"""

import json
import boto3
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from google.cloud import storage as gcs
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


class S3DocumentLayoutDataset(Dataset):
    """Stream training data directly from AWS S3."""
    
    def __init__(self, bucket_name: str, prefix: str, annotations_s3_key: str, 
                 transform=None, aws_access_key_id=None, aws_secret_access_key=None):
        """
        Args:
            bucket_name: S3 bucket name
            prefix: S3 prefix for images (e.g., 'doclaynet/PNG/train')
            annotations_s3_key: S3 key for annotations file (e.g., 'doclaynet/COCO/train.json')
            transform: PyTorch transforms
        """
        # Initialize S3 client
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            self.s3_client = boto3.client('s3')  # Uses default credentials
        
        self.bucket = bucket_name
        self.prefix = prefix
        self.transform = transform
        
        # Download annotations file (small, can cache locally)
        logger.info(f"Loading annotations from s3://{bucket_name}/{annotations_s3_key}")
        try:
            annotations_obj = self.s3_client.get_object(Bucket=bucket_name, Key=annotations_s3_key)
            self.coco_data = json.loads(annotations_obj['Body'].read())
        except Exception as e:
            logger.error(f"Failed to load annotations: {e}")
            raise
        
        # Index images and annotations
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = list(self.images.keys())
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        logger.info(f"Loaded {len(self.image_ids)} images from S3")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Check local cache first
        cache_path = Path("cache") / f"{img_info['file_name']}"
        if cache_path.exists():
            try:
                image = Image.open(cache_path).convert('RGB')
            except Exception:
                # Cache corrupted, download fresh
                image = self._download_and_cache_image(img_info, cache_path)
        else:
            # Download and cache
            image = self._download_and_cache_image(img_info, cache_path)
        
        # Get annotations for this image
        annotations = self.image_annotations.get(img_id, [])
        
        # Convert to training format
        boxes = []
        labels = []
        for ann in annotations:
            # Convert COCO format [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor(img_id)
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def _download_and_cache_image(self, img_info, cache_path):
        """Download image from S3 and cache locally."""
        img_key = f"{self.prefix}/{img_info['file_name']}"
        try:
            # Create cache directory
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download from S3
            img_obj = self.s3_client.get_object(Bucket=self.bucket, Key=img_key)
            image_data = img_obj['Body'].read()
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                f.write(image_data)
            
            # Load image
            image = Image.open(BytesIO(image_data)).convert('RGB')
            logger.info(f"Downloaded and cached: {img_key}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to download image {img_key}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='white')


class GCSDocumentLayoutDataset(Dataset):
    """Stream training data directly from Google Cloud Storage."""
    
    def __init__(self, bucket_name: str, prefix: str, annotations_blob_name: str, 
                 transform=None, credentials_path: Optional[str] = None):
        """
        Args:
            bucket_name: GCS bucket name
            prefix: GCS prefix for images
            annotations_blob_name: GCS blob name for annotations
            credentials_path: Path to service account JSON file
        """
        # Initialize GCS client
        if credentials_path:
            self.client = gcs.Client.from_service_account_json(credentials_path)
        else:
            self.client = gcs.Client()  # Uses default credentials
        
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix
        self.transform = transform
        
        # Download annotations
        logger.info(f"Loading annotations from gs://{bucket_name}/{annotations_blob_name}")
        annotations_blob = self.bucket.blob(annotations_blob_name)
        annotations_data = annotations_blob.download_as_text()
        self.coco_data = json.loads(annotations_data)
        
        # Index data
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = list(self.images.keys())
        
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        logger.info(f"Loaded {len(self.image_ids)} images from GCS")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Stream image from GCS
        blob_name = f"{self.prefix}/{img_info['file_name']}"
        try:
            blob = self.bucket.blob(blob_name)
            image_bytes = blob.download_as_bytes()
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {blob_name}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        # Process annotations
        annotations = self.image_annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor(img_id)
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


class AzureBlobDataset(Dataset):
    """Stream training data from Azure Blob Storage."""
    
    def __init__(self, account_name: str, container_name: str, prefix: str, 
                 annotations_blob_name: str, account_key: str, transform=None):
        """
        Args:
            account_name: Azure storage account name
            container_name: Blob container name
            prefix: Blob prefix for images
            annotations_blob_name: Blob name for annotations
            account_key: Azure storage account key
        """
        # Initialize Azure client
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=account_key
        )
        self.container_name = container_name
        self.prefix = prefix
        self.transform = transform
        
        # Download annotations
        logger.info(f"Loading annotations from Azure blob: {annotations_blob_name}")
        blob_client = self.blob_service_client.get_blob_client(
            container=container_name, blob=annotations_blob_name
        )
        annotations_data = blob_client.download_blob().readall()
        self.coco_data = json.loads(annotations_data)
        
        # Index data
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = list(self.images.keys())
        
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        logger.info(f"Loaded {len(self.image_ids)} images from Azure")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Stream image from Azure
        blob_name = f"{self.prefix}/{img_info['file_name']}"
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )
            image_bytes = blob_client.download_blob().readall()
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {blob_name}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        # Process annotations
        annotations = self.image_annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor(img_id)
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


def create_cloud_dataloader(storage_type: str, config: Dict, batch_size: int = 4, 
                           num_workers: int = 2, transform=None) -> DataLoader:
    """
    Create a DataLoader for cloud-stored datasets.
    
    Args:
        storage_type: 's3', 'gcs', or 'azure'
        config: Storage configuration dictionary
        batch_size: Batch size for training
        num_workers: Number of worker processes
        transform: PyTorch transforms
    
    Returns:
        DataLoader instance
    """
    if storage_type == 's3':
        dataset = S3DocumentLayoutDataset(
            bucket_name=config['bucket_name'],
            prefix=config['prefix'],
            annotations_s3_key=config['annotations_key'],
            transform=transform,
            aws_access_key_id=config.get('aws_access_key_id'),
            aws_secret_access_key=config.get('aws_secret_access_key')
        )
    elif storage_type == 'gcs':
        dataset = GCSDocumentLayoutDataset(
            bucket_name=config['bucket_name'],
            prefix=config['prefix'],
            annotations_blob_name=config['annotations_blob'],
            transform=transform,
            credentials_path=config.get('credentials_path')
        )
    elif storage_type == 'azure':
        dataset = AzureBlobDataset(
            account_name=config['account_name'],
            container_name=config['container_name'],
            prefix=config['prefix'],
            annotations_blob_name=config['annotations_blob'],
            account_key=config['account_key'],
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")
    
    # Custom collate function for object detection
    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


# Example usage configurations
CLOUD_CONFIGS = {
    's3_example': {
        'bucket_name': 'my-training-datasets',
        'prefix': 'doclaynet/PNG/train',
        'annotations_key': 'doclaynet/COCO/train.json',
        'aws_access_key_id': None,  # Use IAM role or ~/.aws/credentials
        'aws_secret_access_key': None
    },
    'gcs_example': {
        'bucket_name': 'my-training-datasets',
        'prefix': 'doclaynet/PNG/train',
        'annotations_blob': 'doclaynet/COCO/train.json',
        'credentials_path': None  # Use default credentials
    },
    'azure_example': {
        'account_name': 'mytrainingdata',
        'container_name': 'datasets',
        'prefix': 'doclaynet/PNG/train',
        'annotations_blob': 'doclaynet/COCO/train.json',
        'account_key': 'your-account-key'
    }
}


if __name__ == "__main__":
    # Example: Create S3 dataloader
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load from S3
    s3_config = CLOUD_CONFIGS['s3_example']
    dataloader = create_cloud_dataloader('s3', s3_config, batch_size=4, transform=transform)
    
    print(f"Created dataloader with {len(dataloader.dataset)} samples")
    
    # Test loading a batch
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}: {len(images)} images")
        if batch_idx == 0:  # Just test first batch
            break

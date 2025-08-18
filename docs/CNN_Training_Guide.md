# CNN Training Guide for Architectural Document Layout Detection

## 🎯 Overview

This guide explains how to download and prepare training data for CNN models specifically designed for architectural document layout detection. The goal is to train models that can identify and classify different regions in architectural plans, drawings, and technical documents.

## Recommended Datasets for Architectural Documents

### 1. **DocLayNet** (Highest Priority)
- **Size**: 7GB
- **Images**: 80,863 document pages
- **Categories**: 11 classes including Caption, Formula, List-item, Page-header, Picture, Section-header, Table, Text, Title
- **Why Best for Architecture**: Diverse document types, high-quality human annotations, includes technical documents
- **Download**: `python scripts/download_training_data.py --dataset doclaynet`

### 2. **PubLayNet** (Good for Large-Scale Training)
- **Size**: 96GB 
- **Images**: 360,000+ document pages
- **Categories**: 5 classes (text, title, list, table, figure)
- **Why Useful**: Massive scale for pre-training, good generalization
- **Note**: Very large download, consider if you have sufficient storage and bandwidth
- **Download**: `python scripts/download_training_data.py --dataset publaynet`

### 3. **FUNSD** (Forms and Structured Documents)
- **Size**: 20MB
- **Images**: 199 forms
- **Categories**: question, answer, header, other
- **Why Useful**: Good for title blocks, form-like architectural drawings, structured layouts
- **Download**: `python scripts/download_training_data.py --dataset funsd`

### 4. **TableBank** (Table Detection)
- **Size**: 20GB
- **Focus**: Table detection and recognition
- **Why Useful**: Architectural drawings often contain schedules, legends, and tabular data
- **Download**: Manual from GitHub (link provided in script)

## Architectural-Specific Considerations

### Categories Relevant to Architectural Documents:
1. **Title Blocks** - Drawing headers with project info
2. **Dimensions** - Measurement lines and text
3. **Symbols** - Architectural symbols and legends  
4. **Text Annotations** - Labels, notes, specifications
5. **Drawing Elements** - Floor plans, elevations, sections
6. **Tables/Schedules** - Door/window schedules, material lists
7. **North Arrows** - Orientation indicators
8. **Scale Indicators** - Drawing scale information

### Custom Dataset Creation:
For best results with architectural documents, you should also create a custom dataset:

```bash
# Create architectural sample structure
python scripts/download_training_data.py --dataset architectural_samples
```

This creates a folder structure where you can add your own architectural PDFs and create annotations.

##  Quick Start - Download Recommended Datasets

```bash
# Download the recommended combination for architectural documents
python scripts/download_training_data.py --recommended

# This downloads:
# - DocLayNet (best overall)
# - FUNSD (good for structured layouts)  
# - Architectural samples (custom structure)
```

## 📋 Training Data Requirements

### Minimum Requirements:
- **Images**: 1,000+ annotated architectural documents
- **Categories**: 8-12 layout categories
- **Format**: COCO JSON format for annotations
- **Resolution**: 1024x1024 pixels minimum
- **Storage**: 10GB+ available space

### Optimal Requirements:
- **Images**: 10,000+ annotated documents
- **Mix**: 70% general documents + 30% architectural-specific
- **Augmentation**: Rotation, scaling, brightness variations
- **Validation**: 20% held-out test set

## 🔧 Data Preprocessing Pipeline

### 1. Image Preprocessing:
```python
# Resize images to consistent dimensions
target_size = (1024, 1024)

# Normalize pixel values
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### 2. Annotation Format (COCO):
```json
{
  "images": [{"id": 1, "file_name": "plan_001.jpg", "width": 1024, "height": 1024}],
  "annotations": [
    {
      "id": 1,
      "image_id": 1, 
      "category_id": 2,
      "bbox": [x, y, width, height],
      "area": 12345,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "text"},
    {"id": 2, "name": "title"},
    {"id": 3, "name": "dimension"}
  ]
}
```

## 🎛️ Model Architecture Recommendations

### For Architectural Documents:

1. **Detectron2 with ResNet-50 Backbone**
   - Good balance of speed and accuracy
   - Pre-trained on COCO dataset
   - Supports instance segmentation

2. **YOLOv8 for Real-time Detection**
   - Faster inference
   - Good for production deployment
   - Easier to deploy

3. **Custom CNN Architecture**
   - Specialized for document layouts
   - Can incorporate architectural domain knowledge

## 📈 Training Configuration

### Recommended Hyperparameters:
```yaml
training:
  batch_size: 8
  learning_rate: 0.001
  epochs: 100
  validation_split: 0.2
  early_stopping: true
  
model:
  architecture: "detectron2"
  backbone: "resnet50"
  num_classes: 11
  input_size: [1024, 1024]
  
augmentation:
  rotation: 5  # degrees
  brightness: 0.1
  contrast: 0.1
  horizontal_flip: true
```

## 📁 Directory Structure After Download

```
data/
├── doclaynet/
│   ├── COCO/
│   ├── PNG/
│   └── annotations/
├── funsd/
│   ├── training_data/
│   └── testing_data/
├── architectural_samples/
│   ├── images/
│   ├── annotations/
│   └── metadata.json
└── training_config.json
```

## 🔄 Training Workflow

1. **Download Data**: Use the download script
2. **Preprocess**: Convert to consistent format
3. **Augment**: Apply data augmentation
4. **Split**: Train/validation/test splits
5. **Train**: Run training with monitoring
6. **Evaluate**: Test on architectural documents
7. **Fine-tune**: Adjust for architectural specifics

## 💡 Tips for Best Results

### Data Quality:
- Ensure high-resolution scans (300+ DPI)
- Include variety of architectural drawing types
- Balance dataset across different categories
- Quality over quantity for annotations

### Training Strategy:
- Start with pre-trained models (transfer learning)
- Fine-tune on architectural-specific data
- Use progressive resizing during training
- Monitor validation metrics closely

### Domain Adaptation:
- Mix general document data with architectural data
- Create synthetic architectural layouts
- Use data augmentation specific to documents
- Consider multi-task learning approaches

## 🚀 Getting Started Commands

```bash
# 1. List available datasets
python scripts/download_training_data.py --list

# 2. Download recommended datasets (7GB+)
python scripts/download_training_data.py --recommended

# 3. Download specific dataset
python scripts/download_training_data.py --dataset doclaynet

# 4. Create training configuration
# (automatically created with --recommended)
```

## 📚 Additional Resources

- **DocLayNet Paper**: https://arxiv.org/abs/2206.01062
- **PubLayNet Paper**: https://arxiv.org/abs/1908.07836
- **Detectron2 Documentation**: https://detectron2.readthedocs.io/
- **COCO Format Guide**: https://cocodataset.org/#format-data

## ⚠️ Important Notes

1. **Storage Requirements**: Ensure sufficient disk space (10GB+ recommended)
2. **Internet Connection**: Large downloads require stable connection
3. **Legal Compliance**: Respect dataset licenses and terms of use
4. **GPU Requirements**: Training requires CUDA-capable GPU for reasonable speed
5. **Memory**: 16GB+ RAM recommended for large datasets

Start with the recommended datasets using `--recommended` flag for the best balance of quality and training time for architectural document analysis.

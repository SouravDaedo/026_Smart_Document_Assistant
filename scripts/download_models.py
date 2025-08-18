"""
Model download script for PlanQuery.
Downloads and sets up required ML models.
"""

import os
import sys
from pathlib import Path
import yaml
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Download sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Downloading sentence transformer model: {model_name}")
        
        model = SentenceTransformer(model_name)
        logger.info(f"Model {model_name} downloaded successfully")
        
        # Test the model
        test_embedding = model.encode("test sentence")
        logger.info(f"Model test successful, embedding dimension: {len(test_embedding)}")
        
    except ImportError:
        logger.warning("sentence-transformers not available, skipping download")
    except Exception as e:
        logger.error(f"Failed to download sentence transformer: {e}")


def download_detectron2_model():
    """Download Detectron2 model for layout detection."""
    try:
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        
        logger.info("Setting up Detectron2 model...")
        
        cfg = get_cfg()
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        
        # This will download the model
        predictor = DefaultPredictor(cfg)
        logger.info("Detectron2 model downloaded successfully")
        
    except ImportError:
        logger.warning("Detectron2 not available, skipping download")
    except Exception as e:
        logger.error(f"Failed to download Detectron2 model: {e}")


def download_yolo_model():
    """Download YOLO model."""
    try:
        from ultralytics import YOLO
        
        logger.info("Downloading YOLO model...")
        model = YOLO('yolov8n.pt')  # This will download the model
        logger.info("YOLO model downloaded successfully")
        
    except ImportError:
        logger.warning("ultralytics not available, skipping download")
    except Exception as e:
        logger.error(f"Failed to download YOLO model: {e}")


def download_trocr_model(model_name: str = "microsoft/trocr-base-printed"):
    """Download TrOCR model."""
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        logger.info(f"Downloading TrOCR model: {model_name}")
        
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        logger.info("TrOCR model downloaded successfully")
        
    except ImportError:
        logger.warning("transformers not available, skipping TrOCR download")
    except Exception as e:
        logger.error(f"Failed to download TrOCR model: {e}")


def setup_tesseract():
    """Check Tesseract installation."""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract found, version: {version}")
        
    except ImportError:
        logger.warning("pytesseract not available")
    except Exception as e:
        logger.warning(f"Tesseract not properly configured: {e}")
        logger.info("Please install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")


def main():
    """Main model download function."""
    logger.info("Starting model downloads...")
    
    # Load configuration
    config_path = Path("config.yaml")
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Download models based on configuration
    processing_config = config.get('processing', {})
    search_config = config.get('search', {})
    
    # Sentence transformer for embeddings
    model_name = search_config.get('vector_store', {}).get('model_name', 'all-MiniLM-L6-v2')
    download_sentence_transformer(model_name)
    
    # Layout detection models
    layout_model = processing_config.get('layout_detection', {}).get('model_type', 'detectron2')
    if layout_model == 'detectron2':
        download_detectron2_model()
    elif layout_model == 'yolo':
        download_yolo_model()
    
    # OCR models
    ocr_engine = processing_config.get('ocr', {}).get('engine', 'tesseract')
    if ocr_engine == 'tesseract':
        setup_tesseract()
    elif ocr_engine == 'trocr':
        trocr_model = processing_config.get('ocr', {}).get('model_name', 'microsoft/trocr-base-printed')
        download_trocr_model(trocr_model)
    
    logger.info("Model download process completed!")


if __name__ == "__main__":
    main()

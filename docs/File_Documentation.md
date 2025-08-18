# PlanQuery File Documentation

## 📁 Complete File Structure and Descriptions

This document provides detailed descriptions of what each file does in the PlanQuery Smart Document Assistant project.

---

## 🏗️ Root Directory Files

### **README.md**
- **Purpose**: Main project documentation and overview
- **Contains**: Installation instructions, usage examples, architecture overview
- **Audience**: Developers and users getting started with PlanQuery

### **requirements.txt**
- **Purpose**: Python package dependencies
- **Contains**: All required Python libraries with version specifications
- **Usage**: `pip install -r requirements.txt`

### **setup.py**
- **Purpose**: Python package installation configuration
- **Contains**: Package metadata, dependencies, entry points
- **Usage**: `pip install -e .` for development installation

### **config.yaml**
- **Purpose**: Global system configuration
- **Contains**: Database URLs, API settings, processing parameters, model paths
- **Configures**: All major system components and their settings

### **.env.example**
- **Purpose**: Environment variable template
- **Contains**: Example configuration for secrets and environment-specific settings
- **Usage**: Copy to `.env` and customize for your environment

### **Dockerfile**
- **Purpose**: Container image definition
- **Contains**: Instructions to build Docker image with all dependencies
- **Usage**: `docker build -t planquery .`

### **docker-compose.yml**
- **Purpose**: Multi-container application orchestration
- **Contains**: PostgreSQL, Redis, API server, and Nginx configuration
- **Usage**: `docker-compose up` to run complete system

---

## 📦 planquery/ - Main Package

### **__init__.py**
- **Purpose**: Package initialization
- **Contains**: Version information and package-level imports
- **Makes**: `planquery` importable as a Python package

---

## 🔧 planquery/core/ - Document Processing Engine

### **pdf_processor.py**
- **Purpose**: PDF document processing and image extraction
- **Functions**:
  - Convert PDF pages to high-resolution images
  - Extract metadata (title, page count, file info)
  - Handle multi-page documents
  - Manage output directory structure
- **Dependencies**: PyMuPDF (fitz), Pillow
- **Output**: PNG images of PDF pages + metadata

### **layout_detector.py**
- **Purpose**: Computer vision-based layout region detection
- **Functions**:
  - Detect text blocks, titles, tables, symbols in document images
  - Support multiple CNN backends (Detectron2, YOLO, custom)
  - Filter and post-process detected regions
  - Provide confidence scores for detections
- **Models**: Pre-trained on COCO, fine-tunable on DocLayNet
- **Output**: Bounding boxes with region type classifications

### **ocr_engine.py**
- **Purpose**: Optical Character Recognition (text extraction)
- **Functions**:
  - Extract text from detected layout regions
  - Support multiple OCR engines (Tesseract, PaddleOCR, EasyOCR)
  - Handle different text orientations and fonts
  - Provide confidence scores for extracted text
- **Dependencies**: pytesseract, paddleocr, easyocr
- **Output**: Text content with confidence scores and character-level coordinates

### **text_classifier.py**
- **Purpose**: Classify extracted text by type and purpose
- **Functions**:
  - Identify text types: dimensions, annotations, titles, specifications
  - Use rule-based patterns and ML models
  - Extract structured information (measurements, room names, etc.)
  - Handle architectural terminology and abbreviations
- **Categories**: Dimension, annotation, title, specification, room_label, equipment_tag
- **Output**: Classified text with type labels and extracted structured data

### **floor_normalizer.py**
- **Purpose**: Standardize floor/level identifications across documents
- **Functions**:
  - Parse various floor naming conventions (L1, Level 1, First Floor, etc.)
  - Create canonical floor IDs for consistent searching
  - Handle basement levels, mezzanines, and roof levels
  - Map discipline-specific floor references
- **Patterns**: Regex-based pattern matching for floor identification
- **Output**: Canonical floor IDs and normalized floor information

### **__init__.py**
- **Purpose**: Core module initialization
- **Contains**: Imports for all core processing classes
- **Exports**: Main processing pipeline components

---

## 🗄️ planquery/indexing/ - Data Storage and Indexing

### **database.py**
- **Purpose**: PostgreSQL database operations and schema management
- **Functions**:
  - Define database schema for documents, pages, regions, and text
  - CRUD operations for all data entities
  - Database connection management and pooling
  - Migration and schema updates
- **Tables**: documents, pages, regions, floors, processing_jobs
- **Features**: Full-text search, spatial indexing, metadata storage

### **vector_store.py**
- **Purpose**: Vector embeddings storage and similarity search
- **Functions**:
  - Generate embeddings for text content using sentence transformers
  - Store vectors in FAISS or pgvector
  - Perform semantic similarity searches
  - Manage embedding model lifecycle
- **Backends**: FAISS (file-based), pgvector (PostgreSQL extension)
- **Models**: sentence-transformers/all-MiniLM-L6-v2 (default)

### **keyword_index.py**
- **Purpose**: Traditional keyword-based search indexing
- **Functions**:
  - Build inverted indexes for fast keyword search
  - Support BM25 ranking algorithm
  - Handle architectural terminology and abbreviations
  - Provide faceted search capabilities
- **Backend**: Whoosh search library
- **Features**: Fuzzy matching, phrase search, boolean queries

### **__init__.py**
- **Purpose**: Indexing module initialization
- **Contains**: Database and search component imports
- **Exports**: DatabaseManager, VectorStore, KeywordIndex classes

---

## 🔍 planquery/search/ - Search and Retrieval Pipeline

### **query_parser.py**
- **Purpose**: Natural language query understanding and parsing
- **Functions**:
  - Parse user queries into structured search parameters
  - Extract intent (search, count, location, comparison)
  - Identify filters (floor, discipline, text type)
  - Handle architectural domain-specific queries
- **Features**: Intent classification, entity extraction, filter parsing
- **Output**: ParsedQuery objects with search parameters and filters

### **retriever.py**
- **Purpose**: Hybrid search implementation combining vector and keyword search
- **Functions**:
  - Execute parallel vector and keyword searches
  - Combine and weight results from different search methods
  - Apply filters (floor, discipline, confidence thresholds)
  - Rank results by relevance and confidence
- **Algorithm**: Weighted combination of semantic and lexical similarity
- **Output**: Ranked list of search results with metadata

### **reranker.py**
- **Purpose**: Advanced result reranking and relevance optimization
- **Functions**:
  - Rerank search results using advanced scoring algorithms
  - Consider query-document similarity, text type relevance, spatial proximity
  - Apply domain-specific ranking factors for architectural documents
  - Optimize result diversity and coverage
- **Features**: Multi-factor scoring, diversity optimization, result clustering
- **Output**: Optimally ranked and filtered search results

### **__init__.py**
- **Purpose**: Search module initialization
- **Contains**: Search pipeline component imports
- **Exports**: QueryParser, HybridRetriever, ResultReranker classes

---

## 🌐 planquery/api/ - Web API and Backend

### **main.py**
- **Purpose**: FastAPI application server and REST API endpoints
- **Functions**:
  - HTTP API for document upload, processing, and search
  - Chat interface for natural language queries
  - Document viewer API endpoints
  - Health checks and system statistics
- **Endpoints**: `/chat`, `/search`, `/upload`, `/documents`, `/viewer`
- **Features**: Async processing, background tasks, CORS support

### **models.py**
- **Purpose**: Pydantic data models for API request/response validation
- **Functions**:
  - Define request and response schemas
  - Data validation and serialization
  - Type hints for API documentation
  - Error handling and validation messages
- **Models**: ChatRequest, SearchResponse, UploadResponse, DocumentInfo
- **Features**: Automatic API documentation, request validation

### **__init__.py**
- **Purpose**: API module initialization
- **Contains**: FastAPI app and model imports
- **Exports**: Main FastAPI application instance

---

## 🎨 planquery/static/ - Web Interface Assets

### **viewer.html**
- **Purpose**: Interactive document viewer web interface
- **Functions**:
  - Display PDF pages with zoom and navigation
  - Overlay bounding boxes for detected regions
  - Show region details on click/hover
  - Provide search and filtering capabilities
- **Features**: Responsive design, keyboard shortcuts, region visualization
- **Dependencies**: Bootstrap 5, Font Awesome icons

### **viewer.js**
- **Purpose**: JavaScript functionality for document viewer
- **Functions**:
  - Handle user interactions (zoom, pan, click)
  - Fetch document data from API
  - Render bounding boxes and region overlays
  - Manage viewer state and navigation
- **Features**: Canvas-based rendering, event handling, API integration
- **Classes**: PlanViewer (main viewer controller)

---

## 🛠️ scripts/ - Utility Scripts

### **setup_db.py**
- **Purpose**: Database initialization and schema setup
- **Functions**:
  - Create database tables and indexes
  - Set up initial data and configurations
  - Handle database migrations
  - Verify database connectivity
- **Usage**: `python scripts/setup_db.py`
- **Features**: Idempotent setup, error handling, progress reporting

### **download_models.py**
- **Purpose**: Download and cache pre-trained models
- **Functions**:
  - Download layout detection models (Detectron2, YOLO)
  - Download embedding models (sentence transformers)
  - Verify model integrity and compatibility
  - Manage model cache and updates
- **Usage**: `python scripts/download_models.py`
- **Models**: Layout detection, text embedding, OCR models

### **download_training_data.py**
- **Purpose**: Download datasets for CNN model training
- **Functions**:
  - Download DocLayNet, PubLayNet, FUNSD datasets
  - Extract and organize training data
  - Create dataset metadata and configurations
  - Support selective dataset downloading
- **Usage**: `python scripts/download_training_data.py --recommended`
- **Datasets**: DocLayNet (7GB), PubLayNet (96GB), FUNSD (20MB)

### **train_cnn_model.py**
- **Purpose**: Train custom CNN models for layout detection
- **Functions**:
  - Train Detectron2, YOLO, or custom CNN models
  - Handle data loading and preprocessing
  - Monitor training progress and validation
  - Save trained models and configurations
- **Usage**: `python scripts/train_cnn_model.py --config config.yaml`
- **Features**: Multi-GPU support, checkpointing, evaluation metrics

### **__init__.py**
- **Purpose**: Scripts module initialization
- **Contains**: Utility function imports
- **Exports**: Common utility functions for scripts

---

## 🧠 planquery/models/ - Model Configurations

### **cnn_config.py**
- **Purpose**: CNN model architecture configuration and factory
- **Functions**:
  - Define configurable CNN architectures
  - Create model instances with custom parameters
  - Support multiple backends (Detectron2, YOLO, custom)
  - Generate training configurations
- **Classes**: CNNModelConfig, CNNModelFactory, LayoutDetectionCNN
- **Features**: Serializable configs, model factory pattern, default presets

---

## 📚 docs/ - Documentation

### **CNN_Training_Guide.md**
- **Purpose**: Comprehensive guide for CNN model training
- **Contains**: Dataset recommendations, training procedures, configuration examples
- **Audience**: Developers training custom layout detection models
- **Topics**: Data preparation, model selection, hyperparameter tuning

### **File_Documentation.md** (this file)
- **Purpose**: Complete file-by-file documentation
- **Contains**: Detailed descriptions of every file's purpose and functionality
- **Audience**: Developers understanding the codebase structure
- **Scope**: All files in the project with their roles and relationships

---

## 🔧 Configuration Files

### **configs/*.yaml**
- **Purpose**: Model-specific configuration files
- **Generated by**: `python planquery/models/cnn_config.py`
- **Contains**: Pre-configured setups for different use cases
- **Files**: `fast_config.yaml`, `balanced_config.yaml`, `accurate_config.yaml`, `custom_config.yaml`

---

## 📊 Data Directories (Created at Runtime)

### **data/**
- **Purpose**: Training datasets storage
- **Contains**: Downloaded datasets (DocLayNet, PubLayNet, etc.)
- **Structure**: Organized by dataset with images and annotations

### **uploads/**
- **Purpose**: User-uploaded PDF files
- **Contains**: Original PDF documents for processing
- **Management**: Automatic cleanup and organization

### **output/**
- **Purpose**: Processed document outputs
- **Contains**: Extracted images, processed data, intermediate results
- **Structure**: Organized by document ID and processing stage

### **indices/**
- **Purpose**: Search index storage
- **Contains**: Vector indexes, keyword indexes, cached embeddings
- **Backends**: FAISS files, Whoosh indexes

### **models/**
- **Purpose**: Downloaded and trained model storage
- **Contains**: Pre-trained models, fine-tuned models, model cache
- **Organization**: By model type and version

### **logs/**
- **Purpose**: Application logging
- **Contains**: Structured logs for debugging and monitoring
- **Rotation**: Automatic log rotation and cleanup

---

## 🚀 Entry Points

### **CLI Interface**
- **File**: `planquery/cli.py`
- **Purpose**: Command-line interface for all operations
- **Commands**: `ingest`, `search`, `serve`, `setup`, `download-models`
- **Usage**: `planquery --help`

### **API Server**
- **File**: `planquery/api/main.py`
- **Purpose**: Web API server
- **Usage**: `uvicorn planquery.api.main:app --reload`
- **Port**: 8000 (default)

### **Web Interface**
- **Files**: `planquery/static/viewer.html`, `planquery/static/viewer.js`
- **Purpose**: Interactive document viewer
- **Access**: `http://localhost:8000/viewer`
- **Features**: Document browsing, region visualization, search integration

---

This documentation provides a complete overview of every file's purpose and functionality in the PlanQuery system. Each file is designed to handle a specific aspect of the document processing and search pipeline, from initial PDF processing through to the final user interface.

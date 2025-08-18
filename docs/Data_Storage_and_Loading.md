# Data Storage and Loading in PlanQuery

## Data Storage Locations

### **1. Training Data Storage**
```
data/
├── doclaynet/                    # DocLayNet dataset (7GB)
│   ├── COCO/
│   │   ├── train.json           # Training annotations (COCO format)
│   │   ├── val.json             # Validation annotations
│   │   └── test.json            # Test annotations
│   ├── PNG/                     # Document images
│   │   ├── train/               # Training images
│   │   ├── val/                 # Validation images
│   │   └── test/                # Test images
│   └── metadata.json            # Dataset metadata
├── publaynet/                   # PubLayNet dataset (96GB)
│   ├── train/
│   ├── val/
│   └── annotations/
├── funsd/                       # FUNSD dataset (20MB)
│   ├── training_data/
│   ├── testing_data/
│   └── annotations/
└── architectural_samples/       # Custom architectural data
    ├── images/
    │   ├── title_block/
    │   ├── dimensions/
    │   ├── symbols/
    │   └── floor_plans/
    ├── annotations/
    └── metadata.json
```

### **2. Runtime Data Storage**
```
uploads/                         # User uploaded PDFs
├── document_001.pdf
├── document_002.pdf
└── temp/                       # Temporary processing files

output/                         # Processed document outputs
├── doc_001/
│   ├── pages/                  # Extracted page images
│   │   ├── page_001.png
│   │   └── page_002.png
│   ├── regions/                # Detected regions
│   │   ├── page_001_regions.json
│   │   └── page_002_regions.json
│   └── metadata.json           # Document metadata

indices/                        # Search indices
├── vector.index                # FAISS vector index
├── keyword/                    # Whoosh keyword index
│   ├── _MAIN_*.toc
│   └── *.seg
└── embeddings_cache/           # Cached embeddings

models/                         # Model storage
├── detectron2/
│   ├── model_final.pth         # Trained Detectron2 model
│   └── config.yaml
├── yolo/
│   ├── best.pt                 # Trained YOLO model
│   └── runs/
├── embeddings/
│   └── sentence-transformers/  # Cached embedding models
└── custom/
    └── layout_detection_cnn.pth

logs/                           # Application logs
├── planquery.log
├── training.log
└── api.log
```

### **3. Database Storage (PostgreSQL)**
```sql
-- Document metadata and structure
documents (id, file_name, title, discipline, total_pages, created_at)
pages (id, document_id, page_number, width, height, image_path)
regions (id, page_id, region_type, text_type, bbox_x1, bbox_y1, bbox_x2, bbox_y2, text_content, confidence)
floors (id, canonical_id, original_names, document_id)
```

---

## 🔄 Data Loading Mechanisms

### **1. Training Data Loading**

#### **DocumentLayoutDataset Class** (in `scripts/train_cnn_model.py`)
```python
class DocumentLayoutDataset(Dataset):
    def __init__(self, data_dir: str, annotations_file: str, transform=None):
        self.data_dir = Path(data_dir)
        
        # Load COCO format annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Index images and annotations
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        
    def __getitem__(self, idx):
        # Load image from disk
        img_path = self.data_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Load corresponding annotations
        annotations = self.image_annotations.get(img_id, [])
        
        return image, target
```

**Data Flow:**
1. **Annotations**: Loaded from JSON files in COCO format
2. **Images**: Loaded on-demand from PNG files
3. **Bounding Boxes**: Converted from COCO format [x, y, w, h] to [x1, y1, x2, y2]
4. **Labels**: Mapped to category IDs

#### **DataLoader Configuration**
```python
# Training data loader
train_dataset = DocumentLayoutDataset(args.data_dir, args.train_annotations)
train_loader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size, 
    shuffle=True,
    num_workers=4,
    collate_fn=custom_collate_fn
)
```

### **2. Runtime Document Processing**

#### **PDF Processing Pipeline**
```python
# 1. PDF Upload (api/main.py)
file_path = api_instance.config['upload_dir'] / file.filename
with open(file_path, 'wb') as f:
    content = await file.read()
    f.write(content)

# 2. PDF to Images (core/pdf_processor.py)
pdf_document = pdf_processor.process_pdf(pdf_path)
for page_info in pdf_document.pages:
    page_image = pdf_processor.get_page_image(page_info)

# 3. Layout Detection (core/layout_detector.py)
detected_regions = layout_detector.detect_regions(page_image)

# 4. OCR Processing (core/ocr_engine.py)
ocr_results = ocr_engine.extract_text_from_regions(page_image, detected_regions)

# 5. Database Storage (indexing/database.py)
document = db_manager.create_document(...)
page = db_manager.create_page(...)
region = db_manager.create_region(...)
```

### **3. Search Index Loading**

#### **Vector Store Loading** (indexing/vector_store.py)
```python
class VectorStore:
    def __init__(self, backend="faiss", index_path="indices/vector.index"):
        # Load pre-built FAISS index
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # Load embedding model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def search(self, query_text, k=10):
        # Generate query embedding
        query_embedding = self.model.encode([query_text])
        
        # Search index
        scores, indices = self.index.search(query_embedding, k)
        return results
```

#### **Keyword Index Loading** (indexing/keyword_index.py)
```python
class KeywordIndex:
    def __init__(self, index_dir="indices/keyword"):
        # Open Whoosh index
        if index.exists_in(index_dir):
            self.ix = index.open_dir(index_dir)
        else:
            self.ix = index.create_in(index_dir, schema)
    
    def search(self, query_string):
        with self.ix.searcher() as searcher:
            query = QueryParser("content", self.ix.schema).parse(query_string)
            results = searcher.search(query)
            return results
```

### **4. Database Loading** (indexing/database.py)
```python
class DatabaseManager:
    def __init__(self, database_url):
        self.engine = create_async_engine(database_url)
        self.SessionLocal = sessionmaker(self.engine)
    
    async def get_document(self, document_id: str):
        async with self.SessionLocal() as session:
            result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            return result.scalar_one_or_none()
    
    async def get_page_regions(self, document_id: str, page_number: int):
        async with self.SessionLocal() as session:
            result = await session.execute(
                select(Region)
                .join(Page)
                .where(Page.document_id == document_id)
                .where(Page.page_number == page_number)
            )
            return result.scalars().all()
```

---

## 🔧 Configuration-Based Loading

### **Data Paths Configuration** (config.yaml)
```yaml
storage:
  upload_dir: "uploads"
  output_dir: "output" 
  index_dir: "indices"
  models_dir: "models"
  data_dir: "data"

database:
  url: "postgresql://localhost/planquery"

search:
  vector_store:
    index_path: "indices/vector.index"
  keyword_index:
    index_dir: "indices/keyword"
```

### **Model Loading Configuration**
```python
# From cnn_config.py
config = CNNModelConfig.load("configs/balanced_config.yaml")

# Model paths are resolved from config
if config.weights_path:
    model.load_state_dict(torch.load(config.weights_path))
```

---

## 📊 Data Loading Performance

### **Lazy Loading Strategies**
1. **Images**: Loaded on-demand during training/inference
2. **Embeddings**: Cached after first computation
3. **Database**: Connection pooling and async queries
4. **Search Indices**: Memory-mapped for fast access

### **Caching Mechanisms**
```python
# Embedding cache
@lru_cache(maxsize=1000)
def get_embedding(text: str):
    return model.encode(text)

# Database query cache
@cached(ttl=300)  # 5 minute cache
async def get_document_metadata(doc_id: str):
    return await db.get_document(doc_id)
```

### **Batch Processing**
```python
# Process multiple documents in batches
def process_batch(pdf_files: List[str], batch_size: int = 4):
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i + batch_size]
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(process_pdf, pdf) for pdf in batch]
```

---

## 🚀 Data Loading Commands

### **Download Training Data**
```bash
# Download recommended datasets
python scripts/download_training_data.py --recommended

# Download specific dataset
python scripts/download_training_data.py --dataset doclaynet
```

### **Load Data for Training**
```bash
# Train with downloaded data
python scripts/train_cnn_model.py \
  --config configs/balanced_config.yaml \
  --data-dir data/doclaynet/PNG/train \
  --train-annotations data/doclaynet/COCO/train.json \
  --val-annotations data/doclaynet/COCO/val.json
```

### **Initialize Database**
```bash
# Set up database schema and load initial data
python scripts/setup_db.py
```

### **Process Documents**
```bash
# CLI document processing
planquery ingest document.pdf --discipline A --title "Floor Plans"

# API document upload
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "discipline=A" \
  -F "title=Floor Plans"
```

---

## 💾 Data Persistence

### **Training Checkpoints**
- **Location**: `models/{model_type}/checkpoints/`
- **Format**: PyTorch `.pth` files
- **Frequency**: Every epoch + best model

### **Search Index Updates**
- **Vector Index**: Rebuilt when new documents added
- **Keyword Index**: Incremental updates
- **Database**: Real-time updates with transactions

### **Backup Strategy**
```bash
# Database backup
pg_dump planquery > backup_$(date +%Y%m%d).sql

# Index backup
tar -czf indices_backup_$(date +%Y%m%d).tar.gz indices/

# Model backup
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

This comprehensive data storage and loading system ensures efficient handling of large datasets while maintaining fast access for real-time document processing and search operations.

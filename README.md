# PlanQuery - Smart Document Assistant for Architectural Plans

Turn unstructured plan sets (PDFs) into a searchable, floor-aware knowledge base—and chat your way to the exact sheet region (with zoomable bbox + snippet) that answers your question.

## What This Does

PlanQuery ingests architectural/MEP/structural sheets, detects layout regions (text, titles, tables), runs OCR, classifies each region into AEC-specific text types (e.g., GeneralNotes, Dimensions, Callout, SpecReference, TitleBlock, ScheduleText), and normalizes floor labels (L1/L2/Roof). A lightweight RAG pipeline then answers questions like "Where can I find Level 2 structural notes?" with sheet suggestions + precise coordinates.

## Key Capabilities

- **Floor-aware retrieval**: Map LEVEL 02/L02/Second Floor → L2 and filter results accordingly
- **Typed text extraction**: Classify regions into actionable categories (Notes, Dimensions, Callouts, Specs, Title-block, Schedules)
- **Fast search at scale**: Hybrid vector + keyword search over region text with metadata filters (floor, discipline, type)
- **Trustworthy answers**: Each response includes sheet ID, region bbox, snippet, and revision date
- **Pluggable OCR & models**: Start simple (Tesseract + linear classifier) or swap in stronger OCR/transformers
- **Open formats**: Expose JSON/CSV + thumbnails for downstream tools

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up database**:
   ```bash
   python scripts/setup_db.py
   ```

3. **Process your first plan set**:
   ```bash
   python -m planquery.cli ingest path/to/plans.pdf
   ```

4. **Start the API server**:
   ```bash
   uvicorn planquery.api.main:app --reload
   ```

5. **Open the web interface**: http://localhost:8000

## Architecture

```
PDFs → Layout Detection → OCR → Text Classification → Floor Normalization → Indexing → RAG Search
```

### Pipeline Steps
1. **Ingest & Structure**: Rasterize PDFs → detect layout regions → OCR → classify region text_type → extract title-block fields → canonicalize floor_id
2. **Indexing**: Store metadata in Postgres + vector index for embeddings + BM25 for keywords/regex
3. **Chat RAG**: Parse intent → apply floor/type filters → hybrid retrieve → rerank → return top sheets/regions

## Project Structure

```
planquery/
├── core/               # Core processing pipeline
│   ├── pdf_processor.py    # PDF rasterization and page handling
│   ├── layout_detector.py  # Region detection (YOLO/Detectron2)
│   ├── ocr_engine.py      # OCR processing (Tesseract/TrOCR)
│   ├── text_classifier.py # AEC text type classification
│   └── floor_normalizer.py # Floor label standardization
├── indexing/           # Search and storage
│   ├── database.py         # Postgres schema and operations
│   ├── vector_store.py     # Vector embeddings (FAISS/pgvector)
│   └── keyword_index.py    # BM25 keyword search
├── search/             # RAG pipeline
│   ├── query_parser.py     # Intent parsing and filters
│   ├── retriever.py        # Hybrid search implementation
│   └── reranker.py         # Result reranking
├── api/                # FastAPI backend
│   ├── main.py             # API server
│   ├── routes/             # API endpoints
│   └── models/             # Pydantic models
├── ui/                 # Web interface
│   ├── static/             # CSS, JS, images
│   └── templates/          # HTML templates
└── scripts/            # Utilities and setup
    ├── setup_db.py         # Database initialization
    └── download_models.py  # Model downloads
```

## Example Queries

- "Where are Level 3 RCP notes?"
- "Show Level 2 mechanical equipment schedules."
- "Which sheet has stair dimensions on Level 1?"
- "Find 07 21 00 references on Level 2."

## Tech Stack

- **Detection**: YOLO/Detectron2 fine-tuned on DocLayNet
- **OCR**: Tesseract (baseline) or TrOCR
- **Embeddings**: Sentence transformers
- **Database**: PostgreSQL with pgvector extension
- **Search**: Hybrid vector + BM25 (Whoosh)
- **API**: FastAPI
- **UI**: Simple HTML/JS viewer with bbox visualization

## License

MIT License - see LICENSE file for details.

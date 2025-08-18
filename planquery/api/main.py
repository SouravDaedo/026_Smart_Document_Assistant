"""
FastAPI main application for PlanQuery.
Provides REST API endpoints for document processing, search, and chat functionality.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from .models import *
from ..core.pdf_processor import PDFProcessor
from ..core.layout_detector import LayoutDetector
from ..core.ocr_engine import OCREngine
from ..core.text_classifier import TextClassifier
from ..core.floor_normalizer import FloorNormalizer
from ..indexing.database import DatabaseManager
from ..indexing.vector_store import VectorStore
from ..indexing.keyword_index import KeywordIndex
from ..search.query_parser import QueryParser
from ..search.retriever import HybridRetriever
from ..search.reranker import ResultReranker


class PlanQueryAPI:
    """Main API application class."""
    
    def __init__(self):
        """Initialize API components."""
        self.app = None
        self.database_manager = None
        self.vector_store = None
        self.keyword_index = None
        self.query_parser = None
        self.retriever = None
        self.reranker = None
        
        # Processing components
        self.pdf_processor = None
        self.layout_detector = None
        self.ocr_engine = None
        self.text_classifier = None
        self.floor_normalizer = None
        
        # Configuration
        self.config = {
            'database_url': os.getenv('DATABASE_URL', 'postgresql://localhost/planquery'),
            'upload_dir': Path('uploads'),
            'output_dir': Path('output'),
            'index_dir': Path('indices'),
            'static_dir': Path('static'),
        }
        
        # Chat sessions
        self.chat_sessions = {}
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing PlanQuery API...")
        
        # Create directories
        for dir_path in [self.config['upload_dir'], self.config['output_dir'], 
                        self.config['index_dir'], self.config['static_dir']]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize database
        self.database_manager = DatabaseManager(self.config['database_url'])
        
        # Initialize search components
        self.vector_store = VectorStore(
            backend="faiss",
            index_path=str(self.config['index_dir'] / 'vector.index'),
            database_manager=self.database_manager
        )
        
        self.keyword_index = KeywordIndex(
            index_dir=str(self.config['index_dir'] / 'keyword')
        )
        
        # Initialize processing components
        self.pdf_processor = PDFProcessor(
            dpi=300,
            output_dir=str(self.config['output_dir'])
        )
        
        self.layout_detector = LayoutDetector(
            model_type="detectron2",
            confidence_threshold=0.5
        )
        
        self.ocr_engine = OCREngine(engine="tesseract")
        self.text_classifier = TextClassifier(model_type="rules")
        self.floor_normalizer = FloorNormalizer()
        
        # Initialize search pipeline
        self.query_parser = QueryParser()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            keyword_index=self.keyword_index,
            database_manager=self.database_manager
        )
        self.reranker = ResultReranker()
        
        logger.info("PlanQuery API initialized successfully")
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up PlanQuery API...")
        
        if self.database_manager:
            self.database_manager.close()
        
        if self.keyword_index:
            self.keyword_index.close()


# Global API instance
api_instance = PlanQueryAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    await api_instance.initialize()
    yield
    await api_instance.cleanup()


# Create FastAPI app
app = FastAPI(
    title="PlanQuery API",
    description="Smart Document Assistant for Architectural Plans",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {}
    
    try:
        # Check database
        stats = api_instance.database_manager.get_database_stats()
        components["database"] = "healthy"
    except Exception as e:
        components["database"] = f"error: {str(e)}"
    
    try:
        # Check vector store
        vector_stats = api_instance.vector_store.get_stats()
        components["vector_store"] = "healthy"
    except Exception as e:
        components["vector_store"] = f"error: {str(e)}"
    
    try:
        # Check keyword index
        keyword_stats = api_instance.keyword_index.get_stats()
        components["keyword_index"] = "healthy"
    except Exception as e:
        components["keyword_index"] = f"error: {str(e)}"
    
    overall_status = "healthy" if all("error" not in status for status in components.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="0.1.0",
        components=components
    )


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with PlanQuery using natural language."""
    start_time = time.time()
    
    try:
        # Generate or use existing session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Parse the query
        parsed_query = api_instance.query_parser.parse_query(request.query)
        
        # Apply additional filters from request
        if request.filters:
            parsed_query.filters.update(request.filters)
        
        # Retrieve results
        retrieval_results = api_instance.retriever.retrieve(
            parsed_query,
            limit=request.limit or 20
        )
        
        # Rerank results
        reranked_results = api_instance.reranker.rerank(
            retrieval_results,
            parsed_query,
            max_results=request.limit
        )
        
        # Convert to API models
        search_results = []
        for result in reranked_results:
            bbox = None
            if result.metadata.get('bbox'):
                bbox_data = result.metadata['bbox']
                bbox = BoundingBox(
                    x1=bbox_data.get('x1', 0),
                    y1=bbox_data.get('y1', 0),
                    x2=bbox_data.get('x2', 0),
                    y2=bbox_data.get('y2', 0)
                )
            
            search_results.append(SearchResult(
                region_id=result.region_id,
                text=result.text,
                score=result.combined_score,
                source=result.source,
                highlights=result.highlights,
                metadata=result.metadata,
                bbox=bbox,
                floor_id=result.metadata.get('floor_id'),
                discipline=result.metadata.get('discipline'),
                page_number=result.metadata.get('page_number'),
                sheet_number=result.metadata.get('sheet_number')
            ))
        
        # Generate natural language response
        response_text = _generate_chat_response(parsed_query, search_results)
        
        # Store session context
        api_instance.chat_sessions[session_id] = {
            'last_query': request.query,
            'last_results': search_results,
            'context': request.context
        }
        
        execution_time = time.time() - start_time
        
        return ChatResponse(
            response=response_text,
            results=search_results,
            parsed_query=ParsedQuery(
                original_query=parsed_query.original_query,
                query_type=QueryType(parsed_query.query_type.value),
                search_scope=SearchScope(parsed_query.search_scope.value),
                search_terms=parsed_query.search_terms,
                filters=parsed_query.filters,
                floor_ids=parsed_query.floor_ids,
                disciplines=parsed_query.disciplines,
                text_types=parsed_query.text_types,
                confidence=parsed_query.confidence,
                metadata=parsed_query.metadata
            ),
            session_id=session_id,
            metadata={
                'execution_time': execution_time,
                'total_results': len(search_results)
            }
        )
        
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Direct search endpoint."""
    start_time = time.time()
    
    try:
        # Create parsed query from request
        from ..search.query_parser import ParsedQuery as CoreParsedQuery, QueryType as CoreQueryType, SearchScope as CoreSearchScope
        
        parsed_query = CoreParsedQuery(
            original_query=request.query,
            query_type=CoreQueryType(request.query_type.value) if request.query_type else CoreQueryType.SEARCH,
            search_scope=CoreSearchScope.GLOBAL,
            search_terms=request.query.split(),
            filters=request.filters,
            floor_ids=request.floor_ids,
            disciplines=request.disciplines,
            text_types=request.text_types,
            region_types=[],
            confidence=1.0,
            metadata={}
        )
        
        # Retrieve and rerank results
        retrieval_results = api_instance.retriever.retrieve(
            parsed_query,
            limit=request.limit or 20,
            min_score=request.min_score or 0.1
        )
        
        reranked_results = api_instance.reranker.rerank(
            retrieval_results,
            parsed_query,
            max_results=request.limit
        )
        
        # Convert to API models
        search_results = []
        for result in reranked_results:
            bbox = None
            if result.metadata.get('bbox'):
                bbox_data = result.metadata['bbox']
                bbox = BoundingBox(
                    x1=bbox_data.get('x1', 0),
                    y1=bbox_data.get('y1', 0),
                    x2=bbox_data.get('x2', 0),
                    y2=bbox_data.get('y2', 0)
                )
            
            search_results.append(SearchResult(
                region_id=result.region_id,
                text=result.text,
                score=result.combined_score,
                source=result.source,
                highlights=result.highlights,
                metadata=result.metadata,
                bbox=bbox,
                floor_id=result.metadata.get('floor_id'),
                discipline=result.metadata.get('discipline'),
                page_number=result.metadata.get('page_number'),
                sheet_number=result.metadata.get('sheet_number')
            ))
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            results=search_results,
            total_results=len(search_results),
            query_info=ParsedQuery(
                original_query=parsed_query.original_query,
                query_type=QueryType.SEARCH,
                search_scope=SearchScope.GLOBAL,
                search_terms=parsed_query.search_terms,
                filters=parsed_query.filters,
                floor_ids=parsed_query.floor_ids,
                disciplines=parsed_query.disciplines,
                text_types=parsed_query.text_types,
                confidence=parsed_query.confidence,
                metadata=parsed_query.metadata
            ),
            execution_time=execution_time,
            metadata={}
        )
        
    except Exception as e:
        logger.error(f"Search request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document upload endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    discipline: Optional[str] = None,
    title: Optional[str] = None
):
    """Upload and process a PDF document."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file
        file_path = api_instance.config['upload_dir'] / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Process document in background
        document_id = str(uuid.uuid4())
        background_tasks.add_task(
            _process_document,
            str(file_path),
            document_id,
            discipline,
            title
        )
        
        return UploadResponse(
            document_id=document_id,
            status="processing",
            message="Document uploaded and processing started",
            pages_processed=0,
            regions_extracted=0,
            processing_time=0.0
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics endpoint
@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    try:
        db_stats = api_instance.database_manager.get_database_stats()
        vector_stats = api_instance.vector_store.get_stats()
        keyword_stats = api_instance.keyword_index.get_stats()
        
        return StatsResponse(
            documents=db_stats.get('documents', 0),
            pages=db_stats.get('pages', 0),
            regions=db_stats.get('regions', 0),
            regions_with_text=db_stats.get('regions_with_text', 0),
            by_discipline=db_stats.get('by_discipline', {}),
            by_floor=db_stats.get('by_floor', {}),
            index_stats={
                'vector_store': vector_stats,
                'keyword_index': keyword_stats
            }
        )
        
    except Exception as e:
        logger.error(f"Stats request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Pattern search endpoint
@app.post("/search/patterns", response_model=SearchResponse)
async def search_patterns(request: PatternSearchRequest):
    """Search for specific AEC patterns."""
    try:
        results = api_instance.retriever.search_patterns(
            request.pattern_type,
            request.filters,
            request.limit or 50
        )
        
        search_results = []
        for result in results:
            bbox = None
            if result.metadata.get('bbox'):
                bbox_data = result.metadata['bbox']
                bbox = BoundingBox(
                    x1=bbox_data.get('x1', 0),
                    y1=bbox_data.get('y1', 0),
                    x2=bbox_data.get('x2', 0),
                    y2=bbox_data.get('y2', 0)
                )
            
            search_results.append(SearchResult(
                region_id=result.region_id,
                text=result.text,
                score=result.combined_score,
                source=result.source,
                highlights=result.highlights,
                metadata=result.metadata,
                bbox=bbox,
                floor_id=result.metadata.get('floor_id'),
                discipline=result.metadata.get('discipline'),
                page_number=result.metadata.get('page_number'),
                sheet_number=result.metadata.get('sheet_number')
            ))
        
        return SearchResponse(
            results=search_results,
            total_results=len(search_results),
            query_info=ParsedQuery(
                original_query=f"pattern:{request.pattern_type}",
                query_type=QueryType.SEARCH,
                search_scope=SearchScope.GLOBAL,
                search_terms=[],
                filters=request.filters or {},
                floor_ids=[],
                disciplines=[],
                text_types=[],
                confidence=1.0,
                metadata={'pattern_type': request.pattern_type}
            ),
            execution_time=0.0,
            metadata={'pattern_type': request.pattern_type}
        )
        
    except Exception as e:
        logger.error(f"Pattern search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Web interfaces
@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Serve the web chat interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PlanQuery - Smart Document Assistant</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .chat-container { max-width: 800px; margin: 50px auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
            .chat-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px 15px 0 0; text-align: center; }
            .chat-body { padding: 30px; }
            .message { margin-bottom: 20px; }
            .user-message { text-align: right; }
            .user-message .badge { background: #667eea; }
            .assistant-message .badge { background: #28a745; }
            .message-content { display: inline-block; max-width: 70%; padding: 15px; border-radius: 15px; }
            .user-message .message-content { background: #e3f2fd; }
            .assistant-message .message-content { background: #f8f9fa; }
            .input-group { margin-top: 20px; }
            #chatMessages { max-height: 400px; overflow-y: auto; }
            .nav-links { text-align: center; margin-top: 20px; }
            .nav-links a { color: white; text-decoration: none; margin: 0 15px; }
            .nav-links a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="chat-container">
                <div class="chat-header">
                    <h2><i class="fas fa-drafting-compass me-2"></i>PlanQuery</h2>
                    <p class="mb-0">Smart Document Assistant for Architectural Plans</p>
                    <div class="nav-links">
                        <a href="/viewer"><i class="fas fa-eye me-1"></i>Document Viewer</a>
                        <a href="/upload"><i class="fas fa-upload me-1"></i>Upload Documents</a>
                    </div>
                </div>
                <div class="chat-body">
                    <div id="chatMessages"></div>
                    <div class="input-group">
                        <input type="text" id="messageInput" class="form-control" placeholder="Ask about your plans..." onkeypress="if(event.key==='Enter') sendMessage()">
                        <button class="btn btn-primary" onclick="sendMessage()">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const messages = document.getElementById('chatMessages');
                const query = input.value.trim();
                if (!query) return;
                
                // Add user message
                messages.innerHTML += `
                    <div class="message user-message">
                        <span class="badge">You</span>
                        <div class="message-content">${query}</div>
                    </div>
                `;
                input.value = '';
                messages.scrollTop = messages.scrollHeight;
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query })
                    });
                    const data = await response.json();
                    
                    // Add assistant message
                    messages.innerHTML += `
                        <div class="message assistant-message">
                            <span class="badge">Assistant</span>
                            <div class="message-content">${data.answer}</div>
                        </div>
                    `;
                                </div>
                            `;
                        });
                    }
                    
                    botMessage.innerHTML = html;
                    chatContainer.appendChild(botMessage);
                    
                } catch (error) {
                    const errorMessage = document.createElement('div');
                    errorMessage.className = 'message bot-message';
                    errorMessage.innerHTML = `<strong>Error:</strong> ${error.message}`;
                    chatContainer.appendChild(errorMessage);
                }
                
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendQuery();
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Helper functions
def _generate_chat_response(parsed_query, results: List[SearchResult]) -> str:
    """Generate natural language response for chat."""
    if not results:
        return f"I couldn't find any information matching '{parsed_query.original_query}'. Try rephrasing your question or checking if the documents have been processed."
    
    result_count = len(results)
    query_type = parsed_query.query_type.value
    
    if query_type == "count":
        return f"Found {result_count} items matching your criteria."
    elif query_type == "location":
        if result_count == 1:
            result = results[0]
            location_info = []
            if result.floor_id:
                location_info.append(f"Floor {result.floor_id}")
            if result.sheet_number:
                location_info.append(f"Sheet {result.sheet_number}")
            if result.page_number:
                location_info.append(f"Page {result.page_number}")
            
            location_str = ", ".join(location_info) if location_info else "the documents"
            return f"I found what you're looking for on {location_str}. The relevant text is: '{result.text[:200]}...'"
        else:
            return f"I found {result_count} locations that match your query. Here are the most relevant results:"
    elif query_type == "list":
        return f"Here are {result_count} items matching your criteria:"
    else:
        return f"I found {result_count} relevant results for your query. Here are the most relevant ones:"


async def _process_document(file_path: str, document_id: str, discipline: Optional[str], title: Optional[str]):
    """Process uploaded document in background."""
    try:
        logger.info(f"Processing document: {file_path}")
        start_time = time.time()
        
        # Process PDF
        pdf_document = api_instance.pdf_processor.process_pdf(file_path)
        
        # Create database record
        document = api_instance.database_manager.create_document(
            file_path=file_path,
            file_name=Path(file_path).name,
            discipline=discipline or pdf_document.discipline,
            title=title or pdf_document.title,
            total_pages=pdf_document.total_pages,
            file_size=Path(file_path).stat().st_size,
            metadata=pdf_document.metadata
        )
        
        total_regions = 0
        
        # Process each page
        for page_info in pdf_document.pages:
            # Load page image
            page_image = api_instance.pdf_processor.get_page_image(page_info)
            
            # Detect layout regions
            detected_regions = api_instance.layout_detector.detect_regions(page_image)
            detected_regions = api_instance.layout_detector.filter_regions(detected_regions)
            
            # Extract text from regions
            ocr_results = api_instance.ocr_engine.extract_text_from_regions(page_image, detected_regions)
            
            # Classify text
            classifications = api_instance.text_classifier.classify_ocr_results(ocr_results)
            
            # Normalize floor information
            page_text = " ".join([ocr.text for ocr in ocr_results])
            floor_info = api_instance.floor_normalizer.normalize_floor_from_text(page_text)
            
            # Create page record
            page = api_instance.database_manager.create_page(
                document_id=document.id,
                page_number=page_info.page_num,
                width=page_info.width,
                height=page_info.height,
                dpi=page_info.dpi,
                image_path=page_info.image_path,
                floor_id=floor_info.canonical_id if floor_info else None,
                metadata=page_info.metadata
            )
            
            # Process regions
            for ocr_result, classification in zip(ocr_results, classifications):
                # Create region record
                region = api_instance.database_manager.create_region(
                    page_id=page.id,
                    region_type=ocr_result.region_type,
                    text_type=classification.text_type,
                    bbox_x1=ocr_result.bbox.x1,
                    bbox_y1=ocr_result.bbox.y1,
                    bbox_x2=ocr_result.bbox.x2,
                    bbox_y2=ocr_result.bbox.y2,
                    confidence=ocr_result.confidence,
                    text_content=ocr_result.text,
                    text_confidence=ocr_result.confidence,
                    metadata=ocr_result.metadata
                )
                
                # Add to search indices
                if ocr_result.text.strip():
                    # Add to vector store
                    api_instance.vector_store.add_text(
                        ocr_result.text,
                        str(region.id),
                        metadata={
                            'floor_id': floor_info.canonical_id if floor_info else None,
                            'discipline': document.discipline,
                            'text_type': classification.text_type,
                            'region_type': ocr_result.region_type,
                            'page_number': page_info.page_num
                        }
                    )
                    
                    # Add to keyword index
                    api_instance.keyword_index.add_document(
                        str(region.id),
                        ocr_result.text,
                        text_type=classification.text_type,
                        region_type=ocr_result.region_type,
                        floor_id=floor_info.canonical_id if floor_info else None,
                        discipline=document.discipline,
                        page_number=page_info.page_num,
                        confidence=ocr_result.confidence,
                        bbox=ocr_result.bbox.to_dict()
                    )
                
                total_regions += 1
        
        # Save indices
        api_instance.vector_store.save_index()
        api_instance.keyword_index.optimize()
        
        processing_time = time.time() - start_time
        logger.info(f"Document processing completed: {total_regions} regions in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")


# Web interface endpoints
@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Serve the web chat interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PlanQuery - Smart Document Assistant</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .chat-container { max-width: 800px; margin: 50px auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
            .chat-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px 15px 0 0; text-align: center; }
            .chat-body { padding: 30px; }
            .message { margin-bottom: 20px; }
            .user-message { text-align: right; }
            .user-message .badge { background: #667eea; }
            .assistant-message .badge { background: #28a745; }
            .message-content { display: inline-block; max-width: 70%; padding: 15px; border-radius: 15px; }
            .user-message .message-content { background: #e3f2fd; }
            .assistant-message .message-content { background: #f8f9fa; }
            .input-group { margin-top: 20px; }
            #chatMessages { max-height: 400px; overflow-y: auto; }
            .nav-links { text-align: center; margin-top: 20px; }
            .nav-links a { color: white; text-decoration: none; margin: 0 15px; }
            .nav-links a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="chat-container">
                <div class="chat-header">
                    <h2><i class="fas fa-drafting-compass me-2"></i>PlanQuery</h2>
                    <p class="mb-0">Smart Document Assistant for Architectural Plans</p>
                    <div class="nav-links">
                        <a href="/viewer"><i class="fas fa-eye me-1"></i>Document Viewer</a>
                        <a href="/upload"><i class="fas fa-upload me-1"></i>Upload Documents</a>
                    </div>
                </div>
                <div class="chat-body">
                    <div id="chatMessages"></div>
                    <div class="input-group">
                        <input type="text" id="messageInput" class="form-control" placeholder="Ask about your plans..." onkeypress="if(event.key==='Enter') sendMessage()">
                        <button class="btn btn-primary" onclick="sendMessage()">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const messages = document.getElementById('chatMessages');
                const query = input.value.trim();
                if (!query) return;
                
                messages.innerHTML += `
                    <div class="message user-message">
                        <span class="badge">You</span>
                        <div class="message-content">${query}</div>
                    </div>
                `;
                input.value = '';
                messages.scrollTop = messages.scrollHeight;
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query })
                    });
                    const data = await response.json();
                    
                    messages.innerHTML += `
                        <div class="message assistant-message">
                            <span class="badge">Assistant</span>
                            <div class="message-content">${data.response}</div>
                        </div>
                    `;
                } catch (error) {
                    messages.innerHTML += `
                        <div class="message assistant-message">
                            <span class="badge bg-danger">Error</span>
                            <div class="message-content">Sorry, I encountered an error. Please try again.</div>
                        </div>
                    `;
                }
                messages.scrollTop = messages.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/viewer", response_class=HTMLResponse)
async def viewer_interface():
    """Serve the document viewer interface."""
    try:
        with open("planquery/static/viewer.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Viewer interface not found")


# Document viewer API endpoints
@app.get("/api/documents")
async def get_documents():
    """Get list of all documents."""
    try:
        documents = api_instance.database_manager.get_all_documents()
        return {
            "success": True,
            "documents": [
                {
                    "id": doc.id,
                    "file_name": doc.file_name,
                    "title": doc.title,
                    "discipline": doc.discipline,
                    "total_pages": doc.total_pages,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None
                }
                for doc in documents
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details."""
    try:
        document = api_instance.database_manager.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "success": True,
            "document": {
                "id": document.id,
                "file_name": document.file_name,
                "title": document.title,
                "discipline": document.discipline,
                "total_pages": document.total_pages,
                "file_size": document.file_size,
                "created_at": document.created_at.isoformat() if document.created_at else None,
                "metadata": document.metadata
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{document_id}/pages/{page_number}/image")
async def get_page_image(document_id: str, page_number: int):
    """Get page image."""
    try:
        page = api_instance.database_manager.get_page(document_id, page_number)
        if not page or not page.image_path:
            raise HTTPException(status_code=404, detail="Page image not found")
        
        if not os.path.exists(page.image_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        return FileResponse(page.image_path, media_type="image/png")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get page image {document_id}/{page_number}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{document_id}/pages/{page_number}/regions")
async def get_page_regions(document_id: str, page_number: int):
    """Get page regions with bounding boxes and text."""
    try:
        regions = api_instance.database_manager.get_page_regions(document_id, page_number)
        
        return {
            "success": True,
            "regions": [
                {
                    "id": region.id,
                    "region_type": region.region_type,
                    "text_type": region.text_type,
                    "bbox_x1": region.bbox_x1,
                    "bbox_y1": region.bbox_y1,
                    "bbox_x2": region.bbox_x2,
                    "bbox_y2": region.bbox_y2,
                    "confidence": region.confidence,
                    "text_content": region.text_content,
                    "text_confidence": region.text_confidence
                }
                for region in regions
            ]
        }
    
    except Exception as e:
        logger.error(f"Failed to get page regions {document_id}/{page_number}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def _generate_chat_response(parsed_query, results: List) -> str:
    """Generate natural language response for chat."""
    if not results:
        return f"I couldn't find any information matching '{parsed_query.original_query}'. Try rephrasing your question or checking if the documents have been processed."
    
    result_count = len(results)
    query_type = parsed_query.query_type.value if hasattr(parsed_query.query_type, 'value') else str(parsed_query.query_type)
    
    if query_type == "count":
        return f"Found {result_count} items matching your criteria."
    elif query_type == "location":
        locations = set()
        for result in results[:5]:  # Top 5 results
            if result.floor_id:
                locations.add(result.floor_id)
            if result.metadata.get('page_number'):
                locations.add(f"Page {result.metadata['page_number']}")
        
        location_text = ", ".join(sorted(locations)) if locations else "various locations"
        return f"Found {result_count} matches in {location_text}. Here are the most relevant results:\n\n" + \
               "\n".join([f"• {result.text[:100]}..." for result in results[:3]])
    
    elif query_type == "comparison":
        return f"Found {result_count} items for comparison. Here are the key findings:\n\n" + \
               "\n".join([f"• {result.text[:100]}..." for result in results[:3]])
    
    else:  # General search
        response = f"Found {result_count} relevant results. Here are the most relevant:\n\n"
        
        for i, result in enumerate(results[:3], 1):
            response += f"{i}. {result.text[:150]}...\n"
            if result.floor_id:
                response += f"   📍 Floor: {result.floor_id}\n"
            if result.discipline:
                response += f"   📋 Discipline: {result.discipline}\n"
            if result.metadata.get('page_number'):
                response += f"   📄 Page: {result.metadata['page_number']}\n"
            response += "\n"
        
        if result_count > 3:
            response += f"... and {result_count - 3} more results."
        
        return response


if __name__ == "__main__":
    uvicorn.run(
        "planquery.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

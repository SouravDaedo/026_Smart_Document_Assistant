"""
Pydantic models for API request/response schemas.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    """Query types supported by the API."""
    SEARCH = "search"
    LOCATION = "location"
    FILTER = "filter"
    COMPARISON = "comparison"
    LIST = "list"
    COUNT = "count"


class SearchScope(str, Enum):
    """Search scope options."""
    GLOBAL = "global"
    FLOOR = "floor"
    DISCIPLINE = "discipline"
    DOCUMENT = "document"


# Request Models
class ChatRequest(BaseModel):
    """Chat request with user query."""
    query: str = Field(..., description="Natural language query")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Search filters")
    limit: Optional[int] = Field(20, description="Maximum number of results")


class SearchRequest(BaseModel):
    """Direct search request."""
    query: str = Field(..., description="Search query")
    query_type: Optional[QueryType] = Field(None, description="Type of query")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Search filters")
    floor_ids: Optional[List[str]] = Field(default_factory=list, description="Floor IDs to search")
    disciplines: Optional[List[str]] = Field(default_factory=list, description="Disciplines to search")
    text_types: Optional[List[str]] = Field(default_factory=list, description="Text types to search")
    limit: Optional[int] = Field(20, description="Maximum number of results")
    min_score: Optional[float] = Field(0.1, description="Minimum relevance score")


class UploadRequest(BaseModel):
    """Document upload request."""
    file_name: str = Field(..., description="Name of the uploaded file")
    discipline: Optional[str] = Field(None, description="Document discipline (A/M/E/S/C)")
    title: Optional[str] = Field(None, description="Document title")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


# Response Models
class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: int = Field(..., description="Left coordinate")
    y1: int = Field(..., description="Top coordinate")
    x2: int = Field(..., description="Right coordinate")
    y2: int = Field(..., description="Bottom coordinate")


class SearchResult(BaseModel):
    """Individual search result."""
    region_id: str = Field(..., description="Unique region identifier")
    text: str = Field(..., description="Extracted text content")
    score: float = Field(..., description="Relevance score")
    source: str = Field(..., description="Search source (vector/keyword/hybrid)")
    highlights: List[str] = Field(default_factory=list, description="Highlighted terms")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Region metadata")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box coordinates")
    floor_id: Optional[str] = Field(None, description="Floor identifier")
    discipline: Optional[str] = Field(None, description="Document discipline")
    page_number: Optional[int] = Field(None, description="Page number")
    sheet_number: Optional[str] = Field(None, description="Sheet number")


class ParsedQuery(BaseModel):
    """Parsed query information."""
    original_query: str = Field(..., description="Original user query")
    query_type: QueryType = Field(..., description="Detected query type")
    search_scope: SearchScope = Field(..., description="Search scope")
    search_terms: List[str] = Field(default_factory=list, description="Extracted search terms")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Extracted filters")
    floor_ids: List[str] = Field(default_factory=list, description="Extracted floor IDs")
    disciplines: List[str] = Field(default_factory=list, description="Extracted disciplines")
    text_types: List[str] = Field(default_factory=list, description="Extracted text types")
    confidence: float = Field(..., description="Parsing confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChatResponse(BaseModel):
    """Chat response with results and context."""
    response: str = Field(..., description="Natural language response")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    parsed_query: Optional[ParsedQuery] = Field(None, description="Parsed query information")
    session_id: str = Field(..., description="Chat session ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class SearchResponse(BaseModel):
    """Search response with results."""
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    query_info: ParsedQuery = Field(..., description="Parsed query information")
    execution_time: float = Field(..., description="Query execution time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class DocumentInfo(BaseModel):
    """Document information."""
    id: str = Field(..., description="Document ID")
    file_name: str = Field(..., description="File name")
    title: Optional[str] = Field(None, description="Document title")
    discipline: Optional[str] = Field(None, description="Document discipline")
    total_pages: int = Field(..., description="Total number of pages")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class PageInfo(BaseModel):
    """Page information."""
    id: str = Field(..., description="Page ID")
    document_id: str = Field(..., description="Parent document ID")
    page_number: int = Field(..., description="Page number")
    floor_id: Optional[str] = Field(None, description="Floor identifier")
    sheet_number: Optional[str] = Field(None, description="Sheet number")
    sheet_name: Optional[str] = Field(None, description="Sheet name")
    width: int = Field(..., description="Page width")
    height: int = Field(..., description="Page height")
    image_url: Optional[str] = Field(None, description="Page image URL")


class RegionInfo(BaseModel):
    """Region information."""
    id: str = Field(..., description="Region ID")
    page_id: str = Field(..., description="Parent page ID")
    region_type: str = Field(..., description="Region type")
    text_type: Optional[str] = Field(None, description="Text classification")
    bbox: BoundingBox = Field(..., description="Bounding box")
    confidence: float = Field(..., description="Detection confidence")
    text_content: Optional[str] = Field(None, description="Extracted text")
    text_confidence: Optional[float] = Field(None, description="OCR confidence")


class UploadResponse(BaseModel):
    """Document upload response."""
    document_id: str = Field(..., description="Created document ID")
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")
    pages_processed: int = Field(..., description="Number of pages processed")
    regions_extracted: int = Field(..., description="Number of regions extracted")
    processing_time: float = Field(..., description="Processing time in seconds")


class StatsResponse(BaseModel):
    """System statistics response."""
    documents: int = Field(..., description="Total documents")
    pages: int = Field(..., description="Total pages")
    regions: int = Field(..., description="Total regions")
    regions_with_text: int = Field(..., description="Regions with extracted text")
    by_discipline: Dict[str, int] = Field(default_factory=dict, description="Documents by discipline")
    by_floor: Dict[str, int] = Field(default_factory=dict, description="Pages by floor")
    index_stats: Dict[str, Any] = Field(default_factory=dict, description="Search index statistics")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    components: Dict[str, str] = Field(default_factory=dict, description="Component status")


# Pattern search models
class PatternSearchRequest(BaseModel):
    """Pattern search request."""
    pattern_type: str = Field(..., description="Type of pattern to search")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Search filters")
    limit: Optional[int] = Field(50, description="Maximum results")


class SimilarRegionsRequest(BaseModel):
    """Similar regions search request."""
    region_id: str = Field(..., description="Reference region ID")
    limit: Optional[int] = Field(10, description="Maximum results")


class ContextRequest(BaseModel):
    """Context around region request."""
    region_id: str = Field(..., description="Center region ID")
    context_radius: Optional[int] = Field(2, description="Number of nearby regions")


# Viewer models
class ViewerRequest(BaseModel):
    """Viewer request for page with regions."""
    page_id: str = Field(..., description="Page ID to view")
    highlight_regions: Optional[List[str]] = Field(default_factory=list, description="Region IDs to highlight")
    show_all_regions: Optional[bool] = Field(False, description="Show all detected regions")


class ViewerResponse(BaseModel):
    """Viewer response with page and region data."""
    page: PageInfo = Field(..., description="Page information")
    regions: List[RegionInfo] = Field(default_factory=list, description="Page regions")
    image_url: str = Field(..., description="Page image URL")
    highlighted_regions: List[str] = Field(default_factory=list, description="Highlighted region IDs")


# Configuration models
class IndexConfig(BaseModel):
    """Index configuration."""
    vector_backend: str = Field("faiss", description="Vector store backend")
    keyword_backend: str = Field("whoosh", description="Keyword index backend")
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Embedding model name")
    vector_dimension: int = Field(384, description="Vector dimension")


class ProcessingConfig(BaseModel):
    """Processing configuration."""
    ocr_engine: str = Field("tesseract", description="OCR engine")
    layout_detector: str = Field("detectron2", description="Layout detection model")
    text_classifier: str = Field("rules", description="Text classification method")
    dpi: int = Field(300, description="PDF rasterization DPI")


class APIConfig(BaseModel):
    """API configuration."""
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, description="API port")
    debug: bool = Field(False, description="Debug mode")
    cors_origins: List[str] = Field(default_factory=list, description="CORS allowed origins")
    max_file_size: int = Field(100 * 1024 * 1024, description="Max upload file size in bytes")


class ConfigResponse(BaseModel):
    """Configuration response."""
    index_config: IndexConfig = Field(..., description="Index configuration")
    processing_config: ProcessingConfig = Field(..., description="Processing configuration")
    api_config: APIConfig = Field(..., description="API configuration")

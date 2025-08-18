"""
Database schema and operations for PlanQuery.
Handles storage of documents, pages, regions, and extracted text with metadata.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    logger.warning("pgvector not available, vector operations will be limited")

Base = declarative_base()


class Document(Base):
    """Represents a processed PDF document."""
    __tablename__ = 'documents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_path = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    title = Column(String)
    discipline = Column(String)  # A/M/E/S/C
    total_pages = Column(Integer, nullable=False)
    file_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    metadata = Column(JSON)
    
    # Relationships
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, file_name='{self.file_name}', discipline='{self.discipline}')>"


class Page(Base):
    """Represents a page within a document."""
    __tablename__ = 'pages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    page_number = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    dpi = Column(Integer, default=300)
    image_path = Column(String)
    floor_id = Column(String)  # Canonical floor ID (L1, L2, LB1, LR, etc.)
    sheet_number = Column(String)  # Sheet number from title block
    sheet_name = Column(String)   # Sheet name/title
    revision = Column(String)     # Revision number/date
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)
    
    # Relationships
    document = relationship("Document", back_populates="pages")
    regions = relationship("Region", back_populates="page", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Page(id={self.id}, page_number={self.page_number}, floor_id='{self.floor_id}')>"


class Region(Base):
    """Represents a detected layout region within a page."""
    __tablename__ = 'regions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    page_id = Column(UUID(as_uuid=True), ForeignKey('pages.id'), nullable=False)
    region_type = Column(String, nullable=False)  # text, title, table, etc.
    text_type = Column(String)  # AEC-specific classification
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)
    confidence = Column(Float, default=0.0)
    text_content = Column(Text)
    text_confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)
    
    # Vector embedding (if pgvector is available)
    if PGVECTOR_AVAILABLE:
        embedding = Column(Vector(384))  # Sentence transformer dimension
    
    # Relationships
    page = relationship("Page", back_populates="regions")
    
    @property
    def bbox(self) -> Dict[str, int]:
        """Get bounding box as dictionary."""
        return {
            'x1': self.bbox_x1,
            'y1': self.bbox_y1,
            'x2': self.bbox_x2,
            'y2': self.bbox_y2
        }
    
    @property
    def area(self) -> int:
        """Calculate region area."""
        return (self.bbox_x2 - self.bbox_x1) * (self.bbox_y2 - self.bbox_y1)
    
    def __repr__(self):
        return f"<Region(id={self.id}, region_type='{self.region_type}', text_type='{self.text_type}')>"


class SearchIndex(Base):
    """Keyword search index for BM25 and regex searches."""
    __tablename__ = 'search_index'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    region_id = Column(UUID(as_uuid=True), ForeignKey('regions.id'), nullable=False)
    keywords = Column(Text)  # Preprocessed keywords for search
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SearchIndex(id={self.id}, region_id={self.region_id})>"


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: str = "postgresql://localhost/planquery"):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            self.engine = create_engine(
                self.database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            logger.info(f"Database initialized: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def create_document(self, file_path: str, file_name: str, 
                       discipline: Optional[str] = None,
                       title: Optional[str] = None,
                       total_pages: int = 0,
                       file_size: Optional[int] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Create a new document record."""
        with self.get_session() as session:
            document = Document(
                file_path=file_path,
                file_name=file_name,
                discipline=discipline,
                title=title,
                total_pages=total_pages,
                file_size=file_size,
                processed_at=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            session.add(document)
            session.commit()
            session.refresh(document)
            
            logger.info(f"Created document: {document.id}")
            return document
    
    def create_page(self, document_id: uuid.UUID, page_number: int,
                   width: int, height: int, dpi: int = 300,
                   image_path: Optional[str] = None,
                   floor_id: Optional[str] = None,
                   sheet_number: Optional[str] = None,
                   sheet_name: Optional[str] = None,
                   revision: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Page:
        """Create a new page record."""
        with self.get_session() as session:
            page = Page(
                document_id=document_id,
                page_number=page_number,
                width=width,
                height=height,
                dpi=dpi,
                image_path=image_path,
                floor_id=floor_id,
                sheet_number=sheet_number,
                sheet_name=sheet_name,
                revision=revision,
                metadata=metadata or {}
            )
            
            session.add(page)
            session.commit()
            session.refresh(page)
            
            return page
    
    def create_region(self, page_id: uuid.UUID, region_type: str,
                     bbox_x1: int, bbox_y1: int, bbox_x2: int, bbox_y2: int,
                     confidence: float = 0.0,
                     text_type: Optional[str] = None,
                     text_content: Optional[str] = None,
                     text_confidence: float = 0.0,
                     embedding: Optional[List[float]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Region:
        """Create a new region record."""
        with self.get_session() as session:
            region_data = {
                'page_id': page_id,
                'region_type': region_type,
                'text_type': text_type,
                'bbox_x1': bbox_x1,
                'bbox_y1': bbox_y1,
                'bbox_x2': bbox_x2,
                'bbox_y2': bbox_y2,
                'confidence': confidence,
                'text_content': text_content,
                'text_confidence': text_confidence,
                'metadata': metadata or {}
            }
            
            # Add embedding if pgvector is available
            if PGVECTOR_AVAILABLE and embedding:
                region_data['embedding'] = embedding
            
            region = Region(**region_data)
            
            session.add(region)
            session.commit()
            session.refresh(region)
            
            return region
    
    def get_document_by_id(self, document_id: uuid.UUID) -> Optional[Document]:
        """Get document by ID."""
        with self.get_session() as session:
            return session.query(Document).filter(Document.id == document_id).first()
    
    def get_documents_by_discipline(self, discipline: str) -> List[Document]:
        """Get all documents of a specific discipline."""
        with self.get_session() as session:
            return session.query(Document).filter(Document.discipline == discipline).all()
    
    def get_pages_by_floor(self, floor_id: str) -> List[Page]:
        """Get all pages for a specific floor."""
        with self.get_session() as session:
            return session.query(Page).filter(Page.floor_id == floor_id).all()
    
    def get_regions_by_text_type(self, text_type: str) -> List[Region]:
        """Get all regions of a specific text type."""
        with self.get_session() as session:
            return session.query(Region).filter(Region.text_type == text_type).all()
    
    def search_regions_by_text(self, query: str, limit: int = 50) -> List[Region]:
        """Search regions by text content."""
        with self.get_session() as session:
            return session.query(Region).filter(
                Region.text_content.ilike(f"%{query}%")
            ).limit(limit).all()
    
    def get_regions_with_filters(self, 
                               floor_id: Optional[str] = None,
                               discipline: Optional[str] = None,
                               text_type: Optional[str] = None,
                               region_type: Optional[str] = None,
                               limit: int = 100) -> List[Region]:
        """Get regions with multiple filters."""
        with self.get_session() as session:
            query = session.query(Region).join(Page).join(Document)
            
            if floor_id:
                query = query.filter(Page.floor_id == floor_id)
            
            if discipline:
                query = query.filter(Document.discipline == discipline)
            
            if text_type:
                query = query.filter(Region.text_type == text_type)
            
            if region_type:
                query = query.filter(Region.region_type == region_type)
            
            return query.limit(limit).all()
    
    def update_region_embedding(self, region_id: uuid.UUID, embedding: List[float]):
        """Update region embedding."""
        if not PGVECTOR_AVAILABLE:
            logger.warning("pgvector not available, cannot update embedding")
            return
        
        with self.get_session() as session:
            region = session.query(Region).filter(Region.id == region_id).first()
            if region:
                region.embedding = embedding
                session.commit()
    
    def vector_search(self, query_embedding: List[float], 
                     limit: int = 10,
                     floor_id: Optional[str] = None,
                     discipline: Optional[str] = None) -> List[Region]:
        """Perform vector similarity search."""
        if not PGVECTOR_AVAILABLE:
            logger.warning("pgvector not available, cannot perform vector search")
            return []
        
        with self.get_session() as session:
            query = session.query(Region).join(Page).join(Document)
            
            # Add filters
            if floor_id:
                query = query.filter(Page.floor_id == floor_id)
            
            if discipline:
                query = query.filter(Document.discipline == discipline)
            
            # Order by vector similarity (L2 distance)
            query = query.order_by(Region.embedding.l2_distance(query_embedding))
            
            return query.limit(limit).all()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_session() as session:
            stats = {
                'documents': session.query(Document).count(),
                'pages': session.query(Page).count(),
                'regions': session.query(Region).count(),
                'regions_with_text': session.query(Region).filter(
                    Region.text_content.isnot(None)
                ).count(),
            }
            
            # Discipline breakdown
            discipline_counts = session.query(
                Document.discipline, 
                session.query(Document).filter(
                    Document.discipline == Document.discipline
                ).count()
            ).group_by(Document.discipline).all()
            
            stats['by_discipline'] = dict(discipline_counts)
            
            # Floor breakdown
            floor_counts = session.query(
                Page.floor_id,
                session.query(Page).filter(
                    Page.floor_id == Page.floor_id
                ).count()
            ).group_by(Page.floor_id).all()
            
            stats['by_floor'] = dict(floor_counts)
            
            return stats
    
    def cleanup_orphaned_records(self):
        """Clean up orphaned records."""
        with self.get_session() as session:
            # Remove regions without pages
            orphaned_regions = session.query(Region).filter(
                ~Region.page_id.in_(session.query(Page.id))
            ).count()
            
            if orphaned_regions > 0:
                session.query(Region).filter(
                    ~Region.page_id.in_(session.query(Page.id))
                ).delete(synchronize_session=False)
                
                logger.info(f"Cleaned up {orphaned_regions} orphaned regions")
            
            # Remove pages without documents
            orphaned_pages = session.query(Page).filter(
                ~Page.document_id.in_(session.query(Document.id))
            ).count()
            
            if orphaned_pages > 0:
                session.query(Page).filter(
                    ~Page.document_id.in_(session.query(Document.id))
                ).delete(synchronize_session=False)
                
                logger.info(f"Cleaned up {orphaned_pages} orphaned pages")
            
            session.commit()
    
    def delete_document(self, document_id: uuid.UUID):
        """Delete document and all related records."""
        with self.get_session() as session:
            document = session.query(Document).filter(Document.id == document_id).first()
            if document:
                session.delete(document)  # Cascade will handle pages and regions
                session.commit()
                logger.info(f"Deleted document: {document_id}")
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")

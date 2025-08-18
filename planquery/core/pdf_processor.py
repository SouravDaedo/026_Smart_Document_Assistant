"""
PDF processing module for converting PDFs to images and extracting metadata.
"""

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class PageInfo:
    """Information about a PDF page."""
    page_num: int
    width: int
    height: int
    dpi: int
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class PDFDocument:
    """Represents a processed PDF document."""
    file_path: str
    total_pages: int
    pages: List[PageInfo]
    metadata: Dict[str, Any]
    title: Optional[str] = None
    discipline: Optional[str] = None  # A/M/E/S/C


class PDFProcessor:
    """Handles PDF rasterization and basic metadata extraction."""
    
    def __init__(self, dpi: int = 300, output_dir: str = "output"):
        self.dpi = dpi
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def process_pdf(self, pdf_path: str) -> PDFDocument:
        """
        Process a PDF file and extract pages as images with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDFDocument with processed pages and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract metadata using PyMuPDF
        doc_metadata = self._extract_pdf_metadata(pdf_path)
        
        # Convert pages to images
        pages = self._convert_to_images(pdf_path)
        
        # Detect discipline from filename or metadata
        discipline = self._detect_discipline(pdf_path, doc_metadata)
        
        document = PDFDocument(
            file_path=str(pdf_path),
            total_pages=len(pages),
            pages=pages,
            metadata=doc_metadata,
            title=doc_metadata.get('title'),
            discipline=discipline
        )
        
        logger.info(f"Processed {len(pages)} pages from {pdf_path}")
        return document
    
    def _extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Add custom fields
            metadata.update({
                'page_count': doc.page_count,
                'file_size': pdf_path.stat().st_size,
                'file_name': pdf_path.name,
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
            })
            
            doc.close()
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
            return {
                'file_name': pdf_path.name,
                'file_size': pdf_path.stat().st_size,
            }
    
    def _convert_to_images(self, pdf_path: Path) -> List[PageInfo]:
        """Convert PDF pages to images."""
        try:
            # Use pdf2image for high-quality conversion
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt='PNG',
                thread_count=4
            )
            
            pages = []
            for i, image in enumerate(images):
                page_num = i + 1
                
                # Save image
                image_filename = f"{pdf_path.stem}_page_{page_num:03d}.png"
                image_path = self.output_dir / image_filename
                image.save(image_path, 'PNG', optimize=True)
                
                page_info = PageInfo(
                    page_num=page_num,
                    width=image.width,
                    height=image.height,
                    dpi=self.dpi,
                    image_path=str(image_path),
                    metadata=self._extract_page_metadata(pdf_path, page_num)
                )
                
                pages.append(page_info)
                
            return pages
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise
    
    def _extract_page_metadata(self, pdf_path: Path, page_num: int) -> Dict[str, Any]:
        """Extract metadata for a specific page."""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]  # 0-indexed
            
            # Get page dimensions
            rect = page.rect
            
            # Extract text for basic analysis
            text = page.get_text()
            
            metadata = {
                'page_num': page_num,
                'width': rect.width,
                'height': rect.height,
                'rotation': page.rotation,
                'text_length': len(text),
                'has_images': len(page.get_images()) > 0,
                'has_drawings': len(page.get_drawings()) > 0,
            }
            
            doc.close()
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract page metadata for page {page_num}: {e}")
            return {'page_num': page_num}
    
    def _detect_discipline(self, pdf_path: Path, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Detect the discipline (A/M/E/S/C) from filename or metadata.
        
        A = Architectural
        M = Mechanical  
        E = Electrical
        S = Structural
        C = Civil
        """
        filename = pdf_path.name.lower()
        title = metadata.get('title', '').lower()
        
        # Common patterns in architectural plan filenames
        discipline_patterns = {
            'A': ['arch', 'architectural', 'floor plan', 'elevation', 'section'],
            'M': ['mech', 'mechanical', 'hvac', 'plumbing', 'mep'],
            'E': ['elec', 'electrical', 'power', 'lighting', 'telecom'],
            'S': ['struct', 'structural', 'foundation', 'framing'],
            'C': ['civil', 'site', 'grading', 'utility', 'survey']
        }
        
        for discipline, patterns in discipline_patterns.items():
            for pattern in patterns:
                if pattern in filename or pattern in title:
                    return discipline
        
        # Try to extract from filename prefix (e.g., "A-101", "M-201")
        parts = filename.split('-')
        if len(parts) >= 2 and len(parts[0]) == 1:
            potential_discipline = parts[0].upper()
            if potential_discipline in discipline_patterns:
                return potential_discipline
        
        return None
    
    def get_page_image(self, page_info: PageInfo) -> np.ndarray:
        """Load a page image as numpy array."""
        if not page_info.image_path or not Path(page_info.image_path).exists():
            raise FileNotFoundError(f"Page image not found: {page_info.image_path}")
        
        image = Image.open(page_info.image_path)
        return np.array(image)
    
    def cleanup_images(self, document: PDFDocument):
        """Remove generated page images to save space."""
        for page in document.pages:
            if page.image_path and Path(page.image_path).exists():
                Path(page.image_path).unlink()
                page.image_path = None

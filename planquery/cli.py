"""
Command-line interface for PlanQuery.
Provides CLI commands for document processing and system management.
"""

import click
import sys
import time
from pathlib import Path
from typing import Optional
import yaml
from loguru import logger

from .core.pdf_processor import PDFProcessor
from .core.layout_detector import LayoutDetector
from .core.ocr_engine import OCREngine
from .core.text_classifier import TextClassifier
from .core.floor_normalizer import FloorNormalizer
from .indexing.database import DatabaseManager
from .indexing.vector_store import VectorStore
from .indexing.keyword_index import KeywordIndex
from .search.query_parser import QueryParser
from .search.retriever import HybridRetriever
from .search.reranker import ResultReranker


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


@click.group()
@click.option('--config', default='config.yaml', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """PlanQuery CLI - Smart Document Assistant for Architectural Plans."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--discipline', help='Document discipline (A/M/E/S/C)')
@click.option('--title', help='Document title')
@click.option('--output-dir', default='output', help='Output directory')
@click.pass_context
def ingest(ctx, pdf_path, discipline, title, output_dir):
    """Ingest and process a PDF document."""
    config = ctx.obj['config']
    
    logger.info(f"Processing document: {pdf_path}")
    start_time = time.time()
    
    try:
        # Initialize components
        database_url = config.get('database', {}).get('url', 'postgresql://localhost/planquery')
        db_manager = DatabaseManager(database_url)
        
        vector_store = VectorStore(
            backend=config.get('search', {}).get('vector_store', {}).get('backend', 'faiss'),
            index_path=config.get('search', {}).get('vector_store', {}).get('index_path', 'indices/vector.index'),
            database_manager=db_manager
        )
        
        keyword_index = KeywordIndex(
            index_dir=config.get('search', {}).get('keyword_index', {}).get('index_dir', 'indices/keyword')
        )
        
        # Processing components
        pdf_processor = PDFProcessor(
            dpi=config.get('processing', {}).get('pdf', {}).get('dpi', 300),
            output_dir=output_dir
        )
        
        layout_detector = LayoutDetector(
            model_type=config.get('processing', {}).get('layout_detection', {}).get('model_type', 'detectron2'),
            confidence_threshold=config.get('processing', {}).get('layout_detection', {}).get('confidence_threshold', 0.5)
        )
        
        ocr_engine = OCREngine(
            engine=config.get('processing', {}).get('ocr', {}).get('engine', 'tesseract')
        )
        
        text_classifier = TextClassifier(
            model_type=config.get('processing', {}).get('text_classification', {}).get('model_type', 'rules')
        )
        
        floor_normalizer = FloorNormalizer()
        
        # Process PDF
        pdf_document = pdf_processor.process_pdf(pdf_path)
        
        # Create database record
        document = db_manager.create_document(
            file_path=pdf_path,
            file_name=Path(pdf_path).name,
            discipline=discipline or pdf_document.discipline,
            title=title or pdf_document.title,
            total_pages=pdf_document.total_pages,
            file_size=Path(pdf_path).stat().st_size,
            metadata=pdf_document.metadata
        )
        
        total_regions = 0
        
        # Process each page
        for page_info in pdf_document.pages:
            click.echo(f"Processing page {page_info.page_num}...")
            
            # Load page image
            page_image = pdf_processor.get_page_image(page_info)
            
            # Detect layout regions
            detected_regions = layout_detector.detect_regions(page_image)
            detected_regions = layout_detector.filter_regions(detected_regions)
            
            # Extract text from regions
            ocr_results = ocr_engine.extract_text_from_regions(page_image, detected_regions)
            
            # Classify text
            classifications = text_classifier.classify_ocr_results(ocr_results)
            
            # Normalize floor information
            page_text = " ".join([ocr.text for ocr in ocr_results])
            floor_info = floor_normalizer.normalize_floor_from_text(page_text)
            
            # Create page record
            page = db_manager.create_page(
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
                region = db_manager.create_region(
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
                    vector_store.add_text(
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
                    keyword_index.add_document(
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
        vector_store.save_index()
        keyword_index.optimize()
        
        processing_time = time.time() - start_time
        
        click.echo(f"✅ Document processed successfully!")
        click.echo(f"   - Pages: {pdf_document.total_pages}")
        click.echo(f"   - Regions: {total_regions}")
        click.echo(f"   - Processing time: {processing_time:.2f}s")
        
        db_manager.close()
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        click.echo(f"❌ Processing failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--limit', default=10, help='Maximum number of results')
@click.option('--floor', help='Filter by floor ID')
@click.option('--discipline', help='Filter by discipline')
@click.pass_context
def search(ctx, query, limit, floor, discipline):
    """Search documents using natural language query."""
    config = ctx.obj['config']
    
    try:
        # Initialize components
        database_url = config.get('database', {}).get('url', 'postgresql://localhost/planquery')
        db_manager = DatabaseManager(database_url)
        
        vector_store = VectorStore(
            backend=config.get('search', {}).get('vector_store', {}).get('backend', 'faiss'),
            index_path=config.get('search', {}).get('vector_store', {}).get('index_path', 'indices/vector.index'),
            database_manager=db_manager
        )
        
        keyword_index = KeywordIndex(
            index_dir=config.get('search', {}).get('keyword_index', {}).get('index_dir', 'indices/keyword')
        )
        
        query_parser = QueryParser()
        retriever = HybridRetriever(vector_store, keyword_index, db_manager)
        reranker = ResultReranker()
        
        # Parse query
        parsed_query = query_parser.parse_query(query)
        
        # Apply CLI filters
        if floor:
            parsed_query.filters['floor_id'] = floor
        if discipline:
            parsed_query.filters['discipline'] = discipline
        
        # Search
        results = retriever.retrieve(parsed_query, limit=limit)
        reranked_results = reranker.rerank(results, parsed_query, max_results=limit)
        
        # Display results
        click.echo(f"\n🔍 Search results for: '{query}'")
        click.echo(f"Found {len(reranked_results)} results\n")
        
        for i, result in enumerate(reranked_results, 1):
            click.echo(f"{i}. Score: {result.combined_score:.3f} | Source: {result.source}")
            click.echo(f"   Text: {result.text[:200]}...")
            
            metadata = result.metadata
            info_parts = []
            if metadata.get('floor_id'):
                info_parts.append(f"Floor: {metadata['floor_id']}")
            if metadata.get('discipline'):
                info_parts.append(f"Discipline: {metadata['discipline']}")
            if metadata.get('page_number'):
                info_parts.append(f"Page: {metadata['page_number']}")
            if metadata.get('text_type'):
                info_parts.append(f"Type: {metadata['text_type']}")
            
            if info_parts:
                click.echo(f"   Info: {' | '.join(info_parts)}")
            
            click.echo()
        
        db_manager.close()
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        click.echo(f"❌ Search failed: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show system statistics."""
    config = ctx.obj['config']
    
    try:
        database_url = config.get('database', {}).get('url', 'postgresql://localhost/planquery')
        db_manager = DatabaseManager(database_url)
        
        stats = db_manager.get_database_stats()
        
        click.echo("📊 PlanQuery Statistics")
        click.echo("=" * 30)
        click.echo(f"Documents: {stats.get('documents', 0)}")
        click.echo(f"Pages: {stats.get('pages', 0)}")
        click.echo(f"Regions: {stats.get('regions', 0)}")
        click.echo(f"Regions with text: {stats.get('regions_with_text', 0)}")
        
        by_discipline = stats.get('by_discipline', {})
        if by_discipline:
            click.echo("\nBy Discipline:")
            for discipline, count in by_discipline.items():
                if discipline:
                    click.echo(f"  {discipline}: {count}")
        
        by_floor = stats.get('by_floor', {})
        if by_floor:
            click.echo("\nBy Floor:")
            for floor, count in sorted(by_floor.items()):
                if floor:
                    click.echo(f"  {floor}: {count}")
        
        db_manager.close()
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        click.echo(f"❌ Stats failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.pass_context
def serve(ctx, host, port, reload):
    """Start the PlanQuery API server."""
    try:
        import uvicorn
        from .api.main import app
        
        click.echo(f"🚀 Starting PlanQuery API server on {host}:{port}")
        
        uvicorn.run(
            "planquery.api.main:app",
            host=host,
            port=port,
            reload=reload
        )
        
    except ImportError:
        click.echo("❌ uvicorn not available. Install with: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server failed: {e}")
        click.echo(f"❌ Server failed: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def setup(ctx):
    """Set up PlanQuery database and directories."""
    try:
        from .scripts.setup_db import main as setup_main
        setup_main()
        click.echo("✅ Setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        click.echo(f"❌ Setup failed: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def download_models(ctx):
    """Download required ML models."""
    try:
        from .scripts.download_models import main as download_main
        download_main()
        click.echo("✅ Model download completed!")
        
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        click.echo(f"❌ Model download failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()

"""
Database setup script for PlanQuery.
Creates database tables and initializes the system.
"""

import os
import sys
from pathlib import Path
import yaml
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from planquery.indexing.database import DatabaseManager


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def setup_database(database_url: str):
    """Set up database tables and initial data."""
    logger.info(f"Setting up database: {database_url}")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(database_url)
        
        # Tables are created automatically in __init__
        logger.info("Database tables created successfully")
        
        # Get initial stats
        stats = db_manager.get_database_stats()
        logger.info(f"Database initialized with {stats.get('documents', 0)} documents")
        
        db_manager.close()
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise


def create_directories(config: dict):
    """Create necessary directories."""
    directories = config.get('directories', {})
    
    default_dirs = {
        'uploads': 'uploads',
        'output': 'output', 
        'indices': 'indices',
        'static': 'static',
        'logs': 'logs'
    }
    
    for dir_key, default_path in default_dirs.items():
        dir_path = Path(directories.get(dir_key, default_path))
        dir_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Created directory: {dir_path}")


def main():
    """Main setup function."""
    logger.info("Starting PlanQuery database setup...")
    
    # Load configuration
    config = load_config()
    
    # Get database URL from config or environment
    database_url = (
        config.get('database', {}).get('url') or 
        os.getenv('DATABASE_URL') or 
        'postgresql://localhost/planquery'
    )
    
    try:
        # Create directories
        create_directories(config)
        
        # Setup database
        setup_database(database_url)
        
        logger.info("PlanQuery setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

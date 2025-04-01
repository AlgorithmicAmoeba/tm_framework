"""
OpenAI Embedding jobs pipeline.
These functions identify new chunks that need embeddings and create jobs for them.
"""
import logging

from sqlalchemy import text
from sqlalchemy.orm import Session
import duckdb
from duckdb import DuckDBPyConnection

from database import get_session
import configuration as cfg


def color_logging_text(message: str, color: str) -> str:
    """Change the color of the logging message."""
    color_map = {
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'purple': '35',
        'cyan': '36',
        'white': '37'
    }
    
    color_code = color_map.get(color, '37')
    return f"\033[{color_code}m{message}\033[0m"


def find_embedding_jobs(
    session: Session,
    duckdb_cache_session: DuckDBPyConnection,
    source_table: str,
    cache_table: str,
    model_name: str = "text-embedding-3-small",
    batch_size: int = 49_995
) -> int:
    """
    Look in the source table for any rows that do not exist in the cache table.
    Place them in the jobs table.
    
    Args:
        session: Database session
        duckdb_cache_session: DuckDB connection for cache operations
        source_table: Source table name to check for chunks
        cache_table: Cache table name to check for existing embeddings
        
    Returns:
        new_jobs_count
    """
    logging.info(f"Finding embedding jobs from {source_table} not in {cache_table}")
    
    # Check what's already in the cache
    cache_query = f"""
        SELECT document_hash
        FROM {cache_table}
    """
    
    try:
        cache_results = duckdb_cache_session.execute(cache_query).fetchall()
        cache_doc_hashes = {row[0] for row in cache_results}
        logging.info(f"Found {len(cache_doc_hashes)} existing documents in cache")
    except Exception as e:
        logging.error(f"Error querying cache: {str(e)}")
        cache_doc_hashes = set()
    
    # Find source documents that need embedding
    source_query = text(f"""
        SELECT c.id, c.raw_document_hash, c.corpus_name, c.chunk_hash, c.content
        FROM {source_table} c
        LEFT JOIN pipeline.embedding_jobs j ON 
            c.id = j.chunk_id AND j.model_name = :model_name
        WHERE j.id IS NULL
    """)
    
    result = session.execute(source_query, {"model_name": model_name})
    all_chunks = [(row[0], row[1], row[2], row[3], row[4]) for row in result]
    
    # Filter out chunks that already exist in cache
    new_chunks = [
        chunk for chunk in all_chunks 
        if chunk[3] not in cache_doc_hashes  # Check if chunk_hash exists in cache
    ]
    
    if not new_chunks:
        logging.info("No new chunks found that need embeddings")
        return 0, len(cache_doc_hashes)
    
    # Count existing jobs
    existing_jobs_query = text("""
        SELECT COUNT(*)
        FROM pipeline.embedding_jobs
        WHERE model_name = :model_name
    """)
    
    existing_count = session.execute(
        existing_jobs_query, 
        {"model_name": model_name}
    ).scalar() or 0
    
    # Create new jobs
    jobs_created = 0
    
    for i in range(0, len(new_chunks), batch_size):
        batch_chunks = new_chunks[i:i+batch_size]
        batch_inserts = 0
        
        for chunk_id, raw_doc_hash, corpus_name, chunk_hash, content in batch_chunks:
            # Create job entry
            insert_job_query = text("""
                INSERT INTO pipeline.embedding_jobs 
                (chunk_id, raw_document_hash, corpus_name, chunk_hash, model_name, status)
                VALUES (:chunk_id, :raw_document_hash, :corpus_name, :chunk_hash, :model_name, :status)
                ON CONFLICT (chunk_id, model_name) DO NOTHING
            """)
            
            session.execute(insert_job_query, {
                "chunk_id": chunk_id,
                "raw_document_hash": raw_doc_hash,
                "corpus_name": corpus_name,
                "chunk_hash": chunk_hash,
                "model_name": model_name,
                "status": "pending"
            })
            
            batch_inserts += 1
            jobs_created += 1
        
        # Commit after each batch
        session.commit()
        logging.debug(f"Processed batch {i//batch_size + 1}, inserted {batch_inserts} jobs")
    
    # Log statistics
    logging.info(f"Embedding job creation complete")
    logging.info(f"  Documents already in cache: {len(cache_doc_hashes)}")
    logging.info(f"  Existing jobs: {existing_count}")
    
    if jobs_created:
        logging.info(color_logging_text(
            f"  New jobs created: {jobs_created}",
            color='green'
        ))
    else:
        logging.info(f"  New jobs created: {jobs_created}")
    
    total_jobs = existing_count + jobs_created
    logging.info(f"  Total jobs in database: {total_jobs}")
    
    return jobs_created


def open_duckdb_connection(db_path: str) -> DuckDBPyConnection:
    """
    Open a connection to the DuckDB database.
    
    Args:
        db_path: Path to the DuckDB database file
        
    Returns:
        DuckDB connection object
    """
    try:
        conn = duckdb.connect(db_path)
        return conn
    except Exception as e:
        logging.error(f"Error connecting to DuckDB at {db_path}: {str(e)}")
        raise


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database
    
    # Path to DuckDB cache
    cache_db_path = str(config.get_data_path() / "embeddings" / "cache.db")
    
    # Create database session
    with get_session(db_config) as session:
        # Find embedding jobs
        logging.info("Starting embedding job discovery pipeline...")
        
        duckdb_cache_session = open_duckdb_connection(cache_db_path)

        new_jobs = find_embedding_jobs(
            session=session,
            duckdb_cache_session=duckdb_cache_session,
            source_table="pipeline.document_chunks",
            cache_table="pipeline.embedding_jobs",
        )
        
        if new_jobs > 0:
            logging.info(color_logging_text(
                f"Successfully created {new_jobs} new embedding jobs",
                color='green'
            ))
        else:
            logging.info("No new embedding jobs needed to be created")

            
        logging.info("Embedding job discovery pipeline completed successfully.")


if __name__ == "__main__":
    main()

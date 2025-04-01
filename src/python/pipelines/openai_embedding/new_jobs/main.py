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
from shared_code import color_logging_text


def find_embedding_jobs(
    session: Session,
    duckdb_cache_session: DuckDBPyConnection,
    source_table: str,
    cache_table: str,
    model_name: str = "text-embedding-3-small",
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
        SELECT c.corpus_name, c.chunk_hash, c.content
        FROM {source_table} c
        LEFT JOIN pipeline.embedding_jobs j ON 
            c.id = j.chunk_id AND j.model_name = :model_name
        WHERE j.id IS NULL
    """)
    
    all_chunks = session.execute(source_query, {"model_name": model_name}).mappings().fetchall()
    
    # Filter out chunks that already exist in cache
    new_chunks = [
        chunk for chunk in all_chunks 
        if chunk["chunk_hash"] not in cache_doc_hashes  # Check if chunk_hash exists in cache
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
    
        
    for chunk in new_chunks:
        # Create job entry
        insert_job_query = text("""
            INSERT INTO pipeline.embedding_jobs 
            (corpus_name, chunk_hash, chunk_content, model_name,)
            VALUES (:corpus_name, :chunk_hash, :chunk_content, :model_name)
            ON CONFLICT (chunk_id, model_name) DO NOTHING
        """)
        
        session.execute(insert_job_query, {
            "corpus_name": chunk["corpus_name"],
            "chunk_hash": chunk["chunk_hash"],
            "chunk_content": chunk["content"],
            "model_name": model_name
        })
        
        jobs_created += 1
        
    session.commit()
    
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


def create_duckdb_cache_table(
    duckdb_cache_session: DuckDBPyConnection,
    cache_table: str
) -> None:
    """
    Create the DuckDB cache table if it doesn't exist.
    
    Args:
        duckdb_cache_session: DuckDB connection for cache operations
        cache_table: Cache table name to create
    """
    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {cache_table} (
            document_hash VARCHAR PRIMARY KEY,
            embedding BLOB
        )
    """
    
    try:
        duckdb_cache_session.execute(create_table_query)
        logging.info(f"Cache table {cache_table} created")
    except Exception as e:
        logging.error(f"Error creating cache table {cache_table}: {str(e)}")
        raise


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
    cache_table = "pipeline.embedding_jobs"
    source_table = "pipeline.document_chunks"
    
    # Create database session
    with get_session(db_config) as session:
        # Find embedding jobs
        logging.info("Starting embedding job discovery pipeline...")
        
        duckdb_cache_session = open_duckdb_connection(cache_db_path)

        # Create DuckDB cache table if it doesn't exist
        create_duckdb_cache_table(
            duckdb_cache_session=duckdb_cache_session,
            cache_table=cache_table,
        )

        new_jobs = find_embedding_jobs(
            session=session,
            duckdb_cache_session=duckdb_cache_session,
            source_table=source_table,
            cache_table=cache_table,
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

"""
Process for ingesting embeddings from the cache into the main database.
This allows for efficient access to previously generated embeddings.
"""
import logging
import pathlib
from typing import List, Optional, Dict, Any, Tuple, Set
import json

import duckdb
from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm

from database import get_session
import configuration as cfg
from shared_code import color_logging_text
from openai_embeddings.cache import Chunk, get_cache_stats


def get_chunks_needing_embeddings(session: Session, batch_size: int = 1000) -> List[str]:
    """
    Get chunks from chunked_document that don't have embeddings yet.
    
    Args:
        session: Database session
        batch_size: Number of chunks to process at once
        
    Returns:
        List of chunk_hash
    """
    query = text("""
        SELECT cd.chunk_hash
        FROM pipeline.chunked_document cd
        LEFT JOIN pipeline.chunk_embedding ce ON cd.chunk_hash = ce.chunk_hash
        WHERE ce.chunk_hash IS NULL
        LIMIT :batch_size
    """)
    
    result = session.execute(query, {"batch_size": batch_size}).scalars().fetchall()
    return result


def get_existing_embeddings(session: Session, chunk_hashes: List[str]) -> Set[str]:
    """
    Check which chunk hashes already have embeddings in the database.
    
    Args:
        session: Database session
        chunk_hashes: List of chunk hashes to check
        
    Returns:
        Set of chunk hashes that already have embeddings
    """
    if not chunk_hashes:
        return set()
        
    placeholders = ','.join([':hash' + str(i) for i in range(len(chunk_hashes))])
    params = {f'hash{i}': hash for i, hash in enumerate(chunk_hashes)}
    
    query = text(f"""
        SELECT chunk_hash
        FROM pipeline.chunk_embedding
        WHERE chunk_hash IN ({placeholders})
    """)
    
    result = session.execute(query, params).fetchall()
    return {row[0] for row in result}


def get_embeddings_from_cache(
    cache_path: pathlib.Path,
    chunk_hashes: List[str],
    batch_size: int = 1000
) -> Dict[str, List[float]]:
    """
    Retrieve embeddings from the cache for specified chunk hashes.
    
    Args:
        cache_path: Path to the cache database
        chunk_hashes: List of chunk hashes to retrieve
        batch_size: Number of embeddings to retrieve at once
        
    Returns:
        Dictionary mapping chunk hashes to embeddings
    """
    embeddings = {}
    
    with duckdb.connect(str(cache_path)) as conn:
        # Process in batches to avoid memory issues
        for i in range(0, len(chunk_hashes), batch_size):
            batch = chunk_hashes[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch))
            
            query = f"""
            SELECT document_hash, embedding
            FROM document
            WHERE document_hash IN ({placeholders})
            """
            
            results = conn.execute(query, batch).fetchall()
            for chunk_hash, embedding in results:
                # Parse the embedding from JSON string to list of floats
                embedding_list = json.loads(embedding)
                embeddings[chunk_hash] = embedding_list
    
    return embeddings


def ingest_embeddings_to_db(
    session: Session,
    embeddings: Dict[str, List[float]],
    target_table: str = "pipeline.chunk_embedding"
) -> int:
    """
    Ingest embeddings into the main database.
    
    Args:
        session: Database session
        embeddings: Dictionary mapping chunk hashes to embeddings
        target_table: Target table to store embeddings
        
    Returns:
        Number of embeddings ingested
    """
    if not embeddings:
        return 0
    
    # Prepare data for insertion
    values = []
    for chunk_hash, embedding in embeddings.items():
        values.append({
            "chunk_hash": chunk_hash,
            "embedding": embedding  # List of floats will be automatically converted to FLOAT[]
        })
    
    # Insert embeddings
    insert_query = text(f"""
        INSERT INTO {target_table} (chunk_hash, embedding)
        VALUES (:chunk_hash, :embedding)
        ON CONFLICT (chunk_hash) DO UPDATE SET
            embedding = EXCLUDED.embedding
    """)
    
    for value in values:
        session.execute(insert_query, value)
    
    session.commit()
    return len(embeddings)


def main():
    """Main entry point for cache ingestion process."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database
    
    # Define cache path
    cache_path = pathlib.Path(config.get_data_path() / "embeddings" / "cache.db")
    
    # Get cache statistics
    cache_stats = get_cache_stats(cache_path)
    logging.info(f"Cache contains {cache_stats['total_documents']} documents")
    
    # Create database session
    with get_session(db_config) as session:
        total_processed = 0
        total_ingested = 0
        
        # Get total count of chunks needing embeddings
        count_query = text("""
            SELECT COUNT(*)
            FROM pipeline.chunked_document cd
            LEFT JOIN pipeline.chunk_embedding ce ON cd.chunk_hash = ce.chunk_hash
            WHERE ce.chunk_hash IS NULL
        """)
        total_chunks = session.execute(count_query).scalar()
        
        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            while True:
                # Get next batch of chunks needing embeddings
                chunk_hashes = get_chunks_needing_embeddings(session)
                if not chunk_hashes:
                    break
                
                # Check which ones already have embeddings
                existing_embeddings = get_existing_embeddings(session, chunk_hashes)
                missing_hashes = [hash for hash in chunk_hashes if hash not in existing_embeddings]
                
                if missing_hashes:
                    # Try to get missing embeddings from cache
                    cache_embeddings = get_embeddings_from_cache(cache_path, missing_hashes)
                    
                    if cache_embeddings:
                        # Ingest found embeddings to database
                        ingested_count = ingest_embeddings_to_db(session, cache_embeddings)
                        total_ingested += ingested_count
                
                total_processed += len(chunk_hashes)
                pbar.update(len(chunk_hashes))
                pbar.set_postfix({'ingested': total_ingested})
        
        if total_ingested > 0:
            logging.info(color_logging_text(
                f"Successfully ingested {total_ingested} embeddings from cache",
                color='green'
            ))
        else:
            logging.info("No new embeddings were ingested from cache")
        
        logging.info("Cache ingestion completed successfully.")


if __name__ == '__main__':
    main() 
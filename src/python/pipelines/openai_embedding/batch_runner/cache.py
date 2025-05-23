"""
Enhanced cache operations for document embeddings, supporting document chunks.
"""
import json
import pathlib
import hashlib
from typing import Optional, Union, Any

import duckdb
import pandas as pd

from pipelines.openai_embedding.chunk import Chunk




def ensure_cache_db(cache_path: pathlib.Path) -> None:
    """
    Ensure the cache database exists and has the correct schema.
    
    Args:
        cache_path: Path to the cache database
    """
    # Make sure directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to database
    with duckdb.connect(str(cache_path)) as conn:
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk (
            document_hash VARCHAR,
            embedding BLOB
        )
        """)



def add_to_cache(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    cache_path: pathlib.Path
) -> int:
    """
    Add document chunk embeddings to the cache.
    
    Args:
        chunks: List of Chunk objects with required fields:
            - document_hash: Hash of original document
            - chunk_hash: Hash of chunk text
            - chunk_start_index: Start index in document
            - chunk_end_index: End index in document
            - corpus_name: Name of corpus
            - text: The chunk text
        embeddings: List of embeddings (each is a list of floats)
        cache_path: Path to the cache database
        
    Returns:
        Number of items added to cache
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match")
    
    # Ensure cache exists
    ensure_cache_db(cache_path)
    
    # Prepare data for insertion
    chunk_rows = []
    
    for chunk, embedding in zip(chunks, embeddings):
        # Add chunk data
        chunk_rows.append({
            'document_hash': chunk.document_hash,
            'embedding': json.dumps(embedding)
        })
    
    # Insert into database
    with duckdb.connect(str(cache_path)) as conn:
        if chunk_rows:
            df_chunks = pd.DataFrame(chunk_rows)
            # Register the DataFrame with DuckDB
            conn.register('df_chunks', df_chunks)
            conn.execute("""
            INSERT INTO document (
                document_hash, embedding
            )
            SELECT 
                document_hash, embedding
            FROM df_chunks
            ON CONFLICT (document_hash) DO UPDATE SET
                document_hash = EXCLUDED.document_hash,
                embedding = EXCLUDED.embedding
            """)
        conn.commit()
    return len(chunk_rows)



def get_cache_stats(cache_path: pathlib.Path) -> dict[str, Any]:
    """
    Get statistics about the cache.
    
    Args:
        cache_path: Path to the cache database
        
    Returns:
        Dictionary with cache statistics
    """
    ensure_cache_db(cache_path)
    
    with duckdb.connect(str(cache_path)) as conn:
        # Get chunk count
        chunk_count = conn.execute("SELECT COUNT(*) FROM document").fetchone()[0]
        
        # Get unique document count
        doc_count = conn.execute("""
        SELECT COUNT(DISTINCT document_hash) FROM document
        """).fetchone()[0]
    
    return {
        "total_documents": doc_count,
        "total_chunks": chunk_count,
    }
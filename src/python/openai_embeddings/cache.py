"""
Simplified cache operations for document embeddings, using raw text as keys.
"""
import json
import pathlib
import hashlib
from typing import List, Optional, Dict, Tuple, Any

import duckdb
import pandas as pd

def hash_text(text: str) -> str:
    """
    Create a hash of the input text to use as a key.
    
    Args:
        text: The text to hash
        
    Returns:
        A string hash representation of the input text
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


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
        CREATE SEQUENCE IF NOT EXISTS id_sequence START 1
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS document (
            id INTEGER DEFAULT nextval('id_sequence') PRIMARY KEY,
            corpus_id INTEGER,
            content_hash VARCHAR,
            embedding_json VARCHAR
        )
        """)


def check_cache(
    texts: List[str], 
    cache_path: pathlib.Path
) -> Tuple[List[bool], List[Optional[List[float]]]]:
    """
    Check if documents are in the cache.
    
    Args:
        texts: List of text strings to check
        cache_path: Path to the cache database
        
    Returns:
        Tuple of (is_cached, embeddings) where:
        - is_cached: List of booleans indicating whether each text is cached
        - embeddings: List of embeddings (or None if not cached)
    """
    # Ensure cache exists
    ensure_cache_db(cache_path)
    
    is_cached = []
    embeddings = []
    with duckdb.connect(str(cache_path)) as conn:
        # Hash each text and check if it exists in the cache
        for text in texts:
            text_hash = hash_text(text)
            query = f"""
            SELECT embedding_json
            FROM document
            WHERE content_hash = ?
            """
            result = conn.execute(query, [text_hash]).fetchone()
            if result:
                is_cached.append(True)
                embeddings.append(json.loads(result[0]))
            else:
                is_cached.append(False)
                embeddings.append(None)
    
    return is_cached, embeddings


def add_to_cache(
    hashes: List[str],
    embeddings: List[List[float]],
    cache_path: pathlib.Path
) -> int:
    """
    Add document embeddings to the cache.
    
    Args:
        hashed_text: List of text hashes
        embeddings: List of embeddings (each is a list of floats)
        cache_path: Path to the cache database
        
    Returns:
        Number of items added to cache
    """
    if len(hashes) != len(embeddings):
        raise ValueError("Number of texts and embeddings must match")
    
    # Ensure cache exists
    ensure_cache_db(cache_path)
    
    # Prepare data for insertion
    rows = []
    for hashed_text, embedding in zip(hashes, embeddings):
        rows.append({
            'content_hash': hashed_text,
            'embedding_json': json.dumps(embedding)
        })
    
    # Insert into database
    with duckdb.connect(str(cache_path)) as conn:
        # Use pandas for bulk insertion
        if rows:
            df = pd.DataFrame(rows)
            conn.execute("""
            INSERT OR REPLACE INTO document (content_hash, embedding_json)
            SELECT content_hash, embedding_json FROM df
            """)
    
    return len(rows)


def get_cache_stats(cache_path: pathlib.Path) -> Dict[str, int]:
    """
    Get statistics about the cache.
    
    Args:
        cache_path: Path to the cache database
        
    Returns:
        Dictionary with cache statistics
    """
    ensure_cache_db(cache_path)
    
    with duckdb.connect(str(cache_path)) as conn:
        # Get total entries
        total = conn.execute("SELECT COUNT(*) FROM document").fetchone()[0]
        
        # Get total size
        size = conn.execute("""
        SELECT SUM(LENGTH(content_hash) + LENGTH(embedding_json)) 
        FROM document
        """).fetchone()[0] or 0
        
        # Get oldest and newest entries
        oldest = conn.execute("""
        SELECT MIN(created_at) FROM document
        """).fetchone()[0]
        
        newest = conn.execute("""
        SELECT MAX(created_at) FROM document
        """).fetchone()[0]
    
    return {
        "total_entries": total,
        "size_bytes": size,
        "oldest_entry": str(oldest) if oldest else None,
        "newest_entry": str(newest) if newest else None
    }



if __name__ == "__main__":
    cache_path = pathlib.Path("ignore/embedding_store.db")
    with duckdb.connect(str(cache_path)) as session:
        # print all table names
        tables = session.execute("SHOW TABLES").fetchall()
        print(tables)

        # describe all tables
        for table in tables:
            table = table[0]
            print(session.execute(f"DESCRIBE {table}").fetch_df())

        # get 1 from from document table
        print(session.execute("SELECT * FROM document LIMIT 1").fetch_df())
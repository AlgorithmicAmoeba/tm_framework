"""
Enhanced cache operations for document embeddings, supporting document chunks.
"""
import json
import pathlib
import hashlib
from typing import Optional, Union, Any
import dataclasses

import duckdb
import pandas as pd


@dataclasses.dataclass
class Chunk:
    """A chunk of text to embed."""
    document_hash: str = ""
    chunk_hash: str = ""
    chunk_start_index: int = 0
    chunk_end_index: int = 0
    corpus_name: str = ""
    text: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_hash": self.document_hash,
            "chunk_hash": self.chunk_hash,
            "chunk_start_index": self.chunk_start_index,
            "chunk_end_index": self.chunk_end_index,
            "corpus_name": self.corpus_name,
            "text": self.text
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create Chunk from dictionary."""
        return cls(
            document_hash=data.get("document_hash", ""),
            chunk_hash=data.get("chunk_hash", ""),
            chunk_start_index=data.get("chunk_start_index", 0),
            chunk_end_index=data.get("chunk_end_index", 0),
            corpus_name=data.get("corpus_name", ""),
            text=data.get("text", "")
        )

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
        CREATE TABLE IF NOT EXISTS chunk (
            id INTEGER DEFAULT nextval('id_sequence') PRIMARY KEY,
            document_hash VARCHAR,              -- Hash of original document
            chunk_hash VARCHAR UNIQUE,          -- Hash of the chunk text
            chunk_start_index INTEGER,          -- Start index of chunk in original doc
            chunk_end_index INTEGER,            -- End index of chunk in original doc
            corpus_name VARCHAR,                -- Name of the corpus
            text TEXT,                          -- The chunk text
            embedding_json VARCHAR,             -- The embedding as JSON
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create indexes for faster lookups
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_document_hash ON chunk(document_hash)
        """)
        
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunk_hash ON chunk(chunk_hash)
        """)


def check_cache(
    chunks: list[Chunk], 
    cache_path: pathlib.Path
) -> tuple[list[bool], list[Optional[list[float]]]]:
    """
    Check if document chunks are in the cache.
    
    Args:
        chunks: List of Chunk objects
        cache_path: Path to the cache database
        
    Returns:
        Tuple of (is_cached, embeddings) where:
        - is_cached: List of booleans indicating whether each chunk is cached
        - embeddings: List of embeddings (or None if not cached)
    """
    # Ensure cache exists
    ensure_cache_db(cache_path)
    
    is_cached = []
    embeddings = []
    with duckdb.connect(str(cache_path)) as conn:
        # Check each chunk
        for chunk in chunks:
            query = """
            SELECT embedding_json
            FROM chunk
            WHERE chunk_hash = ?
            """
            result = conn.execute(query, [chunk.chunk_hash]).fetchone()
            if result:
                is_cached.append(True)
                embeddings.append(json.loads(result[0]))
            else:
                is_cached.append(False)
                embeddings.append(None)
    
    return is_cached, embeddings


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
            'chunk_hash': chunk.chunk_hash,
            'chunk_start_index': chunk.chunk_start_index,
            'chunk_end_index': chunk.chunk_end_index,
            'corpus_name': chunk.corpus_name,
            'text': chunk.text,
            'embedding_json': json.dumps(embedding)
        })
    
    # Insert into database
    with duckdb.connect(str(cache_path)) as conn:
        if chunk_rows:
            df_chunks = pd.DataFrame(chunk_rows)
            conn.execute("""
            INSERT OR REPLACE INTO chunk (
                document_hash, chunk_hash, chunk_start_index, 
                chunk_end_index, corpus_name, text, embedding_json
            )
            SELECT 
                document_hash, chunk_hash, chunk_start_index, 
                chunk_end_index, corpus_name, text, embedding_json
            FROM df_chunks
            """)
    
    return len(chunk_rows)


def get_document_chunks(
    document_hash: str,
    cache_path: pathlib.Path
) -> list[Chunk]:
    """
    Get all chunks for a specific document by its hash.
    
    Args:
        document_hash: Hash of the document
        cache_path: Path to the cache database
        
    Returns:
        List of Chunk objects
    """
    ensure_cache_db(cache_path)
    
    chunks = []
    with duckdb.connect(str(cache_path)) as conn:
        # Get all chunks for this document
        results = conn.execute("""
        SELECT 
            document_hash, chunk_hash, chunk_start_index, 
            chunk_end_index, corpus_name, text, embedding_json
        FROM chunk
        WHERE document_hash = ?
        ORDER BY chunk_start_index
        """, [document_hash]).fetchall()
        
        for row in results:
            chunk = Chunk(
                document_hash=row[0],
                chunk_hash=row[1],
                chunk_start_index=row[2],
                chunk_end_index=row[3],
                corpus_name=row[4],
                text=row[5]
            )
            chunks.append(chunk)
    
    return chunks


def get_chunk_by_hash(
    chunk_hash: str,
    cache_path: pathlib.Path
) -> Optional[Chunk]:
    """
    Get a specific chunk by its hash.
    
    Args:
        chunk_hash: Hash of the chunk
        cache_path: Path to the cache database
        
    Returns:
        Chunk object, or None if not found
    """
    ensure_cache_db(cache_path)
    
    with duckdb.connect(str(cache_path)) as conn:
        # Get chunk info
        result = conn.execute("""
        SELECT 
            document_hash, chunk_hash, chunk_start_index, 
            chunk_end_index, corpus_name, text
        FROM chunk
        WHERE chunk_hash = ?
        """, [chunk_hash]).fetchone()
        
        if result:
            return Chunk(
                document_hash=result[0],
                chunk_hash=result[1],
                chunk_start_index=result[2],
                chunk_end_index=result[3],
                corpus_name=result[4],
                text=result[5]
            )
    
    return None


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
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
        
        # Get unique document count
        doc_count = conn.execute("""
        SELECT COUNT(DISTINCT document_hash) FROM chunk
        """).fetchone()[0]
        
        # Get average chunks per document
        avg_chunks = conn.execute("""
        SELECT AVG(chunk_count) FROM (
            SELECT document_hash, COUNT(*) as chunk_count 
            FROM chunk GROUP BY document_hash
        )
        """).fetchone()[0]
        
        # Get total size
        chunk_size = conn.execute("""
        SELECT SUM(LENGTH(document_hash) + LENGTH(chunk_hash) + LENGTH(text) + LENGTH(embedding_json) + LENGTH(corpus_name)) 
        FROM chunk
        """).fetchone()[0] or 0
        
        # Get oldest and newest entries
        oldest_chunk = conn.execute("""
        SELECT MIN(created_at) FROM chunk
        """).fetchone()[0]
        
        newest_chunk = conn.execute("""
        SELECT MAX(created_at) FROM chunk
        """).fetchone()[0]
    
    return {
        "total_documents": doc_count,
        "total_chunks": chunk_count,
        "avg_chunks_per_document": float(avg_chunks) if avg_chunks else 0,
        "chunk_size_bytes": chunk_size,
        "total_size_bytes": chunk_size,
        "oldest_chunk": str(oldest_chunk) if oldest_chunk else None,
        "newest_chunk": str(newest_chunk) if newest_chunk else None
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

        # get 1 from chunk table
        print(session.execute("SELECT * FROM chunk LIMIT 1").fetch_df())
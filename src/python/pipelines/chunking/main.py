"""
Chunking pipeline for document corpora.
These functions handle document chunking for further processing or embedding.
"""
import hashlib
import logging
import dataclasses
from typing import Any

import tiktoken
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg


def hash_text(text: str) -> str:
    """Generate a hash for the document content."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


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


@dataclasses.dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    raw_document_hash: str
    corpus_name: str
    content: str
    chunk_start: int
    token_count: int
    chunk_hash: str = ""
    
    def __post_init__(self) -> None:
        """Compute chunk hash if not provided."""
        if not self.chunk_hash:
            self.chunk_hash = hash_text(self.content)
    
    @property
    def db_key(self) -> tuple[str, str, str]:
        """
        Get the unique identifier tuple for database operations.
        
        Returns:
            Tuple of (raw_document_hash, corpus_name, chunk_hash)
        """
        return (self.raw_document_hash, self.corpus_name, self.chunk_hash)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for database operations.
        
        Returns:
            Dictionary representation of the chunk
        """
        return {
            "raw_document_hash": self.raw_document_hash,
            "corpus_name": self.corpus_name,
            "chunk_hash": self.chunk_hash,
            "content": self.content,
            "chunk_start": self.chunk_start,
            "token_count": self.token_count
        }


class DocumentChunker:
    """Document chunking class with configurable options."""
    
    def __init__(
            self,
            max_tokens: int,
            tokenizer_name: str = "cl100k_base"
        ):
        """
        Initialize the document chunker.
        
        Args:
            max_tokens: Maximum number of tokens per chunk
            tokenizer_name: Name of the tokenizer to use
        """
        self._max_tokens = max_tokens
        self._tokenizer_name = tokenizer_name
        
        # Initialize tokenizer
        self._tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    def chunk_first_n_tokens(self, doc_hash: str, corpus_name: str, content: str) -> list[DocumentChunk]:
        """
        Create a single chunk with the first N tokens of a document.
        
        Args:
            doc_hash: Hash of the document
            corpus_name: Name of the corpus
            content: Document content
            
        Returns:
            List containing a single DocumentChunk
        """
        # Tokenize the document content
        tokens = self._tokenizer.encode(content)
        
        # Take only the first max_tokens
        if len(tokens) > self._max_tokens:
            tokens = tokens[:self._max_tokens]
        
        # Decode back to text
        chunk_text = self._tokenizer.decode(tokens)
        
        # Create a DocumentChunk
        chunk = DocumentChunk(
            raw_document_hash=doc_hash,
            corpus_name=corpus_name,
            content=chunk_text,
            chunk_start=0,
            token_count=len(tokens)
        )
        
        return [chunk]


def store_chunked_documents(
    session: Session,
    chunked_data: list[DocumentChunk],
    corpus_name: str
):
    """
    Store chunked documents in the database.
    Uses upsert pattern for idempotency.
    
    Args:
        session: Database session
        chunked_data: List of DocumentChunk objects
        corpus_name: Name of the corpus
    """
    logging.info(f"Storing {len(chunked_data)} chunked documents")
    
    # Get existing chunks to track what's already processed
    existing_chunks = session.execute(
        text("""
            SELECT raw_document_hash, corpus_name, chunk_hash 
            FROM pipeline.chunked_document
            WHERE corpus_name = :corpus_name
        """),
        {"corpus_name": corpus_name}
    ).fetchall()
    
    # Create a set of tuples for fast lookup
    existing_chunks_set = {
        (row[0], row[1], row[2]): False 
        for row in existing_chunks
    }
    
    # Track statistics
    chunks_existing = 0
    chunks_inserted = 0
    
    # Process chunks in batches to avoid memory issues
    batch_size = 1000
    for i in range(0, len(chunked_data), batch_size):
        batch_chunks = chunked_data[i:i+batch_size]
        batch_inserts = 0
        
        for chunk in batch_chunks:
            # Use the db_key property to get the unique key
            if chunk.db_key in existing_chunks_set:
                existing_chunks_set[chunk.db_key] = True
                chunks_existing += 1
                continue
            
            # Insert new chunk
            insert_chunk_query = text("""
                INSERT INTO pipeline.chunked_document 
                (raw_document_hash, corpus_name, chunk_hash, content, chunk_start, token_count)
                VALUES (:raw_document_hash, :corpus_name, :chunk_hash, :content, :chunk_start, :token_count)
                ON CONFLICT (corpus_name, raw_document_hash, chunk_hash) 
                DO UPDATE SET 
                    content = EXCLUDED.content,
                    chunk_start = EXCLUDED.chunk_start,
                    token_count = EXCLUDED.token_count
            """)
            
            # Convert the chunk to a dictionary for the query parameters
            session.execute(insert_chunk_query, chunk.to_dict())
            chunks_inserted += 1
            batch_inserts += 1
        
        # Commit after each batch
        session.commit()
        logging.debug(f"Processed batch {i//batch_size + 1}, inserted {batch_inserts} chunks")
    
    # Delete chunks that were not updated (no longer exist in source)
    chunks_deleted = 0
    for chunk_key, processed in existing_chunks_set.items():
        if processed:
            continue
        
        delete_chunk_query = text("""
            DELETE FROM pipeline.chunked_document
            WHERE raw_document_hash = :raw_document_hash 
            AND corpus_name = :corpus_name 
            AND chunk_hash = :chunk_hash
        """)
        
        session.execute(
            delete_chunk_query,
            {
                "raw_document_hash": chunk_key[0],
                "corpus_name": chunk_key[1],
                "chunk_hash": chunk_key[2]
            }
        )
        chunks_deleted += 1
    
    session.commit()
    
    # Log statistics
    logging.info(f"Chunking storage complete")
    logging.info(f"  Chunks already existing: {chunks_existing}")
    
    if chunks_inserted:
        logging.info(color_logging_text(
            f"  Chunks newly inserted: {chunks_inserted}",
            color='green'
        ))
    else:
        logging.info(f"  Chunks newly inserted: {chunks_inserted}")
    
    if chunks_deleted:
        logging.info(color_logging_text(
            f"  Chunks deleted: {chunks_deleted}",
            color='red'
        ))
    else:
        logging.info(f"  Chunks deleted: {chunks_deleted}")
    
    logging.info(f"  Total chunks in database: {chunks_existing + chunks_inserted}")


def chunk_corpus(
        session: Session, 
        corpus_name: str, 
        max_tokens: int,
        source_table: str = "pipeline.used_raw_document"
    ):
    """
    Chunk documents for a specified corpus.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus to chunk
        max_tokens: Maximum number of tokens per chunk
        source_table: Table to fetch documents from
    """
    logging.info(f"Starting chunking for corpus: {corpus_name}")
    
    # Initialize chunker
    chunker = DocumentChunker(max_tokens=max_tokens)
    
    # Fetch documents from the corpus
    fetch_docs_query = text(f"""
        SELECT document_hash, content 
        FROM {source_table}
        WHERE corpus_name = :corpus_name
    """)
    
    result = session.execute(fetch_docs_query, {"corpus_name": corpus_name})
    docs = [(row[0], row[1]) for row in result]
    
    if not docs:
        logging.warning(f"No documents found for corpus: {corpus_name}")
        return
    
    doc_count = len(docs)
    logging.info(f"Found {doc_count} documents for corpus: {corpus_name}")
    
    # Chunk each document
    all_chunks = []
    for doc_hash, content in docs:
        chunks = chunker.chunk_first_n_tokens(doc_hash, corpus_name, content)
        all_chunks.extend(chunks)
    
    # Store the chunked documents
    store_chunked_documents(session, all_chunks, corpus_name)
    
    # Print statistics
    logging.info(f"Chunking complete for corpus: {corpus_name}")
    logging.info(f"Documents processed: {doc_count}")


def chunk_newsgroups(session: Session, max_tokens: int):
    """Chunk 20 Newsgroups corpus."""
    chunk_corpus(session, "newsgroups", max_tokens)


def chunk_wikipedia(session: Session, max_tokens: int):
    """Chunk Wikipedia corpus."""
    chunk_corpus(session, "wikipedia_sample", max_tokens)


def chunk_imdb(session: Session, max_tokens: int):
    """Chunk IMDB reviews corpus."""
    chunk_corpus(session, "imdb_reviews", max_tokens)


def chunk_trec(session: Session, max_tokens: int):
    """Chunk TREC questions corpus."""
    chunk_corpus(session, "trec_questions", max_tokens)


def chunk_twitter_financial(session: Session, max_tokens: int):
    """Chunk Twitter financial news corpus."""
    chunk_corpus(session, "twitter-financial-news", max_tokens)


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database
    
    # Create database session
    with get_session(db_config) as session:
        # Define corpus-specific chunking parameters
        max_tokens = 8190
        
        # Run chunking for each corpus
        logging.info("Starting chunking pipelines...")
        
        logging.info("Chunking newsgroups corpus...")
        chunk_newsgroups(session, max_tokens=max_tokens)
        
        logging.info("Chunking Wikipedia corpus...")
        chunk_wikipedia(session, max_tokens=max_tokens)
        
        logging.info("Chunking IMDB reviews corpus...")
        chunk_imdb(session, max_tokens=max_tokens)
        
        logging.info("Chunking TREC questions corpus...")
        chunk_trec(session, max_tokens=max_tokens)
        
        logging.info("Chunking Twitter financial news corpus...")
        chunk_twitter_financial(session, max_tokens=max_tokens)
        
        logging.info("All chunking pipelines completed successfully.")
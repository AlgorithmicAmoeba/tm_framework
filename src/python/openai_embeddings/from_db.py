"""
Functions for working with database documents and embeddings.
"""
import hashlib
import logging
import pathlib
from typing import Optional, Any, Callable, TypeVar
from collections.abc import Sequence
import math

import tiktoken
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import text
from tqdm import tqdm

from models import Document, Embedder, Embedding
from openai_embeddings.cache import Chunk, hash_text, check_cache, add_to_cache
from openai_embeddings.batch import create_batch_files

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variables for generic function types
D = TypeVar('D', bound=Document)


def get_embedder(session, embedder_name: str = "openai_small") -> Embedder:
    """
    Get an embedder by name.
    
    Args:
        session: SQLAlchemy session
        embedder_name: Name of the embedder
        
    Returns:
        Embedder object
    """
    embedder = session.query(Embedder).filter_by(name=embedder_name).first()
    if not embedder:
        raise ValueError(f"Embedder '{embedder_name}' not found")
    
    return embedder


def first_n_tokens_chunker(
    document: Document, 
    max_tokens: int,
) -> list[Chunk]:
    """
    Create a single chunk from the first N tokens of a document.
    
    Args:
        document: Document object to chunk
        max_tokens: Maximum number of tokens in the chunk
        
    Returns:
        List containing a single Chunk object
    """
    # Initialize tokenizer (OpenAI's cl100k_base used by default)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Tokenize the document content
    tokens = tokenizer.encode(document.content)
    
    # Take only the first max_tokens
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    # Decode back to text
    chunk_text = tokenizer.decode(tokens)
    
    # Create a hash for the document and chunk
    document_hash = hash_text(document.content)
    chunk_hash = hash_text(chunk_text)
    
    # Create a chunk object
    chunk = Chunk(
        document_hash=document_hash,
        chunk_hash=chunk_hash,
        chunk_start_index=0,
        chunk_end_index=len(chunk_text),
        corpus_name=document.corpus.name if document.corpus else "",
        text=chunk_text
    )
    
    return [chunk]


def sliding_window_chunker(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """
    Split a document into overlapping chunks using a sliding window approach.
    
    Args:
        document: Document object to chunk
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of Chunk objects
    """
    # Initialize tokenizer (OpenAI's cl100k_base used by default)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Tokenize the document content
    tokens = tokenizer.encode(document.content)
    
    # Create document hash
    document_hash = hash_text(document.content)
    
    # Calculate the stride (non-overlapping tokens)
    stride = max(1, chunk_size - chunk_overlap)
    
    # Generate chunks
    chunks = []
    for i in range(0, len(tokens), stride):
        # Get chunk tokens
        chunk_tokens = tokens[i:i + chunk_size]
        if not chunk_tokens:
            continue
            
        # Convert back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        
        # Calculate character positions in the original document
        full_text = document.content
        start_text = tokenizer.decode(tokens[:i])
        chunk_start_index = len(start_text)
        chunk_end_index = chunk_start_index + len(chunk_text)
        
        # Create chunk hash
        chunk_hash = hash_text(chunk_text)
        
        # Create chunk
        chunk = Chunk(
            document_hash=document_hash,
            chunk_hash=chunk_hash,
            chunk_start_index=chunk_start_index,
            chunk_end_index=chunk_end_index,
            corpus_name=document.corpus.name if document.corpus else "",
            text=chunk_text
        )
        chunks.append(chunk)
        
        # If we've reached the end of the document, stop
        if i + chunk_size >= len(tokens):
            break
    
    return chunks


def embed_documents(
    session,
    documents: Sequence[Document],
    chunker: Callable[[Document], list[Chunk]],
    cache_db_path: pathlib.Path,
    batch_dir: pathlib.Path,
    embedder_name: str = "openai_small",
) -> dict[str, Any]:
    """
    Process document embeddings:
    1. Check cache for existing embeddings
    2. Create chunks for documents
    3. Prepare new batches for documents not in cache
    4. Store cached embeddings in the database
    
    Args:
        session: SQLAlchemy session
        documents: List of Document objects to embed
        chunker: Function that converts a document into chunks
        cache_db_path: Path to the cache database
        batch_dir: Directory for batch files
        embedder_name: Name of the embedder
        
    Returns:
        Dictionary with statistics about the processing
    """
    logger.info(f"Processing embeddings for {len(documents)} documents")
    
    # Get embedder
    embedder = get_embedder(session, embedder_name)
    
    # Process each document into chunks
    all_chunks = []
    doc_to_chunks = {}
    
    logger.info("Creating chunks from documents")
    for doc in tqdm(documents, desc="Chunking documents"):
        # Create chunks using the provided chunker function
        chunks = chunker(doc)
        all_chunks.extend(chunks)
        doc_to_chunks[doc.id] = chunks
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    
    # Check which chunks are already in the cache
    logger.info("Checking cache for existing chunks")
    is_cached, cached_embeddings = check_cache(all_chunks, cache_db_path)
    
    # Separate cached and uncached chunks
    cached_chunks = []
    uncached_chunks = []
    
    for i, chunk in enumerate(all_chunks):
        if is_cached[i]:
            cached_chunks.append((chunk, cached_embeddings[i]))
        else:
            uncached_chunks.append(chunk)
    
    logger.info(f"Found {len(cached_chunks)} chunks in cache, {len(uncached_chunks)} not in cache")
    
    # Create batches for uncached chunks
    if uncached_chunks:
        logger.info(f"Creating batches for {len(uncached_chunks)} chunks")
        create_batch_files(uncached_chunks, batch_dir)
        logger.info(f"Created batches in {batch_dir}")
    
    # Store cached embeddings in database if there are any
    documents_with_embeddings = set()
    if cached_chunks:
        logger.info(f"Storing {len(cached_chunks)} cached chunk embeddings in database")
        
        # Organize by document
        doc_chunks = {}
        for chunk, embedding in cached_chunks:
            # Extract document ID from the chunk
            # We need to find which document this chunk belongs to
            for doc_id, chunks in doc_to_chunks.items():
                for c in chunks:
                    if c.chunk_hash == chunk.chunk_hash:
                        if doc_id not in doc_chunks:
                            doc_chunks[doc_id] = []
                        doc_chunks[doc_id].append(embedding)
                        break
        
        # Create document embeddings (average of chunk embeddings)
        doc_embeddings = []
        doc_ids = []
        
        for doc_id, embeddings in doc_chunks.items():
            # If we have embeddings for this document, average them
            if embeddings:
                # Calculate average embedding
                avg_embedding = [0.0] * len(embeddings[0])
                for emb in embeddings:
                    for i in range(len(emb)):
                        avg_embedding[i] += emb[i]
                for i in range(len(avg_embedding)):
                    avg_embedding[i] /= len(embeddings)
                
                doc_embeddings.append(avg_embedding)
                doc_ids.append(doc_id)
                documents_with_embeddings.add(doc_id)
        
        # Prepare data for insertion
        if doc_embeddings:
            rows = []
            for doc_id, embedding in zip(doc_ids, doc_embeddings):
                rows.append({
                    'document_id': doc_id,
                    'embedder_id': embedder.id,
                    'vector': embedding
                })
            
            # Insert into database with conflict handling
            stmt = insert(Embedding).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["document_id", "embedder_id"],
                set_={"vector": text("excluded.vector")}
            )
            session.execute(stmt)
            session.commit()
            
            logger.info(f"Stored embeddings for {len(rows)} documents in database")
    
    # Return statistics
    return {
        "total_documents": len(documents),
        "total_chunks": len(all_chunks),
        "cached_chunks": len(cached_chunks),
        "new_chunks": len(uncached_chunks),
        "documents_with_embeddings": len(documents_with_embeddings),
        "documents_needing_processing": len(documents) - len(documents_with_embeddings),
        "avg_chunks_per_document": len(all_chunks) / max(1, len(documents))
    }


if __name__ == "__main__":
    # Example usage
    import pathlib
    import configuration as cfg
    import database
    
    config = cfg.load_config_from_env()
    
    with database.get_session(config.database) as session:
        from sqlalchemy import select
        from models import Document
        
        # Get documents
        documents = session.execute(select(Document).limit(10)).scalars().all()
        
        # Process embeddings
        stats = embed_documents(
            session,
            documents,
            sliding_window_chunker,
            cache_db_path=pathlib.Path("ignore/embedding_cache.db"),
            batch_dir=pathlib.Path("ignore/embedding_batches"),
            embedder_name="openai_small"
        )
        
        print(stats)
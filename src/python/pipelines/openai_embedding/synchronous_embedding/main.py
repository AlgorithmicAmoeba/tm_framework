"""
Synchronous embedding pipeline for document chunks.
This pipeline processes chunks from the chunking pipeline, checks a DuckDB cache,
and either retrieves cached embeddings or generates new ones using OpenAI's API.
"""
import logging
import pathlib
import dataclasses
import asyncio
from typing import Any, Optional, List

import openai
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg
from shared_code import color_logging_text
from openai_embeddings.cache import ensure_cache_db, check_cache, add_to_cache, Chunk


@dataclasses.dataclass
class EmbeddedChunk:
    """Represents a document chunk with its embedding."""
    raw_document_hash: str
    corpus_name: str
    chunk_hash: str
    embedding: List[float]
    model: str
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for database operations.
        
        Returns:
            Dictionary representation of the embedded chunk
        """
        return {
            "raw_document_hash": self.raw_document_hash,
            "corpus_name": self.corpus_name,
            "chunk_hash": self.chunk_hash,
            "embedding": self.embedding,
            "model": self.model
        }


async def generate_embedding(client: openai.Client, text: str, model: str) -> List[float]:
    """
    Generate embedding for a single text using OpenAI API.
    
    Args:
        client: OpenAI API client
        text: Text to embed
        model: Model to use for embedding
        
    Returns:
        Embedding vector as a list of floats
    """
    try:
        response = await client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        raise


async def process_chunks(
    client: openai.Client,
    document_chunks: list[dict],
    cache_path: pathlib.Path,
    model: str = "text-embedding-3-small"
) -> list[EmbeddedChunk]:
    """
    Process document chunks and return embeddings.
    Checks cache first and only calls OpenAI API for chunks not in cache.
    
    Args:
        client: OpenAI API client
        document_chunks: List of document chunk dictionaries with required fields
        cache_path: Path to the cache database
        model: OpenAI embedding model name
            
    Returns:
        List of EmbeddedChunk objects with embeddings
    """
    logging.info(f"Processing {len(document_chunks)} document chunks")
    
    # Ensure cache exists
    ensure_cache_db(cache_path)
    
    # Convert to Chunk objects for cache compatibility
    chunks = []
    for chunk_dict in document_chunks:
        chunk = Chunk(
            document_hash=chunk_dict["raw_document_hash"],
            chunk_hash=chunk_dict["chunk_hash"],
            chunk_start_index=chunk_dict.get("chunk_start", 0),
            chunk_end_index=chunk_dict.get("chunk_start", 0) + len(chunk_dict["content"]),
            corpus_name=chunk_dict["corpus_name"],
            text=chunk_dict["content"]
        )
        chunks.append(chunk)
    
    # Check cache
    is_cached, cached_embeddings = check_cache(chunks, cache_path)
    
    # Count cache hits and misses
    cache_hits = sum(1 for cached in is_cached if cached)
    cache_misses = len(chunks) - cache_hits
    
    logging.info(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")
    
    # Process chunks not in cache asynchronously
    new_chunks = []
    new_embeddings = []
    
    if cache_misses > 0:
        # Get chunks that need embedding
        uncached_chunks = [chunk for i, chunk in enumerate(chunks) if not is_cached[i]]
        
        logging.info(f"Generating embeddings for {len(uncached_chunks)} chunks")
        
        # Create tasks for async embedding generation
        tasks = []
        for chunk in uncached_chunks:
            task = generate_embedding(client, chunk.text, model)
            tasks.append(task)
        
        # Run tasks concurrently
        embeddings_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(embeddings_results):
            chunk = uncached_chunks[i]
            
            if isinstance(result, Exception):
                logging.error(f"Error embedding chunk {chunk.chunk_hash}: {result}")
                continue
                
            new_chunks.append(chunk)
            new_embeddings.append(result)
        
        # Add new embeddings to cache
        if new_chunks:
            logging.info(f"Adding {len(new_chunks)} new embeddings to cache")
            add_to_cache(new_chunks, new_embeddings, cache_path)
    
    # Combine cached and new embeddings
    embedded_chunks = []
    
    for i, chunk in enumerate(chunks):
        if is_cached[i]:
            # Use cached embedding
            embedding = cached_embeddings[i]
        else:
            # Find in new embeddings
            new_index = None
            for j, new_chunk in enumerate(new_chunks):
                if new_chunk.chunk_hash == chunk.chunk_hash:
                    new_index = j
                    break
                    
            if new_index is None:
                logging.warning(f"Could not find embedding for chunk {chunk.chunk_hash}")
                continue
                
            embedding = new_embeddings[new_index]
        
        # Create EmbeddedChunk
        embedded_chunk = EmbeddedChunk(
            raw_document_hash=chunk.document_hash,
            corpus_name=chunk.corpus_name,
            chunk_hash=chunk.chunk_hash,
            embedding=embedding,
            model=model
        )
        
        embedded_chunks.append(embedded_chunk)
    
    return embedded_chunks


def store_embedded_documents(
    session: Session,
    embedded_chunks: list[EmbeddedChunk],
    corpus_name: str
):
    """
    Store embedded document chunks in the database.
    Uses upsert pattern for idempotency.
    
    Args:
        session: Database session
        embedded_chunks: List of EmbeddedChunk objects
        corpus_name: Name of the corpus
    """
    logging.info(f"Storing {len(embedded_chunks)} embedded documents")
    
    # Get existing embeddings to track what's already processed
    existing_embeddings = session.execute(
        text("""
            SELECT raw_document_hash, corpus_name, chunk_hash 
            FROM pipeline.embedded_document
            WHERE corpus_name = :corpus_name
        """),
        {"corpus_name": corpus_name}
    ).fetchall()
    
    # Create a set of tuples for fast lookup
    existing_embeddings_set = {
        (row[0], row[1], row[2]): False 
        for row in existing_embeddings
    }
    
    # Track statistics
    embeddings_existing = 0
    embeddings_inserted = 0
    
    for embed in embedded_chunks:
        # Create lookup key
        key = (embed.raw_document_hash, embed.corpus_name, embed.chunk_hash)
        
        if key in existing_embeddings_set:
            existing_embeddings_set[key] = True
            embeddings_existing += 1
            continue
        
        # Insert new embedding
        insert_embedding_query = text("""
            INSERT INTO pipeline.embedded_document 
            (raw_document_hash, corpus_name, chunk_hash, embedding, model)
            VALUES (:raw_document_hash, :corpus_name, :chunk_hash, :embedding, :model)
            ON CONFLICT (corpus_name, raw_document_hash, chunk_hash) 
            DO UPDATE SET 
                embedding = EXCLUDED.embedding,
                model = EXCLUDED.model
        """)
        
        # Convert the embedding to a dictionary for the query parameters
        session.execute(insert_embedding_query, embed.to_dict())
        embeddings_inserted += 1
    
    # Commit changes
    session.commit()
    
    # Log statistics
    logging.info(f"Embedding storage complete")
    logging.info(f"  Embeddings already existing: {embeddings_existing}")
    
    if embeddings_inserted:
        logging.info(color_logging_text(
            f"  Embeddings newly inserted: {embeddings_inserted}",
            color='green'
        ))
    else:
        logging.info(f"  Embeddings newly inserted: {embeddings_inserted}")
    
    logging.info(f"  Total embeddings in database: {embeddings_existing + embeddings_inserted}")


async def embed_corpus(
        session: Session, 
        client: openai.Client,
        corpus_name: str,
        cache_path: pathlib.Path,
        model: str = "text-embedding-3-small",
        chunk_source_table: str = "pipeline.chunked_document"
    ):
    """
    Embed document chunks for a specified corpus.
    
    Args:
        session: Database session
        client: OpenAI API client
        corpus_name: Name of the corpus to process
        cache_path: Path to DuckDB cache file
        model: OpenAI embedding model to use
        chunk_source_table: Table to fetch chunks from
        
    Returns:
        Number of chunks processed
    """
    logging.info(f"Starting embedding for corpus: {corpus_name}")
    
    # Fetch chunks from the corpus
    fetch_chunks_query = text(f"""
        SELECT raw_document_hash, corpus_name, chunk_hash, content, chunk_start, token_count
        FROM {chunk_source_table}
        WHERE corpus_name = :corpus_name
    """)
    
    result = session.execute(fetch_chunks_query, {"corpus_name": corpus_name})
    
    # Convert to list of dictionaries
    chunks = []
    for row in result:
        chunk = {
            "raw_document_hash": row[0],
            "corpus_name": row[1],
            "chunk_hash": row[2],
            "content": row[3],
            "chunk_start": row[4],
            "token_count": row[5]
        }
        chunks.append(chunk)
    
    if not chunks:
        logging.warning(f"No chunks found for corpus: {corpus_name}")
        return 0
    
    chunk_count = len(chunks)
    logging.info(f"Found {chunk_count} chunks for corpus: {corpus_name}")
    
    # Process chunks and get embeddings
    embedded_chunks = await process_chunks(
        client=client,
        document_chunks=chunks,
        cache_path=cache_path,
        model=model
    )
    
    # Store the embedded chunks
    store_embedded_documents(session, embedded_chunks, corpus_name)
    
    # Print statistics
    logging.info(f"Embedding complete for corpus: {corpus_name}")
    logging.info(f"Chunks processed: {chunk_count}")
    logging.info(f"Embeddings created: {len(embedded_chunks)}")
    
    return chunk_count


async def embed_newsgroups(session: Session, client: openai.Client, cache_path: pathlib.Path):
    """Embed 20 Newsgroups corpus."""
    return await embed_corpus(session, client, "newsgroups", cache_path)


async def embed_wikipedia(session: Session, client: openai.Client, cache_path: pathlib.Path):
    """Embed Wikipedia corpus."""
    return await embed_corpus(session, client, "wikipedia_sample", cache_path)


async def embed_imdb(session: Session, client: openai.Client, cache_path: pathlib.Path):
    """Embed IMDB reviews corpus."""
    return await embed_corpus(session, client, "imdb_reviews", cache_path)


async def embed_trec(session: Session, client: openai.Client, cache_path: pathlib.Path):
    """Embed TREC questions corpus."""
    return await embed_corpus(session, client, "trec_questions", cache_path)


async def embed_twitter_financial(session: Session, client: openai.Client, cache_path: pathlib.Path):
    """Embed Twitter financial news corpus."""
    return await embed_corpus(session, client, "twitter-financial-news", cache_path)


async def main():
    """Main entry point for embedding pipelines."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database
    openai_config = config.openai
    
    # Initialize OpenAI client
    client = openai.Client(api_key=openai_config.api_key)
    
    # Define cache path
    cache_path = pathlib.Path(config.get_data_path() / "cache" / "embedding_cache.db")
    
    # Create database session
    with get_session(db_config) as session:
        # Run embedding for each corpus
        logging.info("Starting embedding pipelines...")
        
        logging.info("Embedding newsgroups corpus...")
        await embed_newsgroups(session, client, cache_path)
        
        logging.info("Embedding Wikipedia corpus...")
        await embed_wikipedia(session, client, cache_path)
        
        logging.info("Embedding IMDB reviews corpus...")
        await embed_imdb(session, client, cache_path)
        
        logging.info("Embedding TREC questions corpus...")
        await embed_trec(session, client, cache_path)
        
        logging.info("Embedding Twitter financial news corpus...")
        await embed_twitter_financial(session, client, cache_path)
        
        logging.info("All embedding pipelines completed successfully.")


if __name__ == '__main__':
    asyncio.run(main())
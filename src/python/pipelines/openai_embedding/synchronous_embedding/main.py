"""
Synchronous embedding pipeline for document chunks.
This pipeline processes jobs from the embedding_job table and stores results in the cache.
"""
import logging
import pathlib
import asyncio
from typing import List

import openai
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg
from shared_code import color_logging_text
from openai_embeddings.cache import ensure_cache_db, add_to_cache, Chunk


def generate_embedding(client: openai.Client, text: str, model: str) -> List[float]:
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
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        raise


async def process_job(
    client: openai.Client,
    job: dict,
    cache_path: pathlib.Path,
    model: str,
    session: Session
) -> bool:
    """
    Process a single embedding job.
    
    Args:
        client: OpenAI API client
        job: Job dictionary with required fields
        cache_path: Path to the cache database
        model: Default model to use if not specified in job
        session: Database session
        
    Returns:
        True if job was processed successfully, False otherwise
    """
    try:
        # Generate embedding
        embedding = generate_embedding(
            client=client,
            text=job["chunk_content"],
            model=job["model_name"] or model
        )
        
        # Create chunk object for cache
        chunk = Chunk(
            document_hash=job["chunk_hash"],
            embedding=embedding
        )
        
        # Add to cache
        add_to_cache([chunk], [embedding], cache_path)
        
        # Delete the processed job
        delete_query = text("""
            DELETE FROM pipeline.embedding_job
            WHERE id = :job_id
        """)
        session.execute(delete_query, {"job_id": job["id"]})
        session.commit()
        
        logging.info(f"Processed job {job['id']}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing job {job['id']}: {e}")
        logging.error("Full stack trace:", exc_info=True)
        return False


async def process_jobs(
    session: Session,
    client: openai.Client,
    cache_path: pathlib.Path,
    model: str = "text-embedding-3-small"
) -> int:
    """
    Process jobs from the embedding_job table and store results in cache.
    
    Args:
        session: Database session
        client: OpenAI API client
        cache_path: Path to the cache database
        model: OpenAI embedding model name
            
    Returns:
        Number of jobs processed
    """
    logging.info("Starting to process embedding jobs")
    
    # Ensure cache exists
    ensure_cache_db(cache_path)
    
    # Get jobs that haven't been processed yet
    jobs_query = text("""
        SELECT id, chunk_hash, chunk_content, model_name
        FROM pipeline.embedding_job
    """)
    
    jobs = session.execute(jobs_query).mappings().fetchall()
    
    if not jobs:
        logging.info("No jobs found")
        return 0
    
    logging.info(f"Found {len(jobs)} jobs to process")
    
    # Create tasks for all jobs
    tasks = [
        process_job(client, job, cache_path, model, session)
        for job in jobs
    ]
    
    # Process all jobs concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count successes and failures
    processed_count = sum(1 for r in results if r is True)
    error_count = len(jobs) - processed_count
    
    # Log statistics
    logging.info(f"Job processing complete")
    logging.info(f"  Jobs processed successfully: {processed_count}")
    
    if error_count:
        logging.warning(f"  Jobs failed: {error_count}")
    
    return processed_count


async def main():
    """Main entry point for embedding pipeline."""
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
    cache_path = pathlib.Path(config.get_data_path() / "embeddings" / "cache.db")
    
    # Create database session
    with get_session(db_config) as session:
        # Process embedding jobs
        logging.info("Starting embedding pipeline...")
        
        processed_count = await process_jobs(
            session=session,
            client=client,
            cache_path=cache_path
        )
        
        if processed_count > 0:
            logging.info(color_logging_text(
                f"Successfully processed {processed_count} embedding jobs",
                color='green'
            ))
        else:
            logging.info("No jobs needed to be processed")
        
        logging.info("Embedding pipeline completed successfully.")


if __name__ == '__main__':
    asyncio.run(main())
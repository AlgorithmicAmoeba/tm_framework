"""
Main script for batch processing OpenAI embeddings from database jobs.
"""
import asyncio
import logging
import pathlib
from typing import List, Tuple

import openai
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg
from pipelines.openai_embedding.chunk import Chunk
from pipelines.openai_embedding.batch_runner.batch import (
    create_batch_files,
    submit_batches,
    check_batch_status,
    download_results,
    process_results
)
from shared_code import color_logging_text


async def get_pending_jobs(session: Session) -> List[dict]:
    """
    Get pending embedding jobs from the database.
    
    Args:
        session: Database session
        
    Returns:
        List of job dictionaries
    """
    jobs_query = text("""
        SELECT id, chunk_hash, chunk_content, model_name
        FROM pipeline.embedding_job
    """)
    
    jobs = session.execute(jobs_query).mappings().fetchall()
    return [dict(job) for job in jobs]


def create_chunks_from_jobs(jobs: List[dict]) -> Tuple[List[Chunk], List[str]]:
    """
    Create Chunk objects and texts from jobs.
    
    Args:
        jobs: List of job dictionaries
        
    Returns:
        Tuple of (chunks, texts)
    """
    chunks = []
    texts = []
    
    for job in jobs:
        chunk = Chunk(
            document_hash=job["chunk_hash"]
        )
        chunks.append(chunk)
        texts.append(job["chunk_content"])
    
    return chunks, texts


async def main():
    """Main entry point for batch embedding pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    log = logging.getLogger("batch_embedding")
    
    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database
    openai_config = config.openai
    
    # Initialize OpenAI client
    client = openai.Client(api_key=openai_config.api_key)
    
    # Define paths
    batch_dir = pathlib.Path(config.get_data_path() / "embeddings" / "batches_02")
    cache_path = pathlib.Path(config.get_data_path() / "embeddings" / "cache.db")
    
    # Create database session
    with get_session(db_config) as session:
        # Get pending jobs
        jobs = await get_pending_jobs(session)
        
        if not jobs:
            log.info("No pending jobs found")
            return
        
        log.info(f"Found {len(jobs)} pending jobs")
        
        # Create chunks and texts
        chunks, texts = create_chunks_from_jobs(jobs)
        
        # Create batch files
        log.info("Creating batch files...")
        create_batch_files(
            chunks=chunks,
            batch_dir=batch_dir,
            texts=texts
        )
        
        # Submit batches
        log.info("Submitting batches to OpenAI...")
        submit_batches(client, batch_dir)
        
        # Check batch status until complete
        while True:
            status_counts = check_batch_status(client, batch_dir)
            log.info(f"Batch status: {status_counts}")
            
            statuses = ["pending", "processing", "in_progress", "validating"]
            if all(status_counts.get(status, 0) == 0 for status in statuses):
                break
                
            await asyncio.sleep(300)  # Wait 5 minute before checking again
        
        # Download results
        log.info("Downloading batch results...")
        download_results(client, batch_dir)
        
        # Process results and store in cache
        log.info("Processing results and storing in cache...")
        process_results(batch_dir, cache_path)
        
        # Delete processed jobs
        delete_query = text("""
            DELETE FROM pipeline.embedding_job
            WHERE id = ANY(:job_ids)
        """)
        session.execute(delete_query, {"job_ids": [job["id"] for job in jobs]})
        session.commit()
        
        log.info(color_logging_text(
            f"Successfully processed {len(jobs)} embedding jobs",
            color='green'
        ))


if __name__ == '__main__':
    asyncio.run(main())

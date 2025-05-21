"""
Enhanced batch processing for OpenAI embeddings with document chunking support.
"""
import dataclasses
import gzip
import json
import pathlib
import logging
from typing import Optional

import openai
from tqdm import tqdm

from openai_embeddings.cache import add_to_cache, Chunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("openai_embeddings.batch")

REQUEST_DIR = pathlib.Path("requests")
RESULTS_DIR = pathlib.Path("results")
ERRORS_DIR = pathlib.Path("errors")
CHUNK_MAP_DIR = pathlib.Path("chunk_maps")
INFO_FILE = "batch_info.json"
    

@dataclasses.dataclass
class BatchInfo:
    """Information about a batch submission."""
    batch_id: str
    input_file: str
    num_requests: int
    status: str
    output_file: Optional[str] = None
    error_file: Optional[str] = None
    chunk_map_file: Optional[str] = None


def save_batch_info(
    batch_dir: pathlib.Path,
    batch_info: list[BatchInfo],
):
    """
    Save batch information to a file.
    
    Args:
        batch_dir: Directory to save batch files
        batch_info: List of BatchInfo objects
    """
    info_file = pathlib.Path(batch_dir / INFO_FILE)
    
    with open(info_file, 'w') as f:
        json.dump([dataclasses.asdict(batch) for batch in batch_info], f, indent=2)


def load_batch_info(
    batch_dir: pathlib.Path,
) -> list[BatchInfo]:
    """
    Load batch information from a file.
    
    Args:
        batch_dir: Directory with batch files
        
    Returns:
        List of BatchInfo objects
    """
    info_file = pathlib.Path(batch_dir / INFO_FILE)
    
    with open(info_file, 'r') as f:
        batch_data = json.load(f)
    
    return [BatchInfo(**data) for data in batch_data]


def create_batch_files(
    chunks: list[Chunk],
    batch_dir: pathlib.Path,
    batch_size: int = 49_990,
    texts: Optional[list[str]] = None,
):
    """
    Create batch files for OpenAI embedding requests from document chunks.
    Generates a chunk mapping file to track chunk information.
    
    Args:
        chunks: List of Chunk objects to embed
        batch_dir: Directory to save batch files
        batch_size: Maximum number of texts per batch
        texts: Optional list of texts to embed, must match length of chunks
    """
    if texts is None:
        raise ValueError("texts parameter is required for creating batch files")
    
    if len(chunks) != len(texts):
        raise ValueError("Number of chunks and texts must match")
        
    output_dir = pathlib.Path(batch_dir / REQUEST_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_map_dir = pathlib.Path(batch_dir / CHUNK_MAP_DIR)
    chunk_map_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into batches
    batch_chunks = [
        chunks[i:i+batch_size]
        for i in range(0, len(chunks), batch_size)
    ]
    batch_texts = [
        texts[i:i+batch_size]
        for i in range(0, len(texts), batch_size)
    ]
    
    # Write each batch to a file
    for i, (batch_items, batch_text_items) in enumerate(zip(batch_chunks, batch_texts)):
        batch_file = output_dir / f"batch_{i}.jsonl"
        chunk_map_file = chunk_map_dir / f"chunk_map_{i}.jsonl"
        
        with open(batch_file, 'w') as f, open(chunk_map_file, 'w') as map_f:
            for j, (chunk, text) in enumerate(zip(batch_items, batch_text_items)):
                # Create a unique custom ID
                custom_id = f"chunk-{i}-{j}"
                
                # Save the mapping between custom ID and chunk
                chunk_map_entry = {
                    "custom_id": custom_id,
                    "chunk": chunk.to_dict()
                }
                map_f.write(f"{json.dumps(chunk_map_entry)}\n")
                
                # Create the batch request
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "text-embedding-3-small",
                        "input": text
                    }
                }
                f.write(f"{json.dumps(request)}\n")


def submit_batches(
    client: openai.Client,
    batch_dir: pathlib.Path,
):
    """
    Submit batch files to OpenAI.
    
    Args:
        client: OpenAI client
        batch_dir: Directory containing batch files
    """

    input_dir = pathlib.Path(batch_dir / REQUEST_DIR)
    chunk_map_dir = pathlib.Path(batch_dir / CHUNK_MAP_DIR)
    
    # Find all batch files recursively
    batch_files = sorted(input_dir.rglob("*.jsonl"))
    if not batch_files:
        log.warning(f"No batch files found in {input_dir}")
        return []
    
    # Submit each batch
    batches = []
    for input_file in tqdm(batch_files, desc="Submitting batch files"):
        # Find corresponding chunk map file
        batch_number = input_file.stem.split('_')[-1]
        chunk_map_file = chunk_map_dir / f"chunk_map_{batch_number}.jsonl"
        
        if not chunk_map_file.exists():
            log.warning(f"No chunk map file found for {input_file}")
            continue
        
        batch_file = client.files.create(
            file=open(input_file, "rb"),
            purpose="batch"
        )

        batch_object = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"source_file": str(input_file)}
        )

        # Count number of requests in the batch file
        with open(input_file, 'r') as f:
            num_requests = sum(1 for _ in f)

        relative_input_file = input_file.relative_to(input_dir)
        relative_chunk_map_file = chunk_map_file.relative_to(chunk_map_dir)
        
        batch_info = BatchInfo(
            batch_id=batch_object.id,
            input_file=str(relative_input_file),
            num_requests=num_requests,
            status="submitted",
            chunk_map_file=str(relative_chunk_map_file)
        )
        batches.append(batch_info)
    
    # Save batch information
    save_batch_info(batch_dir, batches)


def check_batch_status(
    client: openai.Client,
    batch_dir: pathlib.Path,
) -> dict[str, int]:
    """
    Check status of submitted batches.
    
    Args:
        client: OpenAI client
        batch_dir: File with batch information
        
    Returns:
        Dictionary with status counts
    """
    
    # Load batch information
    batches = load_batch_info(batch_dir)
    
    # Check status of each batch
    status_counts = {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
    for batch in tqdm(batches, desc="Checking batch status"):
        try:
            batch_status = client.batches.retrieve(batch.batch_id)
            status = batch_status.status
            batch.status = status
            status_counts[status] = status_counts.get(status, 0) + 1
        except Exception as e:
            log.error(f"Error checking batch {batch.batch_id}: {e}")
    
    # Save updated batch information
    save_batch_info(batch_dir, batches)
    
    return status_counts


def download_results(
    client: openai.Client,
    batch_dir: pathlib.Path,
):
    """
    Download batch results from OpenAI.
    
    Args:
        client: OpenAI client
        batch_dir: Directory with batch information
    """
    output_dir = pathlib.Path(batch_dir / RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_dir = pathlib.Path(batch_dir / ERRORS_DIR)
    error_dir.mkdir(parents=True, exist_ok=True)
    
    # Load batch information
    batches = load_batch_info(batch_dir)
    
    # Download results for each completed batch
    for batch in tqdm(batches, desc="Downloading batch results"):
        try:     
            # Generate output file path
            relative_input_dir = pathlib.Path(batch.input_file).parent
            output_file = output_dir / relative_input_dir / f"result_{batch.batch_id}.jsonl.gz"

            if output_file.exists():
                log.info(f"Output file {output_file} already exists, skipping download")
                continue

            openai_batch = client.batches.retrieve(batch.batch_id)
            batch.status = openai_batch.status

            if openai_batch.status != "completed":
                log.info(f"Batch {batch.batch_id} is not completed, skipping download")
                save_batch_info(batch_dir, batches)
                continue
            
            # Download the file content
            file_response = client.files.content(openai_batch.output_file_id)

            # Write content to file using gzip
            with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                f.write(file_response.text)
                            
            # Download error file if it exists
            if openai_batch.error_file_id is not None:
                error_file = error_dir / relative_input_dir / f"error_{batch.batch_id}.jsonl.gz"
                error_response = client.files.content(openai_batch.error_file_id)
                with gzip.open(error_file, 'wt', encoding='utf-8') as f:
                    f.write(error_response.text)

            # Update batch information
            relative_output_file = output_file.relative_to(output_dir)
            relative_error_file = error_file.relative_to(error_dir) if openai_batch.error_file_id else None
            batch.output_file = str(relative_output_file)
            batch.error_file = str(relative_error_file) if relative_error_file else None

            save_batch_info(batch_dir, batches)
        except Exception as e:
            log.error(f"Error downloading batch {batch.batch_id}: {e}")


def process_results(
    batch_dir: pathlib.Path,
    cache_db_path: pathlib.Path,
):
    """
    Process downloaded results and extract embeddings.
    Maps results back to original document chunks using the chunk mapping files.
    
    Args:
        batch_dir: Directory containing batch files, including requests and results
        cache_db_path: Path to the cache database
    """
    batch_dir = pathlib.Path(batch_dir)
    batches = load_batch_info(batch_dir)
    log.info(f"Processing results from {len(batches)} batches")

    for batch in tqdm(batches, desc="Processing batches"):
        if batch.output_file is None or batch.chunk_map_file is None:
            log.warning(f"Skipping batch {batch.batch_id} with missing output or chunk map files")
            continue
        
        # Load results file
        output_file = batch_dir / RESULTS_DIR / batch.output_file
        if not output_file.exists():
            log.warning(f"Output file {output_file} does not exist")
            continue
        
        # Load chunk mapping file
        chunk_map_file = batch_dir / CHUNK_MAP_DIR / batch.chunk_map_file
        if not chunk_map_file.exists():
            log.warning(f"Chunk map file {chunk_map_file} does not exist")
            continue
            
        log.info(f"Processing batch {batch.batch_id}: {output_file}")
        
        # Load chunk map data from file
        chunk_map = {}
        with open(chunk_map_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                custom_id = entry["custom_id"]
                chunk = Chunk.from_dict(entry["chunk"])
                chunk_map[custom_id] = chunk
        
        # Process result file
        processed_chunks = []
        processed_embeddings = []
        processed_count = 0
        skipped_count = 0
        
        with gzip.open(output_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing embeddings", leave=False):
                result = json.loads(line)
                
                # Skip failed requests
                if result['response']['status_code'] != 200:
                    skipped_count += 1
                    continue
                
                # Get the custom ID and embedding
                custom_id = result['custom_id']
                embedding = result['response']['body']['data'][0]["embedding"]
                
                # Get the chunk details
                if custom_id not in chunk_map:
                    log.warning(f"Could not find chunk for custom_id {custom_id}")
                    continue

                chunk = chunk_map[custom_id]
                chunk.embedding = embedding
                processed_chunks.append(chunk)
                processed_embeddings.append(embedding)
                processed_count += 1

        log.info(f"Processed {processed_count} embeddings, skipped {skipped_count} failed requests")
        
        # Add embeddings to cache
        if processed_chunks and processed_embeddings:
            log.info(f"Adding {len(processed_chunks)} chunk embeddings to cache")
            
            # Pass chunks and embeddings to add_to_cache
            add_to_cache(processed_chunks, processed_embeddings, cache_db_path)
        else:
            log.warning(f"No valid embeddings found in batch {batch.batch_id}")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process OpenAI embedding batches with chunk tracking")
    parser.add_argument("--batch-dir", type=str, required=True, help="Directory for batch files")
    parser.add_argument("--cache-db", type=str, required=True, help="Path to cache database")
    parser.add_argument("--process", action="store_true", help="Process results")
    
    args = parser.parse_args()
    
    if args.process:
        process_results(
            batch_dir=pathlib.Path(args.batch_dir),
            cache_db_path=pathlib.Path(args.cache_db)
        )
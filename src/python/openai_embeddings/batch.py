"""
Simplified batch processing for OpenAI embeddings.
"""
import dataclasses
import gzip
import json
import pathlib
import hashlib
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import duckdb
import openai
from tqdm import tqdm

from openai_embeddings.cache import add_to_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("openai_embeddings.batch")

REQUEST_DIR = pathlib.Path("requests")
RESULTS_DIR = pathlib.Path("results")
ERRORS_DIR = pathlib.Path("errors")
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

def save_batch_info(
    batch_dir: pathlib.Path,
    batch_info: List[BatchInfo],
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
) -> List[BatchInfo]:
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
    texts: List[str],
    batch_dir: pathlib.Path,
    batch_size: int = 49_990,
):
    """
    Create batch files for OpenAI embedding requests.
    
    Args:
        texts: List of text strings to embed
        batch_dir: Directory to save batch files
        batch_size: Maximum number of texts per batch
    """
    output_dir = pathlib.Path(batch_dir / REQUEST_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create batch files for non-cached texts
    batch_files = []

    # Split into batches
    batches = [
        texts[i:i+batch_size]
        for i in range(0, len(texts), batch_size)
    ]
    
    # Write each batch to a file
    for i, batch_texts in enumerate(batches):
        batch_file = output_dir / f"batch_{i}.jsonl"
        batch_files.append(batch_file)
        
        with open(batch_file, 'w') as f:
            for j, text in enumerate(batch_texts):
                request = {
                    "custom_id": f"text-{i}-{j}",
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
    
    # Find all batch files recursively
    batch_files = sorted(input_dir.rglob("*.jsonl"))
    if not batch_files:
        log.warning(f"No batch files found in {input_dir}")
        return
    
    # Submit each batch
    batches = []
    for input_file in tqdm(batch_files, desc="Submitting batch files"):
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
        batch_info = BatchInfo(
            batch_id=batch_object.id,
            input_file=str(relative_input_file),
            num_requests=num_requests,
            status="submitted"
        )
        batches.append(batch_info)
    
    # Save batch information
    save_batch_info(batch_dir, batches)
    
    return batches


def check_batch_status(
    client: openai.Client,
    batch_dir: pathlib.Path,
) -> Dict[str, int]:
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
        info_file: File with batch information
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
    
    Args:
        batch_dir: Directory containing batch files, including requests and results
    """
    batch_dir = pathlib.Path(batch_dir)
    batches = load_batch_info(batch_dir)
    log.info(f"Processing results from {len(batches)} batches")

    for batch in tqdm(batches, desc="Processing batches"):
        if batch.output_file is None:
            log.warning(f"Skipping batch {batch.batch_id} with no output file")
            continue
        
        input_file = batch_dir / REQUEST_DIR / batch.input_file
        output_file = batch_dir / RESULTS_DIR / batch.output_file
        log.info(f"Processing batch {batch.batch_id}: {output_file}")

        # Load and hash the input texts
        hashed_texts = {}
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc="Hashing input texts", leave=False):
                request = json.loads(line)
                custom_id = request["custom_id"]
                text = request["body"]["input"]
                hashed_text = hashlib.md5(text.encode('utf-8')).hexdigest()
                hashed_texts[custom_id] = hashed_text
    
        # Process result file
        embeddings = {}
        processed_count = 0
        skipped_count = 0

        if not output_file.exists():
            log.warning(f"Output file {output_file} does not exist")
            continue
        
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
                
                # Get the hashed text
                if custom_id not in hashed_texts:
                    log.warning(f"Could not find text for custom_id {custom_id}")
                    continue

                hashed_text = hashed_texts[custom_id]
                embeddings[hashed_text] = embedding
                processed_count += 1

        log.info(f"Processed {processed_count} embeddings, skipped {skipped_count} failed requests")
        
        # Add embeddings to cache
        if embeddings:
            log.info(f"Adding {len(embeddings)} embeddings to cache")
            add_to_cache(list(embeddings.keys()), list(embeddings.values()), cache_db_path)
        else:
            log.warning(f"No valid embeddings found in batch {batch.batch_id}")


if __name__ == "__main__":
    process_results(
        batch_dir=pathlib.Path("ignore/batch_embeddings"),
        cache_db_path=pathlib.Path("ignore/embeddings_cache.db")
    )
import dataclasses
import gzip
import json
import pathlib

import openai
import tqdm
import tiktoken
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import Table, MetaData, text

from models import Corpus, Document, DocumentType, Embedder, Embedding


def count_tokens(strings: list[str]) -> int:
    """Count the number of tokens in a list of strings."""
    enc = tiktoken.encoding_for_model("gpt-4o")
    return sum(len(enc.encode(s)) for s in strings)


def get_corpus_documents(session, corpus: str) -> list[Document]:
    """Get all raw documents from a corpus."""
    corpus = session.query(Corpus).filter_by(name=corpus).first()
    raw_document_type_id = session.query(DocumentType).filter_by(name='raw').first().id

    documents = session.query(Document).filter_by(corpus_id=corpus.id, type_id=raw_document_type_id).all()
    return documents


def count_tokens_in_corpus(session, corpus: str) -> int:
    """Count the number of tokens in a corpus."""
    documents = get_corpus_documents(session, corpus)
    raw_texts = [doc.content for doc in documents]
    tokens = count_tokens(raw_texts)
    return tokens


def create_batch_embedding_file(
    documents: list[tuple[str, str]],
    output_path: str,
    batch_size: int = 49_990,
):
    """Create a JSONL file for batch embedding processing."""
    batches = (
        documents[i:i + batch_size]
        for i in range(0, len(documents), batch_size)
    )

    output_path = pathlib.Path(output_path)

    # Write all batches to file
    for i, batch in enumerate(batches):
        output_file = output_path / f"batch_{i}.jsonl"
        # make sure the output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for doc_id, doc_text in batch:
                request = {
                    "custom_id": doc_id,
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "text-embedding-3-small",
                        "input": doc_text
                    }
                }
                f.write(f"{json.dumps(request)}\n")


@dataclasses.dataclass
class Chunk:
    doc_id: int
    chunk_from: int
    chunk_to: int
    content: str
    num_tokens: int
    

def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    # stride_length: int,
) -> list[Chunk]:
    """Chunk documents into smaller pieces. Using tiktoken for tokenization."""
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    chunks = []
    for doc in documents:
        tokens = enc.encode(doc.content)
        
        chunk_tokens = tokens[: chunk_size]
        chunk_content = enc.decode(chunk_tokens)
        
        chunk = Chunk(
            doc_id=doc.id,
            chunk_from=0,
            chunk_to=len(chunk_tokens),
            content=chunk_content,
            num_tokens=len(chunk_tokens),
        )
        chunks.append(chunk)
    
    return chunks


def create_batch_files_across_corpora(
        session,
        output_dir: str
    ):
    """Create batch files for each corpus and return mapping of corpus name to file path."""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    corpora = session.query(Corpus).all()


    total_tokens = 0
    for corpus in tqdm.tqdm(corpora, desc="Creating batch files for corpora"):
        documents = session.query(Document).filter_by(corpus_id=corpus.id).all()

        chunks = chunk_documents(
            documents,
            chunk_size=8191, 
            # stride_length=1024,
        )

        corpus_tokens = sum(chunk.num_tokens for chunk in chunks)
        total_tokens += corpus_tokens

        print(f"Corpus {corpus.name} has {corpus_tokens} tokens")

        # Use simpler custom_id format that matches the format from observed results
        documents = [
            (f"doc-{chunk.doc_id}-chunk-{chunk.chunk_from}", chunk.content)
            for chunk in chunks
        ]
        
        create_batch_embedding_file(documents, output_path / f"{corpus.name}")

    print(f"Total tokens: {total_tokens}")
        
    

def count_tokens_across_corpora(session) -> dict:
    """Count tokens in all corpora and return a dictionary with results."""
    results = {}
    total_count = 0
    
    corpus_names = session.query(Corpus).all()
    for corpus in tqdm.tqdm(corpus_names):
        count = count_tokens_in_corpus(session, corpus.name)
        results[corpus.name] = count
        total_count += count
    
    results['total'] = total_count
    return results


@dataclasses.dataclass
class BatchInfo:
    batch_id: str
    input_file: str
    num_requests: int
    status: str = "pending"

class BatchProcessor:
    def __init__(self, client: openai.Client):
        self.client = client
        self.batches: list[BatchInfo] = []

    def submit_batch(self, input_file: str) -> BatchInfo:
        """Submit a single batch file to OpenAI."""
        batch_file = self.client.files.create(
            file=open(input_file, "rb"),
            purpose="batch"
        )

        batch_object = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"source_file": input_file}
        )

        # Count number of requests in the batch file
        with open(input_file, 'r') as f:
            num_requests = sum(1 for _ in f)

        batch_info = BatchInfo(
            batch_id=batch_object.id,
            input_file=input_file,
            num_requests=num_requests
        )
        self.batches.append(batch_info)
        return batch_info
    
    def check_batch_statuses(self, return_full=True) -> dict:
        """Check the status of all submitted batches."""
        status_summary = {"completed": 0, "processing": 0, "failed": 0}
        batch_details = []
        
        for batch in self.batches:
            try:
                batch_status = self.client.batches.retrieve(batch.batch_id)
                batch.status = batch_status.status
                status_summary[batch_status.status] = status_summary.get(batch_status.status, 0) + 1
                if return_full:
                    batch_details.append(batch_status)
            except Exception as e:
                print(f"Error checking batch {batch.batch_id}: {e}")
                
        return batch_details if return_full else status_summary
    
    def save_batch_info(self, filepath: str):
        """Save batch tracking information to a file."""
        batch_data = [vars(batch) for batch in self.batches]
        with open(filepath, 'w') as f:
            json.dump(batch_data, f, indent=2)

    def load_batch_info(self, filepath: str):
        """Load batch tracking information from a file."""
        with open(filepath, 'r') as f:
            batch_data = json.load(f)
        self.batches = [BatchInfo(**data) for data in batch_data]

    def cancel_all_batches(self):
        """Cancel all submitted batches."""
        for batch in self.batches:
            try:
                self.client.batches.cancel(batch.batch_id)
            except Exception as e:
                print(f"Error cancelling batch {batch.batch_id}: {e}")

    def download_batch_result(self, batch_info: BatchInfo, output_dir: str):
        """Download the result file for a completed batch."""
        output_dir = pathlib.Path(output_dir)

        batch_status = self.client.batches.retrieve(batch_info.batch_id)
        
        input_file = batch_info.input_file
        input_file_dir_name = pathlib.Path(input_file).parent.name

        output_file = output_dir/input_file_dir_name/f"result_{batch_info.batch_id}.jsonl.gz"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the file content
        file_response = self.client.files.content(batch_status.output_file_id)
        
        # Write content to file using gzip
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            f.write(file_response.text)
            
        if batch_status.error_file_id is not None:
            error_file = f"{output_dir}/error_{batch_info.batch_id}.jsonl.gz"
            error_response = self.client.files.content(batch_status.error_file_id)
            with gzip.open(error_file, 'wt', encoding='utf-8') as f:
                f.write(error_response.text)
    
    def submit_batches(self, input_dir: str):
        """Submit all batch files in a directory."""
        input_dir = pathlib.Path(input_dir)
        batch_files = []
        for input_file in input_dir.glob('**/*.jsonl'):
            if input_file.is_file():
                batch_files.append(input_file)
        
        for input_file in tqdm.tqdm(batch_files, desc="Submitting batch files"):
            self.submit_batch(str(input_file))

    def download_all_results(self, output_dir: str):
        """Download all completed batch results."""
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for batch in self.batches:
            self.download_batch_result(batch, output_dir)


def load_result_vectors_into_db(session, results_dir: str, error_dir: str):
    """Load result vectors into the database - optimized for performance."""
    import concurrent.futures

    results_dir = pathlib.Path(results_dir)
    result_files = list(results_dir.glob('**/*.jsonl.gz'))
    
    # Get embedder_id once outside the loop
    embedder_id = session.query(Embedder).filter_by(name='openai_small').first().id
    docs_with_errors = []
    
    # Get metadata for direct table access
    metadata = MetaData()
    embedding_table = Table('embedding', metadata, schema='topic_modelling')
    
    def process_file(file_path):
        file_errors = []
        batch_size = 1000
        embeddings_batch = []
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                doc_id = int(result['custom_id'].split('-')[1])
                
                if result['response']['status_code'] != 200:
                    file_errors.append(doc_id)
                    continue

                embedding = result['response']['body']['data'][0]["embedding"]
                embeddings_batch.append({
                    'document_id': doc_id,
                    'embedder_id': embedder_id,
                    'vector': embedding
                })
                
                # Insert in batches to reduce overhead
                if len(embeddings_batch) >= batch_size:
                    _bulk_insert_embeddings(embeddings_batch)
                    embeddings_batch = []
            
            # Insert any remaining embeddings
            if embeddings_batch:
                _bulk_insert_embeddings(embeddings_batch)
                
        return file_errors
    
    def _bulk_insert_embeddings(embeddings):
        # Using PostgreSQL's INSERT ... ON CONFLICT for bulk upsert
        stmt = insert(Embedding).values(embeddings)
        stmt = stmt.on_conflict_do_update(
            index_elements=["document_id", "embedder_id"],
            set_={"vector": text("excluded.vector")}  # Use text() to reference excluded.vector
        )
        session.execute(stmt)
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in result_files}
        
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_file), total=len(result_files), desc="Processing files"):
            file_path = future_to_file[future]
            try:
                file_errors = future.result()
                if file_errors:
                    docs_with_errors.extend(file_errors)
            except Exception as exc:
                print(f"{file_path} generated an exception: {exc}")
    
    # Save error IDs if any
    if docs_with_errors:
        error_file = pathlib.Path(error_dir) / "errors.json"
        with open(error_file, 'w') as f:
            json.dump(docs_with_errors, f, indent=2)

def process_error_documents(session, error_file_path: str, client: openai.Client = None):
    """
    Process documents that failed during batch embedding by:
    1. Reading the error JSON file
    2. Retrieving affected documents
    3. Shortening content to appropriate token length
    4. Making live OpenAI embedding calls
    5. Saving results to the database
    
    Args:
        session: SQLAlchemy session
        error_file_path: Path to the JSON file containing document IDs with errors
        client: OpenAI client (if None, a new client will be created)
    """
    import time
    
    # Get OpenAI client if not provided
    if client is None:
        client = openai.Client()
        
    # Get embedder_id
    embedder_id = session.query(Embedder).filter_by(name='openai_small').first().id
    if not embedder_id:
        print("Error: Could not find embedder 'openai_small'")
        return
        
    # Read error document IDs
    try:
        with open(error_file_path, 'r') as f:
            error_doc_ids = json.load(f)
    except Exception as e:
        print(f"Error reading error file: {e}")
        return
        
    print(f"Processing {len(error_doc_ids)} documents with errors")
    
    # Retrieve documents
    documents = session.query(Document).filter(Document.id.in_(error_doc_ids)).all()
    doc_map = {doc.id: doc for doc in documents}
    
    # Create tokenizer for limiting text length
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    
    # Define max token limit (8191 for text-embedding-3-small)
    MAX_TOKENS = 8191
    
    # Function to shorten text to token limit
    def shorten_text(text):
        tokens = enc.encode(text)
        if len(tokens) > MAX_TOKENS:
            tokens = tokens[:MAX_TOKENS]
            return enc.decode(tokens)
        return text
    
    # Process documents in batches to avoid rate limits
    batch_size = 100  # Adjust based on OpenAI rate limits
    success_count = 0
    failed_doc_ids = []
    
    for i in range(0, len(error_doc_ids), batch_size):
        batch_doc_ids = error_doc_ids[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(error_doc_ids) + batch_size - 1)//batch_size}")
        
        for doc_id in tqdm.tqdm(batch_doc_ids):
            if doc_id not in doc_map:
                print(f"Warning: Document ID {doc_id} not found in database")
                failed_doc_ids.append(doc_id)
                continue
                
            doc = doc_map[doc_id]
            
            # Shorten text to token limit
            shortened_text = shorten_text(doc.content)
            
            # Make embedding request with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=shortened_text
                    )
                    
                    # Extract embedding vector
                    embedding_vector = response.data[0].embedding
                    
                    # Check if embedding already exists
                    existing = session.query(Embedding).filter_by(
                        document_id=doc_id, 
                        embedder_id=embedder_id
                    ).first()
                    
                    if existing:
                        # Update existing embedding
                        existing.vector = embedding_vector
                    else:
                        # Create new embedding
                        new_embedding = Embedding(
                            document_id=doc_id,
                            embedder_id=embedder_id,
                            vector=embedding_vector
                        )
                        session.add(new_embedding)
                    
                    # Commit each embedding individually to prevent losing all work if one fails
                    session.commit()
                    success_count += 1
                    break
                except Exception as e:
                    print(f"Error processing document {doc_id} (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        failed_doc_ids.append(doc_id)
                    else:
                        # Exponential backoff
                        time.sleep(2 ** attempt)
        
    # Save any remaining failed documents for future processing
    if failed_doc_ids:
        failed_file_path = error_file_path.replace('.json', '_still_failed.json')
        with open(failed_file_path, 'w') as f:
            json.dump(failed_doc_ids, f, indent=2)
        print(f"Saved {len(failed_doc_ids)} still-failing document IDs to {failed_file_path}")
    
    print(f"Successfully processed {success_count}/{len(error_doc_ids)} documents")

def hash_input_from_request_files(input_dir: str, results_dir: str, output_dir: str):
    """
    Read request files and result files to create a mapping of input hashes to embeddings,
    with separate files for each corpus.
    
    Args:
        input_dir: Directory containing request files organized by corpus
        results_dir: Directory containing result files organized by corpus
        output_dir: Directory to write the output JSONL.GZ files (one per corpus)
    """
    import hashlib
    
    input_dir = pathlib.Path(input_dir)
    results_dir = pathlib.Path(results_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all corpus subdirectories in the input directory
    corpus_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    for corpus_dir in tqdm.tqdm(corpus_dirs, desc="Processing corpora"):
        corpus_name = corpus_dir.name
        output_path = output_dir / f"{corpus_name}_hash_embedding_mapping.jsonl.gz"
        
        # Build mapping from document_id to hash for this corpus
        doc_id_to_hash = {}
        
        # Process request files for this corpus
        request_files = list(corpus_dir.glob('*.jsonl'))
        for file_path in tqdm.tqdm(request_files, desc=f"Processing requests for {corpus_name}"):
            with open(file_path, 'r') as f:
                for line in f:
                    request = json.loads(line)
                    input_text = request['body']['input']
                    custom_id = request['custom_id']
                    
                    # Extract document_id from custom_id
                    parts = custom_id.split('-')
                    if len(parts) >= 2:
                        doc_id = int(parts[1])
                    else:
                        print(f"Warning: Could not parse document ID from custom_id {custom_id}")
                        continue
                    
                    # Calculate hash of input text
                    input_hash = hashlib.md5(input_text.encode('utf-8')).hexdigest()
                    
                    # Store mapping
                    doc_id_to_hash[doc_id] = input_hash
        
        print(f"Processed {len(doc_id_to_hash)} unique requests for corpus {corpus_name}")
        
        # Find result files for this corpus
        result_dir = results_dir / corpus_name
        if not result_dir.exists():
            print(f"No results directory found for corpus {corpus_name}")
            continue
            
        result_files = list(result_dir.glob('*.jsonl.gz'))
        if not result_files:
            print(f"No result files found for corpus {corpus_name}")
            continue
        
        # Process result files to match embeddings with hashes
        with gzip.open(output_path, 'wt', encoding='utf-8') as output_file:
            total_mappings = 0
            
            for file_path in tqdm.tqdm(result_files, desc=f"Processing results for {corpus_name}"):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        result = json.loads(line)
                        
                        # Skip failed requests
                        if result['response']['status_code'] != 200:
                            continue
                        
                        # Get the custom ID
                        custom_id = result['custom_id']
                        doc_id = int(custom_id.split('-')[1])
                        
                        # Skip if not in our mapping
                        if doc_id not in doc_id_to_hash:
                            continue
                            
                        embedding = result['response']['body']['data'][0]["embedding"]
                        hash_code = doc_id_to_hash[doc_id]
                        
                        # Write directly to output file
                        record = {'input_hash': hash_code, 'embedding': embedding}
                        output_file.write(json.dumps(record) + '\n')
                        total_mappings += 1
        
        print(f"Wrote {total_mappings} hash-to-embedding mappings for corpus {corpus_name} to {output_path}")

def match_embeddings_for_documents(session, hash_embedding_dir: str, client: openai.Client = None):
    """
    Match documents in database with embeddings from the hash-to-embedding mapping files.
    Process one corpus at a time to optimize memory usage.
    
    Args:
        session: SQLAlchemy session
        hash_embedding_dir: Directory containing hash-to-embedding mapping files by corpus
    """
    import hashlib
    
    hash_embedding_dir = pathlib.Path(hash_embedding_dir)
    
    # Get embedder_id for the model
    embedder_id = session.query(Embedder).filter_by(name='openai_small').first().id
    
    # Tokenizer for text processing
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    MAX_TOKENS = 8191  # Max tokens for text-embedding-3-small
    
    # Get all corpus files
    mapping_files = list(hash_embedding_dir.glob("*_hash_embedding_mapping.jsonl.gz"))
    
    if not mapping_files:
        print(f"No hash mapping files found in {hash_embedding_dir}")
        return
    
    total_matched = 0
    total_unmatched = 0
    total_openai_calls = 0
    
    for mapping_file in tqdm.tqdm(mapping_files, desc="Processing corpora"):
        corpus_name = mapping_file.name.split('_hash_embedding_mapping')[0]
        print(f"Processing corpus: {corpus_name}")
        
        # Get corpus ID from name
        corpus = session.query(Corpus).filter_by(name=corpus_name).first()
        if not corpus:
            print(f"Warning: No corpus found with name '{corpus_name}', skipping")
            continue
        
        # Load hash-to-embedding mapping for this corpus
        hash_to_embedding = {}
        with gzip.open(mapping_file, 'rt', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                hash_to_embedding[item['input_hash']] = item['embedding']
        
        print(f"Loaded {len(hash_to_embedding)} hash-to-embedding mappings for corpus {corpus_name}")
        
        # Get all documents from this corpus that are raw documents
        raw_document_type_id = session.query(DocumentType).filter_by(name='raw').first().id
        documents = session.query(Document).filter_by(corpus_id=corpus.id, type_id=raw_document_type_id).all()
        print(f"Found {len(documents)} documents in corpus {corpus_name}")
        
        # Lists to track progress
        corpus_unmatched = []
        corpus_matched = 0
        openai_calls = 0
        
        # Process documents in batches
        batch_size = 1000
        embeddings_batch = []
        
        for doc in tqdm.tqdm(documents, desc=f"Processing documents in {corpus_name}"):
            # Truncate document text to max token limit
            tokens = enc.encode(doc.content)
            if len(tokens) > MAX_TOKENS:
                tokens = tokens[:MAX_TOKENS]
            
            # Recreate the same text that was used in the embedding request
            truncated_text = enc.decode(tokens)
            input_hash = hashlib.md5(truncated_text.encode('utf-8')).hexdigest()
            
            # Check if we have an embedding for this hash
            if input_hash in hash_to_embedding:
                embedding_vector = hash_to_embedding[input_hash]
                
                embeddings_batch.append({
                    'document_id': doc.id,
                    'embedder_id': embedder_id,
                    'vector': embedding_vector
                })
                
                corpus_matched += 1
            elif client is not None:
                # Make a live OpenAI embedding call
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=truncated_text
                )
                
                embedding_vector = response.data[0].embedding
                
                embeddings_batch.append({
                    'document_id': doc.id,
                    'embedder_id': embedder_id,
                    'vector': embedding_vector
                })
                
                corpus_matched += 1
                openai_calls += 1
            else:
                corpus_unmatched.append(doc.id)
            
            # Insert in batches
            if len(embeddings_batch) >= batch_size:
                _bulk_insert_embeddings(session, embeddings_batch)
                embeddings_batch = []
        
        # Insert any remaining embeddings
        if embeddings_batch:
            _bulk_insert_embeddings(session, embeddings_batch)
        
        print(f"Matched {corpus_matched} documents with embeddings in corpus {corpus_name}")
        print(f"Could not match {len(corpus_unmatched)} documents in corpus {corpus_name}")
        if openai_calls:
            print(f"Made {openai_calls} live OpenAI embedding calls")
        
        # Save unmatched document IDs for this corpus
        if corpus_unmatched:
            with open(f'unmatched_documents_{corpus_name}.json', 'w') as f:
                json.dump(corpus_unmatched, f, indent=2)
        
        total_matched += corpus_matched
        total_unmatched += len(corpus_unmatched)
        total_openai_calls += openai_calls
        
        # Clear hash_to_embedding to free memory before processing next corpus
        hash_to_embedding.clear()
    
    print(f"Total matched: {total_matched}, Total unmatched: {total_unmatched}")
    if total_openai_calls:
        print(f"Total live OpenAI embedding calls: {total_openai_calls}")

def _bulk_insert_embeddings(session, embeddings):
    """Helper function to bulk insert embeddings with conflict handling."""
    # Using PostgreSQL's INSERT ... ON CONFLICT for bulk upsert
    stmt = insert(Embedding).values(embeddings)
    stmt = stmt.on_conflict_do_update(
        index_elements=["document_id", "embedder_id"],
        set_={"vector": text("excluded.vector")}
    )
    session.execute(stmt)
    session.commit()

def main():
    import configuration as cfg
    import database

    config = cfg.load_config_from_env()
    db_config = config.database

    with database.get_session(db_config) as session:
        # Step 1: Create hash mapping from request files and link with embeddings from results
        # hash_input_from_request_files(
        #     "ignore/batch_embeddings/request_files",
        #     "ignore/batch_embeddings/results",
        #     "ignore/batch_embeddings/hash_embedding_mapping.jsonl.gz"
        # )
        
        # Step 2: Match documents with embeddings using the mapping
        # match_embeddings_for_documents(
        #     session,
        #     "ignore/batch_embeddings/hash_embedding_mapping.jsonl.gz"
        # )
        
        # Optional: Process documents that couldn't be matched
        # process_error_documents(session, "unmatched_documents.json")
        
        # Generate batch files if needed
        # create_batch_files_across_corpora(session, "ignore/batch_embeddings/request_files")
        
        # For batch submission and downloading (uncomment as needed)
        # batch_processor = BatchProcessor(openai.Client())
        # batch_processor.submit_batches("ignore/batch_embeddings/request_files")
        # batch_processor.save_batch_info("ignore/batch_embeddings/batch_info.json")
        # batch_processor.load_batch_info("ignore/batch_embeddings/batch_info.json")
        # batch_processor.download_all_results("ignore/batch_embeddings/results")

        # create hash files
        # hash_input_from_request_files("ignore/batch_embeddings/request_files", "ignore/batch_embeddings/results", "ignore/batch_embeddings/hash_embedding_mapping.jsonl.gz")

        # match embeddings
        client = openai.Client()
        match_embeddings_for_documents(session, "ignore/batch_embeddings/hash_embedding_mapping.jsonl.gz", client)
        
        session.commit()



if __name__ == "__main__":
    main()
import dataclasses
import gzip
import hashlib
import json
import pathlib
from typing import Dict, List, Optional, Set, Tuple

import openai
import tqdm
import tiktoken
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import text

from models import Corpus, Document, DocumentType, Embedder, Embedding
from db_document_store import DocStore


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


def get_existing_embeddings(duckdb_path: str, embedder_name: str = "openai_small") -> Set[int]:
    """
    Get document IDs that already have embeddings in the DuckDB database.
    
    Args:
        duckdb_path: Path to the DuckDB database
        embedder_name: Name of the embedder
        
    Returns:
        Set of document IDs with existing embeddings
    """
    try:
        with DocStore(duckdb_path) as doc_store:
            # Use a custom query to get just the document IDs
            result = doc_store.run_query("SELECT id FROM document")
            return {row[0] for row in result}
    except Exception as e:
        print(f"Warning: Could not retrieve existing embeddings from DuckDB: {e}")
        return set()


def create_batch_files_across_corpora(
        session,
        output_dir: str,
        duckdb_path: Optional[str] = None
    ) -> Dict[str, List[Tuple[int, str]]]:
    """
    Create batch files for each corpus and return mapping of corpus name to documents.
    
    Args:
        session: SQLAlchemy session
        output_dir: Output directory for batch files
        duckdb_path: Optional path to DuckDB database to check for existing embeddings
        
    Returns:
        Dictionary mapping corpus names to lists of (doc_id, content) tuples
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    corpora = session.query(Corpus).all()
    
    # Get set of document IDs that already have embeddings in DuckDB
    existing_embeddings = set()
    if duckdb_path:
        existing_embeddings = get_existing_embeddings(duckdb_path)
        print(f"Found {len(existing_embeddings)} documents with existing embeddings in DuckDB")

    # Get the embedder ID
    embedder = session.query(Embedder).filter_by(name='openai_small').first()
    if not embedder:
        raise ValueError("Embedder 'openai_small' not found in the database")
    
    # Get existing embeddings in main database
    existing_embedding_doc_ids = {
        doc_id for (doc_id,) in session.query(Embedding.document_id).filter_by(embedder_id=embedder.id).all()
    }
    print(f"Found {len(existing_embedding_doc_ids)} documents with existing embeddings in main database")
    
    # Combine both sets of existing embeddings
    all_existing_embeddings = existing_embeddings.union(existing_embedding_doc_ids)
    print(f"Total {len(all_existing_embeddings)} unique documents already have embeddings")
    
    # Store documents that need direct DB transfer
    direct_to_db_documents = {}

    total_tokens = 0
    total_processed = 0
    total_skipped = 0
    
    for corpus in tqdm.tqdm(corpora, desc="Creating batch files for corpora"):
        # Get raw documents
        raw_document_type_id = session.query(DocumentType).filter_by(name='raw').first().id
        documents = session.query(Document).filter_by(corpus_id=corpus.id, type_id=raw_document_type_id).all()
        
        # Apply chunking
        chunks = chunk_documents(
            documents,
            chunk_size=8191,
        )
        
        corpus_tokens = sum(chunk.num_tokens for chunk in chunks)
        total_tokens += corpus_tokens
        
        print(f"Corpus {corpus.name} has {len(chunks)} documents and {corpus_tokens} tokens")
        
        # Separate documents that already have embeddings
        batch_documents = []
        direct_documents = []
        
        for chunk in chunks:
            doc_id = chunk.doc_id
            
            if doc_id in all_existing_embeddings:
                # Document already has an embedding, skip batch processing
                total_skipped += 1
                if corpus.name not in direct_to_db_documents:
                    direct_to_db_documents[corpus.name] = []
                direct_documents.append((doc_id, chunk.content))
            else:
                # Document needs embedding, add to batch
                total_processed += 1
                batch_documents.append((f"doc-{doc_id}-chunk-{chunk.chunk_from}", chunk.content))
        
        # Create batch file if we have documents to process
        if batch_documents:
            create_batch_embedding_file(batch_documents, output_path / f"{corpus.name}")
        
    print(f"Total tokens: {total_tokens}")
    print(f"Documents to process: {total_processed}")
    print(f"Documents already processed: {total_skipped}")
    
    return direct_to_db_documents


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


def process_direct_to_db_documents(
    session, 
    direct_documents: Dict[str, List[Tuple[int, str]]], 
    duckdb_path: str,
    embedder_name: str = "openai_small"
):
    """
    Process documents that already have embeddings in DuckDB and 
    transfer them directly to the main database.
    
    Args:
        session: SQLAlchemy session
        direct_documents: Dictionary mapping corpus names to document lists
        duckdb_path: Path to the DuckDB database
        embedder_name: Name of the embedder to use
    """
    embedder = session.query(Embedder).filter_by(name=embedder_name).first()
    if not embedder:
        raise ValueError(f"Embedder '{embedder_name}' not found in the database")
    
    total_transferred = 0
    
    with DocStore(duckdb_path) as doc_store:
        for corpus_name, documents in direct_documents.items():
            print(f"Processing {len(documents)} documents from corpus {corpus_name}")
            
            embeddings_batch = []
            batch_size = 100
            
            for doc_id, _ in tqdm.tqdm(documents, desc=f"Transferring embeddings for {corpus_name}"):
                # Get document from DuckDB
                doc = doc_store.get_document_by_id(doc_id)
                if not doc:
                    continue
                
                # Extract embedding
                embedding_vector = doc["embedding"]
                
                # Add to batch
                embeddings_batch.append({
                    'document_id': doc_id,
                    'embedder_id': embedder.id,
                    'vector': embedding_vector
                })
                
                # Insert in batches
                if len(embeddings_batch) >= batch_size:
                    _bulk_insert_embeddings(session, embeddings_batch)
                    total_transferred += len(embeddings_batch)
                    embeddings_batch = []
            
            # Insert any remaining embeddings
            if embeddings_batch:
                _bulk_insert_embeddings(session, embeddings_batch)
                total_transferred += len(embeddings_batch)
    
    print(f"Total embeddings transferred from DuckDB to main database: {total_transferred}")


def load_result_vectors_into_db(
    session, 
    results_dir: str, 
    error_dir: str, 
    duckdb_path: Optional[str] = None
):
    """
    Load result vectors into the database and optionally into DuckDB.
    
    Args:
        session: SQLAlchemy session
        results_dir: Directory containing result files
        error_dir: Directory to write error files
        duckdb_path: Optional path to DuckDB database to also store embeddings
    """
    import concurrent.futures

    results_dir = pathlib.Path(results_dir)
    result_files = list(results_dir.glob('**/*.jsonl.gz'))
    
    # Get embedder_id once outside the loop
    embedder_id = session.query(Embedder).filter_by(name='openai_small').first().id
    docs_with_errors = []
    
    # Open DuckDB connection if path provided
    if duckdb_path:
        try:
            with DocStore(duckdb_path) as doc_store:
                # Make sure tables exist
                doc_store._create_tables()
        except Exception as e:
            print(f"Warning: Could not initialize DuckDB at {duckdb_path}: {e}")
            duckdb_path = None

    def process_file(file_path):
        file_errors = []
        batch_size = 1000
        embeddings_batch = []
        duckdb_docs = []
        
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
                
                # If DuckDB path is provided, collect data for DuckDB
                if duckdb_path:
                    # Get the document content from the request
                    input_text = result['request']['body']['input']
                    
                    # Add to DuckDB docs list
                    duckdb_docs.append({
                        'id': doc_id,
                        'content': input_text,
                        'embedding_json': json.dumps(embedding)
                    })
                
                # Insert in batches to reduce overhead
                if len(embeddings_batch) >= batch_size:
                    _bulk_insert_embeddings(embeddings_batch)
                    embeddings_batch = []
                    
                    # If DuckDB path is provided, insert into DuckDB
                    if duckdb_path and duckdb_docs:
                        _insert_into_duckdb(duckdb_docs, file_path)
                        duckdb_docs = []
            
            # Insert any remaining embeddings
            if embeddings_batch:
                _bulk_insert_embeddings(embeddings_batch)
                
                # If DuckDB path is provided, insert remaining into DuckDB
                if duckdb_path and duckdb_docs:
                    _insert_into_duckdb(duckdb_docs, file_path)
                
        return file_errors
    
    def _bulk_insert_embeddings(embeddings):
        # Using PostgreSQL's INSERT ... ON CONFLICT for bulk upsert
        stmt = insert(Embedding).values(embeddings)
        stmt = stmt.on_conflict_do_update(
            index_elements=["document_id", "embedder_id"],
            set_={"vector": text("excluded.vector")}  # Use text() to reference excluded.vector
        )
        session.execute(stmt)
    
    def _insert_into_duckdb(docs, file_path):
        # Extract corpus name from file path
        corpus_name = None
        path_parts = pathlib.Path(file_path).parts
        for part in path_parts:
            if part != "results" and part != "batch_embeddings":
                corpus_name = part
                break
        
        if not corpus_name:
            print(f"Warning: Could not determine corpus name from file path {file_path}")
            return
        
        try:
            with DocStore(duckdb_path) as doc_store:
                # Get corpus ID or create if doesn't exist
                corpus_result = doc_store.run_query("SELECT id FROM corpus WHERE name = ?", [corpus_name])
                if not corpus_result:
                    # Get corpus ID from Postgres
                    corpus = session.query(Corpus).filter_by(name=corpus_name).first()
                    if corpus:
                        doc_store.run_query(
                            "INSERT INTO corpus (id, name, description) VALUES (?, ?, ?)",
                            [corpus.id, corpus.name, corpus.description]
                        )
                        corpus_id = corpus.id
                    else:
                        print(f"Warning: Corpus {corpus_name} not found in main database")
                        return
                else:
                    corpus_id = corpus_result[0][0]
                
                # Insert documents in batches
                import pandas as pd
                df = pd.DataFrame(docs)
                
                # Add corpus_id to each document
                df['corpus_id'] = corpus_id
                
                # Insert into DuckDB
                doc_store.conn.execute("INSERT OR REPLACE INTO document SELECT * FROM df")
        except Exception as e:
            print(f"Error inserting into DuckDB: {e}")
    
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

def process_error_documents(session, error_file_path: str, client: openai.Client = None, duckdb_path: Optional[str] = None):
    """
    Process documents that failed during batch embedding by:
    1. Reading the error JSON file
    2. Retrieving affected documents
    3. Shortening content to appropriate token length
    4. Making live OpenAI embedding calls
    5. Saving results to the database and optionally to DuckDB
    
    Args:
        session: SQLAlchemy session
        error_file_path: Path to the JSON file containing document IDs with errors
        client: OpenAI client (if None, a new client will be created)
        duckdb_path: Optional path to DuckDB database to also store embeddings
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
    duckdb_docs = []

    # Open DuckDB connection if path provided
    if duckdb_path:
        try:
            with DocStore(duckdb_path) as doc_store:
                # Make sure tables exist
                doc_store._create_tables()
        except Exception as e:
            print(f"Warning: Could not initialize DuckDB at {duckdb_path}: {e}")
            duckdb_path = None
    
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
                    
                    # If DuckDB path is provided, collect data for DuckDB
                    if duckdb_path:
                        duckdb_docs.append({
                            'id': doc_id,
                            'corpus_id': doc.corpus_id,
                            'content': shortened_text,
                            'embedding_json': json.dumps(embedding_vector)
                        })
                    
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
        
        # Insert batch into DuckDB if path provided
        if duckdb_path and duckdb_docs:
            try:
                with DocStore(duckdb_path) as doc_store:
                    import pandas as pd
                    df = pd.DataFrame(duckdb_docs)
                    doc_store.conn.execute("INSERT OR REPLACE INTO document SELECT * FROM df")
                    duckdb_docs = []
            except Exception as e:
                print(f"Error inserting into DuckDB: {e}")
    
    # Insert any remaining docs into DuckDB
    if duckdb_path and duckdb_docs:
        try:
            with DocStore(duckdb_path) as doc_store:
                import pandas as pd
                df = pd.DataFrame(duckdb_docs)
                doc_store.conn.execute("INSERT OR REPLACE INTO document SELECT * FROM df")
        except Exception as e:
            print(f"Error inserting into DuckDB: {e}")
    
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

def match_embeddings_for_documents(session, hash_embedding_dir: str, client: openai.Client = None, duckdb_path: Optional[str] = None):
    """
    Match documents in database with embeddings from the hash-to-embedding mapping files.
    Process one corpus at a time to optimize memory usage.
    
    Args:
        session: SQLAlchemy session
        hash_embedding_dir: Directory containing hash-to-embedding mapping files by corpus
        client: Optional OpenAI client for live embedding calls
        duckdb_path: Optional path to DuckDB database to also store embeddings
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
    
    # Initialize DuckDB if path provided
    if duckdb_path:
        try:
            with DocStore(duckdb_path) as doc_store:
                # Make sure tables exist
                doc_store._create_tables()
        except Exception as e:
            print(f"Warning: Could not initialize DuckDB at {duckdb_path}: {e}")
            duckdb_path = None
    
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
        duckdb_docs = []
        
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
                
                # If DuckDB path is provided, collect data for DuckDB
                if duckdb_path:
                    duckdb_docs.append({
                        'id': doc.id,
                        'corpus_id': doc.corpus_id,
                        'content': truncated_text,
                        'embedding_json': json.dumps(embedding_vector)
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
                
                # If DuckDB path is provided, collect data for DuckDB
                if duckdb_path:
                    duckdb_docs.append({
                        'id': doc.id,
                        'corpus_id': doc.corpus_id,
                        'content': truncated_text,
                        'embedding_json': json.dumps(embedding_vector)
                    })
                
                corpus_matched += 1
                openai_calls += 1
            else:
                corpus_unmatched.append(doc.id)
            
            # Insert in batches
            if len(embeddings_batch) >= batch_size:
                _bulk_insert_embeddings(session, embeddings_batch)
                embeddings_batch = []
                
                # Insert into DuckDB if path provided
                if duckdb_path and duckdb_docs:
                    try:
                        with DocStore(duckdb_path) as doc_store:
                            # Ensure corpus exists in DuckDB
                            corpus_result = doc_store.run_query("SELECT id FROM corpus WHERE id = ?", [doc.corpus_id])
                            if not corpus_result:
                                doc_store.run_query(
                                    "INSERT INTO corpus (id, name, description) VALUES (?, ?, ?)",
                                    [corpus.id, corpus.name, corpus.description]
                                )
                            
                            # Insert documents
                            import pandas as pd
                            df = pd.DataFrame(duckdb_docs)
                            doc_store.conn.execute("INSERT OR REPLACE INTO document SELECT * FROM df")
                            duckdb_docs = []
                    except Exception as e:
                        print(f"Error inserting into DuckDB: {e}")
        
        # Insert any remaining embeddings
        if embeddings_batch:
            _bulk_insert_embeddings(session, embeddings_batch)
            
            # Insert remaining into DuckDB if path provided
            if duckdb_path and duckdb_docs:
                try:
                    with DocStore(duckdb_path) as doc_store:
                        import pandas as pd
                        df = pd.DataFrame(duckdb_docs)
                        doc_store.conn.execute("INSERT OR REPLACE INTO document SELECT * FROM df")
                except Exception as e:
                    print(f"Error inserting into DuckDB: {e}")
        
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch embedding processing tool")
    parser.add_argument("--prepare", action="store_true", help="Prepare batch files for processing")
    parser.add_argument("--submit", action="store_true", help="Submit batch files to OpenAI")
    parser.add_argument("--download", action="store_true", help="Download batch results")
    parser.add_argument("--load", action="store_true", help="Load results into database")
    parser.add_argument("--process-errors", action="store_true", help="Process documents with errors")
    parser.add_argument("--request-dir", default="ignore/batch_embeddings/request_files", help="Directory for request files")
    parser.add_argument("--result-dir", default="ignore/batch_embeddings/results", help="Directory for result files")
    parser.add_argument("--batch-info", default="ignore/batch_embeddings/batch_info.json", help="Path to batch info file")
    parser.add_argument("--hash-mapping-dir", default="ignore/batch_embeddings/hash_mapping", help="Directory for hash mapping files")
    parser.add_argument("--error-dir", default="ignore/batch_embeddings/errors", help="Directory for error files")
    parser.add_argument("--error-file", default=None, help="Path to error file for processing")
    parser.add_argument("--duckdb-path", default="ignore/embedding_store.db", help="Path to DuckDB database")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    config = cfg.load_config_from_env()
    db_config = config.database

    with database.get_session(db_config) as session:
        # Make OpenAI client
        client = openai.Client()

        # Create directories if they don't exist
        for dir_path in [args.request_dir, args.result_dir, args.hash_mapping_dir, args.error_dir]:
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Step 1: Prepare batch files (check DuckDB first for existing embeddings)
        if args.prepare:
            print("Preparing batch files for processing...")
            direct_to_db_documents = create_batch_files_across_corpora(
                session, 
                args.request_dir,
                args.duckdb_path
            )
            
            # Process documents that already have embeddings in DuckDB
            if direct_to_db_documents:
                print("Processing documents with existing embeddings in DuckDB...")
                process_direct_to_db_documents(
                    session, 
                    direct_to_db_documents, 
                    args.duckdb_path
                )
            else:
                print("No documents with existing embeddings in DuckDB to process")
        
        # Step 2: Submit batch files to OpenAI
        if args.submit:
            print("Submitting batch files to OpenAI...")
            batch_processor = BatchProcessor(client)
            batch_processor.submit_batches(args.request_dir)
            batch_processor.save_batch_info(args.batch_info)
            print(f"Batch information saved to {args.batch_info}")
        
        # Step 3: Download batch results
        if args.download:
            print("Downloading batch results...")
            batch_processor = BatchProcessor(client)
            batch_processor.load_batch_info(args.batch_info)
            batch_processor.download_all_results(args.result_dir)
            print(f"Results downloaded to {args.result_dir}")
        
        # Step 4: Load results into database and DuckDB
        if args.load:
            print("Loading results into database...")
            load_result_vectors_into_db(
                session, 
                args.result_dir, 
                args.error_dir,
                args.duckdb_path
            )
            
            # Create hash mapping files
            print("Creating hash mapping files...")
            hash_input_from_request_files(
                args.request_dir, 
                args.result_dir, 
                args.hash_mapping_dir
            )
            
            # Match embeddings for documents
            print("Matching embeddings for documents...")
            match_embeddings_for_documents(
                session, 
                args.hash_mapping_dir, 
                client,
                args.duckdb_path
            )
        
        # Step 5: Process documents with errors
        if args.process_errors:
            if not args.error_file:
                # Try to find most recent error file
                error_files = list(pathlib.Path(args.error_dir).glob("*.json"))
                if not error_files:
                    print("No error files found in error directory")
                    return
                error_file = max(error_files, key=lambda x: x.stat().st_mtime)
                print(f"Using most recent error file: {error_file}")
            else:
                error_file = args.error_file
            
            print(f"Processing documents with errors from {error_file}...")
            process_error_documents(
                session, 
                error_file,
                client,
                args.duckdb_path
            )
        
        session.commit()


if __name__ == "__main__":
    main()
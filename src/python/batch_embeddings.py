import dataclasses
import json
import pathlib

import openai
import tqdm
import tiktoken

from models import Corpus, Document, DocumentType


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

        documents = [
            (f"doc-{chunk.doc_id}-chunk-{chunk.chunk_from}-{chunk.chunk_to}", chunk.content)
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

    def download_batch_result(self, batch_info: BatchInfo, output_dir: str) -> str:
        """Download the result file for a completed batch."""
        batch_status = self.client.batches.retrieve(batch_info.batch_id)
        if not batch_status.output_file_id:
            raise ValueError(f"No output file found for batch {batch_info.batch_id}")

        output_file = f"{output_dir}/result_{batch_info.batch_id}.jsonl"
        
        # Download the file content
        file_response = self.client.files.content(batch_status.output_file_id)
        
        # Write content to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(file_response.text)
            
        return output_file
    
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
            if batch.status == "completed":
                self.download_batch_result(batch, output_dir)


def main():
    import configuration as cfg
    import database


    config = cfg.load_config_from_env()
    db_config = config.database

    with database.get_session(db_config) as session:
        # counts = count_tokens_across_corpora(session)
        # for corpus, count in counts.items():
        #     print(f"{corpus} token count:", count)

        # Create batch files
        # create_batch_files_across_corpora(session, "ignore/batch_embeddings/request_files")
        pass

    batch_processor = BatchProcessor(openai.Client())

    batch_processor.submit_batches("ignore/batch_embeddings/request_files")
    batch_processor.save_batch_info("ignore/batch_embeddings/batch_info.json")

    print("Checking batch statuses")
    print(batch_processor.check_batch_statuses())



if __name__ == "__main__":
    main()
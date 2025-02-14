import dataclasses
import json
import pathlib

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
        create_batch_files_across_corpora(session, "ignore/batch_embeddings/request_files")


if __name__ == "__main__":
    main()
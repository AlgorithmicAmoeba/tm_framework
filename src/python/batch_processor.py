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
    documents: list[Document],
    output_path: str,
    batch_size: int = 49_990
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
            for document in batch:
                request = {
                    "custom_id": f"doc-{document.id}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "text-embedding-3-small",
                        "input": document.content
                    }
                }
                f.write(f"{json.dumps(request)}\n")


def create_batch_files_across_corpora(session, output_dir: str):
    """Create batch files for each corpus and return mapping of corpus name to file path."""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    corpora = session.query(Corpus).all()
    
    for corpus in tqdm.tqdm(corpora, desc="Creating batch files for corpora"):
        documents = session.query(Document).filter_by(corpus_id=corpus.id).all()
        
        create_batch_embedding_file(documents, output_path / f"{corpus.name}")
        
    

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
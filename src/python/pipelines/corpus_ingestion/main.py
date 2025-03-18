"""
Corpus ingestion pipeline functions for various datasets.
These functions handle the preprocessing and storage of different corpora using raw SQL.
"""
import hashlib
import json
from typing import Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg


def hash_text(text: str) -> str:
    """Generate a hash for the document content."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()    


def store_corpus_documents(session: Session, corpus_name: str, texts: list[str], description: Optional[str] = None):
    """
    Store a corpus and its documents in the database using a delete-write pattern for idempotency.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus
        texts: List of document texts
        description: Optional description of the corpus
    """
    # Insert or update corpus using ON CONFLICT
    upsert_corpus_query = text("""
        INSERT INTO pipeline.corpus (name, description)
        VALUES (:name, :description)
        ON CONFLICT (name) 
        DO UPDATE SET 
            description = COALESCE(:description, pipeline.corpus.description)
    """)
    
    session.execute(
        upsert_corpus_query, 
        {"name": corpus_name, "description": description}
    )
    
    # Delete all existing documents for this corpus to ensure idempotency
    delete_docs_query = text("""
        DELETE FROM pipeline.document 
        WHERE corpus_name = :corpus_name
    """)
    session.execute(delete_docs_query, {"corpus_name": corpus_name})
    session.commit()
    
    # Process documents in batches to avoid memory issues with large datasets
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        for batch_text in batch_texts:
            content_hash = hash_text(batch_text)
            
            insert_doc_query = text("""
                INSERT INTO pipeline.document 
                (corpus_name, content_hash, content)
                VALUES (:corpus_name, :content_hash, :content)
                ON CONFLICT DO NOTHING
            """)
            
            session.execute(
                insert_doc_query,
                {
                    "corpus_name": corpus_name,
                    "content_hash": content_hash,
                    "content": batch_text
                }
            )
        
        # Commit after each batch
        session.commit()


def ingest_newsgroups(session: Session, subset: Optional[int] = None, description: Optional[str] = None):
    """
    Ingest the 20 Newsgroups dataset into the database.
    
    Args:
        session: Database session
        subset: Optional number of documents to ingest
        description: Optional description of the corpus
    """
    from sklearn.datasets import fetch_20newsgroups
    
    # Fetch and clean data
    newsgroups = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
    )
    texts = newsgroups.data
    
    if subset is not None:
        texts = texts[:subset]
    
    if description is None:
        description = "20 Newsgroups dataset - collection of newsgroup documents"
    
    store_corpus_documents(session, "newsgroups", texts, description)


def ingest_wikipedia(session: Session, file_path: str, subset: Optional[int] = None, description: Optional[str] = None):
    """
    Ingest Wikipedia articles from a JSON Lines file.
    
    Args:
        session: Database session
        file_path: Path to the JSONL file containing Wikipedia articles
        subset: Optional number of documents to ingest
        description: Optional description of the corpus
    """
    # Read JSON Lines file
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            texts.append(article['text'])
    
    if subset is not None:
        texts = texts[:subset]
    
    if description is None:
        description = f"Wikipedia sample articles from {file_path}"
    
    store_corpus_documents(session, "wikipedia_sample", texts, description)


def ingest_imdb(session: Session, subset: Optional[int] = None, description: Optional[str] = None):
    """
    Ingest IMDB movie reviews dataset.
    
    Args:
        session: Database session
        subset: Optional number of documents to ingest
        description: Optional description of the corpus
    """
    from datasets import load_dataset
    
    # Load both train and test splits
    dataset = load_dataset("stanfordnlp/imdb")
    train_texts = dataset['train']['text']
    test_texts = dataset['test']['text']
    texts = train_texts + test_texts
    
    if subset is not None:
        texts = texts[:subset]
    
    if description is None:
        description = "IMDB movie reviews dataset for sentiment analysis"
    
    store_corpus_documents(session, "imdb_reviews", texts, description)


def ingest_trec(session: Session, subset: Optional[int] = None, description: Optional[str] = None):
    """
    Ingest TREC question classification dataset.
    
    Args:
        session: Database session
        subset: Optional number of documents to ingest
        description: Optional description of the corpus
    """
    from datasets import load_dataset
    
    # Load both train and test splits
    dataset = load_dataset("CogComp/trec")
    train_texts = dataset['train']['text']
    test_texts = dataset['test']['text']
    texts = train_texts + test_texts
    
    if subset is not None:
        texts = texts[:subset]
    
    if description is None:
        description = "TREC question classification dataset"
    
    store_corpus_documents(session, "trec_questions", texts, description)


def ingest_twitter_financial(session: Session, subset: Optional[int] = None, description: Optional[str] = None):
    """
    Ingest Twitter Financial News Topic dataset.
    
    Args:
        session: Database session
        subset: Optional number of documents to ingest
        description: Optional description of the corpus
    """
    import pandas as pd
    
    splits = {'train': 'topic_train.csv', 'validation': 'topic_valid.csv'}
    df = pd.read_csv("hf://datasets/zeroshot/twitter-financial-news-topic/" + splits["train"])

    texts = df['text'].tolist()
    
    if subset is not None:
        texts = texts[:subset]
    
    if description is None:
        description = "Twitter Financial News Topic dataset"
    
    store_corpus_documents(session, "twitter-financial-news", texts, description)


if __name__ == '__main__':
    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database
    
    try:
        data_path = config.get_data_path()
    except AttributeError:
        # If get_data_path method doesn't exist
        data_path = None
    
    # Create database session
    with get_session(db_config) as session:
        # Example usage of the ingestion functions
        print("Ingesting newsgroups corpus...")
        ingest_newsgroups(session)
        
        print("Ingesting IMDB reviews corpus...")
        ingest_imdb(session)
        
        print("Ingesting TREC questions corpus...")
        ingest_trec(session)
        
        # For Wikipedia, you need to provide a file path
        if data_path:
            wiki_path = data_path / 'raw_data/wikipedia_20k_sample.jsonl'
            print(f"Ingesting Wikipedia sample from {wiki_path}...")
            ingest_wikipedia(session, str(wiki_path))
        
        print("Ingesting Twitter financial news corpus...")
        ingest_twitter_financial(session)
        
        print("All corpora successfully ingested.")
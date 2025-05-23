from typing import Dict, List, Tuple
import numpy as np
from sqlalchemy import text
from configuration import load_config_from_env
from database import get_session


def get_tfidf_vectors(corpus_name: str) -> Tuple[List[str], np.ndarray]:
    """
    Retrieve TF-IDF vectors for all documents in a corpus.
    
    Args:
        corpus_name: Name of the corpus to retrieve vectors for
        
    Returns:
        Tuple containing:
        - List of document hashes
        - 2D numpy array of TF-IDF vectors
    """
    # Load configuration
    config = load_config_from_env()
    db_config = config.database
    
    # Create database session
    with get_session(db_config) as session:
        # Get all TF-IDF vectors for the corpus
        query = text("""
            SELECT raw_document_hash, terms 
            FROM pipeline.tfidf_vector 
            WHERE corpus_name = :corpus_name
            ORDER BY raw_document_hash
        """).bindparams(corpus_name=corpus_name)
        results = session.execute(query).fetchall()
        
        if not results:
            return [], np.array([])
        
        doc_hashes = [row[0] for row in results]
        vectors = np.array([row[1] for row in results])
        
        return doc_hashes, vectors

def get_vocabulary(corpus_name: str) -> Dict[int, str]:
    """
    Retrieve vocabulary mapping for a corpus from the vocabulary_word table.
    
    Args:
        corpus_name: Name of the corpus to retrieve vocabulary for
        
    Returns:
        Dictionary mapping word indices to words
    """
    # Load configuration
    config = load_config_from_env()
    db_config = config.database
    
    # Create database session
    with get_session(db_config) as session:
        query = text("""
            SELECT word_index, word 
            FROM pipeline.vocabulary_word 
            WHERE corpus_name = :corpus_name
            ORDER BY word_index
        """).bindparams(corpus_name=corpus_name)
        results = session.execute(query).fetchall()
        
        if not results:
            return {}
            
        return {word_index: word for word_index, word in results}

def get_sbert_chunk_embeddings(corpus_name: str) -> Tuple[List[str], np.ndarray]:
    """
    Retrieve SBERT embeddings for all chunks in a corpus.
    
    Args:
        corpus_name: Name of the corpus to retrieve embeddings for
        
    Returns:
        Tuple containing:
        - List of chunk hashes
        - 2D numpy array of embeddings
    """
    # Load configuration
    config = load_config_from_env()
    db_config = config.database
    
    # Create database session
    with get_session(db_config) as session:
        # Get all SBERT chunk embeddings for the corpus by joining with chunked_document
        query = text("""
            SELECT cd.chunk_hash, ce.embedding
            FROM pipeline.chunked_document cd
            JOIN pipeline.sbert_chunk_embedding ce ON cd.chunk_hash = ce.chunk_hash
            WHERE cd.corpus_name = :corpus_name
            ORDER BY cd.raw_document_hash
        """).bindparams(corpus_name=corpus_name)
        results = session.execute(query).fetchall()
        
        if not results:
            return [], np.array([])
        
        chunk_hashes = [row[0] for row in results]
        embeddings = np.array([row[1] for row in results])
        
        return chunk_hashes, embeddings

def get_openai_chunk_embeddings(corpus_name: str) -> Tuple[List[str], np.ndarray]:
    """
    Retrieve embeddings for all chunks in a corpus.
    
    Args:
        corpus_name: Name of the corpus to retrieve embeddings for
        
    Returns:
        Tuple containing:
        - List of chunk hashes
        - 2D numpy array of embeddings
    """
    # Load configuration
    config = load_config_from_env()
    db_config = config.database
    
    # Create database session
    with get_session(db_config) as session:
        # Get all chunk embeddings for the corpus by joining with chunked_document
        query = text("""
            SELECT cd.chunk_hash, ce.embedding
            FROM pipeline.chunked_document cd
            JOIN pipeline.chunk_embedding ce ON cd.chunk_hash = ce.chunk_hash
            WHERE cd.corpus_name = :corpus_name
            ORDER BY cd.raw_document_hash
        """).bindparams(corpus_name=corpus_name)
        results = session.execute(query).fetchall()
        
        if not results:
            return [], np.array([])
        
        chunk_hashes = [row[0] for row in results]
        embeddings = np.array([row[1] for row in results])
        
        return chunk_hashes, embeddings
    
def get_chunk_embeddings(corpus_name: str, embedding_type: str) -> Tuple[List[str], np.ndarray]:
    """
    Retrieve embeddings for all chunks in a corpus based on embedding type.
    
    Args:
        corpus_name: Name of the corpus to retrieve embeddings for
        embedding_type: Type of embedding to retrieve ('openai' or 'sbert')
        
    Returns:
        Tuple containing:
        - List of chunk hashes
        - 2D numpy array of embeddings
    """
    if embedding_type == 'openai':
        return get_openai_chunk_embeddings(corpus_name)
    elif embedding_type == 'sbert':
        return get_sbert_chunk_embeddings(corpus_name)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

def get_vocabulary_documents(corpus_name: str) -> List[Tuple[str, str]]:
    """
    Retrieve vocabulary documents for a corpus from the vocabulary_document table.
    
    Args:
        corpus_name: Name of the corpus to retrieve documents for
        
    Returns:
        List of tuples containing (document_hash, content) for each document
    """
    # Load configuration
    config = load_config_from_env()
    db_config = config.database
    
    # Create database session
    with get_session(db_config) as session:
        query = text("""
            SELECT raw_document_hash, content 
            FROM pipeline.preprocessed_document 
            WHERE corpus_name = :corpus_name
            ORDER BY raw_document_hash
        """).bindparams(corpus_name=corpus_name)
        results = session.execute(query).fetchall()
        
        if not results:
            return []
            
        return [(doc_hash, content) for doc_hash, content in results]

if __name__ == "__main__":
    corpus_name = "newsgroups"
    
    doc_hashes, vectors = get_tfidf_vectors(corpus_name)
    print(doc_hashes)
    print(vectors)

    chunk_hashes, embeddings = get_chunk_embeddings(corpus_name)
    print(chunk_hashes)
    print(embeddings)
    
    vocabulary = get_vocabulary(corpus_name)
    print(vocabulary)
    
    vocabulary_docs = get_vocabulary_documents(corpus_name)
    print(f"Found {len(vocabulary_docs)} vocabulary documents")
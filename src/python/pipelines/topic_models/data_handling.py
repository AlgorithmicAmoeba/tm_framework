from typing import Dict, List, Tuple
import numpy as np
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
        query = """
            SELECT raw_document_hash, terms 
            FROM pipeline.tfidf_vector 
            WHERE corpus_name = %s
        """
        results = session.execute(query, (corpus_name,)).fetchall()
        
        if not results:
            return [], np.array([])
        
        doc_hashes = [row[0] for row in results]
        vectors = np.array([row[1] for row in results])
        
        return doc_hashes, vectors

def get_chunk_embeddings(corpus_name: str) -> Tuple[List[str], np.ndarray]:
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
        query = """
            SELECT cd.chunk_hash, ce.embedding
            FROM pipeline.chunked_document cd
            JOIN pipeline.chunk_embedding ce ON cd.chunk_hash = ce.chunk_hash
            WHERE cd.corpus_name = %s
        """
        results = session.execute(query, (corpus_name,)).fetchall()
        
        if not results:
            return [], np.array([])
        
        chunk_hashes = [row[0] for row in results]
        embeddings = np.array([row[1] for row in results])
        
        return chunk_hashes, embeddings
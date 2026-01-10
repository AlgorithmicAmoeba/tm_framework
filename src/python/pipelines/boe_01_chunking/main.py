"""
BOE (Bag of Embeddings) Chunking pipeline for document corpora.
This pipeline chunks documents into fixed character lengths with sliding windows
and filters chunks to contain only vocabulary words from the corpus.
"""
import logging
from typing import Any, List, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session
import spacy

from database import get_session
import configuration as cfg
from shared_code import color_logging_text, hash_string


def create_chunks(text: str, chunk_size: int = 1280, overlap: int = 256) -> List[Tuple[str, int, int]]:
    """
    Create overlapping chunks from text with specified character length and overlap.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum character length of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of tuples containing (chunk_content, start_position, end_position)
    """
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_content = text[start:end]
        chunks.append((chunk_content, start, end))
        
        # Move start position by chunk_size - overlap for sliding window
        start += chunk_size - overlap
        
        # Break if we've covered the entire text
        if end >= len(text):
            break
    
    return chunks


def filter_chunk_vocabulary(chunk_content: str, vocabulary_words: set, spacy_model=None) -> str:
    """
    Filter chunk content to contain only vocabulary words from the corpus.
    Uses lemmatization to match words with the vocabulary.
    
    Args:
        chunk_content: Raw chunk content
        vocabulary_words: Set of vocabulary words from the corpus
        spacy_model: Spacy model for lemmatization (optional)
        
    Returns:
        String containing only vocabulary words from the chunk
    """
    if spacy_model is None:
        # Simple word filtering without lemmatization
        words = chunk_content.split()
        vocabulary_only_words = [word.lower() for word in words if word.lower() in vocabulary_words]
    else:
        # Use spacy for lemmatization
        doc = spacy_model(chunk_content)
        vocabulary_only_words = []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                lemma = token.lemma_.lower()
                if lemma in vocabulary_words:
                    vocabulary_only_words.append(lemma)
    
    return ' '.join(vocabulary_only_words)


def get_corpus_vocabulary(session: Session, corpus_name: str) -> set:
    """
    Retrieve vocabulary words for a corpus from the database.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus
        
    Returns:
        Set of vocabulary words (lowercased)
    """
    query = text("""
        SELECT word FROM pipeline.vocabulary_word 
        WHERE corpus_name = :corpus_name
        ORDER BY word_index
    """)
    
    result = session.execute(query, {"corpus_name": corpus_name})
    vocabulary_words = {row[0].lower() for row in result}
    
    logging.info(f"Retrieved {len(vocabulary_words)} vocabulary words for corpus: {corpus_name}")
    return vocabulary_words


def chunk_documents(session: Session, corpus_name: str, chunk_size: int = 1280, overlap: int = 256) -> dict[str, Any]:
    """
    Chunk all documents in a corpus with sliding windows.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus to chunk
        chunk_size: Maximum character length of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        Dictionary containing chunking statistics
    """
    logging.info(f"Starting chunking for corpus: {corpus_name}")
    
    # Get vocabulary words for the corpus
    vocabulary_words = get_corpus_vocabulary(session, corpus_name)
    
    if not vocabulary_words:
        logging.warning(f"No vocabulary found for corpus: {corpus_name}")
        return {"total_chunks": 0, "processed_documents": 0}
    
    # Initialize spacy model for lemmatization
    try:
        spacy_model = spacy.load('en_core_web_sm', exclude=['tok2vec', 'parser', 'ner'])
        logging.info("Loaded spacy model for lemmatization")
    except OSError:
        logging.warning("Could not load spacy model, using simple word filtering")
        spacy_model = None
    
    # Fetch documents from the corpus
    fetch_docs_query = text("""
        SELECT document_hash, content 
        FROM pipeline.used_raw_document
        WHERE corpus_name = :corpus_name
    """)
    
    result = session.execute(fetch_docs_query, {"corpus_name": corpus_name})
    docs = [(row[0], row[1]) for row in result]
    
    if not docs:
        logging.warning(f"No preprocessed documents found for corpus: {corpus_name}")
        return {"total_chunks": 0, "processed_documents": 0}
    
    logging.info(f"Processing {len(docs)} documents for chunking")
    
    all_chunks = []
    processed_docs = 0
    
    for doc_hash, content in docs:
        if not content or not content.strip():
            continue
            
        # Create chunks from the document
        chunks = create_chunks(content, chunk_size, overlap)
        
        for chunk_content, start_pos, end_pos in chunks:
            # Filter chunk to contain only vocabulary words
            vocabulary_only_content = filter_chunk_vocabulary(chunk_content, vocabulary_words, spacy_model)
            
            # Skip chunks with no vocabulary words
            if not vocabulary_only_content.strip():
                continue
            
            # Create chunk hash
            chunk_hash = hash_string(chunk_content)
            
            all_chunks.append({
                'raw_document_hash': doc_hash,
                'corpus_name': corpus_name,
                'chunk_hash': chunk_hash,
                'content': chunk_content,
                'chunk_vocabulary_words': vocabulary_only_content,
                'chunk_start': start_pos,
                'chunk_end': end_pos
            })
        
        processed_docs += 1
        
        if processed_docs % 100 == 0:
            logging.info(f"Processed {processed_docs} documents, created {len(all_chunks)} chunks so far")
    
    logging.info(f"Chunking complete: {len(all_chunks)} chunks from {processed_docs} documents")
    
    return {
        'chunks': all_chunks,
        'total_chunks': len(all_chunks),
        'processed_documents': processed_docs,
        'vocabulary_size': len(vocabulary_words)
    }


def store_chunked_documents(session: Session, corpus_name: str, chunking_data: dict[str, Any]):
    """
    Store chunked documents in the database.
    Uses a delete-insert pattern for idempotency.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus
        chunking_data: Data returned from the chunking function
    """
    logging.info(f"Storing chunked documents for corpus: {corpus_name}")
    
    # Get existing chunks to track what's already processed
    existing_chunks = session.execute(
        text("""
            SELECT raw_document_hash, chunk_hash 
            FROM pipeline.boe_chunked_document 
            WHERE corpus_name = :corpus_name
        """),
        {"corpus_name": corpus_name}
    ).fetchall()
    existing_chunks_dict = {(row[0], row[1]): False for row in existing_chunks}
    
    # Track statistics
    chunks_existing = 0
    chunks_inserted = 0
    
    # Insert chunks
    chunks = chunking_data['chunks']
    
    for chunk_data in chunks:
        chunk_key = (chunk_data['raw_document_hash'], chunk_data['chunk_hash'])
        
        # Mark chunk as processed if it exists
        if chunk_key in existing_chunks_dict:
            existing_chunks_dict[chunk_key] = True
            chunks_existing += 1
            continue
        
        # Insert chunk
        insert_chunk_query = text("""
            INSERT INTO pipeline.boe_chunked_document 
            (raw_document_hash, corpus_name, chunk_hash, content, chunk_vocabulary_words, chunk_start, chunk_end)
            VALUES (:raw_document_hash, :corpus_name, :chunk_hash, :content, :chunk_vocabulary_words, :chunk_start, :chunk_end)
            ON CONFLICT DO NOTHING
        """)
        
        session.execute(insert_chunk_query, chunk_data)
        chunks_inserted += 1
    
    session.commit()
    
    # Delete chunks that were not marked as processed
    chunks_deleted = 0
    for chunk_key, processed in existing_chunks_dict.items():
        if processed:
            continue
        
        raw_document_hash, chunk_hash = chunk_key
        delete_chunk_query = text("""
            DELETE FROM pipeline.boe_chunked_document
            WHERE corpus_name = :corpus_name 
            AND raw_document_hash = :raw_document_hash 
            AND chunk_hash = :chunk_hash
        """)
        
        session.execute(
            delete_chunk_query,
            {
                "corpus_name": corpus_name,
                "raw_document_hash": raw_document_hash,
                "chunk_hash": chunk_hash
            }
        )
        chunks_deleted += 1
    
    session.commit()
    
    # Log statistics
    logging.info(f"Chunking storage complete for corpus: {corpus_name}")
    logging.info(f"  Chunks existing: {chunks_existing}")
    
    if chunks_inserted:
        logging.info(color_logging_text(
            f"  Chunks inserted: {chunks_inserted}",
            color='green'
        ))
    else:
        logging.info(f"  Chunks inserted: {chunks_inserted}")
    
    if chunks_deleted:
        logging.info(color_logging_text(
            f"  Chunks deleted: {chunks_deleted}",
            color='red'
        ))
    else:
        logging.info(f"  Chunks deleted: {chunks_deleted}")


def process_corpus_chunking(session: Session, corpus_name: str, chunk_size: int = 1280, overlap: int = 256):
    """
    Process chunking for a specified corpus.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus to chunk
        chunk_size: Maximum character length of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        Tuple of (total_chunks, processed_documents)
    """
    logging.info(f"Starting chunking pipeline for corpus: {corpus_name}")
    
    # Chunk the documents
    chunking_data = chunk_documents(session, corpus_name, chunk_size, overlap)
    
    if chunking_data['total_chunks'] == 0:
        logging.warning(f"No chunks created for corpus: {corpus_name}")
        return 0, 0
    
    # Store the chunked data
    store_chunked_documents(session, corpus_name, chunking_data)
    
    # Print statistics
    total_chunks = chunking_data['total_chunks']
    processed_docs = chunking_data['processed_documents']
    vocab_size = chunking_data['vocabulary_size']
    
    logging.info(f"Chunking complete for corpus: {corpus_name}")
    logging.info(f"Documents processed: {processed_docs}")
    logging.info(f"Total chunks created: {total_chunks}")
    logging.info(f"Vocabulary size: {vocab_size}")
    logging.info(f"Average chunks per document: {total_chunks/processed_docs:.1f}")
    
    return total_chunks, processed_docs


def get_available_corpora(session: Session) -> list[str]:
    """
    Get list of available corpora from the database.
    """
    query = text("""
        SELECT DISTINCT corpus_name FROM pipeline.boe_chunked_document
    """)
    result = session.execute(query)
    return [row[0] for row in result]


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Direct logging to file
    logging.getLogger().addHandler(logging.FileHandler("boe_chunking.log"))
    
    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database
    
    # Create database session
    with get_session(db_config) as session:
        # Define chunking parameters
        chunk_size = 1280
        overlap = 256
        
        logging.info("Starting BOE chunking pipelines...")
        logging.info(f"Chunk size: {chunk_size} characters")
        logging.info(f"Overlap: {overlap} characters")
        
        # List of corpora to process
        corpora = get_available_corpora(session)
        
        # Process each corpus
        for corpus_name in corpora:
            logging.info(f"Processing corpus: {corpus_name}")
            try:
                total_chunks, processed_docs = process_corpus_chunking(
                    session, corpus_name, chunk_size, overlap
                )
                logging.info(f"Successfully processed {corpus_name}: {total_chunks} chunks from {processed_docs} documents")
            except Exception as e:
                logging.error(f"Error processing corpus {corpus_name}: {str(e)}")
                continue
        
        logging.info("All BOE chunking pipelines completed.")

"""
BOE (Bag of Embeddings) Chunking pipeline for document corpora.
This pipeline chunks documents into fixed character lengths with sliding windows
and extracts vocabulary words from each chunk.
"""
import logging
import sys
from typing import Any, List, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg
from shared_code import color_logging_text, hash_string

# Import TextPreprocessor from preprocessing pipeline
sys.path.insert(0, str(__file__).replace('/boe_01_chunking/main.py', ''))
from preprocessing.main import TextPreprocessor

# Global text preprocessor instance (lazy-loaded)
_text_preprocessor = None


def get_text_preprocessor() -> TextPreprocessor:
    """Get or initialize the text preprocessor for batch cleaning."""
    global _text_preprocessor
    if _text_preprocessor is None:
        _text_preprocessor = TextPreprocessor(
            remove_stopwords=True,
            lemmatize=True,
            remove_numbers=True,
            remove_urls=True,
            min_chars=3
        )
        logging.info("Initialized TextPreprocessor for chunk cleaning")
    return _text_preprocessor


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


def extract_vocabulary_words_from_cleaned(cleaned_text: str, vocabulary_words: set) -> str:
    """
    Extract vocabulary words from already-cleaned text.
    
    Args:
        cleaned_text: Pre-cleaned text content
        vocabulary_words: Set of vocabulary words from the corpus
        
    Returns:
        Space-separated string of vocabulary words found in the text (with duplicates)
    """
    # Split on whitespace and filter to only vocabulary words, keeping duplicates
    words = cleaned_text.split()
    vocabulary_only_words = [word for word in words if word in vocabulary_words]
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


def get_unchunked_documents(session: Session, corpus_name: str) -> List[Tuple[str, str]]:
    """
    Fetch documents that don't have any chunks yet.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus
        
    Returns:
        List of tuples containing (document_hash, content) for unchunked documents
    """
    query = text("""
        SELECT urd.document_hash, urd.content
        FROM pipeline.used_raw_document urd
        WHERE urd.corpus_name = :corpus_name
        AND NOT EXISTS (
            SELECT 1 FROM pipeline.boe_chunked_document bcd
            WHERE bcd.raw_document_hash = urd.document_hash
            AND bcd.corpus_name = urd.corpus_name
        )
    """)
    result = session.execute(query, {"corpus_name": corpus_name})
    return [(row[0], row[1]) for row in result]


def chunk_documents(
    session: Session, 
    corpus_name: str, 
    docs: List[Tuple[str, str]],
    chunk_size: int = 1280, 
    overlap: int = 256
) -> dict[str, Any]:
    """
    Chunk provided documents with sliding windows and extract vocabulary words.
    Uses batch processing for text cleaning to improve performance.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus to chunk
        docs: List of tuples containing (document_hash, content)
        chunk_size: Maximum character length of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        Dictionary containing chunking statistics
    """
    logging.info(f"Starting chunking for corpus: {corpus_name}")
    
    if not docs:
        logging.info(f"No documents to chunk for corpus: {corpus_name}")
        return {"total_chunks": 0, "processed_documents": 0, "chunks": [], "vocabulary_size": 0}
    
    # Get vocabulary words for the corpus (may be empty)
    vocabulary_words = get_corpus_vocabulary(session, corpus_name)
    
    if not vocabulary_words:
        logging.warning(f"No vocabulary found for corpus: {corpus_name}, chunks will have empty vocabulary_words")
    
    logging.info(f"Processing {len(docs)} documents for chunking")
    
    # Step 1: Create all chunks first (without cleaning)
    chunk_data_list = []  # List of (doc_hash, chunk_content, start_pos, end_pos)
    processed_docs = 0
    
    for doc_hash, content in docs:
        if not content or not content.strip():
            continue
            
        # Create chunks from the document
        chunks = create_chunks(content, chunk_size, overlap)
        
        for chunk_content, start_pos, end_pos in chunks:
            # Skip empty chunks
            if not chunk_content.strip():
                continue
            
            chunk_data_list.append((doc_hash, chunk_content, start_pos, end_pos))
        
        processed_docs += 1
        
        if processed_docs % 100 == 0:
            logging.info(f"Chunked {processed_docs} documents, created {len(chunk_data_list)} chunks so far")
    
    logging.info(f"Created {len(chunk_data_list)} chunks from {processed_docs} documents")
    
    if not chunk_data_list:
        return {"total_chunks": 0, "processed_documents": processed_docs, "chunks": [], "vocabulary_size": len(vocabulary_words)}
    
    # Step 2: Clean all chunk contents in batch using TextPreprocessor
    logging.info(f"Cleaning {len(chunk_data_list)} chunks in batch...")
    chunk_contents = [chunk_content for _, chunk_content, _, _ in chunk_data_list]
    preprocessor = get_text_preprocessor()
    cleaned_texts = preprocessor._clean_texts(chunk_contents)
    logging.info("Batch cleaning complete")
    
    # Step 3: Extract vocabulary words and build final chunk data
    all_chunks = []
    for i, (doc_hash, chunk_content, start_pos, end_pos) in enumerate(chunk_data_list):
        cleaned_text = cleaned_texts[i]
        
        # Extract vocabulary words from cleaned text
        vocabulary_only_content = extract_vocabulary_words_from_cleaned(cleaned_text, vocabulary_words)
        
        # Create chunk hash from original content
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
    Inserts new chunks with ON CONFLICT DO NOTHING for idempotency.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus
        chunking_data: Data returned from the chunking function
    """
    logging.info(f"Storing chunked documents for corpus: {corpus_name}")
    
    # Track statistics
    chunks_inserted = 0
    
    # Insert chunks
    chunks = chunking_data['chunks']
    
    for chunk_data in chunks:
        # Insert chunk with ON CONFLICT DO NOTHING for idempotency
        insert_chunk_query = text("""
            INSERT INTO pipeline.boe_chunked_document 
            (raw_document_hash, corpus_name, chunk_hash, content, chunk_vocabulary_words, chunk_start, chunk_end)
            VALUES (:raw_document_hash, :corpus_name, :chunk_hash, :content, :chunk_vocabulary_words, :chunk_start, :chunk_end)
            ON CONFLICT DO NOTHING
        """)
        
        result = session.execute(insert_chunk_query, chunk_data)
        if result.rowcount > 0:
            chunks_inserted += 1
    
    session.commit()
    
    # Log statistics
    logging.info(f"Chunking storage complete for corpus: {corpus_name}")
    
    if chunks_inserted:
        logging.info(color_logging_text(
            f"  Chunks inserted: {chunks_inserted}",
            color='green'
        ))
    else:
        logging.info(f"  Chunks inserted: {chunks_inserted}")


def process_corpus_chunking(session: Session, corpus_name: str, chunk_size: int = 1280, overlap: int = 256):
    """
    Process chunking for a specified corpus.
    Only processes documents that haven't been chunked yet.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus to chunk
        chunk_size: Maximum character length of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        Tuple of (total_chunks, processed_documents)
    """
    logging.info(f"Starting chunking pipeline for corpus: {corpus_name}")
    
    # Get only documents that haven't been chunked yet
    unchunked_docs = get_unchunked_documents(session, corpus_name)
    
    if not unchunked_docs:
        logging.info(f"All documents in corpus '{corpus_name}' have already been chunked, skipping")
        return 0, 0
    
    logging.info(f"Found {len(unchunked_docs)} unchunked documents to process")
    
    # Chunk only the unchunked documents
    chunking_data = chunk_documents(session, corpus_name, unchunked_docs, chunk_size, overlap)
    
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
        SELECT DISTINCT name FROM pipeline.corpus
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

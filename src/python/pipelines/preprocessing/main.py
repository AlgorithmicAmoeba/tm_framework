"""
Preprocessing pipeline functions for document corpora.
These functions handle text preprocessing, tokenization, and TF-IDF computation using raw SQL.
"""
import re
import string
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy

from database import get_session
import configuration as cfg
from shared_code import color_logging_text, hash_string





class TextPreprocessor:
    """Text preprocessing class with configurable options."""
    
    def __init__(
            self,
            top_n: int | None = None,
            min_words_per_document: int | None = None,
            min_df: float = 0.0,
            max_df: float = 1.0,
            min_chars: int = 3,
            remove_stopwords: bool = True,
            lemmatize: bool = True,
            remove_numbers: bool = True,
            remove_urls: bool = True,
            language: str = 'en_core_web_sm'
        ):
        """
        Initialize the text preprocessor.
        
        Args:
            top_n: Maximum number of words to keep in vocabulary
            min_words_per_document: Minimum words required per document
            min_df: Minimum document frequency for words
            max_df: Maximum document frequency for words
            min_chars: Minimum characters per word
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            remove_numbers: Whether to remove numbers
            remove_urls: Whether to remove URLs
            language: Spacy language model to use
        """
        self._top_n = top_n
        self._min_words_per_document = min_words_per_document
        self._min_df = min_df
        self._max_df = max_df
        self._min_chars = min_chars
        self._remove_stopwords = remove_stopwords
        self._remove_urls = remove_urls
        self._remove_numbers = remove_numbers
        self._lemmatize = lemmatize

        # Initialize processing patterns
        self._patterns = {
            'url': re.compile(r'https?://\S+'),
            'whitespace': re.compile(r'\s+'),
            'apostrophe': re.compile(r"(?<=\w)'|'(?=\w)"),
            'punctuation': re.compile(rf"[{re.escape(string.punctuation)}]"),
            'numeric': re.compile(r'\d')
        }

        # Initialize spacy model if needed
        if remove_stopwords or lemmatize:
            self._spacy_model = spacy.load(
                language,
                exclude=['tok2vec', 'parser', 'ner']
            )
        else:
            self._spacy_model = None

    def process_corpus(self, corpus_name: str, texts: list[str]) -> dict[str, Any]:
        """
        Process a corpus of texts.
        
        Args:
            corpus_name: Name of the corpus
            texts: List of document texts
            
        Returns:
            Dictionary containing processed data
        """
        # Generate document hashes for reference
        doc_hashes = [hash_string(text) for text in texts]
        
        # Clean and tokenize texts
        cleaned_texts = self._clean_texts(texts)
        tokenized_texts = [text.split(' ') for text in cleaned_texts]
        
        # Calculate original vocabulary size before filtering
        count_vectorizer = CountVectorizer()
        count_vectorizer.fit(texts)
        original_vocabulary = count_vectorizer.get_feature_names_out()
        original_vocab_size = len(original_vocabulary)
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(
            norm=None,
            lowercase=False,
            min_df=self._min_df,
            max_df=self._max_df,
            max_features=self._top_n
        )
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        vocabulary_words = vectorizer.get_feature_names_out()

        vocabulary_words_set = set(vocabulary_words)

        # Build documents that only have vocabulary words
        # Each document should be a string of space separated vocabulary words that are in the document
        vocabulary_only_documents = [
            ' '.join([word for word in tokenized_texts[i] if word in vocabulary_words_set])
            for i in range(len(cleaned_texts))
        ]

        # Filter documents if needed
        if self._min_words_per_document:
            text_lens = [len(tokens) for tokens in vocabulary_only_documents]
            valid_indices = [i for i, length in enumerate(text_lens) 
                            if length >= self._min_words_per_document]
            
            # Filter all data structures consistently
            doc_hashes = [doc_hashes[i] for i in valid_indices]
            cleaned_texts = [cleaned_texts[i] for i in valid_indices]
            tokenized_texts = [tokenized_texts[i] for i in valid_indices]
            tfidf_matrix = tfidf_matrix[valid_indices]
            raw_texts = [texts[i] for i in valid_indices]
            vocabulary_only_documents = [vocabulary_only_documents[i] for i in valid_indices]
        
        # Build document-term vectors for database storage
        tfidf_vectors = []
        for i, doc_hash in enumerate(doc_hashes):            # Get full vector as array
            row = tfidf_matrix[i].toarray()[0].tolist()
            
            tfidf_vectors.append({
                'document_hash': doc_hash,
                'corpus_name': corpus_name,
                'terms': row
            })
        
        return {
            'doc_hashes': doc_hashes,
            'cleaned_texts': cleaned_texts,
            'vocabulary': vocabulary_words,
            'tfidf_vectors': tfidf_vectors,
            'original_docs_count': len(texts),
            'filtered_docs_count': len(valid_indices),
            'original_vocab_size': original_vocab_size,
            'filtered_vocab_size': len(vocabulary_words),
            'raw_texts': raw_texts,
            'vocabulary_only_documents': vocabulary_only_documents,
        }
    
    def _clean_texts(self, texts: list[str]) -> list[str]:
        """Clean and process texts."""
        # Pre-processing: URL removal
        if self._remove_urls:
            texts = [self._patterns['url'].sub('', text) for text in texts]
        
        # Spacy processing (lemmatization and stopword removal)
        if self._lemmatize or self._remove_stopwords:
            disable = ["tok2vec", "parser", "ner"]
            doc_objects = self._spacy_model.pipe(texts, batch_size=100, n_process=8, disable=disable)
            
            processed_texts = []
            for doc in doc_objects:
                tokens = []
                for token in doc:
                    if self._remove_stopwords and token.is_stop:
                        continue
                    if self._lemmatize:
                        tokens.append(token.lemma_)
                    else:
                        tokens.append(token.text)
                processed_texts.append(' '.join(tokens))
            texts = processed_texts
        
        # Post-processing steps
        for text_idx in range(len(texts)):
            # Convert to lowercase
            text = texts[text_idx].lower()
            
            # Remove punctuation and normalize whitespace
            text = self._patterns['punctuation'].sub(' ', text)
            
            # Remove numbers if configured
            if self._remove_numbers:
                text = self._patterns['numeric'].sub('', text)
            
            # Normalize whitespace
            text = self._patterns['whitespace'].sub(' ', text)
            text = text.strip()
            
            # Filter by minimum character length
            if self._min_chars:
                text = ' '.join([w for w in text.split() if len(w) >= self._min_chars])
            
            texts[text_idx] = text
        
        return texts


def store_preprocessed_documents(session: Session, corpus_name: str, processed_data: dict[str, Any]):
    """
    Store preprocessed documents and vocabulary in the database.
    Uses a delete-insert pattern for idempotency.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus
        processed_data: Data returned from the preprocessor
    """
    logging.info(f"Storing preprocessed documents for corpus: {corpus_name}")
    
    # Get existing data to track what's already processed
    existing_vocab = session.execute(
        text("SELECT word, word_index FROM pipeline.vocabulary_word WHERE corpus_name = :corpus_name"),
        {"corpus_name": corpus_name}
    ).fetchall()
    existing_vocab_dict = {row[0]: {'index': row[1], 'processed': False} for row in existing_vocab}
    
    existing_docs = session.execute(
        text("SELECT raw_document_hash FROM pipeline.preprocessed_document WHERE corpus_name = :corpus_name"),
        {"corpus_name": corpus_name}
    ).scalars().fetchall()
    existing_docs_dict = {doc_hash: False for doc_hash in existing_docs}

    existing_vocabulary_only_docs = session.execute(
        text("SELECT raw_document_hash FROM pipeline.vocabulary_document WHERE corpus_name = :corpus_name"),
        {"corpus_name": corpus_name}
    ).scalars().fetchall()
    existing_vocabulary_only_docs_dict = {doc_hash: False for doc_hash in existing_vocabulary_only_docs}

    existing_vectors = session.execute(
        text("SELECT raw_document_hash, terms FROM pipeline.tfidf_vector WHERE corpus_name = :corpus_name"),
        {"corpus_name": corpus_name}
    ).fetchall()
    existing_vectors_dict = {row[0]: {'vector': row[1], "processed": False} for row in existing_vectors}

    # logging.info(f"Found {len(existing_vocab_dict)} existing vocabulary words and {len(existing_docs_dict)} existing documents")
    
    # Track statistics
    vocab_existing = 0
    vocab_inserted = 0
    docs_existing = 0
    docs_inserted = 0
    tfidf_existing = 0
    tfidf_inserted = 0
    
    # Insert vocabulary words
    vocabulary = processed_data['vocabulary']
    vocabulary_data = []
    
    for idx, word in enumerate(vocabulary):
        if word in existing_vocab_dict:
            existing_vocab_dict[word]['processed'] = True
            vocab_existing += 1
            continue
            
        vocabulary_data.append({
            "corpus_name": corpus_name,
            "word": word,
            "word_index": idx
        })
    
    # Process vocabulary in batches to avoid memory issues
    batch_size = 1000
    
    # Insert new vocabulary words
    for i in range(0, len(vocabulary_data), batch_size):
        batch = vocabulary_data[i:i+batch_size]
        insert_vocab_query = text("""
            INSERT INTO pipeline.vocabulary_word (corpus_name, word, word_index)
            VALUES (:corpus_name, :word, :word_index)
            ON CONFLICT DO NOTHING
        """)
        
        for item in batch:
            session.execute(insert_vocab_query, item)
            vocab_inserted += 1
        
        session.commit()
        logging.debug(f"Processed vocabulary batch {i//batch_size + 1}, inserted {len(batch)} words")
    
    # Insert preprocessed documents
    doc_hashes = processed_data['doc_hashes']
    cleaned_texts = processed_data['cleaned_texts']
    raw_texts = processed_data['raw_texts']
    vocabulary_only_documents = processed_data['vocabulary_only_documents']

    for i in range(len(doc_hashes)):
        doc_hash = doc_hashes[i]
        content = cleaned_texts[i]
        raw_content = raw_texts[i]
        vocabulary_only_document = vocabulary_only_documents[i]
        content_hash = hash_string(content)
        vocabulary_only_document_hash = hash_string(vocabulary_only_document)
        
        # Mark document as processed if it exists
        if doc_hash in existing_docs_dict:
            existing_docs_dict[doc_hash] = True
            docs_existing += 1
            continue
            
        # Insert preprocessed document
        insert_doc_query = text("""
            INSERT INTO pipeline.preprocessed_document (corpus_name, raw_document_hash, content, content_hash)
            VALUES (:corpus_name, :document_hash, :content, :content_hash)
            ON CONFLICT DO NOTHING
        """)
        
        session.execute(
            insert_doc_query,
            {
                "corpus_name": corpus_name,
                "document_hash": doc_hash,
                "content": content,
                "content_hash": content_hash
            }
        )
        docs_inserted += 1
        
        # Insert raw document that passed filtering
        insert_raw_doc_query = text("""
            INSERT INTO pipeline.used_raw_document (corpus_name, document_hash, content)
            VALUES (:corpus_name, :document_hash, :content)
            ON CONFLICT DO NOTHING
        """)
        
        session.execute(
            insert_raw_doc_query,
            {
                "corpus_name": corpus_name,
                "document_hash": doc_hash,
                "content": raw_content
            }
        )

        # Insert vocabulary-only document
        insert_vocabulary_only_doc_query = text("""
            INSERT INTO pipeline.vocabulary_document (corpus_name, raw_document_hash, content, content_hash)
            VALUES (:corpus_name, :document_hash, :content, :content_hash)
            ON CONFLICT DO NOTHING
        """)

        session.execute(
            insert_vocabulary_only_doc_query,
            {
                "corpus_name": corpus_name,
                "document_hash": doc_hash,
                "content": vocabulary_only_document,
                "content_hash": vocabulary_only_document_hash
            }
        )
    
    session.commit()
    
    # Insert document-term matrix entries
    tfidf_vectors = processed_data['tfidf_vectors']
    
    tfidf_data = []
    for i in range(len(tfidf_vectors)):
        doc_hash = doc_hashes[i]
        tfidf_vector = tfidf_vectors[i]
        existing_tfidf_vector = existing_vectors_dict.get(doc_hash)
        if existing_tfidf_vector is not None and existing_tfidf_vector["vector"] == tfidf_vector["terms"]:
            existing_tfidf_vector["processed"] = True
            tfidf_existing += 1
            continue

        tfidf_data.append(tfidf_vector)

    for i in range(0, len(tfidf_data), batch_size):
        batch = tfidf_vectors[i:i+batch_size]
        insert_term_query = text("""
            INSERT INTO pipeline.tfidf_vector 
            (corpus_name, raw_document_hash, terms)
            VALUES (:corpus_name, :document_hash, :terms)
            ON CONFLICT (corpus_name, raw_document_hash) 
            DO UPDATE SET terms = EXCLUDED.terms
        """)
        
        for item in batch:
            session.execute(
                insert_term_query, 
                {
                    "corpus_name": item['corpus_name'],
                    "document_hash": item['document_hash'],
                    "terms": item['terms']
                }
            )
            tfidf_inserted += 1
        
        session.commit()
        logging.debug(f"Processed TF-IDF batch {i//batch_size + 1}, processed {len(batch)} vectors")
    
    # Delete data that was not marked as processed
    docs_deleted = 0
    for doc_hash, processed in existing_docs_dict.items():
        if processed:
            continue

        assert False, "Going to delete something"
            
        delete_doc_query = text("""
            DELETE FROM pipeline.preprocessed_document
            WHERE corpus_name = :corpus_name AND raw_document_hash = :document_hash
        """)
        
        session.execute(
            delete_doc_query,
            {"corpus_name": corpus_name, "document_hash": doc_hash}
        )
        docs_deleted += 1
    
    vocab_deleted = 0
    for word, info in existing_vocab_dict.items():
        if info['processed']:
            continue

        assert False, "Going to delete something"  
        delete_vocab_query = text("""
            DELETE FROM pipeline.vocabulary_word
            WHERE corpus_name = :corpus_name AND word = :word
        """)
        
        session.execute(
            delete_vocab_query,
            {"corpus_name": corpus_name, "word": word}
        )
        vocab_deleted += 1


    tfidf_deleted = 0
    for doc_hash, info in existing_vectors_dict.items():
        if info['processed']:
            continue

        assert False, "Going to delete something"
        delete_tfidf_query = text("""
            DELETE FROM pipeline.tfidf_vector
            WHERE corpus_name = :corpus_name AND raw_document_hash = :document_hash
        """)

        session.execute(
            delete_tfidf_query,
            {"corpus_name": corpus_name, "document_hash": doc_hash}
        )
        tfidf_deleted += 1
    
    session.commit()
    
    # Log statistics
    logging.info(f"Preprocessing storage complete for corpus: {corpus_name}")
    logging.info(f"  Vocabulary words existing: {vocab_existing}")
    if vocab_inserted:
        logging.info(color_logging_text(
            f"  Vocabulary words inserted: {vocab_inserted}",
            color='green'
        ))
    else:
        logging.info(f"  Vocabulary words inserted: {vocab_inserted}")

    if vocab_deleted:
        logging.info(color_logging_text(
            f"  Vocabulary words deleted: {vocab_deleted}",
            color='red'
        ))
    else:
        logging.info(f"  Vocabulary words deleted: {vocab_deleted}")
    logging.info(f"  Documents existing: {docs_existing}")

    if docs_inserted:
        logging.info(color_logging_text(
            f"  Documents inserted: {docs_inserted}",
            color='green'
        ))
    else:
        logging.info(f"  Documents inserted: {docs_inserted}")

    if docs_deleted:
        logging.info(color_logging_text(
            f"  Documents deleted: {docs_deleted}",
            color='red'
        ))
    else:
        logging.info(f"  Documents deleted: {docs_deleted}")

    logging.info(f"  TF-IDF vectors existing: {tfidf_existing}")
    if tfidf_inserted:
        logging.info(color_logging_text(
            f"  TF-IDF vectors processed: {tfidf_inserted}",
            color='green'
        ))
    else:
        logging.info(f"  TF-IDF vectors processed: {tfidf_inserted}")

    if tfidf_deleted:
        logging.info(color_logging_text(
            f"  TF-IDF vectors deleted: {tfidf_deleted}",
            color='red'
        ))
    else:
        logging.info(f"  TF-IDF vectors deleted: {tfidf_deleted}")


def preprocess_corpus(session: Session, corpus_name: str, preprocessing_params: dict[str, Any] | None = None):
    """
    Preprocess documents for a specified corpus.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus to preprocess
        preprocessing_params: Dictionary of preprocessing parameters
        
    Returns:
        Tuple of (vocabulary_size, document_count)
    """
    logging.info(f"Starting preprocessing for corpus: {corpus_name}")
    
    # Default parameters if none provided
    if preprocessing_params is None:
        preprocessing_params = {
            'top_n': 10000,
            'min_words_per_document': 5,
            'min_df': 0.001,
            'max_df': 0.7,
            'min_chars': 3,
            'remove_stopwords': True,
            'lemmatize': True,
            'remove_numbers': True,
            'remove_urls': True
        }
    
    # Fetch documents from the corpus
    fetch_docs_query = text("""
        SELECT content_hash, content 
        FROM pipeline.document
        WHERE corpus_name = :corpus_name
    """)
    
    result = session.execute(fetch_docs_query, {"corpus_name": corpus_name})
    docs = [(row[0], row[1]) for row in result]
    
    if not docs:
        logging.warning(f"No documents found for corpus: {corpus_name}")
        return 0, 0
    
    doc_hashes, texts = zip(*docs)
    original_doc_count = len(texts)
    
    # Initialize preprocessor with parameters
    preprocessor = TextPreprocessor(**preprocessing_params)
    
    # Process the corpus
    logging.info(f"Processing {original_doc_count} documents for corpus: {corpus_name}")
    processed_data = preprocessor.process_corpus(corpus_name, list(texts))
    
    # Store the processed data
    store_preprocessed_documents(session, corpus_name, processed_data)
    
    # Print statistics
    filtered_count = processed_data['filtered_docs_count']
    original_count = processed_data['original_docs_count']
    filtered_vocab = processed_data['filtered_vocab_size']
    original_vocab = processed_data['original_vocab_size']
    
    logging.info(f"Preprocessing complete for corpus: {corpus_name}")
    logging.info(f"Documents: {filtered_count} of {original_count} ({filtered_count/original_count*100:.1f}%)")
    logging.info(f"Vocabulary: {filtered_vocab} of {original_vocab} ({filtered_vocab/original_vocab*100:.1f}%)")
    
    return filtered_vocab, filtered_count


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # direct logging to file
    logging.getLogger().addHandler(logging.FileHandler("preprocessing.log"))
    
    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database
    
    # Create database session
    with get_session(db_config) as session:
        # Define corpus-specific preprocessing parameters
        newsgroups_params = {
            'top_n': 10000,
            'min_words_per_document': 5,
            'min_df': 0.005,
            'max_df': 0.8,
            'min_chars': 3,
            'remove_stopwords': True,
            'lemmatize': True
        }
        
        wikipedia_params = {
            'top_n': 20000,
            'min_words_per_document': 10,
            'min_df': 0.02,
            'max_df': 0.7,
            'min_chars': 3,
            'remove_stopwords': True,
            'lemmatize': True
        }
        
        imdb_params = {
            'top_n': 15000,
            'min_words_per_document': 150,
            'min_df': 0.003,
            'max_df': 0.1,
            'min_chars': 3,
            'remove_stopwords': True,
            'lemmatize': True
        }
        
        trec_params = {
            'top_n': 5000,
            'min_words_per_document': 2,
            'min_df': 0.0005,
            'max_df': 0.9,
            'min_chars': 3,
            'remove_stopwords': True,
            'lemmatize': True
        }
        
        twitter_params = {
            'top_n': 8000,
            'min_words_per_document': 3,
            'min_df': 0.001,
            'max_df': 0.8,
            'min_chars': 3,
            'remove_stopwords': True,
            'lemmatize': True
        }
        
        # Run preprocessing for each corpus
        logging.info("Starting preprocessing pipelines...")
        
        logging.info("Preprocessing newsgroups corpus...")
        preprocess_corpus(session, "newsgroups", newsgroups_params)
        
        logging.info("Preprocessing Wikipedia corpus...")
        preprocess_corpus(session, "wikipedia_sample", wikipedia_params)
        
        logging.info("Preprocessing IMDB reviews corpus...")
        preprocess_corpus(session, "imdb_reviews", imdb_params)
        
        logging.info("Preprocessing TREC questions corpus...")
        preprocess_corpus(session, "trec_questions", trec_params)
        
        logging.info("Preprocessing Twitter financial news corpus...")
        preprocess_corpus(session, "twitter-financial-news", twitter_params)
        
        logging.info("All preprocessing pipelines completed successfully.")
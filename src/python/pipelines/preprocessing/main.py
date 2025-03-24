"""
Preprocessing pipeline functions for document corpora.
These functions handle text preprocessing, tokenization, and TF-IDF computation using raw SQL.
"""
import hashlib
import re
import string
from typing import Optional, List, Dict, Any, Set

from sqlalchemy import text
from sqlalchemy.orm import Session
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy
import tqdm
import json

from database import get_session
import configuration as cfg


def hash_text(text: str) -> str:
    """Generate a hash for the document content."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


class TextPreprocessor:
    """Text preprocessing class with configurable options."""
    
    def __init__(
            self,
            top_n: Optional[int] = None,
            min_words_per_document: Optional[int] = None,
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

    def process_corpus(self, corpus_name: str, texts: List[str]) -> Dict[str, Any]:
        """
        Process a corpus of texts.
        
        Args:
            corpus_name: Name of the corpus
            texts: List of document texts
            
        Returns:
            Dictionary containing processed data
        """
        # Generate document hashes for reference
        doc_hashes = [hash_text(text) for text in texts]
        
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
        feature_names = vectorizer.get_feature_names_out()
        
        # Filter documents if needed
        valid_indices = list(range(len(texts)))
        if self._min_words_per_document:
            text_lens = [len(tokens) for tokens in tokenized_texts]
            valid_indices = [i for i, length in enumerate(text_lens) 
                            if length >= self._min_words_per_document]
            
            # Filter all data structures consistently
            doc_hashes = [doc_hashes[i] for i in valid_indices]
            cleaned_texts = [cleaned_texts[i] for i in valid_indices]
            tokenized_texts = [tokenized_texts[i] for i in valid_indices]
            tfidf_matrix = tfidf_matrix[valid_indices]
        
        # Build document-term vectors for database storage
        tfidf_vectors = []
        for i, doc_idx in enumerate(valid_indices):
            doc_hash = doc_hashes[doc_idx]
            # Get full vector as array
            row = tfidf_matrix[i].toarray()[0].tolist()
            
            tfidf_vectors.append({
                'document_hash': doc_hash,
                'corpus_name': corpus_name,
                'terms': row
            })
        
        return {
            'doc_hashes': doc_hashes,
            'cleaned_texts': cleaned_texts,
            'vocabulary': feature_names,
            'tfidf_vectors': tfidf_vectors,
            'valid_indices': valid_indices,
            'original_docs_count': len(texts),
            'filtered_docs_count': len(valid_indices),
            'original_vocab_size': original_vocab_size,
            'filtered_vocab_size': len(feature_names),
            'raw_texts': [texts[i] for i in valid_indices]
        }
    
    def _clean_texts(self, texts: List[str]) -> List[str]:
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


def store_preprocessed_documents(session: Session, corpus_name: str, processed_data: Dict[str, Any]):
    """
    Store preprocessed documents and vocabulary in the database.
    Uses a delete-insert pattern for idempotency.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus
        processed_data: Data returned from the preprocessor
    """
    # Delete existing data for this corpus to ensure idempotency
    delete_queries = [
        text("DELETE FROM pipeline.used_raw_document WHERE corpus_name = :corpus_name"),
        text("DELETE FROM pipeline.tfidf_vector WHERE corpus_name = :corpus_name"),
        text("DELETE FROM pipeline.vocabulary_word WHERE corpus_name = :corpus_name"),
        text("DELETE FROM pipeline.preprocessed_document WHERE corpus_name = :corpus_name")
    ]
    
    for query in delete_queries:
        session.execute(query, {"corpus_name": corpus_name})
    
    session.commit()
    
    # Insert vocabulary words
    vocabulary = processed_data['vocabulary']
    vocabulary_data = []
    
    for idx, word in enumerate(vocabulary):
        vocabulary_data.append({
            "corpus_name": corpus_name,
            "word": word,
            "word_index": idx
        })
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    
    # Insert vocabulary words
    for i in range(0, len(vocabulary_data), batch_size):
        batch = vocabulary_data[i:i+batch_size]
        insert_vocab_query = text("""
            INSERT INTO pipeline.vocabulary_word (corpus_name, word, word_index)
            VALUES (:corpus_name, :word, :word_index)
            ON CONFLICT DO NOTHING
        """)
        
        for item in batch:
            session.execute(insert_vocab_query, item)
        
        session.commit()
    
    # Insert preprocessed documents
    doc_hashes = processed_data['doc_hashes']
    cleaned_texts = processed_data['cleaned_texts']
    valid_indices = processed_data['valid_indices']
    raw_texts = processed_data['raw_texts']
    
    for i in range(len(doc_hashes)):
        doc_hash = doc_hashes[i]
        content = cleaned_texts[i]
        raw_content = raw_texts[i]
        
        # Insert preprocessed document
        insert_doc_query = text("""
            INSERT INTO pipeline.preprocessed_document (corpus_name, raw_document_hash, content, content_hash)
            VALUES (:corpus_name, :document_hash, :content, :content_hash)
            ON CONFLICT DO NOTHING
        """)
        
        content_hash = hash_text(content)
        
        session.execute(
            insert_doc_query,
            {
                "corpus_name": corpus_name,
                "document_hash": doc_hash,
                "content": content,
                "content_hash": content_hash
            }
        )
        
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
    
    session.commit()
    
    # Insert document-term matrix entries
    tfidf_vectors = processed_data['tfidf_vectors']
    
    for i in range(0, len(tfidf_vectors), batch_size):
        batch = tfidf_vectors[i:i+batch_size]
        insert_term_query = text("""
            INSERT INTO pipeline.tfidf_vector 
            (corpus_name, raw_document_hash, terms)
            VALUES (:corpus_name, :document_hash, :terms)
            ON CONFLICT DO NOTHING
        """)
        
        for item in batch:
            session.execute(
                insert_term_query, 
                {
                    "corpus_name": item['corpus_name'],
                    "document_hash": item['document_hash'],
                    "terms": json.dumps(item['terms'])
                }
            )
        
        session.commit()


def preprocess_corpus(session: Session, corpus_name: str, preprocessing_params: dict = None):
    """
    Preprocess documents for a specified corpus.
    
    Args:
        session: Database session
        corpus_name: Name of the corpus to preprocess
        preprocessing_params: Dictionary of preprocessing parameters
    """
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
        SELECT document_hash, content 
        FROM pipeline.document
        WHERE corpus_name = :corpus_name
    """)
    
    result = session.execute(fetch_docs_query, {"corpus_name": corpus_name})
    docs = [(row[0], row[1]) for row in result]
    
    if not docs:
        print(f"No documents found for corpus: {corpus_name}")
        return
    
    doc_hashes, texts = zip(*docs)
    original_doc_count = len(texts)
    
    # Initialize preprocessor with parameters
    preprocessor = TextPreprocessor(**preprocessing_params)
    
    # Process the corpus
    print(f"Processing {original_doc_count} documents for corpus: {corpus_name}")
    processed_data = preprocessor.process_corpus(corpus_name, list(texts))
    doc_hashes_used = [doc_hashes[i] for i in processed_data['valid_indices']]
    processed_data['doc_hashes'] = doc_hashes_used
    
    # Store the processed data
    print(f"Storing processed data for corpus: {corpus_name}")
    store_preprocessed_documents(session, corpus_name, processed_data)
    
    # Print statistics
    print(f"Preprocessing complete for corpus: {corpus_name}")
    print(f"Documents: {processed_data['filtered_docs_count']} of {processed_data['original_docs_count']} ({processed_data['filtered_docs_count']/processed_data['original_docs_count']*100:.1f}%)")
    print(f"Vocabulary: {processed_data['filtered_vocab_size']} of {processed_data['original_vocab_size']} ({processed_data['filtered_vocab_size']/processed_data['original_vocab_size']*100:.1f}%)")


def preprocess_newsgroups(session: Session, preprocessing_params: dict = None):
    """Preprocess 20 Newsgroups corpus."""
    preprocess_corpus(session, "newsgroups", preprocessing_params)


def preprocess_wikipedia(session: Session, preprocessing_params: dict = None):
    """Preprocess Wikipedia corpus."""
    preprocess_corpus(session, "wikipedia_sample", preprocessing_params)


def preprocess_imdb(session: Session, preprocessing_params: dict = None):
    """Preprocess IMDB reviews corpus."""
    preprocess_corpus(session, "imdb_reviews", preprocessing_params)


def preprocess_trec(session: Session, preprocessing_params: dict = None):
    """Preprocess TREC questions corpus."""
    preprocess_corpus(session, "trec_questions", preprocessing_params)


def preprocess_twitter_financial(session: Session, preprocessing_params: dict = None):
    """Preprocess Twitter financial news corpus."""
    preprocess_corpus(session, "twitter-financial-news", preprocessing_params)


if __name__ == '__main__':
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
            'min_words_per_document': 10,
            'min_df': 0.003,
            'max_df': 0.7,
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
        print("Starting preprocessing pipelines...")
        
        print("\nPreprocessing newsgroups corpus...")
        preprocess_newsgroups(session, newsgroups_params)
        
        print("\nPreprocessing Wikipedia corpus...")
        preprocess_wikipedia(session, wikipedia_params)
        
        print("\nPreprocessing IMDB reviews corpus...")
        preprocess_imdb(session, imdb_params)
        
        print("\nPreprocessing TREC questions corpus...")
        preprocess_trec(session, trec_params)
        
        print("\nPreprocessing Twitter financial news corpus...")
        preprocess_twitter_financial(session, twitter_params)
        
        print("\nAll preprocessing pipelines completed successfully.")
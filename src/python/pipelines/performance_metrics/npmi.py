import numpy as np
from typing import List, Dict, Any, Set, Tuple
from sqlalchemy import text
import json
from collections import defaultdict
import duckdb
import os
from pathlib import Path
import re

# Create cache directory if it doesn't exist
CACHE_DIR = Path("ignore/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB = CACHE_DIR / "word_freq_cache.db"

def init_cache_db():
    """Initialize the DuckDB cache database."""
    conn = duckdb.connect(str(CACHE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS word_frequencies (
            corpus_name VARCHAR,
            word VARCHAR,
            frequency INTEGER,
            PRIMARY KEY (corpus_name, word)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS word_co_occurrences (
            corpus_name VARCHAR,
            word1 VARCHAR,
            word2 VARCHAR,
            co_occurrence INTEGER,
            PRIMARY KEY (corpus_name, word1, word2)
        )
    """)
    conn.commit()
    return conn

def get_corpus_words(session, corpus_name: str) -> Set[str]:
    """Get all unique words in the corpus."""
    query = text("""
        SELECT DISTINCT word
        FROM pipeline.vocabulary_word
        WHERE corpus_name = :corpus_name
    """)
    result = session.execute(query, {"corpus_name": corpus_name}).fetchall()
    return {row.word for row in result}

def get_corpus_documents(session, corpus_name: str) -> List[str]:
    """Get all document contents for the corpus."""
    query = text("""
        SELECT content
        FROM pipeline.preprocessed_document
        WHERE corpus_name = :corpus_name
    """)
    result = session.execute(query, {"corpus_name": corpus_name}).fetchall()
    return [row.content for row in result]

def calculate_word_frequencies(documents: List[str], words: Set[str]) -> Dict[str, int]:
    """Calculate word frequencies from documents."""
    word_freqs = defaultdict(int)
    for doc in documents:
        doc_words = set(re.findall(r'\b\w+\b', doc.lower()))
        for word in words:
            if word.lower() in doc_words:
                word_freqs[word] += 1
    return dict(word_freqs)

def calculate_co_occurrences(documents: List[str], words: Set[str]) -> Dict[Tuple[str, str], int]:
    """Calculate word co-occurrences from documents."""
    co_occurrences = defaultdict(int)
    words_lower = {w.lower(): w for w in words}  # Map lowercase to original case
    
    for doc in documents:
        # Get all words in document and convert to lowercase
        doc_words = set(re.findall(r'\b\w+\b', doc.lower()))
        # Only keep words that are in our vocabulary
        doc_words = {w for w in doc_words if w in words_lower}
        
        # Convert back to original case for consistency
        doc_words = {words_lower[w] for w in doc_words}
        
        # Calculate co-occurrences
        for word1 in doc_words:
            for word2 in doc_words:
                if word1 < word2:  # Only count each pair once
                    co_occurrences[(word1, word2)] += 1
    
    return dict(co_occurrences)

def cache_corpus_stats(session, corpus_name: str):
    """Calculate and cache word frequencies and co-occurrences for a corpus."""
    conn = init_cache_db()
    
    # Check if corpus is already cached
    cached = conn.execute("""
        SELECT COUNT(*) as count
        FROM word_frequencies
        WHERE corpus_name = ?
    """, [corpus_name]).fetchone()[0] > 0
    
    if cached:
        conn.close()
        return
    
    # Get corpus data
    words = get_corpus_words(session, corpus_name)
    documents = get_corpus_documents(session, corpus_name)
    
    # Calculate statistics
    word_freqs = calculate_word_frequencies(documents, words)
    co_occurrences = calculate_co_occurrences(documents, words)
    
    # Cache word frequencies
    conn.executemany("""
        INSERT INTO word_frequencies (corpus_name, word, frequency)
        VALUES (?, ?, ?)
    """, [(corpus_name, word, freq) for word, freq in word_freqs.items()])
    
    # Cache co-occurrences
    conn.executemany("""
        INSERT INTO word_co_occurrences (corpus_name, word1, word2, co_occurrence)
        VALUES (?, ?, ?, ?)
    """, [(corpus_name, word1, word2, count) 
          for (word1, word2), count in co_occurrences.items()])
    
    conn.commit()
    conn.close()

def get_word_frequencies(corpus_name: str) -> Dict[str, int]:
    """Get word frequencies from cache."""
    conn = init_cache_db()
    result = conn.execute("""
        SELECT word, frequency
        FROM word_frequencies
        WHERE corpus_name = ?
    """, [corpus_name]).fetchall()
    conn.close()
    return {row[0]: row[1] for row in result}

def get_co_occurrence_counts(corpus_name: str, word_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
    """Get co-occurrence counts from cache."""
    conn = init_cache_db()
    co_occurrences = {}
    
    for word1, word2 in word_pairs:
        if word1 > word2:  # Ensure consistent ordering
            word1, word2 = word2, word1
        
        result = conn.execute("""
            SELECT co_occurrence
            FROM word_co_occurrences
            WHERE corpus_name = ? AND word1 = ? AND word2 = ?
        """, [corpus_name, word1, word2]).fetchone()
        
        co_occurrences[(word1, word2)] = result[0] if result else 0
    
    conn.close()
    return co_occurrences

def calculate_npmi(
                  word1_freq: int, word2_freq: int, 
                  co_occurrence: int, total_docs: int) -> float:
    """Calculate NPMI for a word pair."""
    if co_occurrence == 0:
        return -1.0
    
    p1 = word1_freq / total_docs
    p2 = word2_freq / total_docs
    p12 = co_occurrence / total_docs
    
    pmi = np.log(p12 / (p1 * p2))
    npmi = pmi / -np.log(p12)
    
    return float(npmi)

def get_word_frequencies_from_db(session, corpus_name: str) -> Dict[str, int]:
    """Get word frequencies directly from database."""
    documents = get_corpus_documents(session, corpus_name)
    words = get_corpus_words(session, corpus_name)
    return calculate_word_frequencies(documents, words)

def get_co_occurrence_counts_from_db(session, corpus_name: str, word_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
    """Get co-occurrence counts directly from database."""
    documents = get_corpus_documents(session, corpus_name)
    words = {word for pair in word_pairs for word in pair}
    co_occurrences = calculate_co_occurrences(documents, words)
    
    # Only return the requested word pairs
    return {pair: co_occurrences.get(pair, 0) for pair in word_pairs}

def get_all_word_pairs(words: Set[str]) -> List[Tuple[str, str]]:
    """Generate all possible word pairs from a set of words."""
    word_list = sorted(list(words))  # Sort for consistent ordering
    return [(w1, w2) for i, w1 in enumerate(word_list) for w2 in word_list[i+1:]]

def calculate_topic_coherence(
    session, 
    corpus_name: str, 
    topic: Dict[str, Any], 
    word_freqs: Dict[str, int],
    co_occurrences: Dict[Tuple[str, str], int],
    total_docs: int,
    top_n: int = 10
) -> float:
    """Calculate NPMI-based topic coherence for a topic.
    
    Args:
        session: SQLAlchemy session
        corpus_name: Name of the corpus
        topic: Topic dictionary containing 'words' list of (word, score) tuples
        word_freqs: Dictionary of word frequencies for the corpus
        co_occurrences: Dictionary of co-occurrence counts for word pairs
        total_docs: Total number of documents in the corpus
        top_n: Number of top words to consider
    
    Returns:
        float: NPMI coherence score for the topic
    """
    # Get topic words
    topic_words = [word for word, _ in topic.get('words', [])[:top_n]]
    if len(topic_words) < 2:
        return 0.0
    
    # Generate word pairs
    word_pairs = [(w1, w2) for i, w1 in enumerate(topic_words) 
                 for w2 in topic_words[i+1:]]
    
    # Calculate NPMI for each pair
    npmi_values = []
    for word1, word2 in word_pairs:
        if word1 in word_freqs and word2 in word_freqs:
            # Ensure consistent ordering for co-occurrence lookup
            if word1 > word2:
                word1, word2 = word2, word1
                
            npmi = calculate_npmi(
                word1, word2,
                word_freqs[word1],
                word_freqs[word2],
                co_occurrences.get((word1, word2), 0),
                total_docs
            )
            npmi_values.append(npmi)
    
    return float(np.mean(npmi_values)) if npmi_values else 0.0

def calculate_corpus_npmi(session, topics: List[Dict[str, Any]], corpus_name: str, use_cache: bool = False) -> float:
    """Calculate average NPMI across all topics for a list of topics.
    
    Args:
        session: SQLAlchemy session
        topics: List of topic dictionaries, each containing 'words' list of (word, score) tuples
        corpus_name: Name of the corpus to calculate NPMI for
        use_cache: Whether to use cached statistics or calculate from database
    
    Returns:
        float: Average NPMI score across all topics
    """
    # Get total number of documents
    query = text("""
        SELECT COUNT(*) as total
        FROM pipeline.preprocessed_document
        WHERE corpus_name = :corpus_name
    """)
    result = session.execute(query, {"corpus_name": corpus_name}).fetchone()
    total_docs = result.total
    
    # Get all unique words from all topics
    all_topic_words = set()
    for topic in topics:
        topic_words = [word for word, _ in topic.get('words', [])]
        all_topic_words.update(topic_words)
    
    # Get word frequencies once for all topics
    if use_cache:
        cache_corpus_stats(session, corpus_name)
        word_freqs = get_word_frequencies(corpus_name)
    else:
        word_freqs = get_word_frequencies_from_db(session, corpus_name)
    
    # Get all possible word pairs and their co-occurrences
    all_word_pairs = get_all_word_pairs(all_topic_words)
    if use_cache:
        co_occurrences = get_co_occurrence_counts(corpus_name, all_word_pairs)
    else:
        co_occurrences = get_co_occurrence_counts_from_db(session, corpus_name, all_word_pairs)
    
    # Calculate NPMI for each topic
    topic_coherences = []
    for topic in topics:
        coherence = calculate_topic_coherence(
            session, 
            corpus_name, 
            topic, 
            word_freqs,
            co_occurrences,
            total_docs
        )
        topic_coherences.append(coherence)
    
    # Return average coherence across all topics
    return float(np.mean(topic_coherences)) if topic_coherences else 0.0

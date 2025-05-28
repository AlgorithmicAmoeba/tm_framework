import numpy as np
from typing import List, Dict, Any, Set, Tuple
from sqlalchemy import text
from sqlalchemy.orm import Session # Assuming Session is imported for type hinting
import json
import re
from pathlib import Path
import scipy.sparse

# --- Constants and Configuration ---
CACHE_DIR = Path("ignore/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True) # Ensure cache directory exists
CACHE_FILE_TEMPLATE = CACHE_DIR / "{corpus_name}_stats.json"

# --- Helper for JSON keying (if complex keys were needed) ---
# For co-occurrences, we'll use "word1_word2" string keys.

# --- Core Data Fetching from SQL (Original Functions) ---

def get_corpus_words(session: Session, corpus_name: str) -> Set[str]:
    """Get all unique words in the corpus from the database."""
    query = text("""
        SELECT DISTINCT word
        FROM pipeline.vocabulary_word
        WHERE corpus_name = :corpus_name
    """)
    result = session.execute(query, {"corpus_name": corpus_name}).fetchall()
    return {row[0] for row in result} # Assuming 'word' is the first column

def get_corpus_documents(session: Session, corpus_name: str) -> List[str]:
    """Get all document contents for the corpus from the database."""
    query = text("""
        SELECT content
        FROM pipeline.preprocessed_document
        WHERE corpus_name = :corpus_name
    """)
    result = session.execute(query, {"corpus_name": corpus_name}).fetchall()
    return [row[0] for row in result] # Assuming 'content' is the first column

# --- Caching Logic with Sparse Matrices and JSON ---

def _co_occurrence_key(word1: str, word2: str) -> str:
    """Generates a consistent string key for a word pair for JSON storage."""
    return "_".join(sorted((word1, word2)))

def _parse_co_occurrence_key(key_str: str) -> Tuple[str, str]:
    """Parses a string key back into a word pair tuple."""
    parts = key_str.split('_', 1)
    return (parts[0], parts[1])

def _compute_and_cache_stats(session: Session, corpus_name: str) -> Dict[str, Any]:
    """
    Computes word frequencies and co-occurrences using sparse matrices
    and caches them to a JSON file.
    """
    print(f"Computing statistics for corpus: {corpus_name}...")

    # 1. Fetch data from the database
    corpus_vocab_set = get_corpus_words(session, corpus_name)
    documents_content = get_corpus_documents(session, corpus_name)

    if not documents_content or not corpus_vocab_set:
        print(f"Warning: No documents or vocabulary found for corpus {corpus_name}. Caching empty stats.")
        empty_stats = {
            "corpus_name": corpus_name,
            "total_docs": 0,
            "vocabulary": [],
            "word_frequencies": {},
            "co_occurrences": {}
        }
        cache_file_path = Path(str(CACHE_FILE_TEMPLATE).format(corpus_name=corpus_name))
        with open(cache_file_path, 'w') as f:
            json.dump(empty_stats, f, indent=2)
        return empty_stats

    vocab_list = sorted(list(corpus_vocab_set))
    word_to_idx = {word: i for i, word in enumerate(vocab_list)}
    
    num_docs = len(documents_content)
    num_vocab = len(vocab_list)

    # 2. Build document-term sparse matrix (binary: word presence)
    # lil_matrix is efficient for incremental construction
    doc_term_matrix = scipy.sparse.lil_matrix((num_docs, num_vocab), dtype=np.int64)
    for doc_idx, doc_content in enumerate(documents_content):
        # Simple tokenization, consistent with original code
        doc_words = set(re.findall(r'\b\w+\b', doc_content.lower()))
        for word in doc_words:
            if word in word_to_idx: # Only consider words in our corpus_vocab_set
                doc_term_matrix[doc_idx, word_to_idx[word]] = 1
    
    doc_term_matrix_csr = doc_term_matrix.tocsr() # Convert to CSR for efficient calculations

    # 3. Calculate word frequencies (document frequencies)
    # Summing columns of the doc-term matrix gives doc frequency for each word
    word_doc_counts = np.array(doc_term_matrix_csr.sum(axis=0)).flatten()
    word_frequencies = {vocab_list[i]: int(word_doc_counts[i]) for i in range(num_vocab) if word_doc_counts[i] > 0}

    # 4. Calculate co-occurrence matrix
    # M.T @ M gives a (vocab x vocab) matrix where C[i,j] is co-occurrence of word i and word j
    co_occurrence_matrix_sparse = (doc_term_matrix_csr.T @ doc_term_matrix_csr)
    
    co_occurrences_dict = {}
    # Iterate through non-zero elements of the upper triangle of the co-occurrence matrix
    # to avoid duplicates and self-co-occurrences for this dictionary.
    co_occurrence_matrix_coo = co_occurrence_matrix_sparse.tocoo()
    for word_idx1, word_idx2, count in zip(co_occurrence_matrix_coo.row, co_occurrence_matrix_coo.col, co_occurrence_matrix_coo.data):
        if word_idx1 < word_idx2: # Ensure we only take one side of the symmetric matrix
            original_word1 = vocab_list[word_idx1]
            original_word2 = vocab_list[word_idx2]
            key = _co_occurrence_key(original_word1, original_word2)
            co_occurrences_dict[key] = int(count)

    # 5. Prepare data for JSON cache
    stats_data = {
        "corpus_name": corpus_name,
        "total_docs": num_docs,
        "vocabulary": vocab_list, # Storing vocab list for potential future use
        "word_frequencies": word_frequencies,
        "co_occurrences": co_occurrences_dict # Stored with string keys
    }

    # 6. Save to JSON cache file
    cache_file_path = Path(str(CACHE_FILE_TEMPLATE).format(corpus_name=corpus_name))
    with open(cache_file_path, 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    print(f"Successfully computed and cached statistics for {corpus_name} to {cache_file_path}")
    return stats_data

def get_corpus_stats(session: Session, corpus_name: str, force_recompute: bool = False) -> Dict[str, Any]:
    """
    Retrieves corpus statistics (total_docs, word_frequencies, co_occurrences)
    from cache if available, otherwise computes and caches them.

    Returns:
        A dictionary containing:
        - 'total_docs': int
        - 'word_frequencies': Dict[str, int]
        - 'co_occurrences': Dict[Tuple[str, str], int] (Note: keys are tuples here for usage)
        - 'vocabulary': List[str]
    """
    cache_file_path = Path(str(CACHE_FILE_TEMPLATE).format(corpus_name=corpus_name))
    
    if not force_recompute and cache_file_path.exists():
        print(f"Loading statistics for {corpus_name} from cache: {cache_file_path}")
        try:
            with open(cache_file_path, 'r') as f:
                cached_data = json.load(f)
            
            # Convert co-occurrence keys back to tuples
            parsed_co_occurrences = {
                _parse_co_occurrence_key(key_str): count 
                for key_str, count in cached_data.get("co_occurrences", {}).items()
            }
            cached_data["co_occurrences"] = parsed_co_occurrences
            return cached_data
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {cache_file_path}. Recomputing...")
            # Fall through to recompute
        except Exception as e:
            print(f"Error loading cache {cache_file_path}: {e}. Recomputing...")
            # Fall through to recompute

    # If cache doesn't exist, is invalid, or recompute is forced
    computed_data = _compute_and_cache_stats(session, corpus_name)
    # Convert co-occurrence keys to tuples for the returned dict
    parsed_co_occurrences_computed = {
        _parse_co_occurrence_key(key_str): count
        for key_str, count in computed_data.get("co_occurrences", {}).items()
    }
    computed_data["co_occurrences"] = parsed_co_occurrences_computed
    return computed_data

# --- NPMI and Topic Coherence Calculation (Largely Original Logic, Adapted for New Data Source) ---

def calculate_npmi(word1_freq: int, word2_freq: int, 
                   co_occurrence: int, total_docs: int) -> float:
    """Calculate NPMI for a word pair."""
    if co_occurrence == 0 or total_docs == 0 or word1_freq == 0 or word2_freq == 0:
        return -1.0  # Standard value for non-co-occurring or non-existent words
    
    # Probabilities
    # p(word1) = docs_containing_word1 / total_docs
    p1 = word1_freq / total_docs
    # p(word2) = docs_containing_word2 / total_docs
    p2 = word2_freq / total_docs
    # p(word1, word2) = docs_containing_both_word1_and_word2 / total_docs
    p12 = co_occurrence / total_docs

    if p12 == 0.0: # Should be caught by co_occurrence == 0, but as safeguard
        return -1.0
    if p1 * p2 == 0.0: # Should be caught by wordX_freq == 0
        return -1.0

    # PMI: log[ p(w1,w2) / (p(w1) * p(w2)) ]
    pmi = np.log(p12 / (p1 * p2))
    
    # NPMI: PMI / -log[ p(w1,w2) ]
    # Denominator -log(p12) can be 0 if p12 = 1 (pair occurs in all docs).
    # Max value of pmi is -log(p12) when p1=p12 and p2=p12 (perfect correlation)
    # Min value of pmi is negative infinity.
    # NPMI normalizes PMI to [-1, 1]
    denominator = -np.log(p12)
    if denominator == 0: # This happens if p12 = 1.0 (co-occur in every document)
        # If p1=1, p2=1, p12=1, then pmi = log(1/(1*1)) = 0. npmi = 0/0, should be 1 (max coherence)
        # If p1 < 1 or p2 < 1, but p12 = 1 (implies p1=1, p2=1), this case is fine.
        # If p12 is 1, then p1 must be 1 and p2 must be 1.
        # So word1_freq = total_docs, word2_freq = total_docs, co_occurrence = total_docs
        # pmi = log( (total_docs/total_docs) / ((total_docs/total_docs) * (total_docs/total_docs)) ) = log(1) = 0
        # -log(p12) = -log(1) = 0.  NPMI = 0/0.
        # In this specific case (perfect co-occurrence in all docs), NPMI should be 1.
        return 1.0 if pmi == 0 else 0.0 # if pmi is not 0, something is off, default to 0.
                                        # but with p12=1, pmi must be 0.

    npmi = pmi / denominator
    
    # Clamp result to [-1, 1] due to potential floating point inaccuracies
    return float(np.clip(npmi, -1.0, 1.0))


def calculate_topic_coherence(
    topic_words: List[str],
    word_freqs: Dict[str, int],
    co_occurrences: Dict[Tuple[str, str], int],
    total_docs: int,
    top_n: int = 10
) -> float:
    """
    Calculate NPMI-based topic coherence for a list of topic words.
    
    Args:
        topic_words: List of words for the topic.
        word_freqs: Dictionary of word frequencies for the corpus.
        co_occurrences: Dictionary of co-occurrence counts for word pairs (tuple keys).
        total_docs: Total number of documents in the corpus.
        top_n: Number of top words from topic_words to consider.
    
    Returns:
        float: NPMI coherence score for the topic.
    """
    # Get top N topic words
    topic_words = topic_words[:top_n]
    
    if len(topic_words) < 2:
        return 0.0  # Cannot calculate coherence with fewer than 2 words
    
    # Generate unique word pairs from the top_n topic words
    word_pairs = []
    for i, w1 in enumerate(topic_words):
        for j in range(i + 1, len(topic_words)):
            w2 = topic_words[j]
            word_pairs.append(tuple(sorted((w1, w2)))) # Ensure consistent ordering for lookup

    npmi_values = []
    for word1, word2 in word_pairs: # word1, word2 are already sorted here
        freq1 = word_freqs.get(word1)
        freq2 = word_freqs.get(word2)
        
        if freq1 is not None and freq2 is not None:
            # Key for co_occurrences dict is already (sorted_word1, sorted_word2)
            co_occurrence_count = co_occurrences.get((word1, word2), 0)
            
            npmi = calculate_npmi(
                freq1,
                freq2,
                co_occurrence_count,
                total_docs
            )
            npmi_values.append(npmi)
        # else:
            # Word not in corpus vocabulary, or frequency is zero.
            # calculate_npmi handles freq=0 by returning -1.0 if called.
            # Here, we simply skip pairs if a word isn't in word_freqs (e.g. from a different corpus)
            # or if we want to be strict about only using words with known frequencies.
            # For topic coherence, typically words are from the corpus.
            # If a word from topic_words is not in word_freqs, it means it had 0 occurrences.
            # The calculate_npmi function would return -1.0 if freq is 0.
            # To be explicit:
            # if freq1 == 0 or freq2 == 0:
            #    npmi_values.append(-1.0) # Or handle as per desired coherence metric behavior

    return float(np.mean(npmi_values)) if npmi_values else 0.0


def calculate_corpus_npmi_coherence(
    session: Session, 
    topics: List[List[str]],
    corpus_name: str,
    top_n_words_per_topic: int = 10,
    force_recompute_stats: bool = False
) -> float:
    """
    Calculate average NPMI coherence across all topics for a given corpus.
    
    Args:
        session: SQLAlchemy session for database access if stats need recomputing.
        topics_words: List of lists of words for each topic.
        corpus_name: Name of the corpus to calculate NPMI for.
        top_n_words_per_topic: Number of top words from each topic to use for coherence.
        force_recompute_stats: Whether to force re-computation of corpus statistics.
    
    Returns:
        float: Average NPMI score across all topics. Returns 0.0 if no topics or no valid coherences.
    """
    if not topics:
        return 0.0

    # Get corpus statistics (total docs, word frequencies, co-occurrences)
    # This will either load from cache or compute and cache them.
    corpus_stats = get_corpus_stats(session, corpus_name, force_recompute=force_recompute_stats)
    
    total_docs = corpus_stats.get("total_docs", 0)
    word_freqs = corpus_stats.get("word_frequencies", {})
    co_occurrences = corpus_stats.get("co_occurrences", {}) # Expects tuple keys

    if total_docs == 0:
        print(f"Warning: Total documents is 0 for corpus {corpus_name}. NPMI will be based on this.")
        # Coherence will likely be 0 or -1 for all topics if total_docs is 0 and freqs are 0.

    topic_coherence_scores = []
    for i, topic_words in enumerate(topics):
        if not topic_words:
            print(f"Warning: Topic {i} has no words. Skipping coherence calculation for this topic.")
            continue

        coherence = calculate_topic_coherence(
            topic_words,
            word_freqs,
            co_occurrences,
            total_docs,
            top_n=top_n_words_per_topic
        )
        topic_coherence_scores.append(coherence)
        # print(f"Coherence for Topic {i}: {coherence:.4f}") # Optional: for debugging
    
    if not topic_coherence_scores:
        print("No topic coherences were calculated.")
        return 0.0
        
    average_coherence = float(np.mean(topic_coherence_scores))
    print(f"Average NPMI Coherence for corpus '{corpus_name}' across {len(topic_coherence_scores)} topics: {average_coherence:.4f}")
    return average_coherence
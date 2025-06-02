import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session

# --- Helper: Fetch word embeddings for a list of words ---
def get_word_embeddings(session: Session, corpus_name: str) -> Dict[str, np.ndarray]:
    """
    Fetch word embeddings for the given words from the database.
    Returns a dict mapping word -> embedding (as np.ndarray).
    """
    query = text("""
        SELECT w.word, e.vector
        FROM pipeline.vocabulary_word_embeddings e
        JOIN pipeline.vocabulary_word w ON w.id = e.vocabulary_word_id
        WHERE w.corpus_name = :corpus_name
    """)
    result = session.execute(query, {"corpus_name": corpus_name})
    embeddings = {}
    for row in result:
        word, vector = row
        embeddings[word] = np.array(vector, dtype=np.float32)
    return embeddings

# --- Helper: Cosine similarity ---
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

# --- WEPS metric for a pair of topics ---
def calculate_weps_for_topic_pair(
    embeddings: Dict[str, np.ndarray],
    topic_i: List[str],
    topic_j: List[str],
    corpus_name: str,
    top_n: int = 10
) -> float:
    """
    Compute WEPS between two topics (lists of words) as per the formula:
    WEPS(ti, tj) = (1/t^2) * sum_{v in ti} sum_{u in tj} sim(wv, wu)
    where sim is cosine similarity between word embeddings.
    Only considers top_n words from each topic.
    """
    topic_i = topic_i[:top_n]
    topic_j = topic_j[:top_n]
    t = len(topic_i)
    if t == 0 or len(topic_j) == 0:
        return 0.0
    # Get all unique words needed
    all_words = list(set(topic_i) | set(topic_j))
    # Compute sum of similarities
    sim_sum = 0.0
    for wv in topic_i:
        vec_v = embeddings.get(wv)
        for wu in topic_j:
            vec_u = embeddings.get(wu)
            sim_sum += cosine_similarity(vec_v, vec_u)
    return sim_sum / (t * t)

# --- WEPS for a list of topics (average pairwise WEPS) ---
def calculate_corpus_weps(
    session: Session,
    topics: List[List[str]],
    corpus_name: str,
    top_n_words_per_topic: int = 10,
    embeddings: Dict[str, np.ndarray] = None
) -> float:
    """
    Compute average WEPS across all unique topic pairs in the list.
    """
    if not topics or len(topics) < 2:
        return 0.0
    
    if embeddings is None:
        embeddings = get_word_embeddings(session, corpus_name)
    
    n = len(topics)
    weps_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            score = calculate_weps_for_topic_pair(
                embeddings,
                topics[i],
                topics[j],
                corpus_name,
                top_n=top_n_words_per_topic
            )
            weps_scores.append(score)
    return float(np.mean(weps_scores)) if weps_scores else 0.0


def calculate_multiple_topic_models_weps(
    session: Session,
    topic_models_outputs: List[List[List[str]]],
    corpus_name: str,
    top_n_words_per_topic: int = 10,
    force_recompute_stats: bool = False
) -> List[float]:
    """
    Compute WEPS for multiple topic models.
    """
    embeddings = get_word_embeddings(session, corpus_name)
    weps_scores = []
    for topic_model_output in topic_models_outputs:
        weps_scores.append(calculate_corpus_weps(session, topic_model_output, corpus_name, top_n_words_per_topic, embeddings=embeddings))
    return weps_scores
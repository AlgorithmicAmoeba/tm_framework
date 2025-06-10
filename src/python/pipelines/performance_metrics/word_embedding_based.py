import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
import random

import tqdm

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

    embeddings[''] = np.zeros(300)
    return embeddings


# --- Helper: similarity ---
def similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return - np.dot(vec1, vec2) 

# --- WEPS metric for a pair of topics ---
def calculate_weps_for_topic_pair_slow(
    embeddings: Dict[str, np.ndarray],
    topic_i: List[str],
    topic_j: List[str],
    corpus_name: str,  # corpus_name is unused but kept for interface consistency
    top_n: int = 10
) -> float:
    """
    Compute WEPS between two topics using a vectorized approach.
    WEPS(ti, tj) = (1/t^2) * sum_{v in ti} sum_{u in tj} sim(wv, wu)
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
            sim_sum += similarity(vec_v, vec_u)

    res = sim_sum / (t * t)

    return res

# --- WEPS for a list of topics (average pairwise WEPS) ---
def calculate_corpus_weps_slow(
    session: Session,
    topics: List[List[str]],
    corpus_name: str,
    top_n_words_per_topic: int = 10,
    embeddings: Dict[str, np.ndarray] | None = None
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
            score = calculate_weps_for_topic_pair_slow(
                embeddings,
                topics[i],
                topics[j],
                corpus_name,
                top_n=top_n_words_per_topic
            )
            weps_scores.append(score)

    res = float(np.mean(weps_scores)) if weps_scores else 0.0
    return res


def calculate_corpus_weps(
    topics: List[List[str]],
    embeddings: Dict[str, np.ndarray],
) -> float:

    # Create tensor (topics, words, embeddings)
    
    topics_tensor = np.array([[embeddings[w] for w in topic] for topic in topics])

    v_1 = np.einsum('twe,sve->', topics_tensor, topics_tensor)  # TODO: should be 'twe,tve->' then is is COH / WECS
    v_2 = np.einsum('twe,twe->', topics_tensor, topics_tensor)  # TODO: for WEPS this is not needed
    
    w = len(topics[0])
    t = len(topics)
    res = -(v_1 - v_2) / (w * w) / (t * t)  # TODO: might need to be -(v1 - v2) / (w * (w-1)) / t
    
    return res


def calculate_multiple_topic_models_weps(
    session: Session,
    topic_models_outputs: List[List[List[str]]],
    corpus_name: str,
) -> List[dict[str, Any]]:
    """
    Compute WEPS for multiple topic models.
    """
    embeddings = get_word_embeddings(session, corpus_name)
    weps_scores = []
    for topic_model_output in tqdm.tqdm(topic_models_outputs):
        weps_scores.append(calculate_corpus_weps(topic_model_output, embeddings))
    return [{"score": score} for score in weps_scores]

# --- WECS: Word Embedding-based Centroid Similarity ---
def topic_centroid(topic_words: List[str], embeddings: Dict[str, np.ndarray], top_n: int = 10) -> Optional[np.ndarray]:
    """
    Compute the centroid (mean vector) of the embeddings for the given topic words.
    Only considers words present in the embeddings dict, up to top_n words.
    Returns None if no embeddings are found for the topic.
    """
    topic_words = topic_words[:top_n]
    vectors = [embeddings[w] for w in topic_words if w in embeddings]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def calculate_wecs_for_topic_pair(
    embeddings: Dict[str, np.ndarray],
    topic_i: List[str],
    topic_j: List[str],
    top_n: int = 10
) -> float:
    """
    Compute WECS between two topics as cosine similarity between their centroids.
    """
    centroid_i = topic_centroid(topic_i, embeddings, top_n=top_n)
    centroid_j = topic_centroid(topic_j, embeddings, top_n=top_n)
    if centroid_i is None or centroid_j is None:
        return 0.0
    
    res = similarity(centroid_i, centroid_j)
    return res

def calculate_wecs_for_topic_pair_fast(
    embeddings: Dict[str, np.ndarray],
    topic_i: List[str],
    topic_j: List[str],
) -> float:
    """
    Compute WECS between two topics as cosine similarity between their centroids.
    """
    topic_i_tensor = np.array([embeddings[w] for w in topic_i])
    topic_j_tensor = np.array([embeddings[w] for w in topic_j])
    
    v_1 = np.einsum('we,ve->', topic_i_tensor, topic_j_tensor)
    w = len(topic_i)
    res = v_1 / (w * w)
    return res

def calculate_corpus_wecs_fast(
    embeddings: Dict[str, np.ndarray],
    topics: List[List[str]],
) -> float:
    """
    Compute average WECS across all unique topic pairs in the list.
    """
    topics_tensor = np.array([[embeddings[w] for w in topic] for topic in topics])
    v_1 = np.einsum('twe,sve->', topics_tensor, topics_tensor)
    v_2 = np.einsum('twe,twe->', topics_tensor, topics_tensor)
    w = len(topics[0])
    t = len(topics)
    res = -(v_1 - v_2) / (w * w) / (t * t)
    return res
    
def calculate_corpus_wecs(
    session: Session,
    topics: List[List[str]],
    corpus_name: str,
    top_n_words_per_topic: int = 10,
    embeddings: Dict[str, np.ndarray] = None
) -> float:
    """
    Compute average WECS across all unique topic pairs in the list.
    """
    if not topics or len(topics) < 2:
        return 0.0
    if embeddings is None:
        embeddings = get_word_embeddings(session, corpus_name)
    n = len(topics)
    wecs_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            score = calculate_wecs_for_topic_pair(
                embeddings,
                topics[i],
                topics[j],
                top_n=top_n_words_per_topic
            )
            wecs_scores.append(score)
    return float(np.mean(wecs_scores)) if wecs_scores else 0.0

def calculate_multiple_topic_models_wecs(
    session: Session,
    topic_models_outputs: List[List[List[str]]],
    corpus_name: str,
    top_n_words_per_topic: int = 10,
) -> List[dict[str, Any]]:
    """
    Compute WECS for multiple topic models.
    """
    embeddings = get_word_embeddings(session, corpus_name)
    wecs_scores = []
    for topic_model_output in topic_models_outputs:
        wecs_scores.append(calculate_corpus_wecs(session, topic_model_output, corpus_name, top_n_words_per_topic, embeddings=embeddings))
    return [{"score": score} for score in wecs_scores]

def calculate_intruder_shift_slow(
    session: Session,
    topics: List[List[str]],
    corpus_name: str,
    top_n_words_per_topic: int = 10,
    n_repeats: int = 1000,
    embeddings: Dict[str, np.ndarray] = None,
    random_seed: int = 42
) -> float:
    """
    Compute the Intruder Shift (ISH) metric for a list of topics.
    For each topic, repeatedly replace a random word with a random word from another topic,
    compute the centroid shift, and average the cosine similarity between original and shifted centroids.
    Lower ISH means more coherent/diverse topics.
    """
    # intruder_shift_equality(embeddings, topics)
    if not topics or len(topics) < 2:
        return 0.0
    if embeddings is None:
        embeddings = get_word_embeddings(session, corpus_name)
    
    random.seed(random_seed)
    K = len(topics)
    sim_sums = [0.0 for _ in range(K)]
    sim_counts = [0 for _ in range(K)]
    for i, topic in enumerate(topics):
        topic_words = topic[:top_n_words_per_topic]
        if len(topic_words) < 2:
            continue
        
        orig_centroid = topic_centroid(topic_words, embeddings, top_n=top_n_words_per_topic)
        other_topic_indices = [j for j in range(K) if j != i and len(topics[j][:top_n_words_per_topic]) > 0]
        if orig_centroid is None or not other_topic_indices:
            continue
        for _ in range(n_repeats): 
            # Pick a random word index in this topic
            idx_in_topic = random.randint(0, len(topic_words) - 1)
            # Pick a random other topic
            j = random.choice(other_topic_indices)
            other_topic_words = topics[j][:top_n_words_per_topic]
            # Pick a random word from the other topic
            intruder_word = random.choice(other_topic_words)
            # Replace the word in topic i with the intruder
            shifted_words = list(topic_words)
            shifted_words[idx_in_topic] = intruder_word
            # Compute centroids
            shifted_centroid = topic_centroid(shifted_words, embeddings, top_n=top_n_words_per_topic)
            if shifted_centroid is None:
                continue
            sim = similarity(orig_centroid, shifted_centroid)
            sim_sums[i] += sim
            sim_counts[i] += 1
    # Average over repeats for each topic, then over topics
    topic_ish = [sim_sums[i] / sim_counts[i] if sim_counts[i] > 0 else 0.0 for i in range(K)]
    return float(np.mean(topic_ish)) if topic_ish else 0.0

def calculate_intruder_shift(
    embeddings: Dict[str, np.ndarray],
    topics: List[List[str]],
    n_repeats: int = 1000,
    random_seed: int = 42,
) -> float:
    """
    Compute the Intruder Shift (ISH) metric for a list of topics.
    """
    np.random.seed(random_seed)

    t = len(topics)
    w = len(topics[0])

    random_topic_indices = np.random.randint(0, t, size=(t, n_repeats))
    random_word_indices_in = np.random.randint(0, w, size=(t, n_repeats))
    random_word_indices_out = np.random.randint(0, w, size=(t, n_repeats))

    topic_word_tensors = np.array([[embeddings[w] for w in topic] for topic in topics])

    topic_scores = []
    for i in range(t):
        tw_i = topic_word_tensors[i]

        # create a tensor of shape (n_repeats, w, e) by tiling tw_i n_repeats times
        tw_i_shifted = np.repeat(tw_i[np.newaxis, :, :], n_repeats, axis=0)

        tw_i_shifted[np.arange(n_repeats), random_word_indices_in[i]] = topic_word_tensors[random_topic_indices[i, :], random_word_indices_out[i]]

        v_1 = np.einsum('we, tve->', tw_i, tw_i_shifted)
        res = -v_1 / (w * w) / n_repeats
        topic_scores.append(res)

    res = float(np.mean(topic_scores)) if topic_scores else 0.0
    return res


def intruder_shift_equality(
    embeddings: Dict[str, np.ndarray],
    topics: List[List[str]],
    n_repeats: int = 15,
    random_seed: int = 42,
) -> float:
    """
    Compute the Intruder Shift Equality (ISE) metric for a list of topics.
    """
    np.random.seed(random_seed)

    t = len(topics)
    w = len(topics[0])

    random_topic_indices = np.random.randint(0, t, size=(n_repeats))
    random_word_indices_in = np.random.randint(0, w, size=(n_repeats))
    random_word_indices_out = np.random.randint(0, w, size=(n_repeats))

    topic_word_tensors = np.array([[embeddings[w] for w in topic] for topic in topics])

    tw = topic_word_tensors[0]

    # create a tensor of shape (n_repeats, w, e) by tiling tw_i n_repeats times
    tw_shifted = np.repeat(tw[np.newaxis, :, :], n_repeats, axis=0)

    tw_shifted[np.arange(n_repeats), random_word_indices_in] = topic_word_tensors[random_topic_indices, random_word_indices_out]

    v_1 = np.einsum('we, tve->', tw, tw_shifted)
    res_1 = -v_1 / (w * w) / n_repeats

    # Now calculate the value using by doing the replacement in a loop
    res_2 = 0.0
    t_0 = topic_centroid(topics[0], embeddings)
    for i in range(n_repeats):
        tw_shifted_i = np.copy(tw)
        tw_shifted_i[random_word_indices_in[i]] = topic_word_tensors[random_topic_indices[i], random_word_indices_out[i]]
        t_shifted = np.mean(tw_shifted_i, axis=0)
        res_2 += similarity(t_0, t_shifted)
        
    res_2 = res_2 / n_repeats

    assert np.isclose(res_1, res_2)
    
    return res_1



def calculate_multiple_topic_models_intruder_shift(
    session: Session,
    topic_models_outputs: List[List[List[str]]],
    corpus_name: str,
    top_n_words_per_topic: int = 10,
    n_repeats: int = 1000,
) -> List[dict[str, Any]]:
    """
    Compute Intruder Shift (ISH) for multiple topic models.
    """
    embeddings = get_word_embeddings(session, corpus_name)
    ish_scores = []
    for topic_model_output in tqdm.tqdm(topic_models_outputs):
        ish_scores.append(calculate_intruder_shift(embeddings, topic_model_output, n_repeats, random_seed=42))

    return [{"score": score} for score in ish_scores]


if __name__ == "__main__":
    # Do 1000 cosine similarity calculations and time them
    import time

    n_vectors = 351064
    vector_dim = 300

    random_vector_1 = np.random.rand(vector_dim)
    random_vector_2 = np.random.rand(vector_dim)

    start_time = time.time()
    for _ in tqdm.tqdm(range(n_vectors)):
        similarity(random_vector_1, random_vector_2)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session
import tqdm


# --- Helper: Fetch BOE word embeddings for a list of words ---
def get_word_embeddings(
    session: Session,
    corpus_name: str,
    source_model_name: str,
    algorithm: str,
    target_dims: int,
    padding_method: str,
) -> dict[str, np.ndarray]:
    """
    Fetch BOE word embeddings for the given corpus/embedding combo from the database.
    Returns a dict mapping word -> embedding (as np.ndarray).
    """
    query = text("""
        SELECT w.word, w.vector
        FROM pipeline.boe_word_embedding w
        WHERE w.corpus_name = :corpus_name
        AND w.source_model_name = :source_model_name
        AND w.algorithm = :algorithm
        AND w.target_dims = :target_dims
        AND w.padding_method = :padding_method
    """)
    result = session.execute(query, {
        "corpus_name": corpus_name,
        "source_model_name": source_model_name,
        "algorithm": algorithm,
        "target_dims": target_dims,
        "padding_method": padding_method,
    })

    embeddings: dict[str, np.ndarray] = {}
    for row in result:
        word, vector = row
        embeddings[word] = np.array(vector, dtype=np.float32)

    # Provide a default vector for empty tokens, consistent with other pipelines.
    embeddings[''] = np.zeros(target_dims, dtype=np.float32)
    return embeddings


# --- Helper: similarity ---
def similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return -np.dot(vec1, vec2)


# --- WEPS for a list of topics (average pairwise WEPS) ---
def calculate_corpus_weps(
    topics: list[list[str]],
    embeddings: dict[str, np.ndarray],
) -> float:
    topics_tensor = np.array([[embeddings[w] for w in topic] for topic in topics])
    t, w, _ = topics_tensor.shape

    v_1 = np.einsum('twe,sve->', topics_tensor, topics_tensor)
    v_2 = np.einsum('twe,tve->', topics_tensor, topics_tensor)
    score = -(v_1 - v_2) / 2
    count = t * (t - 1) * w * w / 2
    res = score / count

    return res


def calculate_multiple_topic_models_weps(
    session: Session,
    topic_models_outputs: list[list[list[str]]],
    corpus_name: str,
    source_model_name: str,
    algorithm: str,
    target_dims: int,
    padding_method: str,
) -> list[dict[str, Any]]:
    """
    Compute WEPS for multiple BOE topic models.
    """
    embeddings = get_word_embeddings(
        session,
        corpus_name,
        source_model_name,
        algorithm,
        target_dims,
        padding_method,
    )
    weps_scores = []
    for topic_model_output in tqdm.tqdm(topic_models_outputs):
        weps_scores.append(calculate_corpus_weps(topic_model_output, embeddings))
    return [{"score": score} for score in weps_scores]


# --- WECS: Word Embedding-based Centroid Similarity ---

def calculate_corpus_wecs(
    embeddings: dict[str, np.ndarray],
    topics: list[list[str]],
) -> float:
    """
    Compute average WECS across all unique topic pairs in the list.
    """
    topics_tensor = np.array([[embeddings[w] for w in topic] for topic in topics])
    t, w, _ = topics_tensor.shape

    v_1 = np.einsum('twe,tve->', topics_tensor, topics_tensor)
    v_2 = np.einsum('twe,twe->', topics_tensor, topics_tensor)
    score = -(v_1 - v_2) / 2
    count = t * w * (w - 1) / 2
    res = score / count

    return res


def calculate_multiple_topic_models_wecs(
    session: Session,
    topic_models_outputs: list[list[list[str]]],
    corpus_name: str,
    source_model_name: str,
    algorithm: str,
    target_dims: int,
    padding_method: str,
) -> list[dict[str, Any]]:
    """
    Compute WECS for multiple BOE topic models.
    """
    embeddings = get_word_embeddings(
        session,
        corpus_name,
        source_model_name,
        algorithm,
        target_dims,
        padding_method,
    )
    wecs_scores = []
    for topic_model_output in tqdm.tqdm(topic_models_outputs):
        wecs_scores.append(calculate_corpus_wecs(embeddings, topic_model_output))
    return [{"score": score} for score in wecs_scores]


def calculate_intruder_shift(
    embeddings: dict[str, np.ndarray],
    topics: list[list[str]],
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

    topic_word_tensors = np.array([[embeddings[word] for word in topic] for topic in topics])

    topic_scores = []
    for i in range(t):
        tw_i = topic_word_tensors[i]

        # create a tensor of shape (n_repeats, w, e) by tiling tw_i n_repeats times
        tw_i_shifted = np.repeat(tw_i[np.newaxis, :, :], n_repeats, axis=0)

        tw_i_shifted[np.arange(n_repeats), random_word_indices_in[i]] = (
            topic_word_tensors[random_topic_indices[i, :], random_word_indices_out[i]]
        )

        v_1 = np.einsum('we, tve->', tw_i, tw_i_shifted)
        res = -v_1 / (w * w) / n_repeats
        topic_scores.append(res)

    res = float(np.mean(topic_scores)) if topic_scores else 0.0
    return res


def calculate_multiple_topic_models_intruder_shift(
    session: Session,
    topic_models_outputs: list[list[list[str]]],
    corpus_name: str,
    source_model_name: str,
    algorithm: str,
    target_dims: int,
    padding_method: str,
    top_n_words_per_topic: int = 10,
    n_repeats: int = 1000,
) -> list[dict[str, Any]]:
    """
    Compute Intruder Shift (ISH) for multiple BOE topic models.
    """
    embeddings = get_word_embeddings(
        session,
        corpus_name,
        source_model_name,
        algorithm,
        target_dims,
        padding_method,
    )
    ish_scores = []
    for topic_model_output in tqdm.tqdm(topic_models_outputs):
        ish_scores.append(calculate_intruder_shift(embeddings, topic_model_output, n_repeats, random_seed=42))

    return [{"score": score} for score in ish_scores]

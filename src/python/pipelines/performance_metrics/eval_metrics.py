import numpy as np


random_seed = 42
np.random.seed(random_seed)


T, W, E = 7, 11, 13


def sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return - np.dot(v1, v2) 


def weps_score_slow(topics_tensor: np.ndarray) -> float:

    score = 0.0
    count = 0
    for t_i in range(T):
        for t_j in range(t_i + 1, T):
            for w_i in range(W):
                for w_j in range(W):
                    score += sim(topics_tensor[t_i, w_i, :], topics_tensor[t_j, w_j, :])
                    count += 1

    return score, count, score / count


def weps_score_fast(topics_tensor: np.ndarray) -> float:
    v_1 = np.einsum('twe,sve->', topics_tensor, topics_tensor)
    v_2 = np.einsum('twe,tve->', topics_tensor, topics_tensor)
    score = -(v_1 - v_2) / 2
    count = T * (T - 1) * W * W / 2
    return score, count, score / count


def weps_equality(topics_tensor: np.ndarray):
    print(weps_score_slow(topics_tensor))
    print(weps_score_fast(topics_tensor))


def wecs_score_slow(topics_tensor: np.ndarray) -> float:
    score = 0.0
    count = 0
    for t in range(T):
        for w_i in range(W):
            for w_j in range(w_i + 1, W):
                score += sim(topics_tensor[t, w_i, :], topics_tensor[t, w_j, :])
                count += 1
    return score, count, score / count


def wecs_score_fast(topics_tensor: np.ndarray) -> float:
    v_1 = np.einsum('twe,tve->', topics_tensor, topics_tensor)
    v_2 = np.einsum('twe,twe->', topics_tensor, topics_tensor)
    score = -(v_1 - v_2) / 2
    count = T * W * (W - 1) / 2
    return score, count, score / count


def wecs_equality(topics_tensor: np.ndarray):
    print(wecs_score_slow(topics_tensor))
    print(wecs_score_fast(topics_tensor))


def topic_centroid(topic_words: list[str], embeddings: dict[str, np.ndarray], top_n: int = 10) -> np.ndarray:
    """
    Compute the centroid (mean vector) of the embeddings for the given topic words.
    Only considers words present in the embeddings dict, up to top_n words.
    Returns None if no embeddings are found for the topic.
    """
    topic_words = topic_words[:top_n]
    vectors = [embeddings[w] for w in topic_words if w in embeddings]
    if not vectors:
        return np.zeros(embeddings[topic_words[0]].shape)
    return np.mean(vectors, axis=0)

def calculate_intruder_shift_slow(
    topics_tensor: np.ndarray,
    n_repeats: int = 1000,
    random_seed: int = 42
) -> float:
    """
    Compute the Intruder Shift (ISH) metric for a tensor of topics.
    For each topic, repeatedly replace a random word with a random word from another topic,
    compute the centroid shift, and average the cosine similarity between original and shifted centroids.
    Lower ISH means more coherent/diverse topics.
    
    Args:
        topics_tensor: Array of shape (T, W, E) where T is number of topics, W is words per topic, E is embedding dim
        n_repeats: Number of times to repeat the intruder replacement for each topic
        random_seed: Random seed for reproducibility
    """
    if topics_tensor.shape[0] < 2:
        return 0.0
    
    np.random.seed(random_seed)
    T, W, E = topics_tensor.shape
    sim_sums = [0.0 for _ in range(T)]
    sim_counts = [0 for _ in range(T)]
    
    for i in range(T):
        orig_centroid = np.mean(topics_tensor[i], axis=0)
        other_topic_indices = [j for j in range(T) if j != i]
        
        for _ in range(n_repeats):
            # Pick a random word index in this topic
            idx_in_topic = np.random.randint(0, W)
            # Pick a random other topic
            j = np.random.choice(other_topic_indices)
            # Pick a random word from the other topic
            idx_in_other = np.random.randint(0, W)
            intruder_word = topics_tensor[j, idx_in_other]
            
            # Replace the word in topic i with the intruder
            shifted_topic = topics_tensor[i].copy()
            shifted_topic[idx_in_topic] = intruder_word
            shifted_centroid = np.mean(shifted_topic, axis=0)
            
            sim_value = sim(orig_centroid, shifted_centroid)
            sim_sums[i] += sim_value
            sim_counts[i] += 1
            
    # Average over repeats for each topic, then over topics
    topic_ish = [sim_sums[i] / sim_counts[i] if sim_counts[i] > 0 else 0.0 for i in range(T)]
    return float(np.mean(topic_ish)) if topic_ish else 0.0

def calculate_intruder_shift(
    topics_tensor: np.ndarray,
    n_repeats: int = 1000,
    random_seed: int = 42,
) -> float:
    """
    Compute the Intruder Shift (ISH) metric for a tensor of topics using vectorized operations.
    
    Args:
        topics_tensor: Array of shape (T, W, E) where T is number of topics, W is words per topic, E is embedding dim
        n_repeats: Number of times to repeat the intruder replacement for each topic
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    T, W, E = topics_tensor.shape
    
    # Generate random indices for all replacements at once
    random_topic_indices = np.random.randint(0, T, size=(T, n_repeats))
    random_word_indices_in = np.random.randint(0, W, size=(T, n_repeats))
    random_word_indices_out = np.random.randint(0, W, size=(T, n_repeats))
    
    topic_scores = []
    for i in range(T):
        # Create shifted versions of the topic by replacing words
        shifted_topics = np.repeat(topics_tensor[i:i+1], n_repeats, axis=0)
        shifted_topics[np.arange(n_repeats), random_word_indices_in[i]] = topics_tensor[random_topic_indices[i], random_word_indices_out[i]]
        
        # Compute similarity between original and shifted centroids
        orig_centroid = np.mean(topics_tensor[i], axis=0)
        shifted_centroids = np.mean(shifted_topics, axis=1)
        similarities = np.array([sim(orig_centroid, sc) for sc in shifted_centroids])
        
        topic_scores.append(np.mean(similarities))
    
    return float(np.mean(topic_scores)) if topic_scores else 0.0

def intruder_shift_equality(
    topics_tensor: np.ndarray,
    n_repeats: int = 15,
    random_seed: int = 42,
) -> float:
    """
    Verify that the slow and fast implementations of intruder shift give the same results.
    
    Args:
        topics_tensor: Array of shape (T, W, E) where T is number of topics, W is words per topic, E is embedding dim
        n_repeats: Number of times to repeat the intruder replacement for each topic
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    
    res_1 = calculate_intruder_shift_slow(topics_tensor, n_repeats, random_seed)
    res_2 = calculate_intruder_shift(topics_tensor, n_repeats, random_seed)
    
    assert np.isclose(res_1, res_2), f"Slow and fast implementations differ: {res_1} vs {res_2}"
    
    return res_1


if __name__ == '__main__':
    topics_tensor = np.random.rand(T, W, E)
    wecs_equality(topics_tensor)
    weps_equality(topics_tensor)
    intruder_shift_equality(topics_tensor)
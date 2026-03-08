import numpy as np
from turftopic.encoders.base import ExternalEncoder


class BOEWordEncoder(ExternalEncoder):
    """Encoder that returns pre-computed BOE word embeddings instead of encoding from scratch."""

    def __init__(self, word_embedding_lookup: dict[str, np.ndarray], embedding_dim: int):
        self.word_embedding_lookup = word_embedding_lookup
        self.embedding_dim = embedding_dim

    def encode(self, sentences, **kwargs) -> np.ndarray:
        vectors = []
        for word in sentences:
            vec = self.word_embedding_lookup.get(word)
            if vec is None:
                raise ValueError(f"Word '{word}' not found in word embedding lookup")
            vectors.append(vec)
        return np.array(vectors)

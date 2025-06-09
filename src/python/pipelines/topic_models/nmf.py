import logging
import numpy as np
from typing import List, Dict
from sqlalchemy import text
import json
from sklearn.decomposition import NMF

from configuration import load_config_from_env
from database import get_session
from data_handling import get_tfidf_vectors, get_vocabulary

class NMFModel:
    def __init__(self, num_topics: int, max_iter: int = 200, random_state: int = 42):
        self.num_topics = num_topics
        self.model = None
        self.max_iter = max_iter
        self.random_state = random_state

    def train(self, tfidf_vectors: np.ndarray, vocabulary: Dict[int, str]):
        """
        Train NMF model on TF-IDF vectors
        
        Args:
            tfidf_vectors: 2D numpy array of TF-IDF vectors
            vocabulary: Dictionary mapping word indices to words
        """
        self.model = NMF(
            n_components=self.num_topics,
            max_iter=self.max_iter,
            random_state=self.random_state,
            init='nndsvdar'  # Non-negative Double Singular Value Decomposition with AR
        )
        
        # Fit the model
        self.model.fit(tfidf_vectors)
        
        # Store vocabulary for topic extraction
        self._vocabulary = vocabulary

    def get_topics(self, n_words: int = 10) -> List[List[str]]:
        """
        Get the top words for each topic
        
        Args:
            n_words: Number of top words to return per topic
            
        Returns:
            List of lists, where each inner list contains the top words for a topic
        """
        # Get the components (topic-word matrix)
        components = self.model.components_
        
        # For each topic, get the indices of the top words
        topics = []
        for topic_idx in range(self.num_topics):
            # Get indices of top words for this topic
            top_word_indices = np.argsort(components[topic_idx])[-n_words:][::-1]
            # Convert indices to words using vocabulary
            topic_words = [self._vocabulary[idx] for idx in top_word_indices]
            topics.append(topic_words)
            
        return topics

def run_nmf_pipeline(corpus_name: str, num_topics: int = 20, num_iterations: int = 1) -> None:
    """
    Run NMF topic modeling pipeline and store results in database
    
    Args:
        corpus_name: Name of the corpus to analyze
        num_topics: Number of topics to extract
        num_iterations: Number of times to run the model
    """
    # Load configuration
    config = load_config_from_env()
    db_config = config.database
    
    # Get TF-IDF vectors and vocabulary
    _, tfidf_vectors = get_tfidf_vectors(corpus_name)
    vocabulary = get_vocabulary(corpus_name)
    
    if len(tfidf_vectors) == 0:
        raise ValueError(f"No TF-IDF vectors found for corpus: {corpus_name}")
    
    if not vocabulary:
        raise ValueError(f"No vocabulary found for corpus: {corpus_name}")
    
    # Get corpus ID and model ID once
    with get_session(db_config) as session:
        # Get corpus ID
        query = text("""
            SELECT id FROM pipeline.corpus 
            WHERE name = :corpus_name
        """).bindparams(corpus_name=corpus_name)
        corpus_id = session.execute(query).scalar()
        
        if not corpus_id:
            raise ValueError(f"Corpus not found: {corpus_name}")
        
        # Get NMF model ID
        query = text("""
            SELECT id FROM pipeline.topic_model 
            WHERE name = 'NMF'
        """)
        model_id = session.execute(query).scalar()
        
        if not model_id:
            raise ValueError("NMF topic model not found in database")
    
    # Run model multiple times
    for iteration in range(num_iterations):
        # Train NMF model
        logging.info(f"Training NMF model for iteration {iteration}...")
        nmf = NMFModel(num_topics=num_topics)
        nmf.train(tfidf_vectors, vocabulary)
        topics = nmf.get_topics()
        
        # Store results in database
        with get_session(db_config) as session:
            query = text("""
                INSERT INTO pipeline.topic_model_corpus_result 
                (topic_model_id, corpus_id, topics, num_topics, hyperparameters)
                VALUES (:model_id, :corpus_id, :topics, :num_topics, :hyperparameters)
            """).bindparams(
                model_id=model_id,
                corpus_id=corpus_id,
                topics=json.dumps(topics),
                num_topics=num_topics,
                hyperparameters=json.dumps({
                    'max_iter': nmf.max_iter,
                    'random_state': nmf.random_state
                })
            )
            session.execute(query)
            session.commit()

if __name__ == '__main__':
    # Example usage
    run_nmf_pipeline(
        corpus_name="newsgroups",
        num_topics=20,
        num_iterations=3
    )

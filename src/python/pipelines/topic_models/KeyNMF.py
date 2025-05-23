import numpy as np
from turftopic import KeyNMF
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Dict, Optional
from sqlalchemy import text
import json

from configuration import load_config_from_env
from database import get_session
from data_handling import get_chunk_embeddings, get_vocabulary_documents, get_vocabulary
from pipelines.sbert_embedding.main import EMBEDDING_MODEL

class KeyNMFWrapper:
    def __init__(self, num_topics: int, seed_phrase: Optional[str] = None):
        """
        Initialize KeyNMF wrapper
        
        Args:
            num_topics: Number of topics to extract
            seed_phrase: Optional seed phrase for seeded topic modeling
        """
        self.num_topics = num_topics
        self.seed_phrase = seed_phrase
        self.model = None
        self.vectorizer = None

    def train(self, documents: List[str], embeddings: np.ndarray, vocabulary: Dict[int, str]):
        """
        Train KeyNMF model on document embeddings
        
        Args:
            documents: List of preprocessed documents
            embeddings: 2D numpy array of document embeddings
            vocabulary: Dictionary mapping word indices to words
        """
        # Create custom vectorizer with vocabulary restriction
        self.vectorizer = CountVectorizer(vocabulary=list(vocabulary.values()))
        
        # Initialize KeyNMF
        self.model = KeyNMF(
            encoder=EMBEDDING_MODEL,
            n_components=self.num_topics,
            seed_phrase=self.seed_phrase,
            top_n=10
        )
        
        # Fit the model to our documents and embeddings
        self.model.fit(documents, embeddings=embeddings)

    def get_topics(self, n_words: int = 10) -> List[List[str]]:
        """
        Get the top words for each topic
        
        Args:
            n_words: Number of top words to return per topic
            
        Returns:
            List of lists, where each inner list contains the top words for a topic
        """
        topic_info = self.model.get_topics(top_k=n_words)
        
        # Extract top n words for each topic
        topic_words = []
        for _, topic in topic_info:
            words = [word for word, _ in topic]
            topic_words.append(words)
        
        return topic_words

def run_keynmf_pipeline(corpus_name: str, num_topics: int = 20, num_iterations: int = 1) -> None:
    """
    Run KeyNMF modeling pipeline and store results in database
    
    Args:
        corpus_name: Name of the corpus to analyze
        num_topics: Number of topics to extract
        num_iterations: Number of times to run the model
    """
    # Load configuration
    config = load_config_from_env()
    db_config = config.database
    
    # Get document embeddings and vocabulary
    _, embeddings = get_chunk_embeddings(corpus_name)
    vocabulary_docs = get_vocabulary_documents(corpus_name)
    vocabulary = get_vocabulary(corpus_name)
    
    if len(embeddings) == 0:
        raise ValueError(f"No document vectors found for corpus: {corpus_name}")
    
    if not vocabulary_docs:
        raise ValueError(f"No vocabulary documents found for corpus: {corpus_name}")
    
    if not vocabulary:
        raise ValueError(f"No vocabulary found for corpus: {corpus_name}")
    
    # Get document contents
    documents = [content for _, content in vocabulary_docs]
    
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
        
        # Get KeyNMF model ID
        query = text("""
            SELECT id FROM pipeline.topic_model 
            WHERE name = 'KeyNMF'
        """)
        model_id = session.execute(query).scalar()
        
        if not model_id:
            raise ValueError("KeyNMF model not found in database")
    
    # Run model multiple times
    for _ in range(num_iterations):
        # Train KeyNMF model
        keynmf = KeyNMFWrapper(num_topics=num_topics)
        keynmf.train(documents, embeddings, vocabulary)
        topics = keynmf.get_topics()
        
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
                hyperparameters=json.dumps({})
            )
            session.execute(query)
            session.commit()

if __name__ == '__main__':
    run_keynmf_pipeline(
        corpus_name="newsgroups",
        num_topics=20,
        num_iterations=3,
    )

import numpy as np
from turftopic import AutoEncodingTopicModel
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Dict
from sqlalchemy import text
import json

from configuration import load_config_from_env
from database import get_session
from data_handling import get_chunk_embeddings, get_vocabulary_documents, get_vocabulary

class AutoEncodingTopicModelWrapper:
    def __init__(self, num_topics: int, combined: bool = False):
        """
        Initialize AutoEncodingTopicModel wrapper
        
        Args:
            num_topics: Number of topics to extract
            combined: Whether to use CombinedTM (True) or ZeroShotTM (False)
        """
        self.num_topics = num_topics
        self.combined = combined
        self.model = None
        self.vectorizer = None

    def train(self, documents: List[str], embeddings: np.ndarray, vocabulary: Dict[int, str]):
        """
        Train AutoEncodingTopicModel on document embeddings
        
        Args:
            documents: List of preprocessed documents
            embeddings: 2D numpy array of document embeddings
            vocabulary: Dictionary mapping word indices to words
        """
        # Create custom vectorizer with vocabulary restriction
        self.vectorizer = CountVectorizer(vocabulary=list(vocabulary.values()))
        
        # Initialize AutoEncodingTopicModel
        self.model = AutoEncodingTopicModel(
            n_components=self.num_topics,
            combined=self.combined,
            vectorizer=self.vectorizer,
            batch_size=32000,
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

def run_autoencoding_tm_pipeline(corpus_name: str, num_topics: int = 20, num_iterations: int = 1, combined: bool = False, embedding_type: str = "openai") -> None:
    """
    Run AutoEncodingTopicModel pipeline and store results in database
    
    Args:
        corpus_name: Name of the corpus to analyze
        num_topics: Number of topics to extract
        num_iterations: Number of times to run the model
        combined: Whether to use CombinedTM (True) or ZeroShotTM (False)
    """
    # Load configuration
    config = load_config_from_env()
    db_config = config.database
    
    # Get document embeddings and vocabulary
    _, embeddings = get_chunk_embeddings(corpus_name, embedding_type)
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
        
        # Get model ID based on whether we're using CombinedTM or ZeroShotTM
        model_name = "CombinedTM" if combined else "ZeroShotTM"
        query = text("""
            SELECT id FROM pipeline.topic_model 
            WHERE name = :model_name
        """).bindparams(model_name=model_name)
        model_id = session.execute(query).scalar()
        
        if not model_id:
            raise ValueError(f"{model_name} model not found in database")
    
    # Run model multiple times
    for _ in range(num_iterations):
        # Train model
        tm = AutoEncodingTopicModelWrapper(num_topics=num_topics, combined=combined)
        tm.train(documents, embeddings, vocabulary)
        topics = tm.get_topics()
        
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
                hyperparameters=json.dumps({"embeddings": embedding_type})
            )
            session.execute(query)
            session.commit()

        del tm

if __name__ == '__main__':
    # Example usage for ZeroShotTM
    # run_autoencoding_tm_pipeline(
    #     corpus_name="newsgroups",
    #     num_topics=20,
    #     num_iterations=3,
    #     combined=False  # Use ZeroShotTM
    # )
    
    # Example usage for CombinedTM
    run_autoencoding_tm_pipeline(
        corpus_name="wikipedia_sample",
        num_topics=20,
        num_iterations=3,
        combined=True  # Use CombinedTM
    )

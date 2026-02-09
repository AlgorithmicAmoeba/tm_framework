import numpy as np
from turftopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Dict
from sqlalchemy import text
import json

from configuration import load_config_from_env
from database import get_session
from pipelines.topic_models.data_handling import get_chunk_embeddings, get_vocabulary_documents, get_vocabulary

class BERTopicModel:
    def __init__(self, num_topics: int):
        self.num_topics = num_topics
        self.model = None
        self.vectorizer = None

    def train(self, documents: List[str], embeddings: np.ndarray, vocabulary: Dict[int, str]):
        """
        Train BERTopic model on document embeddings
        
        Args:
            documents: List of preprocessed documents
            embeddings: 2D numpy array of document embeddings
            vocabulary: Dictionary mapping word indices to words
        """
        # Create custom vectorizer with vocabulary restriction
        self.vectorizer = CountVectorizer(vocabulary=list(vocabulary.values()))
        
        # Initialize BERTopic with our vectorizer and desired number of topics
        self.model = BERTopic(
            n_reduce_to=self.num_topics,
            vectorizer=self.vectorizer,
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

def run_bertopic_pipeline(corpus_name: str, num_topics: int = 20, num_iterations: int = 1, embedding_type: str = "openai") -> None:
    """
    Run BERTopic modeling pipeline and store results in database
    
    Args:
        corpus_name: Name of the corpus to analyze
        num_topics: Number of topics to extract
        num_iterations: Number of times to run the model
        embedding_type: Type of embeddings to use (default: "openai")
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
        
        # Get BERTopic model ID
        if embedding_type == "openai":
            model_name = "BERTopic"
        elif embedding_type == "sbert":
            model_name = "BERTopic_sbert"
        else:
            raise ValueError(f"Invalid embedding type: {embedding_type}")

        query = text("""
            SELECT id FROM pipeline.topic_model 
            WHERE name = :model_name
        """).bindparams(model_name=model_name)
        model_id = session.execute(query).scalar()
        
        if not model_id:
            raise ValueError("BERTopic model not found in database")
    
    # Run model multiple times
    for _ in range(num_iterations):
        # Train BERTopic model
        bertopic = BERTopicModel(num_topics=num_topics)
        bertopic.train(documents, embeddings, vocabulary)
        topics = bertopic.get_topics()
        
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

if __name__ == '__main__':
    # Example usage
    run_bertopic_pipeline(
        corpus_name="imdb_reviews",
        num_topics=20,
        num_iterations=3
    )

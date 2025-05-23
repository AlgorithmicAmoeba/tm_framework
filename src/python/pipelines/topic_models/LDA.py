import gensim
import numpy as np
from typing import List, Dict
from sqlalchemy import text
import json

from configuration import load_config_from_env
from database import get_session
from data_handling import get_tfidf_vectors, get_vocabulary

class LDA:
    def __init__(self, num_topics: int):
        self.num_topics = num_topics
        self.model = None
        self.corpus_lda = None
        self.idx2word = None

    def train(self, tfidf_vectors: np.ndarray, vocabulary: Dict[int, str]):
        """
        Train LDA model on TF-IDF vectors
        
        Args:
            tfidf_vectors: 2D numpy array of TF-IDF vectors
            vocabulary: Dictionary mapping word indices to words
        """
        self.corpus_lda = gensim.matutils.Scipy2Corpus(tfidf_vectors)
        self.idx2word = vocabulary
        
        self.model = gensim.models.LdaModel(
            self.corpus_lda,
            num_topics=self.num_topics,
            id2word=self.idx2word,
            random_state=42,
        )

    def get_topics(self, n_words: int = 10) -> List[List[str]]:
        """
        Get the top words for each topic
        
        Args:
            n_words: Number of top words to return per topic
            
        Returns:
            List of lists, where each inner list contains the top words for a topic
        """
        topics = self.model.show_topics(
            num_topics=self.num_topics,
            num_words=n_words,
            formatted=False,
        )
        return [[word for word, _ in topic] for _, topic in topics]

def run_lda_pipeline(corpus_name: str, num_topics: int = 20) -> None:
    """
    Run LDA topic modeling pipeline and store results in database
    
    Args:
        corpus_name: Name of the corpus to analyze
        num_topics: Number of topics to extract
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
    
    # Train LDA model
    lda = LDA(num_topics=num_topics)
    lda.train(tfidf_vectors, vocabulary)
    topics = lda.get_topics()
    
    # Store results in database
    with get_session(db_config) as session:
        # Get corpus ID
        query = text("""
            SELECT id FROM pipeline.corpus 
            WHERE name = :corpus_name
        """).bindparams(corpus_name=corpus_name)
        corpus_id = session.execute(query).scalar()
        
        if not corpus_id:
            raise ValueError(f"Corpus not found: {corpus_name}")
        
        # Get LDA model ID
        query = text("""
            SELECT id FROM pipeline.topic_model 
            WHERE name = 'LDA'
        """)
        model_id = session.execute(query).scalar()
        
        if not model_id:
            raise ValueError("LDA topic model not found in database")
        
        # Insert results
        query = text("""
            INSERT INTO pipeline.topic_model_corpus_result 
            (topic_model_id, corpus_id, topics, num_topics)
            VALUES (:model_id, :corpus_id, :topics, :num_topics)
        """).bindparams(
            model_id=model_id,
            corpus_id=corpus_id,
            topics=json.dumps(topics),
            num_topics=num_topics
        )
        session.execute(query)
        session.commit()

if __name__ == '__main__':
    # Example usage
    run_lda_pipeline(
        corpus_name="newsgroups",
        num_topics=20
    )

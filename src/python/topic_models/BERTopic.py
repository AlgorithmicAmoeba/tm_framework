import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

import configuration as cfg
from corpus import Corpus
import database


def embedding_list_to_array(embeddings: list):
    return np.array([
        embedding.vector
        for embedding in embeddings
    ])


class BERTopicModel:
    def __init__(self, corpus: Corpus, num_topics: int):
        self.corpus = corpus
        self.num_topics = num_topics
        self.model = None
        
        # Get vocabulary to restrict the model to
        vocabulary = [vocab_word.word for vocab_word in self.corpus.get_vocabulary()]
        
        # Create custom vectorizer with vocabulary restriction
        self.vectorizer = CountVectorizer(vocabulary=vocabulary)
        
    def train(self):
        # Get document embeddings
        embeddings = self.corpus.get_document_vectors('openai_small')
        embeddings = embedding_list_to_array(embeddings)
        
        # Get preprocessed documents
        documents = self.corpus.get_vocabulary_documents()
        documents = [doc.content for doc in documents]
        
        # Initialize BERTopic with our vectorizer and desired number of topics
        self.model = BERTopic(
            nr_topics=self.num_topics,
            vectorizer_model=self.vectorizer,
            verbose=True
        )
        
        # Fit the model to our documents and embeddings
        self.model.fit(documents, embeddings)
        
    def get_topics(self, n_words=10) -> list[list[str]]:
        topic_info = self.model.get_topics()
        
        # Extract top n words for each topic
        topic_words = []
        for i in range(len(topic_info)):
            if i in topic_info:  # Skip -1 (outlier) topic
                words = [word for word, _ in topic_info[i][:n_words]]
                topic_words.append(words)
        
        return topic_words


def run_bertopic_on_corpus(session, corpus_name: str, num_topics: int):
    corpus = Corpus(session, corpus_name)
    bertopic = BERTopicModel(corpus, num_topics)
    bertopic.train()
    return bertopic.get_topics()


if __name__ == '__main__':
    config = cfg.load_config_from_env()
    db_config = config.database

    with database.get_session(db_config) as session:
        topics = run_bertopic_on_corpus(
            session,
            # corpus_name="wikipedia_sample",
            corpus_name='newsgroups',
            # corpus_name='twitter-financial-news-topic-partial',
            num_topics=20,
        )
        for topic in topics:
            print(topic)
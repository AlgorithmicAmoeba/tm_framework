import numpy as np
import scipy.sparse
from turftopic import AutoEncodingTopicModel

from corpus import Corpus


def embedding_list_to_array(embeddings: list):
    return np.array([
        embedding.vector
        for embedding in embeddings
    ])


class CustomVectorizer:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
    
    def fit_transform(self, *args, **kwargs) -> 'scipy.sparse.csr_matrix':
        embeddings = self.corpus.get_document_vectors('tfidf')
        embeddings = embedding_list_to_array(embeddings)

        # convert to sparse matrix
        embeddings = scipy.sparse.csr_matrix(embeddings)
        return embeddings




class ZeroShotTM:
    def __init__(self, corpus: Corpus, num_topics: int):
        self.corpus = corpus
        self.num_topics = num_topics
        self.model = AutoEncodingTopicModel(
            n_components=num_topics,
            encoder=None,
            vectorizer=CustomVectorizer(corpus),
            combined=False,
        )

    def train(self):
        embeddings = self.corpus.get_document_vectors('openai_small')
        embeddings = embedding_list_to_array(embeddings)

        raw_documents = self.corpus.get_raw_documents()
        raw_documents = [
            document.content
            for document in raw_documents
        ]

        self.model.fit(
            raw_documents=raw_documents,
            embeddings=embeddings,
        )

    def get_topics(self):
        return self.model.get_topics()


def run_zero_shot_tm_on_corpus(session, corpus_name: str, num_topics: int):
    corpus = Corpus(session, corpus_name)
    zstm = ZeroShotTM(corpus, num_topics)
    zstm.train()
    return zstm.model.get_topics()


if __name__ == '__main__':
    import configuration as cfg
    import database

    config = cfg.load_config_from_env()
    db_config = config.database

    with database.get_session(db_config) as session:
        topics = run_zero_shot_tm_on_corpus(
            session,
            # corpus_name='trec_questions',
            # corpus_name='imdb_reviews',
            # corpus_name="wikipedia_sample",
            # corpus_name='twitter-financial-news-topic-partial',
            corpus_name='newsgroups',
            num_topics=10,
        )

        print(topics)
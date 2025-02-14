import gensim
import numpy as np

import configuration as cfg
from corpus import Corpus
import database

class LDA:
    def __init__(self, corpus: Corpus, num_topics: int):
        self.corpus = corpus
        self.num_topics = num_topics
        self.model = None
        self.corpus_lda = None
        self.idx2word = None

    def train(self):
        tfidf_vector_obj = self.corpus.get_document_vectors('tfidf')
        tfidf_vectors = np.array([tfidf_vector.vector for tfidf_vector in tfidf_vector_obj])
        self.corpus_lda = gensim.matutils.Scipy2Corpus(tfidf_vectors)
        _idx2word = {vocab_word.id: vocab_word.word for vocab_word in self.corpus.get_vocabulary()}
        self.idx2word = {idx: word for idx, (_, word) in enumerate(sorted(_idx2word.items()))}
        
        self.model = gensim.models.LdaModel(
            self.corpus_lda,
            num_topics=self.num_topics,
            id2word=self.idx2word,
            # passes=10,
            random_state=42,
        )

    def get_topics(self, n_words=10) -> list[str]:
        topics = self.model.show_topics(
            num_topics=self.num_topics,
            num_words=n_words,
            formatted=False,
        )

        topic_words = [[word for word, _ in topic] for _, topic in topics]

        return topic_words


def run_lda_on_corpus(session, corpus_name: str, num_topics: int):
    corpus = Corpus(session, corpus_name)
    lda = LDA(corpus, num_topics)
    lda.train()
    return lda.get_topics()
    

if __name__ == '__main__':
    config = cfg.load_config_from_env()
    db_config = config.database

    with database.get_session(db_config) as session:
        topics = run_lda_on_corpus(
            session,
            # corpus_name='trec_questions',
            # corpus_name='imdb_reviews',
            corpus_name="wikipedia_sample",
            # corpus_name='twitter-financial-news-topic-partial',
            # corpus_name='newsgroups',
            num_topics=20,
        )
        for topic in topics:
            print(topic)
from corpus import Corpus
import gensim

class LDA:
    def __init__(self, corpus: Corpus, num_topics: int):
        self.corpus = corpus
        self.num_topics = num_topics
        self.model = None
        self.corpus_lda = None

    def train(self):
        dictionary = gensim.corpora.Dictionary([self.corpus.get_vocabulary()])
        tfidf_vector_obj = self.corpus.get_document_vectors('tfidf')
        tfidf_vectors = [tfidf_vector.vector for tfidf_vector in tfidf_vector_obj]
        self.corpus_lda = gensim.matutils.Sparse2Corpus(tfidf_vectors)
        self.model = gensim.models.LdaModel(
            self.corpus_lda,
            num_topics=self.num_topics,
            id2word=dictionary,
            passes=15,
            random_state=42
        )

    def get_topics(self) -> list[str]:
        return self.model.print_topics()
    



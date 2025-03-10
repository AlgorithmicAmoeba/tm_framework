from corpus import Corpus



def run_model_on_corpus(session, model_class, corpus_name: str, num_topics: int):
    corpus = Corpus(session, corpus_name)
    model = model_class(corpus, num_topics)
    model.train()
    topics = model.get_topics()


def save_topics_to_db(session
    
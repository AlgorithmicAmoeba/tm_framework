import concurrent.futures

import sklearn.feature_extraction.text
from sqlalchemy.orm import Session

from preprocessing import UnicodeTokenizer, Vocabulariser
from models import Corpus, Document, VocabularyWord, Embedder, Embedding, DocumentType
import database
import configuration as cfg

class CorpusProcessing:
    def __init__(self, raw_texts, tokenized_texts, transformed_texts, vocabulary, tfidf_matrix):
        self.raw_texts = raw_texts
        self.tokenized_texts = tokenized_texts
        self.transformed_texts = transformed_texts
        self.vocabulary = vocabulary
        self.tfidf_matrix = tfidf_matrix

def preprocess_texts(texts, top_n, executor=None):
    tokenizer = UnicodeTokenizer()
    vocabulariser = Vocabulariser(top_n)

    tokenized_texts = tokenizer.process_texts(texts, executor)
    transformed_texts = vocabulariser.fit(tokenized_texts, executor)

    return CorpusProcessing(
        raw_texts=texts,
        tokenized_texts=tokenized_texts,
        transformed_texts=transformed_texts,
        vocabulary=vocabulariser.vocabulary,
        tfidf_matrix=vocabulariser.tfidf_matrix,
    )

def store_in_database(session: Session, corpus_name: str, corpus_processing: CorpusProcessing):
    corpus = Corpus(name=corpus_name)
    session.add(corpus)

    document_types = {
        'raw': session.query(DocumentType).filter_by(name='raw').first().id,
        'preprocessed': session.query(DocumentType).filter_by(name='preprocessed').first().id,
        'vocabulary_only': session.query(DocumentType).filter_by(name='vocabulary_only').first().id
    }

    for word in corpus_processing.vocabulary:
        vocab_word = session.query(VocabularyWord).filter_by(word=word, corpus_id=corpus.id).first()
        if not vocab_word:
            vocab_word = VocabularyWord(corpus_id=corpus.id, word=word)
            session.add(vocab_word)

    for text, tokenized_text, transformed_text, tfidf_vector in zip(
        corpus_processing.raw_texts, 
        corpus_processing.tokenized_texts, 
        corpus_processing.transformed_texts, 
        corpus_processing.tfidf_matrix
    ):
        raw_document = Document(
            corpus_id=corpus.id,
            content=text,
            language_code='en',
            type_id=document_types['raw']
        )
        session.add(raw_document)

        preprocessed_document = Document(
            corpus_id=corpus.id,
            content=' '.join(tokenized_text),
            language_code='en',
            type_id=document_types['preprocessed']
        )
        session.add(preprocessed_document)

        vocabulary_document = Document(
            corpus_id=corpus.id,
            content=' '.join(transformed_text),
            language_code='en',
            type_id=document_types['vocabulary_only']
        )
        session.add(vocabulary_document)

        embedder = session.query(Embedder).filter_by(name='tfidf').first()

        # Store TF-IDF embeddings
        embedding = Embedding(
            embedder_id=embedder.id,
            document_id=raw_document.id,
            vector=tfidf_vector.toarray().tolist()
        )
        session.add(embedding)

    session.commit()

def run_pipeline(session, corpus_name: str, texts, top_n, executor=None):
    

    corpus_processing = preprocess_texts(texts, top_n=top_n, executor=executor)
    store_in_database(session, corpus_name, corpus_processing)


if __name__ == '__main__':
    import pandas as pd

    splits = {'train': 'topic_train.csv', 'validation': 'topic_valid.csv'}
    df = pd.read_csv("hf://datasets/zeroshot/twitter-financial-news-topic/" + splits["train"])

    texts = df['text'].tolist()[:1000]
    top_n = 3000

    config = cfg.load_config_from_env()
    db_config = config.database

    with database.get_test_session(db_config) as session:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=16)
        run_pipeline(session, "twitter-financial-news-topic", texts, top_n, executor)

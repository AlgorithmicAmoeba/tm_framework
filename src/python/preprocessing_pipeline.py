import concurrent.futures

from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import insert

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
    
    document_types = {
        'raw': session.query(DocumentType).filter_by(name='raw').first().id,
        'preprocessed': session.query(DocumentType).filter_by(name='preprocessed').first().id,
        'vocabulary_only': session.query(DocumentType).filter_by(name='vocabulary_only').first().id
    }

    embedder = session.query(Embedder).filter_by(name='tfidf').first()

    corpus = Corpus(name=corpus_name)
    session.add(corpus)
    session.flush()

    # Add vocabulary words
    vocabulary_words = []
    for word in corpus_processing.vocabulary:
        vocabulary_words.append(dict(corpus_id=corpus.id, word=word))
    session.execute(insert(VocabularyWord), vocabulary_words)

    # Add raw documents
    raw_documents = []
    for text in corpus_processing.raw_texts:
        raw_documents.append(dict(
            corpus_id=corpus.id,
            content=text,
            language_code='en',
            type_id=document_types['raw']
        ))
    raw_docs = session.scalars(insert(Document).returning(Document, sort_by_parameter_order=True), raw_documents)
    raw_doc_ids = [doc.id for doc in raw_docs]

    # Add preprocessed documents
    preprocessed_documents = []
    for tokenized_text in corpus_processing.tokenized_texts:
        preprocessed_documents.append(dict(
            corpus_id=corpus.id,
            content=' '.join(tokenized_text),
            language_code='en',
            type_id=document_types['preprocessed']
        ))
    session.execute(insert(Document), preprocessed_documents)

    # Add vocabulary documents
    vocabulary_documents = []
    for transformed_text in corpus_processing.transformed_texts:
        vocabulary_documents.append(dict(
            corpus_id=corpus.id,
            content=' '.join(transformed_text),
            language_code='en',
            type_id=document_types['vocabulary_only']
        ))
    session.execute(insert(Document), vocabulary_documents)

    # Add embeddings
    embeddings = []
    for doc_id, tfidf_vector in zip(raw_doc_ids, corpus_processing.tfidf_matrix):
        embeddings.append(dict(
            embedder_id=embedder.id,
            document_id=doc_id,
            vector=tfidf_vector.toarray().tolist()
        ))
    session.execute(insert(Embedding), embeddings)
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

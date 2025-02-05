import warnings
import re

from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import insert
import tqdm

from preprocessing import TextPreprocessor
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

def preprocess_texts(
        texts,
        top_n, 
        min_words_per_document=None,
        min_df=0.0,
        max_df=1.0,
        min_chars=3,
        remove_urls=True,
        remove_stopwords=False,
    ):
    preprocessor = TextPreprocessor(
        top_n=top_n,
        min_words_per_document=min_words_per_document,
        min_df=min_df,
        max_df=max_df,
        min_chars=min_chars,
        remove_urls=remove_urls,
        remove_stopwords=remove_stopwords
    )

    transformed_texts = preprocessor.fit_transform(texts)

    return CorpusProcessing(
        raw_texts=texts,
        tokenized_texts=preprocessor.tokenized_texts,
        transformed_texts=transformed_texts,
        vocabulary=preprocessor.vocabulary,
        tfidf_matrix=preprocessor.tfidf_matrix,
    )

def store_in_database(session: Session, corpus_name: str, corpus_processing: CorpusProcessing):
    document_types = {
        'raw': session.query(DocumentType).filter_by(name='raw').first().id,
        'preprocessed': session.query(DocumentType).filter_by(name='preprocessed').first().id,
        'vocabulary_only': session.query(DocumentType).filter_by(name='vocabulary_only').first().id
    }

    embedder = session.query(Embedder).filter_by(name='tfidf').first()

    pbar = tqdm.tqdm(total=7, desc="Starting database storage")
    
    corpus = Corpus(name=corpus_name)
    session.add(corpus)
    session.flush()
    pbar.update(1)

    # Add vocabulary words with index
    pbar.set_description("Adding vocabulary words")
    vocabulary_words = [
        dict(corpus_id=corpus.id, word=word, word_index=idx) 
        for idx, word in enumerate(corpus_processing.vocabulary)
    ]
    session.execute(insert(VocabularyWord), vocabulary_words)
    pbar.update(1)

    # Add raw documents
    pbar.set_description("Adding raw documents")
    raw_documents = [dict(
        corpus_id=corpus.id,
        content=text,
        language_code='en',
        type_id=document_types['raw']
    ) for text in corpus_processing.raw_texts]
    raw_docs = session.scalars(insert(Document).returning(Document, sort_by_parameter_order=True), raw_documents)
    raw_doc_ids = [doc.id for doc in raw_docs]
    pbar.update(1)

    # Add preprocessed documents
    pbar.set_description("Adding preprocessed documents")
    preprocessed_documents = [dict(
        corpus_id=corpus.id,
        content=' '.join(tokenized_text),
        language_code='en',
        type_id=document_types['preprocessed']
    ) for tokenized_text in corpus_processing.tokenized_texts]
    session.execute(insert(Document), preprocessed_documents)
    pbar.update(1)

    # Add vocabulary documents
    pbar.set_description("Adding vocabulary-only documents")
    vocabulary_documents = [dict(
        corpus_id=corpus.id,
        content=' '.join(transformed_text),
        language_code='en',
        type_id=document_types['vocabulary_only']
    ) for transformed_text in corpus_processing.transformed_texts]
    session.execute(insert(Document), vocabulary_documents)
    pbar.update(1)

    # Add embeddings
    pbar.set_description("Adding embeddings")
    embeddings = [dict(
        embedder_id=embedder.id,
        document_id=doc_id,
        vector=tfidf_vector.toarray().tolist()[0]
    ) for doc_id, tfidf_vector in zip(raw_doc_ids, corpus_processing.tfidf_matrix)]
    session.execute(insert(Embedding), embeddings)
    pbar.update(1)

    pbar.set_description("Committing changes")
    session.commit()
    pbar.update(1)
    pbar.close()

def run_pipeline(
    session: Session,
    corpus_name: str,
    texts: list[str],
    top_n: int,
    min_words_per_document: int,
    min_df: float,
    max_df: float,
    min_chars: int,
    remove_urls: bool,
    remove_stopwords: bool,
    ):
    corpus_processing = preprocess_texts(
        texts, 
        top_n=top_n,
        min_words_per_document=min_words_per_document,
        min_df=min_df,
        max_df=max_df,
        min_chars=min_chars, 
        remove_urls=remove_urls,
        remove_stopwords=remove_stopwords,
    )
    delete_corpus(session, "newsgroups", if_exists=True)
    store_in_database(session, corpus_name, corpus_processing)


def delete_corpus(session: Session, corpus_name: str, if_exists: bool = False):
    corpus = session.query(Corpus).filter_by(name=corpus_name).first()
    if corpus is None:
        if if_exists:
            warnings.warn(f"Corpus '{corpus_name}' not found")
            return
        raise ValueError(f"Corpus '{corpus_name}' not found")
    
    # Remove embeddings, documents, vocabulary words
    # Remove embeddings by document
    session.query(Embedding).filter(Embedding.document_id.in_(
        session.query(Document.id).filter_by(corpus_id=corpus.id)
    )).delete(synchronize_session=False)
    session.query(Document).filter_by(corpus_id=corpus.id).delete()
    session.query(VocabularyWord).filter_by(corpus_id=corpus.id).delete()
    session.query(Corpus).filter_by(id=corpus.id).delete()
    session.commit()


def twitter_financial_news_topic_pipeline(session: Session, subset: int = None):
    splits = {'train': 'topic_train.csv', 'validation': 'topic_valid.csv'}
    df = pd.read_csv("hf://datasets/zeroshot/twitter-financial-news-topic/" + splits["train"])

    texts = df['text'].tolist()
    if subset is not None:
        texts = texts[:subset]

    run_pipeline(
        session, 
        "twitter-financial-news-topic-partial", 
        texts, 
        top_n=None,
        remove_urls=True,
        min_words_per_document=5,
        min_df=0.005,
        max_df=1.0,
        min_chars=3,
        remove_stopwords=True,
    )

def newsgroups_pipeline(
        session: Session,
        subset: int = None,
    ):

    from sklearn.datasets import fetch_20newsgroups
    
    # Fetch and clean data
    newsgroups = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
    )
    texts = newsgroups.data
    if subset is not None:
        texts = texts[:subset]
    
    run_pipeline(
        session, 
        "newsgroups", 
        texts, 
        top_n=None,
        remove_urls=True,
        min_words_per_document=5,
        min_df=0.005,
        max_df=1.0,
        min_chars=3,
        remove_stopwords=True,
    )

def newsgroups_pipeline_octis(
        session: Session,
        subset: int = None,
    ):
    """Process the newsgroups dataset using OCTIS preprocessing"""
    from sklearn.datasets import fetch_20newsgroups
    import tempfile
    import os
    from octis_preprocessing import Preprocessing, Dataset

    # Fetch and clean data
    newsgroups = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
    )
    texts = newsgroups.data
    if subset is not None:
        texts = texts[:subset]

    # Create temporary files for OCTIS processing
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f_docs:
        for text in texts:
            text = text.replace('\n', ' ')
            f_docs.write(f"{text}\n")
        docs_path = f_docs.name

    try:
        # Initialize OCTIS preprocessing
        preprocessor = Preprocessing(
            lowercase=True,
            min_chars=3,
            min_words_docs=5,
            min_df=0.005,
            max_df=1.0,
            remove_punctuation=True,
            remove_numbers=True,
            lemmatize=True,
            stopword_list='english',
            language='english',
            verbose=True,
            split=False  # We don't need OCTIS to split the data
        )

        # Process the dataset
        dataset = preprocessor.preprocess_dataset(docs_path)
        corpus = dataset.get_corpus()
        vocabulary = dataset.get_vocabulary()

        # build tfidf matrix using corpus and vocabulary
        # Convert tokenized corpus back to strings for TfidfVectorizer
        corpus_strings = [' '.join(doc) for doc in corpus]

        # Create TfidfVectorizer with fixed vocabulary
        vectorizer = TfidfVectorizer(vocabulary=vocabulary)
        tfidf_matrix = vectorizer.fit_transform(corpus_strings)


        # Store in database
        delete_corpus(session, "newsgroups-octis", if_exists=True)
        corpus_processing = CorpusProcessing(
            raw_texts=texts,
            tokenized_texts=corpus,  # OCTIS already tokenized
            transformed_texts=corpus,  # Using same as tokenized since OCTIS already filtered vocabulary
            vocabulary=vocabulary,
            tfidf_matrix=tfidf_matrix,
        )
        store_in_database(session, "newsgroups-octis", corpus_processing)

    finally:
        # Clean up temporary files
        os.unlink(docs_path)

if __name__ == '__main__':
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    config = cfg.load_config_from_env()
    db_config = config.database

    with database.get_session(db_config) as session:
        twitter_financial_news_topic_pipeline(session)
        
        # Choose which preprocessing pipeline to use
        # use_octis = False
        # subset = None
        
        # if use_octis:
        #     newsgroups_pipeline_octis(session, subset=subset)
        # else:
        #     newsgroups_pipeline(session, subset=subset)



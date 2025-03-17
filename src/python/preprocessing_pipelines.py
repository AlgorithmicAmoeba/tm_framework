import warnings
import re

from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import insert
import tqdm

from preprocessor import TextPreprocessor
from models import (
    Corpus, Document, VocabularyWord, Embedder, Embedding,
    DocumentType, TopicModelCorpusResult, ResultPerformance,
    VocabularyWordEmbedding
)
import database
import configuration as cfg

class CorpusProcessing:
    def __init__(self, raw_texts, tokenized_texts, transformed_texts, vocabulary, tfidf_matrix):
        self.raw_texts = raw_texts
        self.tokenized_texts = tokenized_texts
        self.transformed_texts = transformed_texts
        self.vocabulary = vocabulary
        self.tfidf_matrix = tfidf_matrix

        # warn if the shapes don't match
        if len(self.raw_texts) != len(self.tokenized_texts):
            warnings.warn("Number of raw texts and tokenized texts do not match")

        if len(self.raw_texts) != len(self.transformed_texts):
            warnings.warn("Number of raw texts and transformed texts do not match")

        if len(self.raw_texts) != self.tfidf_matrix.shape[0]:
            warnings.warn("Number of raw texts and tfidf vectors do not match")

        if len(self.vocabulary) != self.tfidf_matrix.shape[1]:
            warnings.warn("Vocabulary length does not match tfidf matrix shape")

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
        raw_texts=preprocessor.raw_texts,
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
        type_id=document_types['raw'],
        parent_id=None,
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
        type_id=document_types['preprocessed'],
        parent_id=raw_doc_ids,
    ) for raw_doc_id, tokenized_text in zip(raw_doc_ids, corpus_processing.tokenized_texts)]
    session.execute(insert(Document), preprocessed_documents)
    pbar.update(1)

    # Add vocabulary documents
    pbar.set_description("Adding vocabulary-only documents")
    vocabulary_documents = [dict(
        corpus_id=corpus.id,
        content=' '.join(transformed_text),
        language_code='en',
        type_id=document_types['vocabulary_only']
    ) for raw_doc_id, transformed_text in zip(raw_doc_ids, corpus_processing.transformed_texts)]
    session.execute(insert(Document), vocabulary_documents)
    pbar.update(1)

    # Add embeddings
    pbar.set_description("Adding embeddings")
    embeddings = [dict(
        embedder_id=embedder.id,
        document_id=raw_doc_ids,
        vector=tfidf_vector.toarray().tolist()[0]
    ) for raw_doc_id, tfidf_vector in zip(raw_doc_ids, corpus_processing.tfidf_matrix)]
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
    delete_corpus(session, corpus_name, if_exists=True)
    store_in_database(session, corpus_name, corpus_processing)


def delete_corpus(session: Session, corpus_name: str, if_exists: bool = False):
    corpus = session.query(Corpus).filter_by(name=corpus_name).first()
    if corpus is None:
        if if_exists:
            warnings.warn(f"Corpus '{corpus_name}' not found")
            return
        raise ValueError(f"Corpus '{corpus_name}' not found")
    
    # Remove embeddings, documents, vocabulary words, TopicModelCorpusResults
    # Remove embeddings by document
    # Remove ResultPerformance by TopicModelCorpusResult
    session.query(Embedding).filter(Embedding.document_id.in_(
        session.query(Document.id).filter_by(corpus_id=corpus.id)
    )).delete(synchronize_session=False)
    session.query(Document).filter_by(corpus_id=corpus.id).delete()
    # Delete vocabulary word embeddings
    session.query(VocabularyWordEmbedding).filter(VocabularyWordEmbedding.vocabulary_word_id.in_(
        session.query(VocabularyWord.id).filter_by(corpus_id=corpus.id)
    )).delete(synchronize_session=False)
    session.query(VocabularyWord).filter_by(corpus_id=corpus.id).delete()
    session.query(ResultPerformance).filter(ResultPerformance.topic_model_corpus_result_id.in_(
        session.query(TopicModelCorpusResult.id).filter_by(corpus_id=corpus.id)
    )).delete(synchronize_session=False)
    session.query(TopicModelCorpusResult).filter_by(corpus_id=corpus.id).delete()
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
        min_df=0.001,
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

def wikipedia_pipeline(
        session: Session,
        file_path: str,
        subset: int = None,
    ):
    """Process Wikipedia articles from a JSON Lines file."""
    import json
    
    # Read JSON Lines file
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            texts.append(article['text'])
    
    if subset is not None:
        texts = texts[:subset]
    
    run_pipeline(
        session, 
        "wikipedia_sample", 
        texts, 
        top_n=None,
        remove_urls=True,
        min_words_per_document=10,
        min_df=0.02,
        max_df=0.7,
        min_chars=3,
        remove_stopwords=True,
    )

def imdb_pipeline(
        session: Session,
        subset: int = None,
    ):
    """Process IMDB movie reviews dataset."""
    from datasets import load_dataset
    
    # Load both train and test splits
    dataset = load_dataset("stanfordnlp/imdb")
    train_texts = dataset['train']['text']
    test_texts = dataset['test']['text']
    texts = train_texts + test_texts
    
    if subset is not None:
        texts = texts[:subset]
    
    run_pipeline(
        session, 
        "imdb_reviews", 
        texts, 
        top_n=None,
        remove_urls=True,
        min_words_per_document=10,
        min_df=0.003,  # Lower threshold since reviews use diverse vocabulary
        max_df=0.7,    # Stricter upper bound for common terms
        min_chars=3,
        remove_stopwords=True,
    )

def trec_pipeline(
        session: Session,
        subset: int = None,
    ):
    """Process TREC question classification dataset."""
    from datasets import load_dataset
    
    # Load both train and test splits
    dataset = load_dataset("CogComp/trec")
    train_texts = dataset['train']['text']
    test_texts = dataset['test']['text']
    texts = train_texts + test_texts
    
    if subset is not None:
        texts = texts[:subset]
    
    run_pipeline(
        session, 
        "trec_questions", 
        texts, 
        top_n=None,
        remove_urls=True,
        min_words_per_document=2,
        min_df=0.0005,
        max_df=0.9,
        min_chars=3,
        remove_stopwords=True,
    )

if __name__ == '__main__':
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    config = cfg.load_config_from_env()
    db_config = config.database

    with database.get_session(db_config) as session:
        # twitter_financial_news_topic_pipeline(session)
        newsgroups_pipeline(session)
        wikipedia_pipeline(session, 'ignore/raw_data/wikipedia_20k_sample.jsonl')
        imdb_pipeline(session)
        trec_pipeline(session)



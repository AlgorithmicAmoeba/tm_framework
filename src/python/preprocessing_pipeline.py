from sqlalchemy.orm import Session
from preprocessing import UnicodeTokenizer, Vocabulariser
from models import Corpus, Document, VocabularyWord, Embedding, DocumentType
from database import get_session
import configuration as cfg

class CorpusProcessing:
    def __init__(self, raw_texts, tokenized_texts, transformed_texts, vocabulary, tfidf_matrix):
        self.raw_texts = raw_texts
        self.tokenized_texts = tokenized_texts
        self.transformed_texts = transformed_texts
        self.vocabulary = vocabulary
        self.tfidf_matrix = tfidf_matrix

def preprocess_texts(texts):
    tokenizer = UnicodeTokenizer()
    vocabulariser = Vocabulariser(top_n=3)

    tokenized_texts = [tokenizer.tokenize(text) for text in texts]
    vocabulariser.fit(tokenized_texts)
    transformed_texts = vocabulariser.transform(tokenized_texts)

    return CorpusProcessing(
        raw_texts=texts,
        tokenized_texts=tokenized_texts,
        transformed_texts=transformed_texts,
        vocabulary=vocabulariser.vocabulary,
        tfidf_matrix=vocabulariser.tfidf_matrix
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

        # Store TF-IDF embeddings
        embedding = Embedding(
            embedder_id=1,  # Assuming 1 is the ID for TF-IDF embedder
            document_id=raw_document.id,
            vector=tfidf_vector.tolist()
        )
        session.add(embedding)

    session.commit()

def run_pipeline(corpus_name: str, texts):
    db_config = cfg.DatabaseConfig()
    session = get_session(db_config)()

    corpus_processing = preprocess_texts(texts)
    store_in_database(session, corpus_name, corpus_processing)

    session.close()

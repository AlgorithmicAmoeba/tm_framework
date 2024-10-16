from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Float, ForeignKey, ARRAY
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

Base = declarative_base()

class Corpus(Base):
    __tablename__ = 'corpus'
    __table_args__ = {"schema": "topic_modelling"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    def __repr__(self):
        return (
            f"<Corpus("
            f"id={self.id}, "
            f"name='{self.name}')>"
        )

class DocumentType(Base):
    __tablename__ = 'document_type'
    __table_args__ = {"schema": "topic_modelling"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    def __repr__(self):
        return (
            f"<DocumentType("
            f"id={self.id}, "
            f"name='{self.name}')>"
        )

class Document(Base):
    __tablename__ = 'document'
    __table_args__ = {"schema": "topic_modelling"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    corpus_id = Column(Integer, ForeignKey('topic_modelling.corpus.id'))
    content = Column(Text)
    language_code = Column(String(10))
    type_id = Column(Integer, ForeignKey('topic_modelling.document_type.id'))
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    
    corpus = relationship("Corpus")
    document_type = relationship("DocumentType")

    def __repr__(self):
        return (
            f"<Document("
            f"id={self.id}, "
            f"corpus_id={self.corpus_id})>"
        )

class VocabularyWord(Base):
    __tablename__ = 'vocabulary_word'
    __table_args__ = {"schema": "topic_modelling"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    corpus_id = Column(Integer, ForeignKey('topic_modelling.corpus.id'))
    word = Column(String(255), unique=True, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    
    corpus = relationship("Corpus")

    def __repr__(self):
        return (
            f"<VocabularyWord("
            f"id={self.id}, "
            f"word='{self.word}', "
            f"corpus_id={self.corpus_id})>"
        )

class Embedder(Base):
    __tablename__ = 'embedder'
    __table_args__ = {"schema": "topic_modelling"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    def __repr__(self):
        return (
            f"<Embedder("
            f"id={self.id}, "
            f"name='{self.name}')>"
        )

class Embedding(Base):
    __tablename__ = 'embedding'
    __table_args__ = {"schema": "topic_modelling"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    embedder_id = Column(Integer, ForeignKey('topic_modelling.embedder.id'))
    document_id = Column(Integer, ForeignKey('topic_modelling.document.id'))
    vector = Column(ARRAY(Float))
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    
    embedder = relationship("Embedder")
    document = relationship("Document")

    def __repr__(self):
        return (
            f"<Embedding("
            f"id={self.id}, "
            f"embedder_id={self.embedder_id}, "
            f"document_id={self.document_id})>"
        )

class VocabularyWordEmbedding(Base):
    __tablename__ = 'vocabulary_word_embedding'
    __table_args__ = {"schema": "topic_modelling"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vocabulary_word_id = Column(Integer, ForeignKey('topic_modelling.vocabulary_word.id'))
    embedder_id = Column(Integer, ForeignKey('topic_modelling.embedder.id'))
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    
    vocabulary_word = relationship("VocabularyWord")
    embedder = relationship("Embedder")

    def __repr__(self):
        return (
            f"<VocabularyWordEmbedding("
            f"id={self.id}, "
            f"vocabulary_word_id={self.vocabulary_word_id}, "
            f"embedder_id={self.embedder_id})>"
        )

class TopicModel(Base):
    __tablename__ = 'topic_model'
    __table_args__ = {"schema": "topic_modelling"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    def __repr__(self):
        return (
            f"<TopicModel("
            f"id={self.id}, "
            f"name='{self.name}')>"
        )

class TopicModelCorpusResult(Base):
    __tablename__ = 'topic_model_corpus_result'
    __table_args__ = {"schema": "topic_modelling"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    topic_model_id = Column(Integer, ForeignKey('topic_modelling.topic_model.id'))
    corpus_id = Column(Integer, ForeignKey('topic_modelling.corpus.id'))
    topics = Column(JSONB)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    
    topic_model = relationship("TopicModel")
    corpus = relationship("Corpus")

    def __repr__(self):
        return (
            f"<TopicModelCorpusResult("
            f"id={self.id}, "
            f"topic_model_id={self.topic_model_id}, "
            f"corpus_id={self.corpus_id})>"
        )

class PerformanceMetric(Base):
    __tablename__ = 'performance_metric'
    __table_args__ = {"schema": "topic_modelling"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    def __repr__(self):
        return (
            f"<PerformanceMetric("
            f"id={self.id}, "
            f"name='{self.name}')>"
        )

class ResultPerformance(Base):
    __tablename__ = 'result_performance'
    __table_args__ = {"schema": "topic_modelling"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    topic_model_corpus_result_id = Column(Integer, ForeignKey('topic_modelling.topic_model_corpus_result.id'))
    performance_metric_id = Column(Integer, ForeignKey('topic_modelling.performance_metric.id'))
    value = Column(Float)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    
    topic_model_corpus_result = relationship("TopicModelCorpusResult")
    performance_metric = relationship("PerformanceMetric")

    def __repr__(self):
        return (
            f"<ResultPerformance("
            f"id={self.id}, "
            f"topic_model_corpus_result_id={self.topic_model_corpus_result_id}, "
            f"performance_metric_id={self.performance_metric_id}, "
            f"value={self.value})>"
        )

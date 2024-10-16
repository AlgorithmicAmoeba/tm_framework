from models import (
    Corpus, DocumentType, Document, VocabularyWord, Embedder, Embedding,
    VocabularyWordEmbedding, TopicModel, TopicModelCorpusResult,
    PerformanceMetric, ResultPerformance
)

def test_create_corpus(test_session):
    corpus = Corpus(name="Test Corpus", description="A test corpus")
    test_session.add(corpus)
    test_session.commit()
    
    retrieved = test_session.query(Corpus).filter_by(name="Test Corpus").first()
    assert retrieved is not None
    assert retrieved.name == "Test Corpus"
    assert retrieved.description == "A test corpus"

def test_create_document_type(test_session):
    doc_type = DocumentType(name="Test Type", description="A test document type")
    test_session.add(doc_type)
    test_session.commit()
    
    retrieved = test_session.query(DocumentType).filter_by(name="Test Type").first()
    assert retrieved is not None
    assert retrieved.name == "Test Type"
    assert retrieved.description == "A test document type"

def test_create_document(test_session):
    corpus = Corpus(name="Test Corpus for Document")
    doc_type = DocumentType(name="Test Document Type")
    test_session.add(corpus)
    test_session.add(doc_type)
    test_session.commit()

    document = Document(corpus_id=corpus.id, content="Test content", language_code="en", type_id=doc_type.id)
    test_session.add(document)
    test_session.commit()

    retrieved = test_session.query(Document).filter_by(content="Test content").first()
    assert retrieved is not None
    assert retrieved.corpus_id == corpus.id
    assert retrieved.language_code == "en"
    assert retrieved.type_id == doc_type.id

def test_create_vocabulary_word(test_session):
    corpus = Corpus(name="Test Corpus for Vocabulary")
    test_session.add(corpus)
    test_session.commit()

    vocab_word = VocabularyWord(corpus_id=corpus.id, word="test")
    test_session.add(vocab_word)
    test_session.commit()

    retrieved = test_session.query(VocabularyWord).filter_by(word="test").first()
    assert retrieved is not None
    assert retrieved.corpus_id == corpus.id

def test_create_embedder(test_session):
    embedder = Embedder(name="Test Embedder", description="A test embedder")
    test_session.add(embedder)
    test_session.commit()

    retrieved = test_session.query(Embedder).filter_by(name="Test Embedder").first()
    assert retrieved is not None
    assert retrieved.name == "Test Embedder"

def test_create_embedding(test_session):
    embedder = Embedder(name="Test Embedder for Embedding")
    document = Document(content="Test content for embedding")
    test_session.add(embedder)
    test_session.add(document)
    test_session.commit()

    embedding = Embedding(embedder_id=embedder.id, document_id=document.id, vector=[0.1, 0.2, 0.3])
    test_session.add(embedding)
    test_session.commit()

    retrieved = test_session.query(Embedding).filter_by(document_id=document.id).first()
    assert retrieved is not None
    assert retrieved.embedder_id == embedder.id
    assert retrieved.vector == [0.1, 0.2, 0.3]

def test_create_vocabulary_word_embedding(test_session):
    vocab_word = VocabularyWord(word="test_word")
    embedder = Embedder(name="Test Embedder for Vocabulary Word Embedding")
    test_session.add(vocab_word)
    test_session.add(embedder)
    test_session.commit()

    vocab_word_embedding = VocabularyWordEmbedding(vocabulary_word_id=vocab_word.id, embedder_id=embedder.id)
    test_session.add(vocab_word_embedding)
    test_session.commit()

    retrieved = test_session.query(VocabularyWordEmbedding).filter_by(vocabulary_word_id=vocab_word.id).first()
    assert retrieved is not None
    assert retrieved.embedder_id == embedder.id

def test_create_topic_model(test_session):
    topic_model = TopicModel(name="Test Topic Model", description="A test topic model")
    test_session.add(topic_model)
    test_session.commit()

    retrieved = test_session.query(TopicModel).filter_by(name="Test Topic Model").first()
    assert retrieved is not None
    assert retrieved.name == "Test Topic Model"

def test_create_topic_model_corpus_result(test_session):
    topic_model = TopicModel(name="Test Topic Model for Corpus Result")
    corpus = Corpus(name="Test Corpus for Topic Model Result")
    test_session.add(topic_model)
    test_session.add(corpus)
    test_session.commit()

    result = TopicModelCorpusResult(topic_model_id=topic_model.id, corpus_id=corpus.id, topics={"topic1": 0.5})
    test_session.add(result)
    test_session.commit()

    retrieved = test_session.query(TopicModelCorpusResult).filter_by(corpus_id=corpus.id).first()
    assert retrieved is not None
    assert retrieved.topics == {"topic1": 0.5}

def test_create_performance_metric(test_session):
    metric = PerformanceMetric(name="Test Metric", description="A test performance metric")
    test_session.add(metric)
    test_session.commit()

    retrieved = test_session.query(PerformanceMetric).filter_by(name="Test Metric").first()
    assert retrieved is not None
    assert retrieved.name == "Test Metric"

def test_create_result_performance(test_session):
    result = TopicModelCorpusResult()
    metric = PerformanceMetric(name="Test Metric for Result Performance")
    test_session.add(result)
    test_session.add(metric)
    test_session.commit()

    result_performance = ResultPerformance(topic_model_corpus_result_id=result.id, performance_metric_id=metric.id, value=0.95)
    test_session.add(result_performance)
    test_session.commit()

    retrieved = test_session.query(ResultPerformance).filter_by(value=0.95).first()
    assert retrieved is not None
    assert retrieved.performance_metric_id == metric.id

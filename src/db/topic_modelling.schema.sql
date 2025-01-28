CREATE SCHEMA IF NOT EXISTS topic_modelling;

CREATE TABLE topic_modelling.corpus (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE topic_modelling.document_type (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO topic_modelling.document_type (name, description) VALUES 
    ('raw', 'Raw text document'), 
    ('preprocessed', 'Preprocessed text document'), 
    ('vocabulary_only', 'Vocabulary word list');

CREATE TABLE topic_modelling.document (
    id SERIAL PRIMARY KEY,
    corpus_id INTEGER REFERENCES topic_modelling.corpus(id),
    content TEXT,
    language_code VARCHAR(10),
    type_id INTEGER REFERENCES topic_modelling.document_type(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE topic_modelling.vocabulary_word (
    id SERIAL PRIMARY KEY,
    corpus_id INTEGER REFERENCES topic_modelling.corpus(id),
    word VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_id, word)
);

CREATE TABLE topic_modelling.embedder (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO topic_modelling.embedder (name, description) VALUES 
    ('bow', 'Bag of Words'), 
    ('tfidf', 'Term Frequency-Inverse Document Frequency'), 
    ('SBERT', 'Sentence-BERT');

CREATE TABLE topic_modelling.embedding (
    id SERIAL PRIMARY KEY,
    embedder_id INTEGER REFERENCES topic_modelling.embedder(id),
    document_id INTEGER REFERENCES topic_modelling.document(id),
    vector FLOAT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE topic_modelling.vocabulary_word_embedding (
    id SERIAL PRIMARY KEY,
    vocabulary_word_id INTEGER REFERENCES topic_modelling.vocabulary_word(id),
    embedder_id INTEGER REFERENCES topic_modelling.embedder(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE topic_modelling.topic_model (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE topic_modelling.topic_model_corpus_result (
    id SERIAL PRIMARY KEY,
    topic_model_id INTEGER REFERENCES topic_modelling.topic_model(id),
    corpus_id INTEGER REFERENCES topic_modelling.corpus(id),
    topics jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE topic_modelling.performance_metric (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE topic_modelling.result_performance (
    id SERIAL PRIMARY KEY,
    topic_model_corpus_result_id INTEGER REFERENCES topic_modelling.topic_model_corpus_result(id),
    performance_metric_id INTEGER REFERENCES topic_modelling.performance_metric(id),
    value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

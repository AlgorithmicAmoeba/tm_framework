CREATE SCHEMA IF NOT EXISTS pipeline;

-- This table stores preprocessed documents
CREATE TABLE IF NOT EXISTS pipeline.preprocessed_document (
    id SERIAL PRIMARY KEY,
    raw_document_hash VARCHAR(255) NOT NULL,
    corpus_name VARCHAR(255) NOT NULL,
    content TEXT,
    content_hash VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_name, raw_document_hash)
);

-- This table stores documents with only vocabulary words
CREATE TABLE IF NOT EXISTS pipeline.vocabulary_document (
    id SERIAL PRIMARY KEY,
    raw_document_hash VARCHAR(255) NOT NULL,
    corpus_name VARCHAR(255) NOT NULL,
    content TEXT,
    content_hash VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_name, raw_document_hash)
);

-- This table stores vocabulary words extracted from documents
CREATE TABLE IF NOT EXISTS pipeline.vocabulary_word (
    id SERIAL PRIMARY KEY,
    corpus_name VARCHAR(255) NOT NULL,
    word VARCHAR(255) NOT NULL,
    word_index INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_name, word),
    UNIQUE(corpus_name, word_index)
);

-- This table stores document-term matrix (TF-IDF values) as JSON arrays
CREATE TABLE IF NOT EXISTS pipeline.tfidf_vector (
    id SERIAL PRIMARY KEY,
    raw_document_hash VARCHAR(255) NOT NULL,
    corpus_name VARCHAR(255) NOT NULL,
    terms FLOAT[] NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_name, raw_document_hash)
);

-- This table stores raw documents that are actually used after preprocessing filters
CREATE TABLE IF NOT EXISTS pipeline.used_raw_document (
    id SERIAL PRIMARY KEY,
    document_hash VARCHAR(255) NOT NULL,
    corpus_name VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_name, document_hash)
);

CREATE TRIGGER set_current_timestamp_updated_at
BEFORE UPDATE ON pipeline.preprocessed_document
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();

CREATE TRIGGER set_current_timestamp_updated_at
BEFORE UPDATE ON pipeline.vocabulary_document
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();

CREATE TRIGGER set_current_timestamp_updated_at
BEFORE UPDATE ON pipeline.vocabulary_word
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();

CREATE TRIGGER set_current_timestamp_updated_at
BEFORE UPDATE ON pipeline.tfidf_vector
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();

CREATE TRIGGER set_current_timestamp_updated_at
BEFORE UPDATE ON pipeline.used_raw_document
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();



CREATE SCHEMA IF NOT EXISTS pipeline;

-- This table stores chunked documents
CREATE TABLE IF NOT EXISTS pipeline.boe_chunked_document (
    id SERIAL PRIMARY KEY,
    raw_document_hash VARCHAR(255) NOT NULL,
    corpus_name VARCHAR(255) NOT NULL,
    chunk_hash VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    chunk_vocabulary_words TEXT NOT NULL,  -- string of vocabulary words in the chunk with duplicates
    chunk_start INTEGER NOT NULL,
    chunk_end INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_name, raw_document_hash, chunk_hash)
);

-- Trigger for updated_at timestamp
CREATE TRIGGER set_current_timestamp_updated_at
BEFORE UPDATE ON pipeline.boe_chunked_document
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();
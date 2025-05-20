CREATE SCHEMA IF NOT EXISTS pipeline;

-- This table stores embeddings for document chunks
CREATE TABLE IF NOT EXISTS pipeline.embedded_document (
    id SERIAL PRIMARY KEY,
    raw_document_hash VARCHAR(255) NOT NULL,
    corpus_name VARCHAR(255) NOT NULL,
    chunk_hash VARCHAR(255) NOT NULL,
    embedding VECTOR(1536) NOT NULL,  -- OpenAI embeddings are 1536 dimensions
    model VARCHAR(255) NOT NULL,      -- Name of embedding model used
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_name, raw_document_hash, chunk_hash)
);

-- Trigger for updated_at timestamp
CREATE TRIGGER set_current_timestamp_updated_at
BEFORE UPDATE ON pipeline.embedded_document
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();

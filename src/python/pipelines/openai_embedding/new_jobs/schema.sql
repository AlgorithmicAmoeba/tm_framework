CREATE SCHEMA IF NOT EXISTS pipeline;

CREATE TABLE IF NOT EXISTS pipeline.embedding_job (
    id SERIAL PRIMARY KEY,
    chunk_hash VARCHAR(255) NOT NULL,
    chunk_content TEXT NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (chunk_hash)
);

-- Trigger for updated_at timestamp
CREATE TRIGGER set_current_timestamp_updated_at
BEFORE UPDATE ON pipeline.embedding_job
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();
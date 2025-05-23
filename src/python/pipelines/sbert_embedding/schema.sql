CREATE SCHEMA IF NOT EXISTS pipeline;


-- Schema for chunk embeddings table
CREATE TABLE IF NOT EXISTS pipeline.sbert_chunk_embedding (
    chunk_hash VARCHAR PRIMARY KEY,
    embedding FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
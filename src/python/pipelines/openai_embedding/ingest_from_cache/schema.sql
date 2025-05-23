CREATE SCHEMA IF NOT EXISTS pipeline;


-- Schema for chunk embeddings table
CREATE TABLE IF NOT EXISTS pipeline.chunk_embedding (
    chunk_hash VARCHAR PRIMARY KEY,
    embedding FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_chunk_embedding_hash 
ON pipeline.chunk_embedding(chunk_hash);

-- Add trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION pipeline.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_chunk_embedding_updated_at
    BEFORE UPDATE ON pipeline.chunk_embedding
    FOR EACH ROW
    EXECUTE FUNCTION pipeline.update_updated_at_column(); 
-- Table for storing BOE embeddings
CREATE TABLE IF NOT EXISTS pipeline.boe_embedding (
    id SERIAL PRIMARY KEY,
    boe_chunked_document_hash VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    vector FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(boe_chunked_document_hash, model_name)
);

CREATE TABLE IF NOT EXISTS pipeline.boe_embedding_sparse (
    id SERIAL PRIMARY KEY,
    boe_chunked_document_hash VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    sparse_vector JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(boe_chunked_document_hash, model_name)
);

-- Trigger to update updated_at column
CREATE OR REPLACE FUNCTION pipeline.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_updated_at_trigger 
    BEFORE UPDATE ON pipeline.boe_embedding 
    FOR EACH ROW 
    EXECUTE FUNCTION pipeline.update_updated_at_column();

CREATE TRIGGER update_updated_at_trigger 
    BEFORE UPDATE ON pipeline.boe_embedding_sparse 
    FOR EACH ROW 
    EXECUTE FUNCTION pipeline.update_updated_at_column();
-- Table for storing dimensionality-reduced BOE embeddings
CREATE TABLE IF NOT EXISTS pipeline.boe_embedding_reduced (
    id SERIAL PRIMARY KEY,
    boe_chunked_document_hash VARCHAR(255) NOT NULL,
    source_model_name VARCHAR(255) NOT NULL,  -- e.g., 'all-MiniLM-L6-v2', 'naver/splade-v3'
    source_type VARCHAR(50) NOT NULL,          -- 'dense' or 'sparse'
    algorithm VARCHAR(50) NOT NULL,            -- 'umap' or 'pca'
    target_dims INTEGER NOT NULL,              -- 10 or 20
    corpus_name VARCHAR(255) NOT NULL,
    vector FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(boe_chunked_document_hash, source_model_name, algorithm, target_dims)
);

-- Index for faster lookups by corpus and model
CREATE INDEX IF NOT EXISTS idx_boe_embedding_reduced_corpus_model 
    ON pipeline.boe_embedding_reduced(corpus_name, source_model_name);

-- Index for faster lookups by algorithm and dimensions
CREATE INDEX IF NOT EXISTS idx_boe_embedding_reduced_algo_dims 
    ON pipeline.boe_embedding_reduced(algorithm, target_dims);

-- Table for storing document embeddings aggregated from chunk embeddings
CREATE TABLE IF NOT EXISTS pipeline.boe_document_embedding (
    id SERIAL PRIMARY KEY,
    corpus_name VARCHAR(255) NOT NULL,
    raw_document_hash VARCHAR(255) NOT NULL,
    source_model_name VARCHAR(255) NOT NULL,  -- e.g., 'all-MiniLM-L6-v2', 'naver/splade-v3'
    algorithm VARCHAR(50) NOT NULL,            -- 'umap' or 'pca'
    target_dims INTEGER NOT NULL,              -- 20, 50, or 100
    padding_method VARCHAR(50) NOT NULL,
    vector FLOAT[] NOT NULL,
    chunk_count INTEGER NOT NULL,              -- original chunks before padding
    padded_to INTEGER NOT NULL,                -- fixed target length after padding
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_name, raw_document_hash, source_model_name, algorithm, target_dims, padding_method)
);

-- Index for faster lookups by corpus and model
CREATE INDEX IF NOT EXISTS idx_boe_document_embedding_corpus_model
    ON pipeline.boe_document_embedding(corpus_name, source_model_name);

-- Index for faster lookups by algorithm and dimensions
CREATE INDEX IF NOT EXISTS idx_boe_document_embedding_algo_dims
    ON pipeline.boe_document_embedding(algorithm, target_dims);

-- Index for document lookups
CREATE INDEX IF NOT EXISTS idx_boe_document_embedding_doc_hash
    ON pipeline.boe_document_embedding(raw_document_hash);

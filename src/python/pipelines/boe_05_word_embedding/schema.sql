-- Table for storing word embeddings derived from chunk embeddings
CREATE TABLE IF NOT EXISTS pipeline.boe_word_embedding (
    id SERIAL PRIMARY KEY,
    corpus_name VARCHAR(255) NOT NULL,
    word VARCHAR(255) NOT NULL,
    source_model_name VARCHAR(255) NOT NULL,  -- e.g., 'all-MiniLM-L6-v2', 'naver/splade-v3'
    algorithm VARCHAR(50) NOT NULL,            -- 'umap' or 'pca'
    target_dims INTEGER NOT NULL,              -- 20, 50, or 100
    padding_method VARCHAR(50) NOT NULL,
    vector FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_name, word, source_model_name, algorithm, target_dims, padding_method)
);

-- Index for faster lookups by corpus and model
CREATE INDEX IF NOT EXISTS idx_boe_word_embedding_corpus_model
    ON pipeline.boe_word_embedding(corpus_name, source_model_name);

-- Index for faster lookups by algorithm and dimensions
CREATE INDEX IF NOT EXISTS idx_boe_word_embedding_algo_dims
    ON pipeline.boe_word_embedding(algorithm, target_dims);

-- Index for word lookups
CREATE INDEX IF NOT EXISTS idx_boe_word_embedding_word
    ON pipeline.boe_word_embedding(word);

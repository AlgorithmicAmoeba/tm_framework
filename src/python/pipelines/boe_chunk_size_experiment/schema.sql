-- Schema for the BOE chunk-size experiment.
--
-- This experiment is fully self-contained: it owns its own document-embedding,
-- word-embedding, topic-model and timing tables (all prefixed boe_cse_, where
-- "cse" = chunk size experiment) so that it never touches the production
-- pipeline.boe_* tables. The only shared inputs are the upstream chunk
-- embeddings (pipeline.boe_embedding) and the corpus vocabulary documents.
--
-- The experiment varies target_chunk_count (the number of chunk slots each
-- document is padded/truncated to before stacking) for a fixed embedding slice:
--   source_model_name = 'all-MiniLM-L6-v2', algorithm = 'none', target_dims = 0,
--   padding_method = 'noise_only'.
--
-- Apply with: . .env && psql $DB_URI -f src/python/pipelines/boe_chunk_size_experiment/schema.sql

CREATE SCHEMA IF NOT EXISTS pipeline;

-- ---------------------------------------------------------------------------
-- Document embeddings (mirror of pipeline.boe_document_embedding)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline.boe_cse_document_embedding (
    id SERIAL PRIMARY KEY,
    corpus_name VARCHAR(255) NOT NULL,
    raw_document_hash VARCHAR(255) NOT NULL,
    source_model_name VARCHAR(255) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,            -- 'none', 'umap', or 'pca'
    target_dims INTEGER NOT NULL,              -- 0 for unreduced
    padding_method VARCHAR(50) NOT NULL,
    target_chunk_count INTEGER NOT NULL,
    vector FLOAT[] NOT NULL,
    chunk_count INTEGER NOT NULL,              -- original number of chunks for the document
    padded_to INTEGER NOT NULL,                -- chunk slots after padding/truncation
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (corpus_name, raw_document_hash, source_model_name, algorithm, target_dims, padding_method, target_chunk_count)
);

CREATE INDEX IF NOT EXISTS idx_boe_cse_document_embedding_lookup
    ON pipeline.boe_cse_document_embedding(corpus_name, source_model_name, algorithm, target_dims, padding_method, target_chunk_count);

CREATE INDEX IF NOT EXISTS idx_boe_cse_document_embedding_chunk_count
    ON pipeline.boe_cse_document_embedding(corpus_name, target_chunk_count);

-- ---------------------------------------------------------------------------
-- Word embeddings (mirror of pipeline.boe_word_embedding, chunk-count aware)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline.boe_cse_word_embedding (
    id SERIAL PRIMARY KEY,
    corpus_name VARCHAR(255) NOT NULL,
    word VARCHAR(255) NOT NULL,
    source_model_name VARCHAR(255) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,            -- 'none', 'umap', or 'pca'
    target_dims INTEGER NOT NULL,              -- 0 for unreduced
    padding_method VARCHAR(50) NOT NULL,
    target_chunk_count INTEGER NOT NULL,
    vector FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (corpus_name, word, source_model_name, algorithm, target_dims, padding_method, target_chunk_count)
);

CREATE INDEX IF NOT EXISTS idx_boe_cse_word_embedding_lookup
    ON pipeline.boe_cse_word_embedding(corpus_name, source_model_name, algorithm, target_dims, padding_method, target_chunk_count);

CREATE INDEX IF NOT EXISTS idx_boe_cse_word_embedding_word
    ON pipeline.boe_cse_word_embedding(word);

-- ---------------------------------------------------------------------------
-- Topic-model registry (mirror of pipeline.boe_topic_model)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline.boe_cse_topic_model (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO pipeline.boe_cse_topic_model (name, description) VALUES
    ('BERTopic', 'BERTopic model using BOE document embeddings (chunk-size experiment)'),
    ('ZeroShotTM', 'ZeroShotTM model using BOE document embeddings (chunk-size experiment)'),
    ('CombinedTM', 'CombinedTM model using BOE document embeddings (chunk-size experiment)'),
    ('KeyNMF', 'KeyNMF model using BOE document embeddings (chunk-size experiment)'),
    ('SemanticSignalSeparation', 'SemanticSignalSeparation model using BOE document embeddings (chunk-size experiment)'),
    ('GMM', 'GMM model using BOE document embeddings (chunk-size experiment)')
ON CONFLICT (name) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Topic-model results (mirror of pipeline.boe_topic_model_corpus_result)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline.boe_cse_topic_model_corpus_result (
    id SERIAL PRIMARY KEY,
    topic_model_id INTEGER REFERENCES pipeline.boe_cse_topic_model(id),
    corpus_name VARCHAR(255) NOT NULL,
    source_model_name VARCHAR(255) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    target_dims INTEGER NOT NULL,
    padding_method VARCHAR(50) NOT NULL,
    target_chunk_count INTEGER NOT NULL,
    topics jsonb,
    num_topics INTEGER NOT NULL,
    hyperparameters jsonb DEFAULT '{}',
    soft_delete BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_boe_cse_tm_result_corpus
    ON pipeline.boe_cse_topic_model_corpus_result(corpus_name);

CREATE INDEX IF NOT EXISTS idx_boe_cse_tm_result_model_id
    ON pipeline.boe_cse_topic_model_corpus_result(topic_model_id);

CREATE INDEX IF NOT EXISTS idx_boe_cse_tm_result_combo
    ON pipeline.boe_cse_topic_model_corpus_result(
        corpus_name, topic_model_id, num_topics,
        source_model_name, algorithm, target_dims, padding_method, target_chunk_count
    );

-- ---------------------------------------------------------------------------
-- Performance metrics (mirror of pipeline.boe_topic_model_performance)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline.boe_cse_topic_model_performance (
    id SERIAL PRIMARY KEY,
    boe_cse_topic_model_corpus_result_id INTEGER REFERENCES pipeline.boe_cse_topic_model_corpus_result(id),
    metric_name VARCHAR(50) NOT NULL,
    metric_value JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(boe_cse_topic_model_corpus_result_id, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_boe_cse_tm_performance_result_id
    ON pipeline.boe_cse_topic_model_performance(boe_cse_topic_model_corpus_result_id);

-- ---------------------------------------------------------------------------
-- Timing results (mirror of pipeline.timing_result, with an explicit
-- target_chunk_count column for easy chunk-size analysis)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline.boe_cse_timing_result (
    id SERIAL PRIMARY KEY,
    pipeline_stage VARCHAR(100) NOT NULL,        -- 'boe_doc_embedding', 'boe_word_embedding', 'boe_topic_model'
    corpus_name VARCHAR(255) NOT NULL,
    model_name VARCHAR(255),                     -- topic model name or embedding model name (nullable)
    num_topics INTEGER,                          -- NULL for embedding stages
    target_chunk_count INTEGER NOT NULL,
    repeat_number INTEGER NOT NULL DEFAULT 1,
    duration_seconds DOUBLE PRECISION NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_boe_cse_timing_stage
    ON pipeline.boe_cse_timing_result(pipeline_stage);

CREATE INDEX IF NOT EXISTS idx_boe_cse_timing_corpus
    ON pipeline.boe_cse_timing_result(corpus_name, target_chunk_count);

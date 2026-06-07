-- Schema for the BOE "normal word-embedding" ablation experiment.
--
-- This experiment is a word-embedding ablation against the BOE@1 condition. It
-- reuses the EXISTING first-chunk document embeddings stored in
-- pipeline.boe_cse_document_embedding (source_model_name='all-MiniLM-L6-v2',
-- algorithm='none', target_dims=0, padding_method='noise_only',
-- target_chunk_count=1) as document representations, but replaces the
-- BOE-derived (TF-IDF + Ridge) word embeddings with "normal" word embeddings:
-- each vocabulary word is encoded DIRECTLY with
-- SentenceTransformer('all-MiniLM-L6-v2'), landing in the SAME 384-d space as
-- the document embeddings.
--
-- All experiment-owned tables are prefixed boe_nwe_ ("normal word embedding")
-- so production / other experiments are never touched.
--
-- Apply with: . .env && psql $DB_URI -f src/python/pipelines/boe_normal_we_experiment/schema.sql

CREATE SCHEMA IF NOT EXISTS pipeline;

-- ---------------------------------------------------------------------------
-- Normal (directly-encoded) word embeddings.
-- One row per (corpus, word) encoded directly with all-MiniLM-L6-v2 -> 384-d.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline.boe_nwe_word_embedding (
    id SERIAL PRIMARY KEY,
    corpus_name VARCHAR(255) NOT NULL,
    word VARCHAR(255) NOT NULL,
    source_model_name VARCHAR(255) NOT NULL,   -- always 'all-MiniLM-L6-v2'
    vector FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (corpus_name, word, source_model_name)
);

CREATE INDEX IF NOT EXISTS idx_boe_nwe_word_embedding_lookup
    ON pipeline.boe_nwe_word_embedding(corpus_name, source_model_name);

CREATE INDEX IF NOT EXISTS idx_boe_nwe_word_embedding_word
    ON pipeline.boe_nwe_word_embedding(word);

-- ---------------------------------------------------------------------------
-- Topic-model results.
-- `family` stores the model family name directly ('KeyNMF' /
-- 'SemanticSignalSeparation') so no separate model registry is needed.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline.boe_nwe_topic_model_corpus_result (
    id SERIAL PRIMARY KEY,
    family VARCHAR(255) NOT NULL,              -- 'KeyNMF' or 'SemanticSignalSeparation'
    corpus_name VARCHAR(255) NOT NULL,
    source_model_name VARCHAR(255) NOT NULL,   -- always 'all-MiniLM-L6-v2'
    num_topics INTEGER NOT NULL,
    topics jsonb,
    hyperparameters jsonb DEFAULT '{}',
    soft_delete BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_boe_nwe_tm_result_corpus
    ON pipeline.boe_nwe_topic_model_corpus_result(corpus_name);

CREATE INDEX IF NOT EXISTS idx_boe_nwe_tm_result_combo
    ON pipeline.boe_nwe_topic_model_corpus_result(
        corpus_name, family, num_topics, source_model_name
    );

-- ---------------------------------------------------------------------------
-- Performance metrics (mirror of pipeline.boe_cse_topic_model_performance).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline.boe_nwe_topic_model_performance (
    id SERIAL PRIMARY KEY,
    boe_nwe_topic_model_corpus_result_id INTEGER
        REFERENCES pipeline.boe_nwe_topic_model_corpus_result(id),
    metric_name VARCHAR(50) NOT NULL,
    metric_value JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(boe_nwe_topic_model_corpus_result_id, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_boe_nwe_tm_performance_result_id
    ON pipeline.boe_nwe_topic_model_performance(boe_nwe_topic_model_corpus_result_id);

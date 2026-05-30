-- Table for storing timing results of topic models and BOE embedding computations
CREATE TABLE IF NOT EXISTS pipeline.timing_result (
    id SERIAL PRIMARY KEY,
    pipeline_stage VARCHAR(100) NOT NULL,        -- 'standard_topic_model', 'boe_topic_model', 'boe_doc_embedding', 'boe_word_embedding'
    corpus_name VARCHAR(255) NOT NULL,
    model_name VARCHAR(255),                     -- topic model name or embedding model name (nullable for some stages)
    num_topics INTEGER,                          -- NULL for embedding stages
    repeat_number INTEGER NOT NULL DEFAULT 1,
    duration_seconds DOUBLE PRECISION NOT NULL,
    metadata JSONB DEFAULT '{}',                 -- for additional parameters like embedding combo details
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster lookups by pipeline stage
CREATE INDEX IF NOT EXISTS idx_timing_result_pipeline_stage
    ON pipeline.timing_result(pipeline_stage);

-- Index for faster lookups by corpus name
CREATE INDEX IF NOT EXISTS idx_timing_result_corpus_name
    ON pipeline.timing_result(corpus_name);

-- Index for faster lookups by model name
CREATE INDEX IF NOT EXISTS idx_timing_result_model_name
    ON pipeline.timing_result(model_name);

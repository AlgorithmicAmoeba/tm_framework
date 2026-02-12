CREATE SCHEMA IF NOT EXISTS pipeline;

-- Table to store performance metrics for BOE topic models
CREATE TABLE IF NOT EXISTS pipeline.boe_topic_model_performance (
    id SERIAL PRIMARY KEY,
    boe_topic_model_corpus_result_id INTEGER REFERENCES pipeline.boe_topic_model_corpus_result(id),
    metric_name VARCHAR(50) NOT NULL,
    metric_value JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(boe_topic_model_corpus_result_id, metric_name)
);

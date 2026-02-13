CREATE SCHEMA IF NOT EXISTS pipeline;

CREATE TABLE IF NOT EXISTS pipeline.boe_topic_model_performance (
    id SERIAL PRIMARY KEY,
    boe_topic_model_corpus_result_id INTEGER REFERENCES pipeline.boe_topic_model_corpus_result(id),
    metric_name VARCHAR(50) NOT NULL,
    metric_value JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(boe_topic_model_corpus_result_id, metric_name)
);

CREATE TRIGGER set_current_timestamp_updated_at
BEFORE UPDATE ON pipeline.boe_topic_model_performance
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();

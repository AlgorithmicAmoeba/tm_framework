CREATE SCHEMA IF NOT EXISTS pipeline;

-- Table to store performance metrics for topic models
CREATE TABLE IF NOT EXISTS pipeline.topic_model_performance (
    id SERIAL PRIMARY KEY,
    topic_model_corpus_result_id INTEGER REFERENCES pipeline.topic_model_corpus_result(id),
    metric_name VARCHAR(50) NOT NULL,
    metric_value JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(topic_model_corpus_result_id, metric_name)
);

-- Create trigger for updated_at
CREATE TRIGGER set_current_timestamp_updated_at
BEFORE UPDATE ON pipeline.topic_model_performance
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();

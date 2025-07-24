CREATE SCHEMA IF NOT EXISTS pipeline;

-- Table to store corpus-level features (statistical metrics)
CREATE TABLE IF NOT EXISTS pipeline.corpus_features (
    id SERIAL PRIMARY KEY,
    corpus_name VARCHAR(255) REFERENCES pipeline.corpus(name),
    feature_name VARCHAR(255) NOT NULL,
    feature_value FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_name, feature_name)
);

-- Table to store raw distribution data (feature vectors)
CREATE TABLE IF NOT EXISTS pipeline.corpus_distributions (
    id SERIAL PRIMARY KEY,
    corpus_name VARCHAR(255) REFERENCES pipeline.corpus(name),
    distribution_name VARCHAR(255) NOT NULL,
    data_points FLOAT[] NOT NULL,
    num_points INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_name, distribution_name)
);

-- Trigger to update the updated_at timestamp
CREATE TRIGGER set_current_timestamp_updated_at_corpus_features
BEFORE UPDATE ON pipeline.corpus_features
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();

CREATE TRIGGER set_current_timestamp_updated_at_corpus_distributions
BEFORE UPDATE ON pipeline.corpus_distributions
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();
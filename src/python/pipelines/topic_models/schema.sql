CREATE SCHEMA IF NOT EXISTS pipeline;

CREATE TABLE pipeline.topic_model (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE pipeline.topic_model_corpus_result (
    id SERIAL PRIMARY KEY,
    topic_model_id INTEGER REFERENCES pipeline.topic_model(id),
    corpus_id INTEGER REFERENCES pipeline.corpus(id),
    topics jsonb,
    soft_delete BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


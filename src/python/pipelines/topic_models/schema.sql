CREATE SCHEMA IF NOT EXISTS pipeline;

CREATE TABLE pipeline.topic_model (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert LDA as a topic model
INSERT INTO pipeline.topic_model (name, description) 
VALUES ('LDA', 'Latent Dirichlet Allocation topic model')
ON CONFLICT (name) DO NOTHING;

-- Insert BERTopic as a topic model
INSERT INTO pipeline.topic_model (name, description) 
VALUES ('BERTopic', 'BERTopic model using sentence transformers and clustering')
ON CONFLICT (name) DO NOTHING;

-- Insert ZeroShotTM as a topic model
INSERT INTO pipeline.topic_model (name, description) 
VALUES ('ZeroShotTM', 'ZeroShotTM model using sentence transformers and clustering')
ON CONFLICT (name) DO NOTHING;

-- Insert CombinedTM as a topic model
INSERT INTO pipeline.topic_model (name, description) 
VALUES ('CombinedTM', 'CombinedTM model using sentence transformers and clustering')
ON CONFLICT (name) DO NOTHING;


CREATE TABLE pipeline.topic_model_corpus_result (
    id SERIAL PRIMARY KEY,
    topic_model_id INTEGER REFERENCES pipeline.topic_model(id),
    corpus_id INTEGER REFERENCES pipeline.corpus(id),
    topics jsonb,
    num_topics INTEGER NOT NULL,
    soft_delete BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


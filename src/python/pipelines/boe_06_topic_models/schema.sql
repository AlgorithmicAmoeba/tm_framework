CREATE SCHEMA IF NOT EXISTS pipeline;

CREATE TABLE IF NOT EXISTS pipeline.boe_topic_model (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO pipeline.boe_topic_model (name, description)
VALUES ('BERTopic', 'BERTopic model using BOE document embeddings')
ON CONFLICT (name) DO NOTHING;

INSERT INTO pipeline.boe_topic_model (name, description)
VALUES ('ZeroShotTM', 'ZeroShotTM model using BOE document embeddings')
ON CONFLICT (name) DO NOTHING;

INSERT INTO pipeline.boe_topic_model (name, description)
VALUES ('CombinedTM', 'CombinedTM model using BOE document embeddings')
ON CONFLICT (name) DO NOTHING;

INSERT INTO pipeline.boe_topic_model (name, description)
VALUES ('KeyNMF', 'KeyNMF model using BOE document embeddings')
ON CONFLICT (name) DO NOTHING;

INSERT INTO pipeline.boe_topic_model (name, description)
VALUES ('SemanticSignalSeparation', 'SemanticSignalSeparation model using BOE document embeddings')
ON CONFLICT (name) DO NOTHING;

INSERT INTO pipeline.boe_topic_model (name, description)
VALUES ('GMM', 'GMM model using BOE document embeddings')
ON CONFLICT (name) DO NOTHING;

CREATE TABLE IF NOT EXISTS pipeline.boe_topic_model_corpus_result (
    id SERIAL PRIMARY KEY,
    topic_model_id INTEGER REFERENCES pipeline.boe_topic_model(id),
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

CREATE INDEX IF NOT EXISTS idx_boe_topic_model_corpus
    ON pipeline.boe_topic_model_corpus_result(corpus_name);

CREATE INDEX IF NOT EXISTS idx_boe_topic_model_id
    ON pipeline.boe_topic_model_corpus_result(topic_model_id);

CREATE INDEX IF NOT EXISTS idx_boe_topic_model_corpus_num_topics
    ON pipeline.boe_topic_model_corpus_result(corpus_name, topic_model_id, num_topics);

CREATE INDEX IF NOT EXISTS idx_boe_topic_model_embedding_params
    ON pipeline.boe_topic_model_corpus_result(
        source_model_name,
        algorithm,
        target_dims,
        padding_method,
        target_chunk_count
    );

-- To run this migration:
-- psql -h localhost -U postgres -d phd -f add_topic_models.sql

INSERT INTO topic_modelling.topic_model (name, description)
VALUES 
    ('lda', 'Latent Dirichlet Allocation - A generative probabilistic model that assumes documents are mixtures of topics'),
    ('zero-shot-tm', 'Zero-shot Topic Model - Uses large language models for topic classification without training'),
    ('bertopic', 'BERTopic - A topic modeling technique that leverages BERT embeddings and clustering algorithms')
ON CONFLICT (name) DO NOTHING;

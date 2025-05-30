-- Table for storing vocabulary word embeddings
CREATE TABLE IF NOT EXISTS pipeline.vocabulary_word_embeddings (
    id SERIAL PRIMARY KEY,
    vocabulary_word_id INTEGER NOT NULL REFERENCES pipeline.vocabulary_word(id),
    vector FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(vocabulary_word_id)
);

-- Trigger to update updated_at column
CREATE OR REPLACE FUNCTION pipeline.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_updated_at_trigger 
    BEFORE UPDATE ON pipeline.vocabulary_word_embeddings 
    FOR EACH ROW 
    EXECUTE FUNCTION pipeline.update_updated_at_column();
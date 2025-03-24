CREATE SCHEMA IF NOT EXISTS pipeline;

CREATE TABLE IF NOT EXISTS pipeline.corpus (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline.document (
    id SERIAL PRIMARY KEY,
    corpus_name VARCHAR(255),
    content_hash VARCHAR(255),
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(corpus_name, content_hash)
);


CREATE FUNCTION set_current_timestamp_updated_at()
    RETURNS TRIGGER AS $$
DECLARE
_new record;
BEGIN
  _new := NEW;
  _new."updated_at" = now();
RETURN _new;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_current_timestamp_updated_at
BEFORE UPDATE ON pipeline.corpus
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();

CREATE TRIGGER set_current_timestamp_updated_at
BEFORE UPDATE ON pipeline.document
FOR EACH ROW
EXECUTE FUNCTION set_current_timestamp_updated_at();


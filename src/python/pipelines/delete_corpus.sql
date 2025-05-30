-- Delete all data related to a specific corpus name
-- First delete from tables that reference other tables
DELETE FROM pipeline.topic_model_performance 
WHERE topic_model_corpus_result_id IN (
    SELECT id FROM pipeline.topic_model_corpus_result 
    WHERE corpus_id IN (SELECT id FROM pipeline.corpus WHERE name = 'imdb_reviews')
);

DELETE FROM pipeline.topic_model_corpus_result 
WHERE corpus_id IN (SELECT id FROM pipeline.corpus WHERE name = 'imdb_reviews');

DELETE FROM pipeline.sbert_chunk_embedding 
WHERE chunk_hash IN (
    SELECT chunk_hash FROM pipeline.chunked_document 
    WHERE corpus_name = 'imdb_reviews'
);

DELETE FROM pipeline.chunk_embedding 
WHERE chunk_hash IN (
    SELECT chunk_hash FROM pipeline.chunked_document 
    WHERE corpus_name = 'imdb_reviews'
);

DELETE FROM pipeline.embedding_job 
WHERE chunk_hash IN (
    SELECT chunk_hash FROM pipeline.chunked_document 
    WHERE corpus_name = 'imdb_reviews'
);

-- Delete from main tables
DELETE FROM pipeline.chunked_document 
WHERE corpus_name = 'imdb_reviews';

DELETE FROM pipeline.preprocessed_document 
WHERE corpus_name = 'imdb_reviews';

DELETE FROM pipeline.vocabulary_word 
WHERE corpus_name = 'imdb_reviews';

DELETE FROM pipeline.tfidf_vector 
WHERE corpus_name = 'imdb_reviews';

DELETE FROM pipeline.used_raw_document 
WHERE corpus_name = 'imdb_reviews';

DELETE FROM pipeline.vocabulary_document 
WHERE corpus_name = 'imdb_reviews';

-- DELETE FROM pipeline.document 
-- WHERE corpus_name = 'imdb_reviews';

-- -- Finally delete the corpus itself
-- DELETE FROM pipeline.corpus 
-- WHERE name = 'imdb_reviews'; 
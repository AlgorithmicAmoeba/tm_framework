ALTER TABLE topic_modelling.embedding 
ADD CONSTRAINT uix_embedding_document_embedder UNIQUE (document_id, embedder_id);

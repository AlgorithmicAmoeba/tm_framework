-- Add column parent_id to document table, which is a foreign key to the document table itself. Is nullable
ALTER TABLE topic_modelling.document
ADD COLUMN parent_id INT,
ADD CONSTRAINT fk_document_parent FOREIGN KEY (parent_id) REFERENCES topic_modelling.document(id);

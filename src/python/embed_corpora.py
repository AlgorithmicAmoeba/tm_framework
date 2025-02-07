import warnings
import tqdm
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.sql import insert

import database
import configuration as cfg
from models import Corpus, Document, Embedding, DocumentType
from hf_embedder import HFEmbedder  # Assuming HFEmbedder is the class for the embedder

def embed_corpora():
    config = cfg.load_config_from_env()
    db_config = config.database


    with database.get_session(db_config) as session:
        raw_type_id = session.query(DocumentType).filter_by(name='raw').first().id
        # Fetch all corpuses
        corpuses = session.query(Corpus).all()
        for corpus in corpuses:
            # Fetch raw documents for the corpus
            raw_documents = session.query(Document).filter_by(corpus_id=corpus.id, type_id=raw_type_id).all()
            raw_texts = [doc.content for doc in raw_documents]

            # Instantiate the embedder
            embedder = HFEmbedder()  # Adjust parameters if needed

            batch_size = 32  # Define batch size
            # Embed the raw texts in batches with progress bar
            pbar_batches = tqdm.tqdm(total=len(raw_texts) // batch_size + (1 if len(raw_texts) % batch_size > 0 else 0), desc=f"Embedding texts for corpus '{corpus.name}'")
            embeddings = []
            for i in range(0, len(raw_texts), batch_size):
                batch = raw_texts[i:i + batch_size]
                embeddings.extend(embedder.embed(batch))  # Assuming this method returns a list of embeddings
                pbar_batches.update(1)

            # Prepare data for bulk insert
            embedding_data = []
            for doc, embedding in zip(raw_documents, embeddings):
                embedding_data.append(dict(embedder_id=1, document_id=doc.id, vector=embedding.tolist()))  # Assuming embedder_id is 1

            # Insert embeddings into the database
            pbar = tqdm.tqdm(total=len(embedding_data), desc=f"Storing embeddings for corpus '{corpus.name}'")            
            session.execute(insert(Embedding), embedding_data)
            session.commit()
            pbar.update(len(embedding_data))
            pbar.close()

if __name__ == '__main__':
    embed_corpora()

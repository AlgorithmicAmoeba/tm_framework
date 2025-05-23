from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from tqdm import tqdm

from configuration import load_config_from_env
from database import get_session


def make_sbert_embeddings():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    max_seq_length = 128

    batch_size = 10_000
    model_batch_size = 1000

    config = load_config_from_env()
    db_config = config.database

    # Load the data: find chunks in pipeline.chunked_document that are not in pipeline.sbert_chunk_embedding
    with get_session(db_config) as session:
        query = text("""
            SELECT chunk_hash, content
            FROM pipeline.chunked_document
            WHERE chunk_hash NOT IN (SELECT chunk_hash FROM pipeline.sbert_chunk_embedding)
        """)
        result_iter = session.execute(query)

    # Get total count for progress bar
    with get_session(db_config) as session:
        count_query = text("""
            SELECT COUNT(*)
            FROM pipeline.chunked_document
            WHERE chunk_hash NOT IN (SELECT chunk_hash FROM pipeline.sbert_chunk_embedding)
        """)
        total_count = session.execute(count_query).scalar()

    with tqdm(total=total_count, desc="Processing chunks") as pbar:
        while True:
            data = result_iter.fetchmany(batch_size)
            if not data:
                break

            chunk_hashes, batch = zip(*data)

            # keep only the first 800 characters of the content
            batch = [content[:800] for content in batch]

            embeddings = model.encode(batch, batch_size=model_batch_size, max_seq_length=max_seq_length)
            embeddings = embeddings.tolist()

            # Insert the embeddings into the database
            with get_session(db_config) as session:
                session.execute(text("""
                    INSERT INTO pipeline.sbert_chunk_embedding (chunk_hash, embedding)
                    VALUES (:chunk_hash, :embedding)
                """),
                    [{"chunk_hash": chunk_hash, "embedding": embedding} for chunk_hash, embedding in zip(chunk_hashes, embeddings)]
                )
                session.commit()
            pbar.update(len(data))


if __name__ == "__main__":
    make_sbert_embeddings()
        
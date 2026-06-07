"""
Stage 1 of the BOE normal word-embedding ablation: word embeddings.

For each corpus that has BOE@1 document embeddings (target_chunk_count=1) in
pipeline.boe_cse_document_embedding, fetch the corpus vocabulary words and
encode each word DIRECTLY with SentenceTransformer('all-MiniLM-L6-v2') -> 384-d,
then insert into pipeline.boe_nwe_word_embedding.

This is the key contrast against the BOE pipeline: instead of deriving word
vectors from TF-IDF + Ridge, words are embedded by the sentence-transformer in
the SAME space as the document embeddings.

Run with:
    set -a && . .env && set +a
    uv run src/python/pipelines/boe_normal_we_experiment/word_embedding.py
"""
import logging

from sentence_transformers import SentenceTransformer
from sqlalchemy import text

import configuration as cfg
from database import get_session
from pipelines.boe_normal_we_experiment.common import (
    SOURCE_MODEL_NAME,
    DOC_ALGORITHM,
    DOC_TARGET_DIMS,
    DOC_PADDING_METHOD,
    DOC_TARGET_CHUNK_COUNT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

INSERT_BATCH_SIZE = 5000
ENCODE_BATCH_SIZE = 256


def get_corpora_with_doc_embeddings(session) -> list[str]:
    """Corpora that have BOE@1 (tcc=1) document embeddings to reuse."""
    query = text("""
        SELECT DISTINCT corpus_name
        FROM pipeline.boe_cse_document_embedding
        WHERE source_model_name = :source_model_name
          AND algorithm = :algorithm
          AND target_dims = :target_dims
          AND padding_method = :padding_method
          AND target_chunk_count = :target_chunk_count
        ORDER BY corpus_name
    """)
    return [row[0] for row in session.execute(query, {
        "source_model_name": SOURCE_MODEL_NAME,
        "algorithm": DOC_ALGORITHM,
        "target_dims": DOC_TARGET_DIMS,
        "padding_method": DOC_PADDING_METHOD,
        "target_chunk_count": DOC_TARGET_CHUNK_COUNT,
    })]


def get_vocabulary_words(session, corpus_name: str) -> list[str]:
    """Ordered vocabulary words for a corpus."""
    query = text("""
        SELECT word
        FROM pipeline.vocabulary_word
        WHERE corpus_name = :corpus_name
        ORDER BY word_index
    """)
    return [row[0] for row in session.execute(query, {"corpus_name": corpus_name})]


def count_existing_word_embeddings(session, corpus_name: str) -> int:
    query = text("""
        SELECT COUNT(*)
        FROM pipeline.boe_nwe_word_embedding
        WHERE corpus_name = :corpus_name
          AND source_model_name = :source_model_name
    """)
    return session.execute(query, {
        "corpus_name": corpus_name,
        "source_model_name": SOURCE_MODEL_NAME,
    }).scalar() or 0


def store_word_embeddings(session, corpus_name, words, vectors) -> None:
    insert_query = text("""
        INSERT INTO pipeline.boe_nwe_word_embedding
        (corpus_name, word, source_model_name, vector)
        VALUES (:corpus_name, :word, :source_model_name, :vector)
        ON CONFLICT (corpus_name, word, source_model_name)
        DO NOTHING
    """)

    batch = []
    for word, vector in zip(words, vectors):
        batch.append({
            "corpus_name": corpus_name,
            "word": word,
            "source_model_name": SOURCE_MODEL_NAME,
            "vector": [float(x) for x in vector],
        })
        if len(batch) >= INSERT_BATCH_SIZE:
            session.execute(insert_query, batch)
            session.commit()
            batch = []
    if batch:
        session.execute(insert_query, batch)
        session.commit()


def process_corpus(session, model, corpus_name: str) -> None:
    words = get_vocabulary_words(session, corpus_name)
    if not words:
        logging.warning("No vocabulary words for %s, skipping", corpus_name)
        return

    existing = count_existing_word_embeddings(session, corpus_name)
    if existing >= len(words):
        logging.info(
            "Word embeddings already complete for %s (%d/%d), skipping",
            corpus_name, existing, len(words),
        )
        return

    logging.info("Encoding %d words for %s (existing=%d)", len(words), corpus_name, existing)
    vectors = model.encode(
        words,
        batch_size=ENCODE_BATCH_SIZE,
        convert_to_tensor=False,
        show_progress_bar=True,
    )
    store_word_embeddings(session, corpus_name, words, vectors)
    logging.info("Stored word embeddings for %s (%d words, dim=%d)",
                 corpus_name, len(words), len(vectors[0]))


def main(only_corpus: str | None = None) -> None:
    config = cfg.load_config_from_env()

    logging.info("Loading SentenceTransformer('%s')", SOURCE_MODEL_NAME)
    model = SentenceTransformer(SOURCE_MODEL_NAME)

    with get_session(config.database) as session:
        corpora = get_corpora_with_doc_embeddings(session)
        logging.info("Corpora with BOE@1 doc embeddings: %s", corpora)

        if only_corpus is not None:
            if only_corpus not in corpora:
                logging.warning("Requested corpus %s has no BOE@1 doc embeddings", only_corpus)
                return
            corpora = [only_corpus]

        for corpus_name in corpora:
            try:
                process_corpus(session, model, corpus_name)
            except Exception:
                logging.exception("Error processing corpus %s", corpus_name)
                continue

    logging.info("Normal word-embedding stage complete")


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    main(only_corpus=only)

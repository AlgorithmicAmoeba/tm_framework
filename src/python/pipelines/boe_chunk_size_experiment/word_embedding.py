"""
Stage 2 of the BOE chunk-size experiment: word embeddings.

Derives word embeddings (TF-IDF + Ridge) from the experiment's own document
embeddings (pipeline.boe_cse_document_embedding), once per target_chunk_count,
and writes them into the experiment's own table
(pipeline.boe_cse_word_embedding). Timing of the derive step is recorded into
pipeline.boe_cse_timing_result.

This stage reuses ONLY the core math of the production pipeline
(BOEWordEmbeddingPipeline.derive_word_embeddings); all DB access targets the
chunk-size-experiment tables.
"""
import logging
import time

import numpy as np
from sqlalchemy import text

import configuration as cfg
from database import get_session
from pipelines.boe_05_word_embedding.main import BOEWordEmbeddingPipeline
from pipelines.boe_chunk_size_experiment.common import (
    SOURCE_MODEL_NAME,
    ALGORITHM,
    TARGET_DIMS,
    PADDING_METHOD,
    EXPERIMENT_CHUNK_COUNTS,
    DOC_EMBEDDING_TABLE,
    WORD_EMBEDDING_TABLE,
    store_timing_result,
)

INSERT_BATCH_SIZE = 5000


def word_embeddings_exist(session, corpus_name, target_chunk_count):
    """Return True if word embeddings already exist for this combination."""
    query = text("""
        SELECT 1 FROM pipeline.boe_cse_word_embedding
        WHERE corpus_name = :corpus_name
          AND source_model_name = :source_model_name
          AND algorithm = :algorithm
          AND target_dims = :target_dims
          AND padding_method = :padding_method
          AND target_chunk_count = :target_chunk_count
        LIMIT 1
    """)
    result = session.execute(query, {
        "corpus_name": corpus_name,
        "source_model_name": SOURCE_MODEL_NAME,
        "algorithm": ALGORITHM,
        "target_dims": TARGET_DIMS,
        "padding_method": PADDING_METHOD,
        "target_chunk_count": target_chunk_count,
    })
    return result.fetchone() is not None


def fetch_document_data(session, corpus_name, target_chunk_count):
    """
    Fetch document vocabulary content and document embeddings from the
    experiment's own document-embedding table.

    Returns:
        Tuple of (raw_document_hashes, vocabulary_documents, embeddings_matrix)
    """
    logging.info(
        "Fetching document data for %s (chunks=%d)",
        corpus_name,
        target_chunk_count,
    )

    query = text("""
        SELECT d.raw_document_hash, d.content, e.vector
        FROM pipeline.vocabulary_document d
        JOIN pipeline.boe_cse_document_embedding e
          ON d.raw_document_hash = e.raw_document_hash AND d.corpus_name = e.corpus_name
        WHERE d.corpus_name = :corpus_name
          AND e.source_model_name = :source_model_name
          AND e.algorithm = :algorithm
          AND e.target_dims = :target_dims
          AND e.padding_method = :padding_method
          AND e.target_chunk_count = :target_chunk_count
        ORDER BY d.raw_document_hash
    """)

    result = session.execute(query, {
        "corpus_name": corpus_name,
        "source_model_name": SOURCE_MODEL_NAME,
        "algorithm": ALGORITHM,
        "target_dims": TARGET_DIMS,
        "padding_method": PADDING_METHOD,
        "target_chunk_count": target_chunk_count,
    })

    raw_document_hashes = []
    vocabulary_documents = []
    vectors = []

    for row in result:
        raw_document_hashes.append(row[0])
        vocabulary_documents.append(row[1])
        vectors.append(row[2])

    if not vectors:
        logging.warning(
            "No document data found for %s (chunks=%d)",
            corpus_name,
            target_chunk_count,
        )
        return [], [], np.array([])

    embeddings_matrix = np.array(vectors, dtype=np.float32)
    logging.info(
        "Fetched %d documents with embeddings shape %s",
        len(raw_document_hashes),
        embeddings_matrix.shape,
    )

    return raw_document_hashes, vocabulary_documents, embeddings_matrix


def store_word_embeddings(session, corpus_name, target_chunk_count, vocabulary_words, word_embeddings):
    """Batched insert of word embeddings into pipeline.boe_cse_word_embedding."""
    logging.info(
        "Storing %d word embeddings for %s (chunks=%d)",
        len(vocabulary_words),
        corpus_name,
        target_chunk_count,
    )

    insert_query = text("""
        INSERT INTO pipeline.boe_cse_word_embedding
        (corpus_name, word, source_model_name, algorithm, target_dims, padding_method, target_chunk_count, vector)
        VALUES (:corpus_name, :word, :source_model_name, :algorithm, :target_dims, :padding_method, :target_chunk_count, :vector)
        ON CONFLICT (corpus_name, word, source_model_name, algorithm, target_dims, padding_method, target_chunk_count)
        DO NOTHING
    """)

    batch = []
    for i, word in enumerate(vocabulary_words):
        batch.append({
            "corpus_name": corpus_name,
            "word": word,
            "source_model_name": SOURCE_MODEL_NAME,
            "algorithm": ALGORITHM,
            "target_dims": TARGET_DIMS,
            "padding_method": PADDING_METHOD,
            "target_chunk_count": target_chunk_count,
            "vector": word_embeddings[i].tolist(),
        })

        if len(batch) >= INSERT_BATCH_SIZE:
            session.execute(insert_query, batch)
            session.commit()
            batch = []

    if batch:
        session.execute(insert_query, batch)
        session.commit()

    logging.info("Stored word embeddings for %s (chunks=%d)", corpus_name, target_chunk_count)


def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.FileHandler("boe_cse_word_embedding.log"))

    config = cfg.load_config_from_env()
    db_config = config.database

    with get_session(db_config) as session:
        print("BOE Chunk-Size Experiment: Word Embedding Stage")
        print("=" * 60)
        print("Method: TF-IDF (document-level) + Ridge Regression")
        print(f"Source table: {DOC_EMBEDDING_TABLE}")
        print(f"Target table: {WORD_EMBEDDING_TABLE}")
        print("=" * 60)

        for corpus_name, chunk_counts in EXPERIMENT_CHUNK_COUNTS.items():
            for target_chunk_count in chunk_counts:
                print(f"\nProcessing {corpus_name} (chunks={target_chunk_count})")
                print("-" * 40)

                try:
                    # Skip if word embeddings already exist
                    if word_embeddings_exist(session, corpus_name, target_chunk_count):
                        logging.info(
                            "Word embeddings already exist for %s (chunks=%d), skipping",
                            corpus_name,
                            target_chunk_count,
                        )
                        print("  skipped (already exists)")
                        continue

                    # Fetch document data from the experiment's doc-embedding table
                    raw_document_hashes, vocabulary_documents, document_embeddings = fetch_document_data(
                        session, corpus_name, target_chunk_count
                    )

                    if len(raw_document_hashes) == 0:
                        logging.warning(
                            "No documents found for %s (chunks=%d), skipping",
                            corpus_name,
                            target_chunk_count,
                        )
                        print("  skipped (no documents)")
                        continue

                    # Time the derive step
                    pipeline = BOEWordEmbeddingPipeline()
                    start = time.perf_counter()
                    vocabulary_words, word_embeddings = pipeline.derive_word_embeddings(
                        vocabulary_documents, document_embeddings
                    )
                    duration = time.perf_counter() - start

                    if len(vocabulary_words) == 0:
                        logging.warning(
                            "Empty vocabulary for %s (chunks=%d), skipping",
                            corpus_name,
                            target_chunk_count,
                        )
                        print("  skipped (empty vocabulary)")
                        continue

                    # Record timing
                    store_timing_result(
                        session,
                        pipeline_stage="boe_word_embedding",
                        corpus_name=corpus_name,
                        target_chunk_count=target_chunk_count,
                        duration_seconds=duration,
                        model_name=SOURCE_MODEL_NAME,
                        num_topics=None,
                        repeat_number=1,
                        metadata={
                            "num_documents": len(raw_document_hashes),
                            "vocabulary_size": len(vocabulary_words),
                            "algorithm": ALGORITHM,
                            "target_dims": TARGET_DIMS,
                            "padding_method": PADDING_METHOD,
                        },
                    )

                    # Store word embeddings
                    store_word_embeddings(
                        session, corpus_name, target_chunk_count, vocabulary_words, word_embeddings
                    )

                    print(
                        f"  done: vocab size {len(vocabulary_words)}, "
                        f"{len(raw_document_hashes)} docs, {duration:.2f}s"
                    )

                except Exception as e:
                    logging.error(
                        "Error processing %s (chunks=%d): %s",
                        corpus_name,
                        target_chunk_count,
                        str(e),
                    )
                    continue

    print("\n" + "=" * 60)
    print("WORD EMBEDDING DERIVATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

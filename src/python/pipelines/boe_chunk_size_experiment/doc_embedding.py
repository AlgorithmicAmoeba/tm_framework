"""
BOE chunk-size experiment — Stage 1: document embeddings.

Generates document embeddings for multiple ``target_chunk_count`` values for a
fixed embedding slice (all-MiniLM-L6-v2, unreduced, noise_only padding) and
writes them into the experiment's own table
``pipeline.boe_cse_document_embedding``. Compute timing for each
(corpus, target_chunk_count) combination is recorded in
``pipeline.boe_cse_timing_result``.

The production BOEDocEmbeddingPipeline is reused for all heavy lifting; only the
final insert targets the experiment table instead of the production one.
"""
import logging
import time
from typing import Any

import configuration as cfg
from database import get_session
from sqlalchemy import text

from pipelines.boe_04_doc_embedding.main import BOEDocEmbeddingPipeline
from pipelines.boe_chunk_size_experiment.common import (
    ALGORITHM,
    DOC_EMBEDDING_TABLE,
    EXPERIMENT_CHUNK_COUNTS,
    KNN_K,
    NOISE_SCALE,
    PADDING_METHOD,
    SOURCE_MODEL_NAME,
    TARGET_DIMS,
    store_timing_result,
)


def count_existing_doc_embeddings(
    session,
    corpus_name: str,
    target_chunk_count: int,
) -> int:
    """Count rows already in the experiment doc-embedding table for a combo."""
    query = text(f"""
        SELECT COUNT(*) FROM {DOC_EMBEDDING_TABLE}
        WHERE corpus_name = :corpus_name
        AND source_model_name = :source_model_name
        AND algorithm = :algorithm
        AND target_dims = :target_dims
        AND padding_method = :padding_method
        AND target_chunk_count = :target_chunk_count
    """)
    result = session.execute(query, {
        "corpus_name": corpus_name,
        "source_model_name": SOURCE_MODEL_NAME,
        "algorithm": ALGORITHM,
        "target_dims": TARGET_DIMS,
        "padding_method": PADDING_METHOD,
        "target_chunk_count": target_chunk_count,
    })
    return int(result.scalar() or 0)


def store_doc_embeddings_cse(
    session,
    corpus_name: str,
    target_chunk_count: int,
    doc_embeddings: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    """
    Store document embeddings into the experiment table.

    Mirrors BOEDocEmbeddingPipeline.store_document_embeddings but targets
    pipeline.boe_cse_document_embedding instead of the production table.

    Returns:
        Tuple of (inserted_count, existing_count)
    """
    total = len(doc_embeddings)
    logging.info(f"Storing {total} document embeddings into {DOC_EMBEDDING_TABLE}")

    insert_query = text(f"""
        INSERT INTO {DOC_EMBEDDING_TABLE}
        (corpus_name, raw_document_hash, source_model_name, algorithm,
         target_dims, padding_method, target_chunk_count, vector, chunk_count, padded_to)
        VALUES (:corpus_name, :raw_document_hash, :source_model_name, :algorithm,
                :target_dims, :padding_method, :target_chunk_count, :vector, :chunk_count, :padded_to)
        ON CONFLICT (corpus_name, raw_document_hash, source_model_name, algorithm, target_dims, padding_method, target_chunk_count)
        DO NOTHING
    """)

    batch_size = 5000
    rows: list[dict[str, Any]] = []
    inserted_count = 0

    def flush_batch(batch: list[dict[str, Any]]) -> int:
        if not batch:
            return 0
        result = session.execute(insert_query, batch)
        return int(result.rowcount or 0)

    for doc_hash, embed_data in doc_embeddings.items():
        rows.append({
            "corpus_name": corpus_name,
            "raw_document_hash": doc_hash,
            "source_model_name": SOURCE_MODEL_NAME,
            "algorithm": ALGORITHM,
            "target_dims": TARGET_DIMS,
            "padding_method": PADDING_METHOD,
            "target_chunk_count": target_chunk_count,
            "vector": embed_data["vector"].tolist(),
            "chunk_count": embed_data["chunk_count"],
            "padded_to": embed_data["padded_to"],
        })

        if len(rows) >= batch_size:
            inserted_count += flush_batch(rows)
            logging.info(f"Flushed {len(rows)} rows to the database. Total inserted: {inserted_count}")
            rows.clear()

    if rows:
        inserted_count += flush_batch(rows)

    session.commit()
    existing_count = total - inserted_count
    logging.info(f"Document embeddings stored: {inserted_count} inserted, {existing_count} existing")
    return inserted_count, existing_count


def main():
    """Main execution function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Direct logging to file
    logging.getLogger().addHandler(logging.FileHandler("boe_cse_doc_embedding.log"))

    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database

    with get_session(db_config) as session:
        print("BOE Chunk-Size Experiment — Document Embeddings")
        print("=" * 60)
        print(f"Source model: {SOURCE_MODEL_NAME}")
        print(f"Algorithm: {ALGORITHM}, target_dims: {TARGET_DIMS}")
        print(f"Padding method: {PADDING_METHOD}")
        print(f"Target table: {DOC_EMBEDDING_TABLE}")
        print("=" * 60)

        for corpus_name, chunk_counts in EXPERIMENT_CHUNK_COUNTS.items():
            for target_chunk_count in chunk_counts:
                logging.info("=" * 60)
                logging.info(
                    f"Processing corpus: {corpus_name} "
                    f"(target_chunk_count={target_chunk_count})"
                )
                logging.info("=" * 60)

                try:
                    pipeline = BOEDocEmbeddingPipeline(
                        target_chunk_count=target_chunk_count,
                        knn_k=KNN_K,
                        noise_scale=NOISE_SCALE,
                        padding_method=PADDING_METHOD,
                    )

                    # Skip check: already complete?
                    existing = count_existing_doc_embeddings(
                        session, corpus_name, target_chunk_count
                    )
                    expected = pipeline.count_expected_documents_unreduced(
                        session, corpus_name, SOURCE_MODEL_NAME
                    )
                    if expected > 0 and existing >= expected:
                        logging.info(
                            "Document embeddings already complete for %s "
                            "(target_chunk_count=%d): %d/%d, skipping",
                            corpus_name,
                            target_chunk_count,
                            existing,
                            expected,
                        )
                        continue

                    # Fetch unreduced chunk embeddings
                    chunk_hashes, raw_doc_hashes, chunk_starts, embeddings = (
                        pipeline.fetch_unreduced_chunk_embeddings(
                            session, corpus_name, SOURCE_MODEL_NAME
                        )
                    )
                    if len(chunk_hashes) == 0:
                        logging.warning(
                            "No unreduced chunk embeddings found for %s/%s, skipping",
                            corpus_name,
                            SOURCE_MODEL_NAME,
                        )
                        continue

                    # Group chunks by document
                    doc_groups = pipeline.group_chunks_by_document(
                        chunk_hashes, raw_doc_hashes, chunk_starts, embeddings
                    )

                    # Time the compute step
                    start = time.perf_counter()
                    doc_embeddings = pipeline.compute_document_embeddings(
                        doc_groups, embeddings
                    )
                    duration = time.perf_counter() - start

                    # Record timing
                    store_timing_result(
                        session,
                        pipeline_stage="boe_doc_embedding",
                        corpus_name=corpus_name,
                        target_chunk_count=target_chunk_count,
                        duration_seconds=duration,
                        model_name=SOURCE_MODEL_NAME,
                        num_topics=None,
                        repeat_number=1,
                        metadata={
                            "num_documents": len(doc_embeddings),
                            "num_chunks": len(chunk_hashes),
                            "algorithm": ALGORITHM,
                            "target_dims": TARGET_DIMS,
                            "padding_method": PADDING_METHOD,
                        },
                    )

                    # Store doc embeddings into the experiment table
                    inserted, existing_after = store_doc_embeddings_cse(
                        session, corpus_name, target_chunk_count, doc_embeddings
                    )

                    print(
                        f"\n{corpus_name} (target_chunk_count={target_chunk_count}): "
                        f"{inserted} inserted, {existing_after} existing, "
                        f"compute took {duration:.2f}s"
                    )

                except Exception as e:
                    logging.error(
                        "Error processing corpus %s (target_chunk_count=%s): %s",
                        corpus_name,
                        target_chunk_count,
                        str(e),
                    )
                    raise

    print("\n" + "=" * 60)
    print("BOE CSE DOCUMENT EMBEDDING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
Stage 2 of the BOE normal word-embedding ablation: topic-model training.

Trains ONLY KeyNMF and SemanticSignalSeparation (S^3) using:
  - DOCUMENT embeddings = first-chunk all-MiniLM-L6-v2 embeddings reused from
    pipeline.boe_cse_document_embedding at target_chunk_count=1 (BOE@1).
  - WORD embeddings = "normal" word embeddings directly encoded with
    all-MiniLM-L6-v2 (pipeline.boe_nwe_word_embedding).

This mirrors the chunk-size experiment runner's alignment / build / insert /
skip-complete logic, but with no chunk-count sweep and the simplified
boe_nwe_* result schema (family stored as a string column).

Run with:
    set -a && . .env && set +a
    while true; do
        uv run src/python/pipelines/boe_normal_we_experiment/experiment_runner.py
        rc=$?
        if [ $rc -eq 0 ]; then break; fi
        echo "Exit code: $rc - retrying..."
        sleep 1
    done
"""

import json
import logging
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm

from configuration import load_config_from_env
from database import get_session
from pipelines.topic_models.KeyNMF import KeyNMFWrapper
from pipelines.topic_models.semantic_signal_separation import SemanticSignalSeparationWrapper
from pipelines.topic_models.data_handling import get_vocabulary_documents, get_vocabulary, cleanup_model

from pipelines.boe_normal_we_experiment.common import (
    SOURCE_MODEL_NAME,
    DOC_ALGORITHM,
    DOC_TARGET_DIMS,
    DOC_PADDING_METHOD,
    DOC_TARGET_CHUNK_COUNT,
    FAMILIES,
    NUM_TOPICS_LIST,
    TARGET_RESULTS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_available_corpora(session: Session) -> list[str]:
    """Corpora that have BOTH BOE@1 doc embeddings AND normal word embeddings."""
    query = text("""
        SELECT DISTINCT d.corpus_name
        FROM pipeline.boe_cse_document_embedding d
        WHERE d.source_model_name = :source_model_name
          AND d.algorithm = :algorithm
          AND d.target_dims = :target_dims
          AND d.padding_method = :padding_method
          AND d.target_chunk_count = :target_chunk_count
          AND EXISTS (
              SELECT 1 FROM pipeline.boe_nwe_word_embedding w
              WHERE w.corpus_name = d.corpus_name
                AND w.source_model_name = :source_model_name
          )
        ORDER BY d.corpus_name
    """)
    return [row[0] for row in session.execute(query, {
        "source_model_name": SOURCE_MODEL_NAME,
        "algorithm": DOC_ALGORITHM,
        "target_dims": DOC_TARGET_DIMS,
        "padding_method": DOC_PADDING_METHOD,
        "target_chunk_count": DOC_TARGET_CHUNK_COUNT,
    })]


def count_existing_results(
    session: Session,
    corpus_name: str,
    family: str,
    num_topics: int,
) -> int:
    query = text("""
        SELECT COUNT(*)
        FROM pipeline.boe_nwe_topic_model_corpus_result
        WHERE corpus_name = :corpus_name
          AND family = :family
          AND num_topics = :num_topics
          AND source_model_name = :source_model_name
          AND soft_delete = FALSE
    """)
    return session.execute(query, {
        "corpus_name": corpus_name,
        "family": family,
        "num_topics": num_topics,
        "source_model_name": SOURCE_MODEL_NAME,
    }).scalar() or 0


def fetch_document_embeddings(session: Session, corpus_name: str) -> dict[str, np.ndarray]:
    """Fetch reused BOE@1 doc embeddings keyed by raw_document_hash."""
    query = text("""
        SELECT raw_document_hash, vector
        FROM pipeline.boe_cse_document_embedding
        WHERE corpus_name = :corpus_name
          AND source_model_name = :source_model_name
          AND algorithm = :algorithm
          AND target_dims = :target_dims
          AND padding_method = :padding_method
          AND target_chunk_count = :target_chunk_count
        ORDER BY raw_document_hash
    """)
    result = session.execute(query, {
        "corpus_name": corpus_name,
        "source_model_name": SOURCE_MODEL_NAME,
        "algorithm": DOC_ALGORITHM,
        "target_dims": DOC_TARGET_DIMS,
        "padding_method": DOC_PADDING_METHOD,
        "target_chunk_count": DOC_TARGET_CHUNK_COUNT,
    })
    return {row[0]: np.array(row[1], dtype=np.float32) for row in result}


def fetch_word_embeddings(session: Session, corpus_name: str) -> dict[str, np.ndarray]:
    """Fetch normal (directly-encoded) word embeddings keyed by word."""
    query = text("""
        SELECT word, vector
        FROM pipeline.boe_nwe_word_embedding
        WHERE corpus_name = :corpus_name
          AND source_model_name = :source_model_name
    """)
    result = session.execute(query, {
        "corpus_name": corpus_name,
        "source_model_name": SOURCE_MODEL_NAME,
    })
    return {row[0]: np.array(row[1], dtype=np.float32) for row in result}


def align_documents_and_embeddings(
    corpus_name: str,
    embedding_map: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    vocab_docs = get_vocabulary_documents(corpus_name)
    if not vocab_docs:
        return [], np.array([])

    documents: list[str] = []
    embeddings: list[np.ndarray] = []
    missing = 0

    for doc_hash, content in vocab_docs:
        vector = embedding_map.get(doc_hash)
        if vector is None:
            missing += 1
            continue
        documents.append(content)
        embeddings.append(vector)

    if missing:
        logging.warning("Missing embeddings for %s documents in corpus %s", missing, corpus_name)

    if not embeddings:
        return [], np.array([])

    return documents, np.vstack(embeddings)


def build_model(family: str, num_topics: int) -> tuple[Any, dict[str, Any]]:
    if family == "KeyNMF":
        model = KeyNMFWrapper(num_topics=num_topics)
        hparams = {"seed_phrase": model.seed_phrase}
        return model, hparams

    if family == "SemanticSignalSeparation":
        model = SemanticSignalSeparationWrapper(num_topics=num_topics)
        hparams = {"feature_importance": model.feature_importance}
        return model, hparams

    raise ValueError(f"Unknown topic-model family: {family}")


def run_experiments(target_results: int = TARGET_RESULTS) -> None:
    config = load_config_from_env()

    with get_session(config.database) as session:
        corpora = get_available_corpora(session)

    logging.info("Corpora (with doc embeddings + normal word embeddings): %s", corpora)
    logging.info("Families: %s", FAMILIES)

    total_steps = len(corpora) * len(FAMILIES) * len(NUM_TOPICS_LIST)
    pbar = tqdm(total=total_steps, desc="BOE normal-WE experiments")

    for corpus_name in corpora:
        vocabulary = get_vocabulary(corpus_name)
        if not vocabulary:
            logging.warning("No vocabulary found for corpus: %s", corpus_name)
            pbar.update(len(FAMILIES) * len(NUM_TOPICS_LIST))
            continue

        with get_session(config.database) as session:
            embedding_map = fetch_document_embeddings(session, corpus_name)
            word_embeddings = fetch_word_embeddings(session, corpus_name)

        documents, embeddings = align_documents_and_embeddings(corpus_name, embedding_map)
        if len(documents) == 0:
            logging.warning("No aligned embeddings for corpus %s", corpus_name)
            pbar.update(len(FAMILIES) * len(NUM_TOPICS_LIST))
            continue

        for family in FAMILIES:
            for num_topics in NUM_TOPICS_LIST:
                with get_session(config.database) as session:
                    existing = count_existing_results(session, corpus_name, family, num_topics)

                iterations_to_run = target_results - existing
                if iterations_to_run <= 0:
                    pbar.update(1)
                    continue

                pbar.set_description(
                    f"{family} on {corpus_name}, topics={num_topics}, iters={iterations_to_run}"
                )

                for iteration in range(1, iterations_to_run + 1):
                    try:
                        model, model_hparams = build_model(family, num_topics)
                        model.train(documents, embeddings, vocabulary, word_embeddings=word_embeddings)
                        topics = model.get_topics()
                        cleanup_model(model)

                        hyperparameters = {
                            "word_embedding": "normal_direct_all-MiniLM-L6-v2",
                            "doc_embedding": "boe_cse first-chunk tcc=1",
                            "model": model_hparams,
                        }

                        with get_session(config.database) as session:
                            insert_query = text("""
                                INSERT INTO pipeline.boe_nwe_topic_model_corpus_result
                                (family, corpus_name, source_model_name, num_topics,
                                 topics, hyperparameters)
                                VALUES (:family, :corpus_name, :source_model_name, :num_topics,
                                        :topics, :hyperparameters)
                            """)
                            session.execute(insert_query, {
                                "family": family,
                                "corpus_name": corpus_name,
                                "source_model_name": SOURCE_MODEL_NAME,
                                "num_topics": num_topics,
                                "topics": json.dumps(topics),
                                "hyperparameters": json.dumps(hyperparameters),
                            })
                            session.commit()
                    except Exception:
                        logging.exception(
                            "Error running %s on %s (num_topics=%d)",
                            family, corpus_name, num_topics,
                        )

                pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    run_experiments()

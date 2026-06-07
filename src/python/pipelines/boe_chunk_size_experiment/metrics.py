"""Stage 4 of the BOE chunk-size experiment: performance metrics.

Computes the same four coherence metrics as the main BOE experiment
(see src/python/pipelines/boe_performance_metrics/main.py) for every
boe_cse_topic_model_corpus_result row and stores them in
pipeline.boe_cse_topic_model_performance.

The metric implementations are the shared ones in
pipelines.performance_metrics, so values are directly comparable with the
main BOE experiment and the existing-model baseline: NPMI uses corpus
statistics from pipeline.preprocessed_document / vocabulary_word, and
WEPS / WECS / ISH use the shared pipeline.vocabulary_word_embeddings
(NOT the boe_cse_word_embedding vectors, which are topic-model inputs).

Run with:
    set -a && . .env && set +a
    uv run src/python/pipelines/boe_chunk_size_experiment/metrics.py
"""

import json
import logging
import traceback
from typing import Any

from sqlalchemy import text

from configuration import load_config_from_env
from database import get_session
from pipelines.performance_metrics.npmi import calculate_multiple_topic_models_npmi
from pipelines.performance_metrics.word_embedding_based import (
    calculate_multiple_topic_models_intruder_shift,
    calculate_multiple_topic_models_wecs,
    calculate_multiple_topic_models_weps,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_cse_topic_model_results(session, corpus_name: str, metric_name: str) -> list[Any]:
    """Fetch CSE topic-model results that are missing a specific metric."""
    query = text(
        """
        SELECT
            r.id,
            r.corpus_name,
            m.name AS model_name,
            r.topics
        FROM pipeline.boe_cse_topic_model_corpus_result r
        JOIN pipeline.boe_cse_topic_model m
            ON m.id = r.topic_model_id
        WHERE r.soft_delete = FALSE
          AND r.corpus_name = :corpus_name
          AND NOT EXISTS (
              SELECT 1
              FROM pipeline.boe_cse_topic_model_performance p
              WHERE p.boe_cse_topic_model_corpus_result_id = r.id
                AND p.metric_name = :metric_name
          )
        """
    )
    return session.execute(
        query,
        {
            "corpus_name": corpus_name,
            "metric_name": metric_name,
        },
    ).fetchall()


def save_cse_performance_metric(
    session,
    cse_topic_model_result_id: int,
    metric_name: str,
    metric_value: dict[str, Any],
) -> None:
    """Upsert a CSE metric row."""
    query = text(
        """
        INSERT INTO pipeline.boe_cse_topic_model_performance
            (boe_cse_topic_model_corpus_result_id, metric_name, metric_value)
        VALUES (:result_id, :metric_name, :metric_value)
        ON CONFLICT (boe_cse_topic_model_corpus_result_id, metric_name)
        DO UPDATE SET metric_value = EXCLUDED.metric_value
        """
    )
    session.execute(
        query,
        {
            "result_id": cse_topic_model_result_id,
            "metric_name": metric_name,
            "metric_value": json.dumps(metric_value),
        },
    )
    session.commit()


def get_cse_corpus_list(session) -> list[str]:
    """Get distinct corpus names present in CSE topic-model results."""
    query = text(
        """
        SELECT DISTINCT corpus_name
        FROM pipeline.boe_cse_topic_model_corpus_result
        WHERE soft_delete = FALSE
        ORDER BY corpus_name
        """
    )
    return [row[0] for row in session.execute(query).fetchall()]


def main() -> None:
    config = load_config_from_env()

    metrics = {
        "NPMI": calculate_multiple_topic_models_npmi,
        "WEPS": calculate_multiple_topic_models_weps,
        "WECS": calculate_multiple_topic_models_wecs,
        "ISH": calculate_multiple_topic_models_intruder_shift,
    }

    with get_session(config.database) as session:
        logger.info("Connected to database successfully")

        corpus_list = get_cse_corpus_list(session)
        logger.info("Found %d CSE corpora", len(corpus_list))

        for corpus_name in corpus_list:
            for metric_name, metric_function in metrics.items():
                try:
                    logger.info("Calculating %s for CSE corpus: %s", metric_name, corpus_name)
                    results = get_cse_topic_model_results(session, corpus_name, metric_name)
                    logger.info("Found %d CSE results to evaluate", len(results))

                    if len(results) == 0:
                        continue

                    topic_models_outputs = [result.topics for result in results]
                    metric_scores = metric_function(
                        session=session,
                        topic_models_outputs=topic_models_outputs,
                        corpus_name=corpus_name,
                    )

                    for result, metric_score in zip(results, metric_scores):
                        save_cse_performance_metric(session, result.id, metric_name, metric_score)

                except Exception as exc:
                    logger.error(
                        "Error calculating %s for CSE corpus %s: %s",
                        metric_name,
                        corpus_name,
                        str(exc),
                    )
                    logger.error(traceback.format_exc())
                    continue


if __name__ == "__main__":
    main()

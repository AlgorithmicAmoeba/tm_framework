import logging
import traceback
from typing import List, Dict, Any, Tuple
import json
from collections import defaultdict

from configuration import load_config_from_env
from database import get_session
from sqlalchemy import text

from npmi import calculate_multiple_topic_models_npmi
from word_embedding_based import (
    calculate_multiple_topic_models_weps,
    calculate_multiple_topic_models_wecs,
    calculate_multiple_topic_models_intruder_shift,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_boe_topic_model_results(
    session,
    corpus: str,
    metric_name: str,
) -> List[Dict[str, Any]]:
    """Get BOE topic model results that haven't been evaluated yet for a metric."""
    query = text("""
        SELECT tmr.id,
               tmr.corpus_name,
               tm.name as model_name,
               tmr.topics,
               tmr.source_model_name,
               tmr.algorithm,
               tmr.target_dims,
               tmr.padding_method,
               tmr.target_chunk_count
        FROM pipeline.boe_topic_model_corpus_result tmr
        JOIN pipeline.boe_topic_model tm ON tm.id = tmr.topic_model_id
        WHERE NOT EXISTS (
            SELECT 1
            FROM pipeline.boe_topic_model_performance tmp
            WHERE tmp.boe_topic_model_corpus_result_id = tmr.id
            AND tmp.metric_name = :metric_name
        )
        AND tmr.soft_delete = FALSE
        AND tmr.corpus_name = :corpus
    """)
    return session.execute(query, {
        "corpus": corpus,
        "metric_name": metric_name,
    }).fetchall()


def save_performance_metric(
    session,
    topic_model_result_id: int,
    metric_name: str,
    metric_value: Dict[str, Any],
) -> None:
    """Save a performance metric result to the database."""
    query = text("""
        INSERT INTO pipeline.boe_topic_model_performance
        (boe_topic_model_corpus_result_id, metric_name, metric_value)
        VALUES (:topic_model_result_id, :metric_name, :metric_value)
        ON CONFLICT (boe_topic_model_corpus_result_id, metric_name)
        DO UPDATE SET metric_value = EXCLUDED.metric_value
    """)
    session.execute(query, {
        "topic_model_result_id": topic_model_result_id,
        "metric_name": metric_name,
        "metric_value": json.dumps(metric_value),
    })
    session.commit()


def get_corpus_list(session) -> List[str]:
    """Get all corpus names with BOE topic model results from the database."""
    query = text("""
        SELECT DISTINCT corpus_name
        FROM pipeline.boe_topic_model_corpus_result
        WHERE soft_delete = FALSE
        ORDER BY corpus_name
    """)
    return [row[0] for row in session.execute(query).fetchall()]


def embeddings_exist(
    session,
    corpus_name: str,
    source_model_name: str,
    algorithm: str,
    target_dims: int,
    padding_method: str,
) -> bool:
    query = text("""
        SELECT 1
        FROM pipeline.boe_word_embedding
        WHERE corpus_name = :corpus_name
        AND source_model_name = :source_model_name
        AND algorithm = :algorithm
        AND target_dims = :target_dims
        AND padding_method = :padding_method
        LIMIT 1
    """)
    result = session.execute(query, {
        "corpus_name": corpus_name,
        "source_model_name": source_model_name,
        "algorithm": algorithm,
        "target_dims": target_dims,
        "padding_method": padding_method,
    })
    return result.fetchone() is not None


def group_results_by_combo(
    results: List[Any],
) -> Dict[Tuple[str, str, int, str], List[Any]]:
    grouped: Dict[Tuple[str, str, int, str], List[Any]] = defaultdict(list)
    for result in results:
        key = (
            result.source_model_name,
            result.algorithm,
            result.target_dims,
            result.padding_method,
        )
        grouped[key].append(result)
    return grouped


def main() -> None:
    """Main function to run BOE performance metrics calculations."""
    config = load_config_from_env()
    db_config = config.database

    metrics_word_embedding = {
        'WEPS': calculate_multiple_topic_models_weps,
        'WECS': calculate_multiple_topic_models_wecs,
        'ISH': calculate_multiple_topic_models_intruder_shift,
    }

    with get_session(db_config) as session:
        logger.info("Connected to database successfully")

        corpus_list = get_corpus_list(session)
        logger.info("Found %d corpora", len(corpus_list))

        for corpus_name in corpus_list:
            # NPMI (corpus-only, no embedding combo)
            try:
                metric_name = 'NPMI'
                logger.info("Calculating %s for corpus: %s", metric_name, corpus_name)
                results = get_boe_topic_model_results(session, corpus_name, metric_name)
                logger.info("Found %d topic model results to evaluate", len(results))

                if len(results) > 0:
                    topic_models_outputs = [result.topics for result in results]
                    metric_scores = calculate_multiple_topic_models_npmi(
                        session=session,
                        topic_models_outputs=topic_models_outputs,
                        corpus_name=corpus_name,
                    )

                    for result, metric_score in zip(results, metric_scores):
                        save_performance_metric(session, result.id, metric_name, metric_score)

            except Exception as e:
                logger.error("Error calculating %s for corpus: %s: %s", metric_name, corpus_name, str(e))
                logger.error(traceback.format_exc())

            # Word-embedding-based metrics (per combo)
            for metric_name, metric_function in metrics_word_embedding.items():
                try:
                    logger.info("Calculating %s for corpus: %s", metric_name, corpus_name)
                    results = get_boe_topic_model_results(session, corpus_name, metric_name)
                    logger.info("Found %d topic model results to evaluate", len(results))

                    if len(results) == 0:
                        continue

                    grouped = group_results_by_combo(results)

                    for (source_model_name, algorithm, target_dims, padding_method), group_results in grouped.items():
                        if not embeddings_exist(
                            session,
                            corpus_name,
                            source_model_name,
                            algorithm,
                            target_dims,
                            padding_method,
                        ):
                            logger.error(
                                "Missing BOE word embeddings for corpus=%s combo=%s/%s/%s/%s; skipping.",
                                corpus_name,
                                source_model_name,
                                algorithm,
                                target_dims,
                                padding_method,
                            )
                            continue

                        topic_models_outputs = [result.topics for result in group_results]
                        metric_scores = metric_function(
                            session=session,
                            topic_models_outputs=topic_models_outputs,
                            corpus_name=corpus_name,
                            source_model_name=source_model_name,
                            algorithm=algorithm,
                            target_dims=target_dims,
                            padding_method=padding_method,
                        )

                        for result, metric_score in zip(group_results, metric_scores):
                            save_performance_metric(session, result.id, metric_name, metric_score)

                except Exception as e:
                    logger.error("Error calculating %s for corpus: %s: %s", metric_name, corpus_name, str(e))
                    logger.error(traceback.format_exc())
                    continue


if __name__ == "__main__":
    main()

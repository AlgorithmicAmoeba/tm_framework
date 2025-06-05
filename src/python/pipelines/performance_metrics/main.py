import logging
import traceback
from typing import List, Dict, Any
import json
from collections import defaultdict
from configuration import load_config_from_env
from database import get_session
from sqlalchemy import text
from npmi import calculate_multiple_topic_models_npmi
from word_embedding_based import calculate_multiple_topic_models_weps, calculate_multiple_topic_models_wecs, calculate_multiple_topic_models_intruder_shift

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_topic_model_results(session, corpus: str, metric_name: str ) -> List[Dict[str, Any]]:
    """Get topic model results that haven't been evaluated yet.
    
    Args:
        session: Database session
        corpus: Corpus name to filter results
        metric_name: Metric name to filter results that haven't been evaluated for this metric
    """
    query = text("""
        SELECT tmr.id, tmr.corpus_id, tm.name as model_name, c.name as corpus_name, tmr.topics
        FROM pipeline.topic_model_corpus_result tmr
        JOIN pipeline.topic_model tm ON tm.id = tmr.topic_model_id
        JOIN pipeline.corpus c ON c.id = tmr.corpus_id
        WHERE NOT EXISTS (
            SELECT 1 
            FROM pipeline.topic_model_performance tmp 
            WHERE tmp.topic_model_corpus_result_id = tmr.id
            AND tmp.metric_name = :metric_name
        )
        AND tmr.soft_delete = FALSE
        AND c.name = :corpus
    """)
    return session.execute(query, {
        "corpus": corpus,
        "metric_name": metric_name
    }).fetchall()

def save_performance_metric(session, topic_model_result_id: int, metric_name: str, metric_value: Dict[str, Any]):
    """Save a performance metric result to the database."""
    query = text("""
        INSERT INTO pipeline.topic_model_performance 
        (topic_model_corpus_result_id, metric_name, metric_value)
        VALUES (:topic_model_result_id, :metric_name, :metric_value)
        ON CONFLICT (topic_model_corpus_result_id, metric_name) 
        DO UPDATE SET metric_value = EXCLUDED.metric_value
    """)
    session.execute(query, {
        "topic_model_result_id": topic_model_result_id,
        "metric_name": metric_name,
        "metric_value": json.dumps(metric_value)
    })
    session.commit()

def get_corpus_list(session) -> List[str]:
    """Get all corpus names from the database."""
    query = text("SELECT name FROM pipeline.corpus")
    return [row[0] for row in session.execute(query).fetchall()]

def main():
    """Main function to run performance metrics calculations."""
    config = load_config_from_env()
    db_config = config.database

    metrics = {
        'NPMI': calculate_multiple_topic_models_npmi,
        'WEPS': calculate_multiple_topic_models_weps,
        # 'WECS': calculate_multiple_topic_models_wecs,  same as WEPS for sim = dot product
        'ISH': calculate_multiple_topic_models_intruder_shift,
    }

    with get_session(db_config) as session:
        logger.info("Connected to database successfully")

        corpus_list = get_corpus_list(session)
        logger.info(f"Found {len(corpus_list)} corpora")

        for corpus_name in corpus_list:
            for metric_name, metric_function in metrics.items():
                try:
                    logger.info(f"Calculating {metric_name} for corpus: {corpus_name}")
                    results = get_topic_model_results(session, corpus_name, metric_name)
                    logger.info(f"Found {len(results)} topic model results to evaluate")

                    if len(results) == 0:
                        logger.info(f"No topic model results to evaluate for corpus: {corpus_name}")
                        continue
                    
                    topic_models_outputs = [result.topics for result in results]
                    metric_scores = metric_function(
                        session=session,
                        topic_models_outputs=topic_models_outputs,
                        corpus_name=corpus_name
                    )
                    
                    for result, metric_score in zip(results, metric_scores):
                        save_performance_metric(session, result.id, metric_name, metric_score)

                except Exception as e:
                    logger.error(f"Error calculating {metric_name} for corpus: {corpus_name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
                    

if __name__ == "__main__":
    main()

import logging
from typing import List, Dict, Any
import json
from collections import defaultdict
from configuration import load_config_from_env
from database import get_session
from sqlalchemy import text
from npmi import calculate_multiple_topic_models_npmi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_topic_model_results(session) -> List[Dict[str, Any]]:
    """Get all topic model results that haven't been evaluated yet."""
    query = text("""
        SELECT tmr.id, tmr.corpus_id, tm.name as model_name, c.name as corpus_name, tmr.topics
        FROM pipeline.topic_model_corpus_result tmr
        JOIN pipeline.topic_model tm ON tm.id = tmr.topic_model_id
        JOIN pipeline.corpus c ON c.id = tmr.corpus_id
        WHERE NOT EXISTS (
            SELECT 1 
            FROM pipeline.topic_model_performance tmp 
            WHERE tmp.topic_model_corpus_result_id = tmr.id
        )
        AND tmr.soft_delete = FALSE
    """)
    return session.execute(query).fetchall()

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

def main():
    """Main function to run performance metrics calculations."""
    try:
        config = load_config_from_env()
        db_config = config.database

        with get_session(db_config) as session:
            logger.info("Connected to database successfully")

            # Get topic model results that need evaluation
            results = get_topic_model_results(session)
            logger.info(f"Found {len(results)} topic model results to evaluate")

            # Group results by corpus
            corpus_groups = defaultdict(list)
            result_ids_by_corpus = defaultdict(list)
            
            for result in results:
                corpus_groups[result.corpus_name].append(result.topics)
                result_ids_by_corpus[result.corpus_name].append(result.id)

            # Process each corpus group
            for corpus_name, topics_list in corpus_groups.items():
                try:
                    logger.info(f"Calculating NPMI for corpus: {corpus_name} with {len(topics_list)} topic models")
                    
                    # Calculate NPMI for all topic models in this corpus at once
                    npmi_scores = calculate_multiple_topic_models_npmi(
                        session=session,
                        topic_models_outputs=topics_list,
                        corpus_name=corpus_name
                    )
                    
                    # Save results for each topic model
                    for result_id, npmi_score in zip(result_ids_by_corpus[corpus_name], npmi_scores):
                        metric_value = {
                            "score": float(npmi_score),
                        }
                        save_performance_metric(session, result_id, 'NPMI', metric_value)
                        logger.info(f"NPMI score for result {result_id}: {npmi_score:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error processing corpus {corpus_name}: {str(e)}")
                    continue

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()

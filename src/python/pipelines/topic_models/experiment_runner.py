import itertools
from typing import List
from tqdm import tqdm
from sqlalchemy import text

from configuration import load_config_from_env
from database import get_session
from LDA import run_lda_pipeline
from BERTopic import run_bertopic_pipeline
from ZeroshotTM import run_autoencoding_tm_pipeline
from KeyNMF import run_keynmf_pipeline

def get_corpus_list() -> List[str]:
    """Get list of all corpora from the database"""
    config = load_config_from_env()
    with get_session(config.database) as session:
        query = text("SELECT name FROM pipeline.corpus")
        result = session.execute(query)
        return [row[0] for row in result]

def get_topic_model_list() -> List[str]:
    """Get list of all topic models from the database"""
    config = load_config_from_env()
    with get_session(config.database) as session:
        query = text("SELECT name FROM pipeline.topic_model")
        result = session.execute(query)
        return [row[0] for row in result]

def count_existing_results(corpus_name: str, model_name: str, num_topics: int, config) -> int:
    """Count number of non-soft-deleted results for a corpus-model pair"""
    with get_session(config.database) as session:
        query = text("""
            SELECT COUNT(*) 
            FROM pipeline.topic_model_corpus_result tmr
            JOIN pipeline.topic_model tm ON tmr.topic_model_id = tm.id
            JOIN pipeline.corpus c ON tmr.corpus_id = c.id
            WHERE c.name = :corpus_name 
            AND tm.name = :model_name
            AND tmr.soft_delete = FALSE
            AND tmr.num_topics = :num_topics
        """).bindparams(corpus_name=corpus_name, model_name=model_name, num_topics=num_topics)
        return session.execute(query).scalar()

def run_experiments(num_topics: int = 20, target_results: int = 10):
    """
    Run all topic models against all corpora until target number of results is reached
    
    Args:
        num_topics: Number of topics to extract
        num_iterations: Number of times to run each model
        target_results: Target number of non-soft-deleted results per corpus-model pair
    """
    config = load_config_from_env()
    corpora = get_corpus_list()
    models = get_topic_model_list()

    # Print the corpora and models
    print(f"Corpora: {corpora}")
    print(f"Models: {models}")
    
    # Create progress bar for all corpus-model pairs
    total_pairs = len(corpora) * len(models)
    pbar = tqdm(total=total_pairs, desc="Running experiments")
    
    for model_name, corpus_name in itertools.product(models, corpora):
        # Check how many results we already have
        existing_results = count_existing_results(corpus_name, model_name, num_topics, config)
        
        if existing_results < target_results:
            # Calculate how many more iterations we need
            iterations_to_run = target_results - existing_results

            # Update pbar
            pbar.set_description(f"Running {model_name} on {corpus_name} (iterations: {iterations_to_run})")
            
            # Run the appropriate pipeline
            try:
                if model_name == 'LDA':
                    run_lda_pipeline(corpus_name, num_topics, iterations_to_run)
                elif model_name == 'BERTopic':
                    run_bertopic_pipeline(corpus_name, num_topics, iterations_to_run)
                elif model_name == 'ZeroShotTM':
                    run_autoencoding_tm_pipeline(corpus_name, num_topics, iterations_to_run, combined=False)
                elif model_name == 'CombinedTM':
                    run_autoencoding_tm_pipeline(corpus_name, num_topics, iterations_to_run, combined=True)
                elif model_name == 'KeyNMF':
                    run_keynmf_pipeline(corpus_name, num_topics, iterations_to_run)
            except Exception as e:
                print(f"Error running {model_name} on {corpus_name}: {str(e)}")
        
        pbar.update(1)
    
    pbar.close()

if __name__ == '__main__':
    run_experiments(
        num_topics=20,
        target_results=5
    )

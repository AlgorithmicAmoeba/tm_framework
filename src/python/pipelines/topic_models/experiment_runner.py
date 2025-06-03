import itertools
from typing import List
from tqdm import tqdm
from sqlalchemy import text
import logging

from configuration import load_config_from_env
from database import get_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    

def pretty_print_model_corpus_table():
    """
    Print a table showing number of valid experiments for each model-corpus pair.
    Colors indicate if target number of results is met (green), partially met (yellow), or none (red).
    """
    from rich.console import Console
    from rich.table import Table
    from sqlalchemy import text
    
    config = load_config_from_env()
    
    # Get list of models and corpora
    models = get_topic_model_list()
    corpora = get_corpus_list()
    
    # Get all distinct num_topics values from results
    with get_session(config.database) as session:
        query = text("""
            SELECT DISTINCT num_topics 
            FROM pipeline.topic_model_corpus_result
            ORDER BY num_topics
        """)
        num_topics_list = [row[0] for row in session.execute(query)]
    
    console = Console()
    
    # Create a table for each num_topics value
    for num_topics in num_topics_list:
        table = Table(title=f"Number of Valid Experiments (Topics={num_topics})")
        
        # Add columns
        table.add_column("Corpus", style="white")
        for model in models:
            table.add_column(model)
            
        # Add rows
        for corpus in corpora:
            row = [corpus]
            for model in models:
                # Get count of valid results
                with get_session(config.database) as session:
                    query = text("""
                        SELECT COUNT(*) FROM pipeline.topic_model_corpus_result r
                        JOIN pipeline.topic_model m ON r.topic_model_id = m.id
                        JOIN pipeline.corpus c ON r.corpus_id = c.id
                        WHERE m.name = :model_name
                        AND c.name = :corpus_name
                        AND r.num_topics = :num_topics
                        AND r.soft_delete = false
                    """).bindparams(
                        model_name=model,
                        corpus_name=corpus,
                        num_topics=num_topics
                    )
                    count = session.execute(query).scalar()
                
                # Color code based on count
                if count == 0:
                    cell = f"[red]{count}[/red]"
                elif count >= 5:  # Using 5 as target_results
                    cell = f"[green]{count}[/green]"
                else:
                    cell = f"[yellow]{count}[/yellow]"
                    
                row.append(cell)
            
            table.add_row(*row)
        
        console.print(table)
        console.print("\n")


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

    from LDA import run_lda_pipeline
    from BERTopic import run_bertopic_pipeline
    from ZeroshotTM import run_autoencoding_tm_pipeline
    from KeyNMF import run_keynmf_pipeline
    from semantic_signal_separation import run_s3_pipeline
    from gmm import run_gmm_pipeline

    # Print the corpora and models
    logging.info(f"Corpora: {corpora}")
    logging.info(f"Models: {models}")
    
    # Create progress bar for all corpus-model pairs
    total_pairs = len(corpora) * len(models)
    pbar = tqdm(total=total_pairs, desc="Running experiments")
    
    for model_name, corpus_name in itertools.product(models, corpora):
        # if model_name == "BERTopic":
        #     pbar.update(1)
        #     continue

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
                elif model_name == 'SemanticSignalSeparation':
                    run_s3_pipeline(corpus_name, num_topics, iterations_to_run)
                elif model_name == 'GMM':
                    run_gmm_pipeline(corpus_name, num_topics, iterations_to_run)
            except Exception as e:
                logging.exception(f"Error running {model_name} on {corpus_name}")
        
        pbar.update(1)
    
    pbar.close()

if __name__ == '__main__':
    import faulthandler
    faulthandler.enable()
    # faulthandler.disable()

    pretty_print_model_corpus_table()
    num_topicss = [
        10,
        20,
        50,
        100,
        200,
    ]
    for num_topics in num_topicss:
        run_experiments(
            num_topics=num_topics,
            target_results=10
        )
        logging.info(f"Finished running experiments for {num_topics} topics")

    pretty_print_model_corpus_table()

from rich.console import Console
from rich.table import Table
from sqlalchemy import text
import json
import numpy as np
from typing import List, Dict, Any

from configuration import load_config_from_env
from database import get_session

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

def pretty_print_metrics_table(metric_name: str):
    """
    Print a table showing average and standard deviation of a metric for each model-corpus pair.
    
    Args:
        metric_name: Name of the metric to display (must match metric_name in database)
    """
    config = load_config_from_env()
    console = Console()
    
    # Get list of models and corpora
    models = get_topic_model_list()
    corpora = get_corpus_list()
    
    # Get all distinct num_topics values from results
    with get_session(config.database) as session:
        query = text("""
            SELECT DISTINCT r.num_topics 
            FROM pipeline.topic_model_corpus_result r
            JOIN pipeline.topic_model_performance p ON r.id = p.topic_model_corpus_result_id
            WHERE p.metric_name = :metric_name
            ORDER BY r.num_topics
        """).bindparams(metric_name=metric_name)
        num_topics_list = [row[0] for row in session.execute(query)]
    
    # Create a table for each num_topics value
    for num_topics in num_topics_list:
        table = Table(title=f"{metric_name} Metrics (Topics={num_topics})")
        
        # Add columns
        table.add_column("Corpus", style="white")
        for model in models:
            table.add_column(model)
            
        # Add rows
        for corpus in corpora:
            row = [corpus]
            for model in models:
                # Get metric values
                with get_session(config.database) as session:
                    query = text("""
                        SELECT p.metric_value 
                        FROM pipeline.topic_model_performance p
                        JOIN pipeline.topic_model_corpus_result r ON p.topic_model_corpus_result_id = r.id
                        JOIN pipeline.topic_model m ON r.topic_model_id = m.id
                        JOIN pipeline.corpus c ON r.corpus_id = c.id
                        WHERE m.name = :model_name
                        AND c.name = :corpus_name
                        AND r.num_topics = :num_topics
                        AND p.metric_name = :metric_name
                        AND r.soft_delete = false
                    """).bindparams(
                        model_name=model,
                        corpus_name=corpus,
                        num_topics=num_topics,
                        metric_name=metric_name
                    )
                    results = session.execute(query).fetchall()
                
                if not results:
                    cell = "[red]No data[/red]"
                else:
                    # Extract values from JSONB
                    values = []
                    for result in results:
                        metric_value = result[0]
                        if isinstance(metric_value, str):
                            metric_value = json.loads(metric_value)
                        if isinstance(metric_value, dict):
                            # If it's a dict, try to get a single numeric value
                            values.extend([v for v in metric_value.values() if isinstance(v, (int, float))])
                        elif isinstance(metric_value, (int, float)):
                            values.append(metric_value)
                    
                    if values:
                        avg = np.mean(values)
                        std = np.std(values)
                        cell = f"[green]{avg:.3f} ± {std:.3f}[/green]"
                    else:
                        cell = "[red]Invalid data[/red]"
                    
                row.append(cell)
            
            table.add_row(*row)
        
        console.print(table)
        console.print("\n")

if __name__ == '__main__':
    # Example usage
    pretty_print_metrics_table("NPMI")

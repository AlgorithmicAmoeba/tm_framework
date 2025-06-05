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
    The best performing model for each dataset is shown in green, second best in purple, and third best in blue.
    All other values are shown in white.
    
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
            model_values = []
            
            # First collect all values for this corpus
            for model in models:
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
                    model_values.append((model, None))
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
                        model_values.append((model, (avg, std)))
                    else:
                        model_values.append((model, None))
            
            # Sort models by their average values (if available)
            valid_models = [(model, values) for model, values in model_values if values is not None]
            sorted_models = sorted(valid_models, key=lambda x: x[1][0], reverse=True)
            
            # Create a mapping of model to its rank (1-based)
            model_ranks = {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}
            
            # Now add the cells with appropriate colors
            for model, values in model_values:
                if values is None:
                    cell = "[red]No data[/red]"
                else:
                    avg, std = values
                    rank = model_ranks.get(model)
                    if rank == 1:
                        color = "green"
                    elif rank == 2:
                        color = "purple"
                    elif rank == 3:
                        color = "blue"
                    else:
                        color = "white"
                    cell = f"[{color}]{avg:.3f} ± {std:.3f}[/{color}]"
                row.append(cell)
            
            table.add_row(*row)
        
        console.print(table)
        console.print("\n")

def pretty_print_top_models_table(joined: bool = True, top_n: int = 3):
    """
    Print a table showing the top performing models for each corpus, number of topics, and metric combination.
    
    Args:
        joined: If True, creates a single table with split rows for each metric.
               If False, creates separate tables for each metric.
        top_n: Number of top performing models to show in each cell (default: 3)
    """
    config = load_config_from_env()
    console = Console()
    
    # Get list of models and corpora
    models = get_topic_model_list()
    corpora = get_corpus_list()
    metrics = ["NPMI", "WEPS", "ISH"]
    
    # Define colors for models
    model_colors = {
        model: color for model, color in zip(
            models,
            [
                "red", "green", "blue", "yellow", "magenta", "cyan",
                "bright_red", "bright_green", "bright_blue", "bright_yellow",
                "bright_magenta", "bright_cyan", "white", "bright_white"
            ]
        )
    }
    
    # Get all distinct num_topics values from results
    with get_session(config.database) as session:
        query = text("""
            SELECT DISTINCT r.num_topics 
            FROM pipeline.topic_model_corpus_result r
            JOIN pipeline.topic_model_performance p ON r.id = p.topic_model_corpus_result_id
            ORDER BY r.num_topics
        """)
        num_topics_list = [row[0] for row in session.execute(query)]
    
    def get_top_models(corpus: str, metric: str, num_topics: int) -> str:
        """Helper function to get top N models for a given combination"""
        model_values = []
        for model in models:
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
                    metric_name=metric
                )
                results = session.execute(query).fetchall()
            
            if not results:
                continue
            
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
                model_values.append((model, avg))
        
        # Sort models by their average values
        sorted_models = sorted(model_values, key=lambda x: x[1], reverse=True)
        
        # Get top N models
        top_models = [model for model, _ in sorted_models[:top_n]]
        
        # Create cell content with colored model names
        if len(top_models) == 0:
            return "[red]No data[/red]"
        else:
            colored_models = [f"[{model_colors[model]}]{model}[/{model_colors[model]}]" for model in top_models]
            return " / ".join(colored_models)
    
    if joined:
        # Create a single table
        table = Table(title=f"Top {top_n} Models by Corpus, Metric, and Number of Topics")
        
        # Add columns
        table.add_column("Corpus", style="white")
        table.add_column("Metric", style="white")
        for num_topics in num_topics_list:
            table.add_column(f"Topics={num_topics}")
        
        # Add rows
        for corpus in corpora:
            # For each metric
            for metric in metrics:
                row = [corpus, metric]
                
                # For each number of topics
                for num_topics in num_topics_list:
                    row.append(get_top_models(corpus, metric, num_topics))
                
                table.add_row(*row)
            
            # Add a separator row between corpora
            if corpus != corpora[-1]:  # Don't add separator after the last corpus
                table.add_row(*[""] * (len(num_topics_list) + 2))
        
        console.print(table)
        console.print("\n")
    else:
        # Create separate tables for each metric
        for metric in metrics:
            table = Table(title=f"Top {top_n} Models by {metric}")
            
            # Add columns
            table.add_column("Corpus", style="white")
            for num_topics in num_topics_list:
                table.add_column(f"Topics={num_topics}")
            
            # Add rows
            for corpus in corpora:
                row = [corpus]
                
                # For each number of topics
                for num_topics in num_topics_list:
                    row.append(get_top_models(corpus, metric, num_topics))
                
                table.add_row(*row)
            
            console.print(table)
            console.print("\n")

if __name__ == '__main__':
    # Example usage
    # pretty_print_metrics_table("NPMI")
    # pretty_print_metrics_table("WEPS")
    # pretty_print_metrics_table("ISH")
    pretty_print_top_models_table(joined=True, top_n=1)  # Single table with top 3 models
    # pretty_print_top_models_table(joined=False, top_n=2)  # Separate tables with top 2 models

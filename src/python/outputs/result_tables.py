from rich.console import Console
from rich.table import Table
from sqlalchemy import text
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

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

def get_num_topics_list() -> List[int]:
    """Get list of all distinct num_topics values from results"""
    config = load_config_from_env()
    with get_session(config.database) as session:
        query = text("""
            SELECT DISTINCT r.num_topics 
            FROM pipeline.topic_model_corpus_result r
            JOIN pipeline.topic_model_performance p ON r.id = p.topic_model_corpus_result_id
            ORDER BY r.num_topics
        """)
        return [row[0] for row in session.execute(query)]

def extract_metric_values(metric_value: Any) -> List[float]:
    """Extract numeric values from metric_value, handling different formats"""
    if isinstance(metric_value, str):
        metric_value = json.loads(metric_value)
    if isinstance(metric_value, dict):
        return [v for v in metric_value.values() if isinstance(v, (int, float))]
    elif isinstance(metric_value, (int, float)):
        return [metric_value]
    return []

def get_model_metrics_for_corpus(
    model: str,
    corpus: str,
    num_topics: int,
    metric_name: str
) -> Optional[Tuple[float, float]]:
    """Get average and standard deviation of a metric for a model-corpus pair"""
    config = load_config_from_env()
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
        return None
    
    values = []
    for result in results:
        values.extend(extract_metric_values(result[0]))
    
    if values:
        return np.mean(values), np.std(values)
    return None

def get_model_metrics_all(
    corpus: str,
    model: str,
    num_topics: int,
    metrics: List[str]
) -> Optional[Dict[str, float]]:
    """Get all metric values for a model-corpus pair"""
    metric_values = {}
    for metric in metrics:
        result = get_model_metrics_for_corpus(model, corpus, num_topics, metric)
        if result is None:
            return None
        metric_values[metric] = result[0]  # Use mean value
    return metric_values

def get_model_colors(models: List[str]) -> Dict[str, str]:
    """Get color mapping for models"""
    colors = [
        "red", "green", "blue", "yellow", "magenta", "cyan",
        "bright_red", "bright_green", "bright_blue", "bright_yellow",
        "bright_magenta", "bright_cyan", "white", "bright_white"
    ]
    return {model: color for model, color in zip(models, colors)}

def get_ranked_models(
    model_values: List[Tuple[str, Tuple[float, float]]]
) -> Dict[str, int]:
    """Get ranking of models based on their metric values"""
    valid_models = [(model, values) for model, values in model_values if values is not None]
    sorted_models = sorted(valid_models, key=lambda x: x[1][0], reverse=True)
    return {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}

def get_rank_color(rank: int) -> str:
    """Get color based on rank"""
    if rank == 1:
        return "green"
    elif rank == 2:
        return "purple"
    elif rank == 3:
        return "blue"
    return "white"

def is_dominated(
    model_metrics: Dict[str, float],
    all_metrics: List[Dict[str, float]],
    metrics: List[str]
) -> bool:
    """Check if a model is dominated by any other model"""
    for other_metrics in all_metrics:
        if other_metrics == model_metrics:
            continue
        
        dominated = True
        for metric in metrics:
            if metric in ["NPMI", "WEPS"]:  # Higher is better
                if other_metrics[metric] <= model_metrics[metric]:
                    dominated = False
                    break
            else:  # ISH - lower is better
                if other_metrics[metric] >= model_metrics[metric]:
                    dominated = False
                    break
        
        if dominated:
            return True
    
    return False

def pretty_print_metrics_table(metric_name: str):
    """
    Print a table showing average and standard deviation of a metric for each model-corpus pair.
    The best performing model for each dataset is shown in green, second best in purple, and third best in blue.
    All other values are shown in white.
    
    Args:
        metric_name: Name of the metric to display (must match metric_name in database)
    """
    console = Console()
    
    # Get list of models and corpora
    models = get_topic_model_list()
    corpora = get_corpus_list()
    num_topics_list = get_num_topics_list()
    
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
                result = get_model_metrics_for_corpus(model, corpus, num_topics, metric_name)
                model_values.append((model, result))
            
            # Get model rankings
            model_ranks = get_ranked_models(model_values)
            
            # Now add the cells with appropriate colors
            for model, values in model_values:
                if values is None:
                    cell = "[red]No data[/red]"
                else:
                    avg, std = values
                    rank = model_ranks.get(model)
                    color = get_rank_color(rank)
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
    console = Console()
    
    # Get list of models and corpora
    models = get_topic_model_list()
    corpora = get_corpus_list()
    metrics = ["NPMI", "WEPS", "ISH"]
    num_topics_list = get_num_topics_list()
    
    # Get model colors
    model_colors = get_model_colors(models)
    
    def get_top_models(corpus: str, metric: str, num_topics: int) -> str:
        """Helper function to get top N models for a given combination"""
        model_values = []
        for model in models:
            result = get_model_metrics_for_corpus(model, corpus, num_topics, metric)
            if result is not None:
                model_values.append((model, result[0]))  # Use mean value
        
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

def pretty_print_pareto_front_table():
    """
    Print a table showing the number of models on the Pareto front for each corpus and number of topics combination.
    The Pareto front is calculated based on NPMI, WEPS, and ISH metrics.
    """
    console = Console()
    
    # Get list of models and corpora
    models = get_topic_model_list()
    corpora = get_corpus_list()
    metrics = ["NPMI", "WEPS", "ISH"]
    num_topics_list = get_num_topics_list()
    
    # Create the table
    table = Table(title="Number of Models on Pareto Front")
    
    # Add columns
    table.add_column("Corpus", style="white")
    for num_topics in num_topics_list:
        table.add_column(f"Topics={num_topics}")
    
    # Add rows
    for corpus in corpora:
        row = [corpus]
        
        # For each number of topics
        for num_topics in num_topics_list:
            # Get metrics for all models
            all_metrics = []
            for model in models:
                metrics_dict = get_model_metrics_all(corpus, model, num_topics, metrics)
                if metrics_dict:
                    all_metrics.append(metrics_dict)
            
            # Count models on Pareto front
            pareto_count = 0
            for model_metrics in all_metrics:
                if not is_dominated(model_metrics, all_metrics, metrics):
                    pareto_count += 1
            
            row.append(str(pareto_count))
        
        table.add_row(*row)
    
    console.print(table)
    console.print("\n")

def pretty_print_model_points_table():
    """
    Print a table showing the total points for each model broken down by metric.
    Points are calculated by ranking models from 1 to N for each corpus, num_topics pair,
    where a model with rank n gets 1/n points.
    """
    console = Console()
    
    # Get list of models and corpora
    models = get_topic_model_list()
    corpora = get_corpus_list()
    metrics = ["NPMI", "WEPS", "ISH"]
    num_topics_list = get_num_topics_list()
    
    # Initialize points dictionary for each model and metric
    model_points = {
        model: {metric: 0.0 for metric in metrics}
        for model in models
    }
    
    # Calculate points for each combination
    for corpus in corpora:
        for num_topics in num_topics_list:
            for metric in metrics:
                # Get metrics for all models
                model_values = []
                for model in models:
                    result = get_model_metrics_for_corpus(model, corpus, num_topics, metric)
                    if result is not None:
                        model_values.append((model, result[0]))  # Use mean value
                
                # Sort models by their values
                sorted_models = sorted(model_values, key=lambda x: x[1], reverse=(metric != "ISH"))
                
                # Award points based on rank
                for rank, (model, _) in enumerate(sorted_models, 1):
                    model_points[model][metric] += 1.0 / rank
    
    # Create the table
    table = Table(title="Points by Model and Metric")
    
    # Add columns
    table.add_column("Model", style="white")
    for metric in metrics:
        table.add_column(metric, justify="right")
    table.add_column("Total", justify="right")
    
    # Calculate total points for each model
    model_totals = {
        model: sum(points.values())
        for model, points in model_points.items()
    }
    
    # Sort models by total points
    sorted_models = sorted(model_totals.items(), key=lambda x: x[1], reverse=True)
    
    # Add rows
    for model, _ in sorted_models:
        row = [model]
        for metric in metrics:
            row.append(f"{model_points[model][metric]:.1f}")
        row.append(f"{model_totals[model]:.1f}")
        table.add_row(*row)
    
    console.print(table)
    console.print("\n")

if __name__ == '__main__':
    # Example usage
    # pretty_print_metrics_table("NPMI")
    # pretty_print_metrics_table("WEPS")
    # pretty_print_metrics_table("ISH")
    # pretty_print_top_models_table(joined=False, top_n=2)  # Separate tables with top 2 models
    pretty_print_top_models_table(joined=True, top_n=1)  # Single table with top 3 models
    pretty_print_pareto_front_table()  # Show number of models on Pareto front
    pretty_print_model_points_table()  # Show points by model and metric

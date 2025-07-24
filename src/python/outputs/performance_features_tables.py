"""
Generate performance features tables for topic modeling analysis.
Creates CSV files with instances as datasets, features from corpus_features, 
and performance metrics as the target variables.
"""
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg


def extract_corpus_features(session: Session) -> pd.DataFrame:
    """
    Extract corpus features from the database.
    
    Args:
        session: Database session
        
    Returns:
        DataFrame with corpus_name as index and features as columns
    """
    query = text("""
        SELECT 
            corpus_name,
            feature_name,
            feature_value
        FROM pipeline.corpus_features
        ORDER BY corpus_name, feature_name
    """)
    
    result = session.execute(query)
    rows = result.fetchall()
    
    if not rows:
        logging.warning("No corpus features found in database")
        return pd.DataFrame()
    
    # Convert to DataFrame and pivot
    df = pd.DataFrame(rows, columns=['corpus_name', 'feature_name', 'feature_value'])
    features_df = df.pivot(index='corpus_name', columns='feature_name', values='feature_value')
    
    # Add 'feature_' prefix to column names to match expected format
    features_df.columns = [f'feature_{col}' for col in features_df.columns]
    
    logging.info(f"Extracted {len(features_df)} datasets with {len(features_df.columns)} features")
    
    return features_df


def extract_performance_metrics(session: Session) -> dict[str, pd.DataFrame]:
    """
    Extract performance metrics from the database, grouped by metric name.
    
    Args:
        session: Database session
        
    Returns:
        Dictionary mapping metric names to DataFrames with corpus_name and metric values
    """
    query = text("""
        SELECT DISTINCT
            c.name as corpus_name,
            tm.name as topic_model_name,
            tmp.metric_name,
            CASE 
                WHEN tmp.metric_value ? 'score' THEN (tmp.metric_value->>'score')::float
                ELSE tmp.metric_value::float
            END as score_value
        FROM pipeline.topic_model_performance tmp
        JOIN pipeline.topic_model_corpus_result tmcr ON tmp.topic_model_corpus_result_id = tmcr.id
        JOIN pipeline.corpus c ON tmcr.corpus_id = c.id
        JOIN pipeline.topic_model tm ON tmcr.topic_model_id = tm.id
        ORDER BY c.name, tm.name, tmp.metric_name
    """)
    
    result = session.execute(query)
    rows = result.fetchall()
    
    if not rows:
        logging.warning("No performance metrics found in database")
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=['corpus_name', 'topic_model_name', 'metric_name', 'score_value'])
    
    # Group by metric name
    metrics_dict = {}
    
    for metric_name in df['metric_name'].unique():
        metric_df = df[df['metric_name'] == metric_name].copy()
        
        # Create pivot table with corpus_name as index and topic_model_name as columns
        pivot_df = metric_df.pivot_table(
            index='corpus_name',
            columns='topic_model_name',
            values='score_value',
            aggfunc='first'  # Take first value if duplicates exist
        )
        
        # Add 'algo_' prefix to column names to match expected format
        pivot_df.columns = [f'algo_{col}' for col in pivot_df.columns]
        
        metrics_dict[metric_name] = pivot_df
        
        logging.info(f"Extracted metric '{metric_name}' for {len(pivot_df)} datasets with {len(pivot_df.columns)} algorithms")
    
    return metrics_dict


def create_performance_features_table(features_df: pd.DataFrame, metric_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    Create a performance features table by combining corpus features with performance metrics.
    
    Args:
        features_df: DataFrame with corpus features
        metric_df: DataFrame with performance metric values
        metric_name: Name of the performance metric
        
    Returns:
        Combined DataFrame in the format similar to metadata.csv
    """
    # Find common datasets (corpus names)
    common_datasets = features_df.index.intersection(metric_df.index)
    
    if len(common_datasets) == 0:
        logging.warning(f"No common datasets found between features and metric '{metric_name}'")
        return pd.DataFrame()
    
    # Filter to common datasets
    features_subset = features_df.loc[common_datasets].copy()
    metric_subset = metric_df.loc[common_datasets].copy()
    
    # Create combined DataFrame
    combined_df = pd.concat([features_subset, metric_subset], axis=1)
    
    # Reset index to make corpus_name a column
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'corpus_name': 'Instances'}, inplace=True)
    
    # Add empty Source column
    combined_df.insert(1, 'Source', '')
    
    # Sort columns: Instances, Source, features, then algorithms
    feature_cols = [col for col in combined_df.columns if col.startswith('feature_')]
    algo_cols = [col for col in combined_df.columns if col.startswith('algo_')]
    
    column_order = ['Instances', 'Source'] + sorted(feature_cols) + sorted(algo_cols)
    combined_df = combined_df[column_order]
    
    logging.info(f"Created table for metric '{metric_name}' with {len(combined_df)} datasets, "
                f"{len(feature_cols)} features, and {len(algo_cols)} algorithms")
    
    return combined_df


def generate_performance_features_tables(output_dir: Path = None):
    """
    Generate CSV files for each performance metric with features and performance data.
    
    Args:
        output_dir: Directory to save the CSV files (defaults to current directory)
    """
    if output_dir is None:
        output_dir = Path.cwd()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration and create database session
    config = cfg.load_config_from_env()
    db_config = config.database
    
    with get_session(db_config) as session:
        # Extract corpus features
        logging.info("Extracting corpus features...")
        features_df = extract_corpus_features(session)
        
        if features_df.empty:
            logging.error("No corpus features found. Cannot generate tables.")
            return
        
        # Extract performance metrics
        logging.info("Extracting performance metrics...")
        metrics_dict = extract_performance_metrics(session)
        
        if not metrics_dict:
            logging.error("No performance metrics found. Cannot generate tables.")
            return
        
        # Generate a CSV for each metric
        generated_files = []
        
        for metric_name, metric_df in metrics_dict.items():
            logging.info(f"Processing metric: {metric_name}")
            
            # Create combined table
            combined_df = create_performance_features_table(features_df, metric_df, metric_name)
            
            if combined_df.empty:
                logging.warning(f"Skipping metric '{metric_name}' - no data to output")
                continue
            
            # Save to CSV
            filename = f"performance_features_{metric_name.lower()}.csv"
            filepath = output_dir / filename
            
            combined_df.to_csv(filepath, index=False)
            generated_files.append(filepath)
            
            logging.info(f"Saved table for metric '{metric_name}' to {filepath}")
        
        logging.info(f"Generated {len(generated_files)} performance features tables:")
        for filepath in generated_files:
            logging.info(f"  - {filepath}")


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate tables in results_output directory
    output_dir = Path(__file__).parent / "ignore"
    
    logging.info("Starting performance features tables generation...")
    generate_performance_features_tables(output_dir)
    logging.info("Performance features tables generation completed.")
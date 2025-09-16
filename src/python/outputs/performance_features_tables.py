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


def _get_feature_code(feature_name: str) -> str:
    """
    Map feature names to unique 4-5 letter codes.
    
    Args:
        feature_name: The full feature name
        
    Returns:
        Unique code for the feature
    """
    feature_codes = {
        # Words per document metrics
        'words_per_document_average': 'wpdav',
        'words_per_document_iqr_range': 'wpdiq',
        'words_per_document_kurtosis': 'wpdku',
        'words_per_document_median': 'wpdmd',
        'words_per_document_std_dev': 'wpdsd',
        'words_per_document_fano_factor': 'wpdff',
        'words_per_document_kl_div_uniform': 'wpdkl',
        
        # Sentences per document metrics
        'sentences_per_document_average': 'spdav',
        'sentences_per_document_iqr_range': 'spdiq',
        'sentences_per_document_kurtosis': 'spdku',
        'sentences_per_document_median': 'spdmd',
        'sentences_per_document_std_dev': 'spdsd',
        'sentences_per_document_fano_factor': 'spdff',
        'sentences_per_document_kl_div_uniform': 'spdkl',
        
        # Characters per word metrics
        'characters_per_word_average': 'cpwav',
        'characters_per_word_iqr_range': 'cpwiq',
        'characters_per_word_kurtosis': 'cpwku',
        'characters_per_word_median': 'cpwmd',
        'characters_per_word_std_dev': 'cpwsd',
        'characters_per_word_fano_factor': 'cpwff',
        'characters_per_word_kl_div_uniform': 'cpwkl',
        
        # Words per sentence metrics
        'words_per_sentence_average': 'wpsav',
        'words_per_sentence_iqr_range': 'wpsiq',
        'words_per_sentence_kurtosis': 'wpsku',
        'words_per_sentence_median': 'wpsmd',
        'words_per_sentence_std_dev': 'wpssd',
        'words_per_sentence_fano_factor': 'wpsff',
        'words_per_sentence_kl_div_uniform': 'wpskl',
        
        # Compression ratio metrics
        'compression_ratio_average': 'cmpav',
        'compression_ratio_iqr_range': 'cmpiq',
        'compression_ratio_kurtosis': 'cmpku',
        'compression_ratio_median': 'cmpmd',
        'compression_ratio_std_dev': 'cmpsd',
        'compression_ratio_fano_factor': 'cmpff',
        'compression_ratio_kl_div_uniform': 'cmpkl',
        
        # Word entropy metrics
        'word_entropy_average': 'wenav',
        'word_entropy_iqr_range': 'weniq',
        'word_entropy_kurtosis': 'wenku',
        'word_entropy_median': 'wenmd',
        'word_entropy_std_dev': 'wensd',
        'word_entropy_fano_factor': 'wenff',
        'word_entropy_kl_div_uniform': 'wenkl',
        
        # Dataset-level metrics
        'num_documents': 'numdoc',
        'document_compression_ratio': 'doccmp',
        'num_topics': 'numtop',
    }
    
    return feature_codes.get(feature_name, feature_name[:5].lower())


def extract_corpus_features(session: Session) -> pd.DataFrame:
    """
    Extract corpus features from the database, creating one row for each (corpus, num_topics) combination.
    
    Args:
        session: Database session
        
    Returns:
        DataFrame with (corpus_name, num_topics) as index and features as columns
    """
    # Get all unique (corpus, num_topics) combinations and their features
    query = text("""
        WITH corpus_topic_combinations AS (
            SELECT DISTINCT
                c.name as corpus_name,
                tmcr.num_topics
            FROM pipeline.topic_model_corpus_result tmcr
            JOIN pipeline.corpus c ON tmcr.corpus_id = c.id
            WHERE tmcr.soft_delete = FALSE
        )
        SELECT 
            ctc.corpus_name,
            ctc.num_topics,
            cf.feature_name,
            cf.feature_value
        FROM corpus_topic_combinations ctc
        JOIN pipeline.corpus_features cf ON ctc.corpus_name = cf.corpus_name
        
        UNION ALL
        
        SELECT 
            ctc.corpus_name,
            ctc.num_topics,
            'num_topics' as feature_name,
            ctc.num_topics as feature_value
        FROM corpus_topic_combinations ctc
        
        ORDER BY corpus_name, num_topics, feature_name
    """)
    
    result = session.execute(query)
    rows = result.fetchall()
    
    if not rows:
        logging.warning("No corpus features found in database")
        return pd.DataFrame()
    
    # Convert to DataFrame and pivot
    df = pd.DataFrame(rows, columns=['corpus_name', 'num_topics', 'feature_name', 'feature_value'])
    
    # Create multi-index for (corpus_name, num_topics) and pivot on feature_name
    df_indexed = df.set_index(['corpus_name', 'num_topics'])
    features_df = df_indexed.pivot_table(
        index=['corpus_name', 'num_topics'], 
        columns='feature_name', 
        values='feature_value',
        aggfunc='first'
    )
    
    # Add 'feature_' prefix with unique codes to column names to match expected format
    features_df.columns = [f'feature_{_get_feature_code(col)}_{col}' for col in features_df.columns]
    
    logging.info(f"Extracted {len(features_df)} (dataset, num_topics) combinations with {len(features_df.columns)} features")
    
    return features_df


def extract_performance_metrics(session: Session) -> dict[str, pd.DataFrame]:
    """
    Extract performance metrics from the database, grouped by metric name.
    Creates rows for each (corpus_name, num_topics) combination.
    
    Args:
        session: Database session
        
    Returns:
        Dictionary mapping metric names to DataFrames with (corpus_name, num_topics) and metric values
    """
    query = text("""
        SELECT DISTINCT
            c.name as corpus_name,
            tmcr.num_topics,
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
        WHERE tmcr.soft_delete = FALSE
        ORDER BY c.name, tmcr.num_topics, tm.name, tmp.metric_name
    """)
    
    result = session.execute(query)
    rows = result.fetchall()
    
    if not rows:
        logging.warning("No performance metrics found in database")
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=['corpus_name', 'num_topics', 'topic_model_name', 'metric_name', 'score_value'])
    
    # Group by metric name
    metrics_dict = {}
    
    for metric_name in df['metric_name'].unique():
        metric_df = df[df['metric_name'] == metric_name].copy()
        
        # Create pivot table with (corpus_name, num_topics) as index and topic_model_name as columns
        pivot_df = metric_df.pivot_table(
            index=['corpus_name', 'num_topics'],
            columns='topic_model_name',
            values='score_value',
            aggfunc='first'  # Take first value if duplicates exist
        )
        
        # Add 'algo_' prefix to column names to match expected format
        pivot_df.columns = [f'algo_{col}' for col in pivot_df.columns]
        
        metrics_dict[metric_name] = pivot_df
        
        logging.info(f"Extracted metric '{metric_name}' for {len(pivot_df)} (dataset, num_topics) combinations with {len(pivot_df.columns)} algorithms")
    
    return metrics_dict


def create_performance_features_table(features_df: pd.DataFrame, metric_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    Create a performance features table by combining corpus features with performance metrics.
    
    Args:
        features_df: DataFrame with (corpus_name, num_topics) multi-index and corpus features
        metric_df: DataFrame with (corpus_name, num_topics) multi-index and performance metric values
        metric_name: Name of the performance metric
        
    Returns:
        Combined DataFrame in the format similar to metadata.csv
    """
    # Find common (corpus_name, num_topics) combinations
    common_combinations = features_df.index.intersection(metric_df.index)
    
    if len(common_combinations) == 0:
        logging.warning(f"No common (dataset, num_topics) combinations found between features and metric '{metric_name}'")
        return pd.DataFrame()
    
    # Filter to common combinations
    features_subset = features_df.loc[common_combinations].copy()
    metric_subset = metric_df.loc[common_combinations].copy()
    
    # Create combined DataFrame
    combined_df = pd.concat([features_subset, metric_subset], axis=1)
    
    # Reset index to make corpus_name and num_topics columns
    combined_df.reset_index(inplace=True)
    
    # Create instance name that combines corpus_name and num_topics
    combined_df['Instances'] = combined_df['corpus_name'] + '_' + combined_df['num_topics'].astype(str)
    
    # Drop the separate corpus_name and num_topics columns since they're now in Instances
    combined_df.drop(['corpus_name', 'num_topics'], axis=1, inplace=True)
    
    # Add empty Source column
    combined_df.insert(1, 'Source', '')
    
    # Sort columns: Instances, Source, features, then algorithms
    feature_cols = [col for col in combined_df.columns if col.startswith('feature_')]
    algo_cols = [col for col in combined_df.columns if col.startswith('algo_')]
    
    column_order = ['Instances', 'Source'] + sorted(feature_cols) + sorted(algo_cols)
    combined_df = combined_df[column_order]
    
    logging.info(f"Created table for metric '{metric_name}' with {len(combined_df)} (dataset, num_topics) combinations, "
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

            # Find metric df minumum and add it to the metric df
            min_value = metric_df.min().min()
            metric_df = metric_df.add(-min_value)

            # log the median of the metric df
            logging.info(f"Median of {metric_name}: {metric_df.median().median()}")
            
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
    output_dir = Path(__file__).parent / "ignore" / "isa"
    
    logging.info("Starting performance features tables generation...")
    generate_performance_features_tables(output_dir)
    logging.info("Performance features tables generation completed.")
"""
Create 2D projection plots of dataset instances using projection matrices.

This module loads performance feature tables and projection matrices to visualize
datasets in a 2D latent space defined by the projection matrices.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_projection_matrix(matrix_path: Path) -> Tuple[np.ndarray, list[str], list[str]]:
    """
    Load projection matrix from JSON file.
    
    Args:
        matrix_path: Path to the projection matrix JSON file
        
    Returns:
        Tuple of (projection_matrix, latent_variables, features)
    """
    with open(matrix_path, 'r') as f:
        data = json.load(f)
    
    projection_matrix = np.array(data['projection_matrix_A'])
    latent_variables = data['latent_variables']
    features = data['features']
    
    return projection_matrix, latent_variables, features


def load_performance_features_table(csv_path: Path) -> pd.DataFrame:
    """
    Load performance features table from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with performance features data
    """
    df = pd.read_csv(csv_path)
    return df


def extract_feature_values(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """
    Extract feature values from DataFrame using feature names.
    
    Args:
        df: DataFrame with feature columns
        feature_names: List of feature names to extract
        
    Returns:
        NumPy array of feature values (n_instances, n_features)
    """
    # Map short feature names to full column names
    feature_columns = []
    
    for feature_name in feature_names:
        # Find column that contains the feature name
        matching_cols = [col for col in df.columns if feature_name in col and col.startswith('feature_')]
        if matching_cols:
            feature_columns.append(matching_cols[0])
        else:
            raise ValueError(f"Feature '{feature_name}' not found in DataFrame columns")
    
    # Extract feature values
    feature_values = df[feature_columns].values
    
    # Handle any NaN values by filling with column mean
    feature_values = pd.DataFrame(feature_values).fillna(pd.DataFrame(feature_values).mean()).values
    
    return feature_values


def project_to_2d(feature_values: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """
    Project feature values to 2D space using projection matrix.
    
    Args:
        feature_values: Feature values array (n_instances, n_features)
        projection_matrix: Projection matrix (2, n_features)
        
    Returns:
        2D projected coordinates (n_instances, 2)
    """
    # Standardize features (mean=0, std=1)
    feature_values_std = (feature_values - np.mean(feature_values, axis=0)) / np.std(feature_values, axis=0)
    
    # Project to 2D space
    projected_coords = feature_values_std @ projection_matrix.T
    
    return projected_coords


def create_projection_plot(projected_coords: np.ndarray, instance_names: list[str], 
                          latent_variables: list[str], title: str, 
                          save_path: Path = None) -> plt.Figure:
    """
    Create a scatter plot of projected coordinates.
    
    Args:
        projected_coords: 2D projected coordinates (n_instances, 2)
        instance_names: List of instance names for labeling
        latent_variables: Names of the latent variables (e.g., ['Z1', 'Z2'])
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(projected_coords[:, 0], projected_coords[:, 1], 
                        alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    # Add instance labels
    for i, name in enumerate(instance_names):
        ax.annotate(name, (projected_coords[i, 0], projected_coords[i, 1]), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel(f'{latent_variables[0]} (First Latent Dimension)')
    ax.set_ylabel(f'{latent_variables[1]} (Second Latent Dimension)')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def plot_projections_for_metric(metric_name: str, output_dir: Path):
    """
    Create projection plots for a specific metric.
    
    Args:
        metric_name: Name of the metric (e.g., 'ish', 'npmi')
        output_dir: Directory containing input files and for output plots
    """
    
    output_dir = Path(output_dir)
    
    # Load projection matrix
    matrix_path = output_dir / f"proj_mat_{metric_name}.json"
    if not matrix_path.exists():
        print(f"Projection matrix not found: {matrix_path}")
        return
    
    projection_matrix, latent_variables, features = load_projection_matrix(matrix_path)
    
    # Load performance features table
    csv_path = output_dir / f"performance_features_{metric_name}.csv"
    if not csv_path.exists():
        print(f"Performance features table not found: {csv_path}")
        return
    
    df = load_performance_features_table(csv_path)
    
    # Extract feature values
    try:
        feature_values = extract_feature_values(df, features)
    except ValueError as e:
        print(f"Error extracting features: {e}")
        return
    
    # Project to 2D space
    projected_coords = project_to_2d(feature_values, projection_matrix)
    
    # Get instance names
    instance_names = df['Instances'].tolist()
    
    # Create plot
    title = f"2D Projection of Dataset Instances - {metric_name.upper()} Metric"
    save_path = output_dir / f"projection_plot_{metric_name}.png"
    
    fig = create_projection_plot(projected_coords, instance_names, latent_variables, 
                                 title, save_path)
    
    # Show plot
    plt.show()
    
    return fig


def plot_all_projections(output_dir: Path):
    """
    Create projection plots for all available metrics.
    
    Args:
        output_dir: Directory containing input files and for output plots
    """
    
    output_dir = Path(output_dir)
    
    # Find all projection matrix files
    matrix_files = list(output_dir.glob("proj_mat_*.json"))
    
    if not matrix_files:
        print(f"No projection matrix files found in {output_dir}")
        return
    
    # Extract metric names from filenames
    metrics = []
    for matrix_file in matrix_files:
        metric_name = matrix_file.stem.replace("proj_mat_", "")
        metrics.append(metric_name)
    
    print(f"Found projection matrices for metrics: {metrics}")
    
    # Create plots for each metric
    figures = {}
    for metric in metrics:
        print(f"\nCreating projection plot for metric: {metric}")
        try:
            fig = plot_projections_for_metric(metric, output_dir)
            if fig:
                figures[metric] = fig
        except Exception as e:
            print(f"Error creating plot for metric '{metric}': {e}")
    
    return figures


if __name__ == '__main__':
    # Create plots for all available metrics
    output_dir = Path(__file__).parent / "ignore" / "isa"
    
    print("Creating 2D projection plots...")
    figures = plot_all_projections(output_dir)
    
    if figures:
        print(f"\nSuccessfully created {len(figures)} projection plots")
        for metric in figures.keys():
            print(f"  - {metric}")
    else:
        print("No plots were created")
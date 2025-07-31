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
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr
from itertools import combinations


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


def cloister_algorithm(feature_matrix: np.ndarray, epsilon: float = 0.5, p_value: float = 0.05) -> np.ndarray:
    """
    CLOISTER: Correlated Limits of the Instance Space's Theoretical or Experimental Regions.
    
    Args:
        feature_matrix: Matrix of instance features (n_instances, n_features)
        epsilon: Minimum correlation value for co-linear features
        p_value: P-value threshold for uncorrelated features
        
    Returns:
        Matrix of boundary coordinates in feature space
    """
    n_instances, n_features = feature_matrix.shape
    
    # Step 1: Compute correlation matrix
    correlation_matrix = np.corrcoef(feature_matrix.T)
    
    # Step 2: Compute p-values for correlations
    p_ff = np.ones_like(correlation_matrix)
    for i in range(n_features):
        for j in range(i+1, n_features):
            if n_instances > 2:  # Need at least 3 points for correlation
                _, p_val = pearsonr(feature_matrix[:, i], feature_matrix[:, j])
                p_ff[i, j] = p_val
                p_ff[j, i] = p_val
    
    # Step 3: Find min and max bounds for each feature
    f_l = np.min(feature_matrix, axis=0)  # Lower bounds
    f_u = np.max(feature_matrix, axis=0)  # Upper bounds
    
    # Step 4: Generate all possible vertex combinations (hypercube vertices)
    # Each vertex is a combination of upper/lower bounds for each feature
    from itertools import product
    
    vertex_combinations = []
    
    # Generate all 2^n vertices of the hypercube
    for combination in product([0, 1], repeat=n_features):
        vertex = np.zeros(n_features)
        for i, choice in enumerate(combination):
            vertex[i] = f_u[i] if choice == 1 else f_l[i]
        vertex_combinations.append(vertex)
    
    # Step 5: Filter vertices based on correlation constraints
    valid_vertices = []
    
    for vertex in vertex_combinations:
        valid = True
        
        # Check all pairs of features for correlation constraints
        for i in range(n_features):
            for j in range(i+1, n_features):
                if p_ff[i, j] < p_value:  # Features are significantly correlated
                    corr = correlation_matrix[i, j]
                    
                    # If strongly positively correlated (corr > epsilon)
                    if corr > epsilon:
                        # Cannot have one feature high and another low
                        if (np.isclose(vertex[i], f_u[i]) and np.isclose(vertex[j], f_l[j])) or \
                           (np.isclose(vertex[i], f_l[i]) and np.isclose(vertex[j], f_u[j])):
                            valid = False
                            break
                    
                    # If strongly negatively correlated (corr < -epsilon)
                    elif corr < -epsilon:
                        # Cannot have both features high or both low
                        if (np.isclose(vertex[i], f_u[i]) and np.isclose(vertex[j], f_u[j])) or \
                           (np.isclose(vertex[i], f_l[i]) and np.isclose(vertex[j], f_l[j])):
                            valid = False
                            break
            
            if not valid:
                break
        
        if valid:
            valid_vertices.append(vertex)
    
    # Step 6: Convert to array and remove duplicates
    if valid_vertices:
        vertex_matrix = np.array(valid_vertices)
        # Remove duplicate vertices
        vertex_matrix = np.unique(vertex_matrix, axis=0)
    else:
        # If no valid vertices found, return the hypercube corners
        print("Warning: No valid vertices found, using hypercube corners")
        vertex_matrix = np.array([f_l, f_u])
        
    print(f"CLOISTER generated {len(vertex_matrix)} boundary vertices")
    return vertex_matrix


def compute_bounded_projection(feature_values: np.ndarray, projection_matrix: np.ndarray, 
                              epsilon: float = 0.5, p_value: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute both instance projections and boundary using CLOISTER algorithm.
    
    Args:
        feature_values: Feature values array (n_instances, n_features)
        projection_matrix: Projection matrix (2, n_features)
        epsilon: Minimum correlation value for CLOISTER
        p_value: P-value threshold for CLOISTER
        
    Returns:
        Tuple of (projected_coords, boundary_coords)
    """
    # Standardize features (mean=0, std=1)
    feature_values_std = (feature_values - np.mean(feature_values, axis=0)) / np.std(feature_values, axis=0)
    
    # Project instances to 2D space
    projected_coords = feature_values_std @ projection_matrix.T
    
    # Compute CLOISTER boundary vertices
    boundary_vertices = cloister_algorithm(feature_values_std, epsilon, p_value)
    
    # Project boundary vertices to 2D space
    if len(boundary_vertices) > 0:
        boundary_coords = boundary_vertices @ projection_matrix.T
    else:
        boundary_coords = np.array([])
    
    return projected_coords, boundary_coords


def create_projection_plot(projected_coords: np.ndarray, instance_names: list[str], 
                          latent_variables: list[str], title: str, 
                          save_path: Path = None, boundary_coords: np.ndarray = None) -> plt.Figure:
    """
    Create a scatter plot of projected coordinates with optional boundary.
    
    Args:
        projected_coords: 2D projected coordinates (n_instances, 2)
        instance_names: List of instance names for labeling
        latent_variables: Names of the latent variables (e.g., ['Z1', 'Z2'])
        title: Plot title
        save_path: Optional path to save the plot
        boundary_coords: Optional boundary coordinates from CLOISTER
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot boundary if provided
    if boundary_coords is not None and len(boundary_coords) > 2:
        try:
            # Compute convex hull of boundary points
            hull = ConvexHull(boundary_coords)
            
            # Plot boundary as filled area (no label)
            ax.fill(boundary_coords[hull.vertices, 0], boundary_coords[hull.vertices, 1], 
                   alpha=0.15, color='lightgray', zorder=1)
            
            # Plot boundary edges more prominently (with label for first line only)
            for i, simplex in enumerate(hull.simplices):
                label = 'Theoretical Boundary' if i == 0 else None
                ax.plot(boundary_coords[simplex, 0], boundary_coords[simplex, 1], 
                       'purple', alpha=0.8, linewidth=2, zorder=2, label=label)
            
            # Plot boundary vertices
            # ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1], 
            #           c='purple', s=40, alpha=0.8, marker='s', 
            #           label='Boundary Vertices', zorder=3)
            
            print(f"Plotted boundary with {len(boundary_coords)} vertices")
            
        except Exception as e:
            print(f"Warning: Could not plot boundary - {e}")
            # Plot boundary points without hull as fallback
            ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1], 
                      c='purple', s=40, alpha=0.8, marker='s', 
                      label='Boundary Vertices', zorder=3)
    
    # Create scatter plot for instances
    scatter = ax.scatter(projected_coords[:, 0], projected_coords[:, 1], 
                        alpha=0.8, s=80, edgecolors='black', linewidth=1,
                        c='red', label='Dataset Instances')
    
    # Add instance labels
    for i, name in enumerate(instance_names):
        ax.annotate(name, (projected_coords[i, 0], projected_coords[i, 1]), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=9, alpha=0.9, weight='bold')
    
    # Set labels and title
    ax.set_xlabel(f'{latent_variables[0]} (First Latent Dimension)', fontsize=12)
    ax.set_ylabel(f'{latent_variables[1]} (Second Latent Dimension)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    if boundary_coords is not None and len(boundary_coords) > 2:
        ax.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def create_bounded_projection_plot(projected_coords: np.ndarray, boundary_coords: np.ndarray,
                                  instance_names: list[str], latent_variables: list[str], 
                                  title: str, save_path: Path = None) -> plt.Figure:
    """
    Create a projection plot specifically highlighting the CLOISTER boundary.
    
    Args:
        projected_coords: 2D projected coordinates (n_instances, 2)
        boundary_coords: Boundary coordinates from CLOISTER
        instance_names: List of instance names for labeling
        latent_variables: Names of the latent variables
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    return create_projection_plot(projected_coords, instance_names, latent_variables, 
                                 title, save_path, boundary_coords)


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
    
    # Get instance names
    instance_names = df['Instances'].tolist()
    
    # Create basic projection plot
    projected_coords = project_to_2d(feature_values, projection_matrix)
    title = f"2D Projection of Dataset Instances - {metric_name.upper()} Metric"
    save_path = output_dir / f"projection_plot_{metric_name}.png"
    
    fig1 = create_projection_plot(projected_coords, instance_names, latent_variables, 
                                 title, save_path)
    
    # Create bounded projection using CLOISTER
    print(f"Computing CLOISTER boundary for {metric_name}...")
    projected_coords_bounded, boundary_coords = compute_bounded_projection(
        feature_values, projection_matrix, epsilon=0.5, p_value=0.05)
    
    # Create bounded projection plot
    title_bounded = f"Matilda Projection Plot - {metric_name.upper()} Metric"
    save_path_bounded = output_dir / f"bounded_projection_plot_{metric_name}.png"
    
    fig2 = create_bounded_projection_plot(projected_coords_bounded, boundary_coords,
                                         instance_names, latent_variables, 
                                         title_bounded, save_path_bounded)
    
    return fig1, fig2


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
        print(f"\nCreating projection plots for metric: {metric}")
        try:
            result = plot_projections_for_metric(metric, output_dir)
            if result:
                if isinstance(result, tuple):
                    fig_basic, fig_bounded = result
                    figures[f"{metric}_basic"] = fig_basic
                    figures[f"{metric}_bounded"] = fig_bounded
                else:
                    figures[metric] = result
        except Exception as e:
            print(f"Error creating plots for metric '{metric}': {e}")
    
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
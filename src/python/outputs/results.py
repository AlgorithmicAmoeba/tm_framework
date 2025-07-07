"""
Topic Modeling Results Analysis
Creates comprehensive visualizations and tables for topic modeling results analysis.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import mannwhitneyu, rankdata
from sqlalchemy import text

from configuration import load_config_from_env
from database import get_session


def get_all_results_data() -> Dict[str, Any]:
    """Load all results data from database into structured format"""
    config = load_config_from_env()
    
    with get_session(config.database) as session:
        # Get all results with model, corpus, metrics info
        query = text("""
            SELECT 
                m.name as model_name,
                c.name as corpus_name,
                r.num_topics,
                p.metric_name,
                p.metric_value
            FROM pipeline.topic_model_performance p
            JOIN pipeline.topic_model_corpus_result r ON p.topic_model_corpus_result_id = r.id
            JOIN pipeline.topic_model m ON r.topic_model_id = m.id
            JOIN pipeline.corpus c ON r.corpus_id = c.id
            WHERE r.soft_delete = false
            ORDER BY m.name, c.name, r.num_topics, p.metric_name
        """)
        
        results = session.execute(query).fetchall()
    
    # Structure data
    data = {}
    for result in results:
        model, corpus, num_topics, metric, value = result
        
        # Extract metric values
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            values = [v for v in value.values() if isinstance(v, (int, float))]
        elif isinstance(value, (int, float)):
            values = [value]
        else:
            values = []
        
        if not values:
            continue
        
        # Store in nested structure
        if model not in data:
            data[model] = {}
        if corpus not in data[model]:
            data[model][corpus] = {}
        if num_topics not in data[model][corpus]:
            data[model][corpus][num_topics] = {}
        if metric not in data[model][corpus][num_topics]:
            data[model][corpus][num_topics][metric] = []
        
        data[model][corpus][num_topics][metric].extend(values)
    
    return data


def get_pretty_model_name(model_name: str) -> str:
    """Convert model names to pretty LaTeX format"""
    replace_dict = {
        "SemanticSignalSeparation": r"$\text{S}^3$",
        "BERTopic": r"BERTopic",
        "TopMost": r"TopMost",
        "TurFTopic": r"TurFTopic",
        "LDA": r"LDA",
        "NMF": r"NMF",
        "CTM": r"CTM",
        "ETM": r"ETM",
        "PLDA": r"PLDA",
        "ProdLDA": r"ProdLDA",
        "NVDM": r"NVDM"
    }
    
    # Replace underscores with hyphens and apply pretty names
    pretty_name = model_name.replace("_", "-")
    return replace_dict.get(model_name, pretty_name)


def create_performance_summary_table(data: Dict[str, Any], output_dir: Path) -> None:
    """Create overall performance summary table with statistical significance"""
    print("Creating performance summary table...")
    
    metrics = ["NPMI", "WEPS", "WECS", "ISH"]
    models = list(data.keys())
    
    # Collect all metric values for ranking
    all_values = {metric: {model: [] for model in models} for metric in metrics}
    
    # Extract all values across datasets and topic numbers
    for model in models:
        for corpus in data[model]:
            for num_topics in data[model][corpus]:
                for metric in metrics:
                    if metric in data[model][corpus][num_topics]:
                        values = data[model][corpus][num_topics][metric]
                        all_values[metric][model].extend(values)
    
    # Calculate statistics and rankings
    results_df = pd.DataFrame(index=models)
    
    for metric in metrics:
        means = []
        stds = []
        ranks = []
        
        # Calculate means and std for each model
        model_means = {}
        for model in models:
            if all_values[metric][model]:
                model_means[model] = np.mean(all_values[metric][model])
                means.append(model_means[model])
                stds.append(np.std(all_values[metric][model]))
            else:
                model_means[model] = np.nan
                means.append(np.nan)
                stds.append(np.nan)
        
        # Calculate rankings (higher is better for all metrics except ISH)
        valid_means = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
        if metric == "ISH":
            # Lower is better for ISH
            sorted_indices = sorted(valid_means, key=lambda x: x[1])
        else:
            # Higher is better for other metrics
            sorted_indices = sorted(valid_means, key=lambda x: x[1], reverse=True)
        
        # Assign ranks
        model_ranks = [np.nan] * len(models)
        for rank, (idx, _) in enumerate(sorted_indices, 1):
            model_ranks[idx] = rank
        
        # Statistical significance testing
        significance = []
        
        # Get all models with valid data
        valid_models = [(i, model) for i, model in enumerate(models) if all_values[metric][model]]
        
        if not valid_models:
            significance = [""] * len(models)
        else:
            # Helper function to test if model A is significantly better than model B
            def is_significantly_better(model_a_idx, model_b_idx):
                try:
                    if metric == "ISH":
                        # Lower is better for ISH
                        _, p_value = mannwhitneyu(
                            all_values[metric][models[model_a_idx]],
                            all_values[metric][models[model_b_idx]],
                            alternative='less'
                        )
                    else:
                        # Higher is better for other metrics
                        _, p_value = mannwhitneyu(
                            all_values[metric][models[model_a_idx]],
                            all_values[metric][models[model_b_idx]],
                            alternative='greater'
                        )
                    return p_value < 0.05
                except:
                    return False
            
            # Step 1: Find the best class
            best_class = []
            remaining_models = [idx for idx, _ in valid_models]
            
            # Start with the top-ranked model
            top_model_idx = min(remaining_models, key=lambda x: model_ranks[x])
            candidate_best = [top_model_idx]
            
            # Add models that are not significantly different from the top model
            for model_idx in remaining_models:
                if model_idx != top_model_idx:
                    # Check if this model is significantly different from the top model
                    if not is_significantly_better(top_model_idx, model_idx) and not is_significantly_better(model_idx, top_model_idx):
                        candidate_best.append(model_idx)
            
            # Verify that all models in candidate_best are not significantly different from each other
            valid_best = True
            for i in range(len(candidate_best)):
                for j in range(i + 1, len(candidate_best)):
                    if is_significantly_better(candidate_best[i], candidate_best[j]) or is_significantly_better(candidate_best[j], candidate_best[i]):
                        valid_best = False
                        break
                if not valid_best:
                    break
            
            if valid_best:
                # Check if the best class is significantly better than all other models
                other_models = [idx for idx in remaining_models if idx not in candidate_best]
                is_best_class = True
                
                if other_models:
                    for best_idx in candidate_best:
                        for other_idx in other_models:
                            if not is_significantly_better(best_idx, other_idx):
                                is_best_class = False
                                break
                        if not is_best_class:
                            break
                
                if is_best_class:
                    best_class = candidate_best
            
            # Step 2: Find the second-best class from remaining models
            second_best_class = []
            remaining_models = [idx for idx in remaining_models if idx not in best_class]
            
            if remaining_models and best_class:
                # Start with the best-ranked remaining model
                if remaining_models:
                    top_remaining_idx = min(remaining_models, key=lambda x: model_ranks[x])
                    candidate_second_best = [top_remaining_idx]
                    
                    # Add models that are not significantly different from the top remaining model
                    for model_idx in remaining_models:
                        if model_idx != top_remaining_idx:
                            if not is_significantly_better(top_remaining_idx, model_idx) and not is_significantly_better(model_idx, top_remaining_idx):
                                candidate_second_best.append(model_idx)
                    
                    # Verify that all models in candidate_second_best are not significantly different from each other

                    # Check if second-best class is significantly worse than all best models
                    significantly_worse_than_best = True
                    for second_idx in candidate_second_best:
                        for best_idx in best_class:
                            if not is_significantly_better(best_idx, second_idx):
                                significantly_worse_than_best = False
                                break
                        if not significantly_worse_than_best:
                            break
                    
                    # Check if second-best class is significantly better than all other models
                    other_models = [idx for idx in remaining_models if idx not in candidate_second_best]
                    significantly_better_than_others = True
                    
                    if other_models:
                        for second_idx in candidate_second_best:
                            for other_idx in other_models:
                                if not is_significantly_better(second_idx, other_idx):
                                    significantly_better_than_others = False
                                    break
                            if not significantly_better_than_others:
                                break
                    
                    if significantly_worse_than_best and significantly_better_than_others:
                        second_best_class = candidate_second_best
            
            # Assign significance markers
            for i, model in enumerate(models):
                if i in best_class:
                    significance.append("**")
                elif i in second_best_class:
                    significance.append("*")
                else:
                    significance.append("")
        
        # Format column
        col_values = []
        for i, model in enumerate(models):
            if not np.isnan(means[i]):
                sig = significance[i]
                if sig == "**":
                    col_values.append(f"\\textbf{{{means[i]:.3f} ± {stds[i]:.3f}}}")
                elif sig == "*":
                    col_values.append(f"\\underline{{{means[i]:.3f} ± {stds[i]:.3f}}}")
                else:
                    col_values.append(f"{means[i]:.3f} ± {stds[i]:.3f}")
            else:
                col_values.append("--")
        
        results_df[metric] = col_values
    
    # Sort by average rank
    avg_ranks = []
    for model in models:
        ranks = []
        for metric in metrics:
            model_idx = models.index(model)
            valid_means = [(i, m) for i, m in enumerate([np.mean(all_values[metric][mod]) if all_values[metric][mod] else np.nan for mod in models]) if not np.isnan(m)]
            if metric == "ISH":
                sorted_indices = sorted(valid_means, key=lambda x: x[1])
            else:
                sorted_indices = sorted(valid_means, key=lambda x: x[1], reverse=True)
            
            for rank, (idx, _) in enumerate(sorted_indices, 1):
                if idx == model_idx:
                    ranks.append(rank)
                    break
        
        avg_ranks.append(np.mean(ranks) if ranks else np.inf)
    
    # Sort DataFrame by average rank
    sort_order = np.argsort(avg_ranks)
    results_df = results_df.iloc[sort_order]
    
    # Apply pretty model names to index
    results_df.index = [get_pretty_model_name(model) for model in results_df.index]
    
    # Save as LaTeX table
    latex_table = results_df.to_latex(
        escape=False,
        column_format='l' + 'c' * len(metrics),
        caption="Overall Performance Summary. \\textbf{Bold} indicates statistically significant best performer, \\underline{Underlined} indicates statistically significant second-best performer (Mann-Whitney U test, p < 0.05).",
        label="tab:performance_summary"
    )
    
    with open(output_dir / "performance_summary.tex", "w") as f:
        f.write(latex_table)
    
    print(f"Performance summary table saved to {output_dir / 'performance_summary.tex'}")


def create_algorithm_heatmaps(data: Dict[str, Any], output_dir: Path) -> None:
    """Create algorithm comparison heatmaps for each metric"""
    print("Creating algorithm heatmaps...")
    
    metrics = ["NPMI", "WEPS", "WECS", "ISH"]
    models = list(data.keys())
    
    # Set up matplotlib for publication quality
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.linewidth': 0.5,
        'figure.dpi': 600
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Create win matrix
        win_matrix = np.zeros((len(models), len(models)))
        
        # Compare each pair of models
        for i, model_a in enumerate(models):
            for j, model_b in enumerate(models):
                if i == j:
                    continue
                
                wins = 0
                total_comparisons = 0
                
                # Compare across all datasets and topic numbers
                for corpus in data[model_a]:
                    if corpus not in data[model_b]:
                        continue
                    
                    for num_topics in data[model_a][corpus]:
                        if num_topics not in data[model_b][corpus]:
                            continue
                        
                        if (metric in data[model_a][corpus][num_topics] and 
                            metric in data[model_b][corpus][num_topics]):
                            
                            values_a = data[model_a][corpus][num_topics][metric]
                            values_b = data[model_b][corpus][num_topics][metric]
                            
                            if values_a and values_b:
                                # Statistical test
                                try:
                                    if metric == "ISH":
                                        # Lower is better for ISH
                                        _, p_value = mannwhitneyu(values_a, values_b, alternative='less')
                                    else:
                                        # Higher is better for other metrics
                                        _, p_value = mannwhitneyu(values_a, values_b, alternative='greater')
                                    
                                    if p_value < 0.05:
                                        wins += 1
                                    total_comparisons += 1
                                except:
                                    continue
                
                win_matrix[i, j] = wins
        
        # Create heatmap
        im = ax.imshow(win_matrix, cmap='gray', aspect='auto')
        
        # Set ticks and labels
        pretty_names = [get_pretty_model_name(model) for model in models]
        ax.set_xticks(range(len(models)))
        ax.set_yticks(range(len(models)))
        ax.set_xticklabels(pretty_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(pretty_names, fontsize=8)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(models)):
                if i != j:
                    # Use white text for dark colors (high values) and black text for light colors (low values)
                    text_color = "white" if win_matrix[i, j] < np.max(win_matrix)/2 else "black"
                    text = ax.text(j, i, f'{int(win_matrix[i, j])}',
                                 ha="center", va="center", color=text_color,
                                 fontsize=7)
        
        ax.set_title(f'{metric}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Algorithm B', fontsize=9)
        ax.set_ylabel('Algorithm A', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Wins', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / "algorithm_heatmaps.png", dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Algorithm heatmaps saved to {output_dir / 'algorithm_heatmaps.png'}")


def create_pareto_plots(data: Dict[str, Any], output_dir: Path) -> None:
    """Create 2D Pareto plots for selected metric pairs"""
    print("Creating Pareto plots...")
    
    metric_pairs = [
        ("NPMI", "WEPS"),
        ("NPMI", "WECS"), 
        ("WEPS", "ISH"),
        ("WECS", "ISH")
    ]
    
    models = list(data.keys())
    
    # Set up matplotlib
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'figure.dpi': 600
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for plot_idx, (metric1, metric2) in enumerate(metric_pairs):
        ax = axes[plot_idx]
        
        # Calculate average performance for each model
        model_performance = {}
        for model in models:
            values1 = []
            values2 = []
            
            for corpus in data[model]:
                for num_topics in data[model][corpus]:
                    if (metric1 in data[model][corpus][num_topics] and 
                        metric2 in data[model][corpus][num_topics]):
                        values1.extend(data[model][corpus][num_topics][metric1])
                        values2.extend(data[model][corpus][num_topics][metric2])
            
            if values1 and values2:
                model_performance[model] = (np.mean(values1), np.mean(values2))
        
        if not model_performance:
            continue
        
        # Determine Pareto optimal points
        pareto_models = []
        non_pareto_models = []
        
        for model in model_performance:
            is_pareto = True
            perf1, perf2 = model_performance[model]
            
            for other_model in model_performance:
                if other_model == model:
                    continue
                
                other_perf1, other_perf2 = model_performance[other_model]
                
                # Check if dominated
                if metric1 in ["NPMI", "WEPS", "WECS"] and metric2 in ["NPMI", "WEPS", "WECS"]:
                    # Both higher is better
                    if other_perf1 >= perf1 and other_perf2 >= perf2 and (other_perf1 > perf1 or other_perf2 > perf2):
                        is_pareto = False
                        break
                elif metric1 in ["NPMI", "WEPS", "WECS"] and metric2 == "ISH":
                    # Higher is better for metric1, lower is better for metric2
                    if other_perf1 >= perf1 and other_perf2 <= perf2 and (other_perf1 > perf1 or other_perf2 < perf2):
                        is_pareto = False
                        break
                elif metric1 == "ISH" and metric2 in ["NPMI", "WEPS", "WECS"]:
                    # Lower is better for metric1, higher is better for metric2
                    if other_perf1 <= perf1 and other_perf2 >= perf2 and (other_perf1 < perf1 or other_perf2 > perf2):
                        is_pareto = False
                        break
                elif metric1 == "ISH" and metric2 == "ISH":
                    # Both lower is better
                    if other_perf1 <= perf1 and other_perf2 <= perf2 and (other_perf1 < perf1 or other_perf2 < perf2):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_models.append(model)
            else:
                non_pareto_models.append(model)
        
        # Plot non-Pareto models
        if non_pareto_models:
            x_vals = [model_performance[model][0] for model in non_pareto_models]
            y_vals = [model_performance[model][1] for model in non_pareto_models]
            ax.scatter(x_vals, y_vals, c='lightgray', s=50, alpha=0.6, marker='x', label='Non-Pareto')
        
        # Plot Pareto models
        if pareto_models:
            x_vals = [model_performance[model][0] for model in pareto_models]
            y_vals = [model_performance[model][1] for model in pareto_models]
            
            # Sort Pareto points for proper line connection
            pareto_points = list(zip(x_vals, y_vals, pareto_models))
            if metric1 in ["NPMI", "WEPS", "WECS"]:
                # Higher is better for metric1
                pareto_points = sorted(pareto_points, key=lambda x: x[0], reverse=True)
            else:
                # Lower is better for metric1 (ISH)
                pareto_points = sorted(pareto_points, key=lambda x: x[0])
            
            x_vals_sorted = [p[0] for p in pareto_points]
            y_vals_sorted = [p[1] for p in pareto_points]
            pareto_models_sorted = [p[2] for p in pareto_points]
            
            ax.scatter(x_vals_sorted, y_vals_sorted, c='black', s=80, alpha=0.8, marker='o', label='Pareto Optimal')
            
            # Connect Pareto points with dotted line
            ax.plot(x_vals_sorted, y_vals_sorted, 'k--', linewidth=2, alpha=0.8, zorder=4)
            
            # Add labels for Pareto optimal points
            for i, model in enumerate(pareto_models_sorted):
                x, y = x_vals_sorted[i], y_vals_sorted[i]
                ax.annotate(get_pretty_model_name(model), (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold')
        
        ax.set_xlabel(metric1, fontsize=11)
        ax.set_ylabel(metric2, fontsize=11)
        ax.set_title(f'{metric1} vs {metric2}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_plots.png", dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Pareto plots saved to {output_dir / 'pareto_plots.png'}")


def create_pareto_table(data: Dict[str, Any], output_dir: Path) -> None:
    """Create table showing count of non-Pareto optimal models"""
    print("Creating Pareto table...")
    
    metrics = ["NPMI", "WEPS", "WECS", "ISH"]
    models = list(data.keys())
    
    # Get all datasets and topic numbers
    datasets = set()
    topic_numbers = set()
    
    for model in models:
        for corpus in data[model]:
            datasets.add(corpus)
            for num_topics in data[model][corpus]:
                topic_numbers.add(num_topics)
    
    datasets = sorted(list(datasets))
    topic_numbers = sorted(list(topic_numbers))
    
    # Create results matrix
    results_matrix = []
    
    for dataset in datasets:
        row = [dataset]
        
        for num_topics in topic_numbers:
            # Get all models with complete data for this dataset-topic combination
            complete_models = []
            for model in models:
                if (dataset in data[model] and 
                    num_topics in data[model][dataset]):
                    
                    # Check if all metrics are available
                    has_all_metrics = True
                    model_metrics = {}
                    
                    for metric in metrics:
                        if metric in data[model][dataset][num_topics]:
                            model_metrics[metric] = np.mean(data[model][dataset][num_topics][metric])
                        else:
                            has_all_metrics = False
                            break
                    
                    if has_all_metrics:
                        complete_models.append((model, model_metrics))
            
            if not complete_models:
                row.append("--")
                continue
            
            # Calculate Pareto front
            non_pareto_count = 0
            
            for model, model_metrics in complete_models:
                is_dominated = False
                
                for other_model, other_metrics in complete_models:
                    if other_model == model:
                        continue
                    
                    # Check if current model is dominated by other model
                    dominates = True
                    for metric in metrics:
                        if metric in ["NPMI", "WEPS", "WECS"]:
                            # Higher is better
                            if other_metrics[metric] <= model_metrics[metric]:
                                dominates = False
                                break
                        elif metric == "ISH":
                            # Lower is better
                            if other_metrics[metric] >= model_metrics[metric]:
                                dominates = False
                                break
                    
                    if dominates:
                        is_dominated = True
                        break
                
                if is_dominated:
                    non_pareto_count += 1
            
            row.append(str(non_pareto_count))
        
        results_matrix.append(row)
    
    # Create DataFrame
    columns = ["Dataset"] + [f"{nt}" for nt in topic_numbers]
    df = pd.DataFrame(results_matrix, columns=columns)
    
    # Save as LaTeX table
    latex_table = df.to_latex(
        index=False,
        column_format='l' + 'c' * len(topic_numbers),
        caption="Number of non-Pareto optimal models for each dataset-topic combination. Pareto optimality calculated using all four metrics (NPMI, WEPS, WECS, ISH).",
        label="tab:pareto_counts"
    )
    
    with open(output_dir / "pareto_table.tex", "w") as f:
        f.write(latex_table)
    
    print(f"Pareto table saved to {output_dir / 'pareto_table.tex'}")


def main():
    """Main function to orchestrate all analysis"""
    print("Starting topic modeling results analysis...")
    
    # Create output directory
    output_dir = Path("src/python/outputs/results_output")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data from database...")
    data = get_all_results_data()
    
    if not data:
        print("No data found in database!")
        return
    
    print(f"Loaded data for {len(data)} models")
    
    # Create all outputs
    create_performance_summary_table(data, output_dir)
    create_algorithm_heatmaps(data, output_dir)
    create_pareto_plots(data, output_dir)
    create_pareto_table(data, output_dir)
    
    print(f"\nAll analysis complete! Results saved to {output_dir}")
    print("Generated files:")
    print("- performance_summary.tex")
    print("- algorithm_heatmaps.png")
    print("- pareto_plots.png")
    print("- pareto_table.tex")


if __name__ == "__main__":
    main()
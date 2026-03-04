import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sqlalchemy import text

from configuration import load_config_from_env
from database import get_session


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_METRICS = ("NPMI", "WEPS", "WECS", "ISH")
SUMMARY_METRICS = ("NPMI", "WEPS", "WECS", "NISH")
DEFAULT_FAMILIES = ("BERTopic", "CombinedTM", "ZeroShotTM")
TOP_K_LABELS = 10
PARETO_METRIC_PAIRS = (
    ("NPMI", "WEPS"),
    ("NPMI", "WECS"),
    ("WEPS", "NISH"),
    ("WECS", "NISH"),
)


@dataclass(frozen=True)
class MetricRow:
    corpus_name: str
    model_name: str
    num_topics: int
    target_chunk_count: int | None
    padding_method: str | None
    metric_name: str
    score: float
    source: str


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BOE vs standard topic-model metrics and generate legacy plus "
            "output-style comparison plots and tables."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "ignore" / "plots",
        help="Directory to write plot files.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics (default: NPMI,WEPS,WECS,ISH).",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=1,
        help="Minimum aligned pairs required to produce a metric plot.",
    )
    parser.add_argument(
        "--families",
        type=str,
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated model families for BOE/openai/sbert comparisons.",
    )
    parser.add_argument(
        "--generate-output-style-plots",
        type=parse_bool,
        default=True,
        help="Whether to generate algorithm heatmaps and Pareto plots (default: true).",
    )
    parser.add_argument(
        "--generate-family-variant-comparisons",
        type=parse_bool,
        default=True,
        help=(
            "Whether to generate per-family boe/openai/sbert distribution and "
            "delta-matrix plots (default: true)."
        ),
    )
    parser.add_argument(
        "--generate-rank-table",
        type=parse_bool,
        default=True,
        help="Whether to generate performance summary rank table (default: true).",
    )
    return parser.parse_args()


def metric_group_name(model_name: str) -> str:
    if model_name in {"BERTopic", "BERTopic_sbert"}:
        return "BERTopic"
    if model_name in {"ZeroShotTM", "ZeroShotTM_sbert"}:
        return "ZeroShotTM"
    return model_name


def model_family_name(model_name: str) -> str:
    if model_name.endswith("_sbert"):
        return model_name[: -len("_sbert")]
    return model_name


def embedding_variant(model_name: str, source: str) -> str:
    if source == "BOE":
        return "boe"
    if model_name.endswith("_sbert"):
        return "sbert"
    return "openai"


def metric_display_name(metric_name: str) -> str:
    if metric_name == "ISH":
        return "NISH"
    return metric_name


def parse_metric_value(metric_value: Any) -> float | None:
    if metric_value is None:
        return None

    value = metric_value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                return float(stripped)
            except ValueError:
                return None

    if isinstance(value, dict):
        score = value.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        for item in value.values():
            if isinstance(item, (int, float)):
                return float(item)
        return None

    if isinstance(value, list):
        for item in value:
            if isinstance(item, (int, float)):
                return float(item)
        return None

    if isinstance(value, (int, float)):
        return float(value)

    return None


def fetch_boe_rows(session, allowed_metrics: set[str]) -> list[MetricRow]:
    query = text(
        """
        SELECT
            r.corpus_name,
            m.name AS model_name,
            r.num_topics,
            r.target_chunk_count,
            r.padding_method,
            p.metric_name,
            p.metric_value
        FROM pipeline.boe_topic_model_performance p
        JOIN pipeline.boe_topic_model_corpus_result r
            ON p.boe_topic_model_corpus_result_id = r.id
        JOIN pipeline.boe_topic_model m
            ON r.topic_model_id = m.id
        WHERE r.soft_delete = FALSE
        """
    )
    rows = session.execute(query).fetchall()
    parsed: list[MetricRow] = []
    dropped = 0
    for row in rows:
        metric_name = row.metric_name
        if metric_name not in allowed_metrics:
            continue

        score = parse_metric_value(row.metric_value)
        if score is None or not np.isfinite(score):
            dropped += 1
            continue

        if metric_name == "ISH":
            score = -score

        parsed.append(
            MetricRow(
                corpus_name=row.corpus_name,
                model_name=row.model_name,
                num_topics=row.num_topics,
                target_chunk_count=int(row.target_chunk_count),
                padding_method=str(row.padding_method) if row.padding_method is not None else None,
                metric_name=metric_display_name(metric_name),
                score=score,
                source="BOE",
            )
        )

    logger.info("Fetched %d BOE rows, dropped %d invalid rows", len(parsed), dropped)
    return parsed


def fetch_standard_rows(session, allowed_metrics: set[str]) -> list[MetricRow]:
    query = text(
        """
        SELECT
            c.name AS corpus_name,
            m.name AS model_name,
            r.num_topics,
            p.metric_name,
            p.metric_value
        FROM pipeline.topic_model_performance p
        JOIN pipeline.topic_model_corpus_result r
            ON p.topic_model_corpus_result_id = r.id
        JOIN pipeline.topic_model m
            ON r.topic_model_id = m.id
        JOIN pipeline.corpus c
            ON r.corpus_id = c.id
        WHERE r.soft_delete = FALSE
        """
    )
    rows = session.execute(query).fetchall()
    parsed: list[MetricRow] = []
    dropped = 0
    for row in rows:
        metric_name = row.metric_name
        if metric_name not in allowed_metrics:
            continue

        score = parse_metric_value(row.metric_value)
        if score is None or not np.isfinite(score):
            dropped += 1
            continue

        if metric_name == "ISH":
            score = -score

        parsed.append(
            MetricRow(
                corpus_name=row.corpus_name,
                model_name=row.model_name,
                num_topics=row.num_topics,
                target_chunk_count=None,
                padding_method=None,
                metric_name=metric_display_name(metric_name),
                score=score,
                source="STANDARD",
            )
        )

    logger.info(
        "Fetched %d standard rows, dropped %d invalid rows", len(parsed), dropped
    )
    return parsed


def rows_to_dataframe(rows: list[MetricRow]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "corpus_name": r.corpus_name,
                "model_name": r.model_name,
                "model_group": metric_group_name(r.model_name),
                "family_name": model_family_name(r.model_name),
                "num_topics": r.num_topics,
                "target_chunk_count": r.target_chunk_count,
                "padding_method": r.padding_method,
                "metric_name": r.metric_name,
                "score": r.score,
                "source": r.source,
            }
            for r in rows
        ]
    )
    frame["embedding_variant"] = [
        embedding_variant(model_name, source)
        for model_name, source in zip(frame["model_name"], frame["source"], strict=False)
    ]
    frame["variant_model_name"] = (
        frame["family_name"] + "_" + frame["embedding_variant"]
    )
    return frame


def aggregate_side(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    subset = df[df["source"] == source_name].copy()
    grouped = (
        subset.groupby(
            ["corpus_name", "model_group", "num_topics", "metric_name"],
            as_index=False,
        )["score"]
        .agg(mean="mean", count="count")
        .rename(
            columns={
                "mean": f"{source_name.lower()}_mean",
                "count": f"n_{source_name.lower()}",
            }
        )
    )
    return grouped


def build_aligned_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    boe = aggregate_side(df, "BOE")
    std = aggregate_side(df, "STANDARD")
    merged = boe.merge(
        std,
        on=["corpus_name", "model_group", "num_topics", "metric_name"],
        how="inner",
    )
    merged["delta"] = merged["boe_mean"] - merged["standard_mean"]
    return merged


def sanitize_metric(metric_name: str) -> str:
    return metric_name.lower().replace("/", "_").replace(" ", "_")


def prettify_model_name(model_name: str) -> str:
    return model_name.replace("_", "-")


def requested_variant_models(families: list[str]) -> list[str]:
    variants: list[str] = []
    for family in families:
        variants.extend(
            [
                f"{family}_boe",
                f"{family}_openai",
                f"{family}_sbert",
            ]
        )
    return variants


def normalize_query_metrics(metrics: set[str]) -> set[str]:
    normalized = set()
    for metric in metrics:
        if metric == "NISH":
            normalized.add("ISH")
        else:
            normalized.add(metric)
    return normalized


def save_distribution_plot(metric_df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    boe_vals = metric_df["boe_mean"].to_numpy()
    std_vals = metric_df["standard_mean"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 6))

    parts = ax.violinplot([boe_vals, std_vals], positions=[1, 2], showmeans=True)
    for body in parts["bodies"]:
        body.set_alpha(0.45)

    ax.boxplot([boe_vals, std_vals], positions=[1, 2], widths=0.2, showfliers=False)

    rng = np.random.default_rng(42)
    ax.scatter(1 + rng.normal(0, 0.03, size=len(boe_vals)), boe_vals, s=18, alpha=0.6)
    ax.scatter(2 + rng.normal(0, 0.03, size=len(std_vals)), std_vals, s=18, alpha=0.6)

    ax.set_xticks([1, 2], labels=["BOE", "Standard"])
    ax.set_title(f"{metric}: Distribution Comparison")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25, axis="y")
    ax.text(
        0.5,
        -0.12,
        (
            f"pairs={len(metric_df)} | median(BOE)={np.median(boe_vals):.4f} | "
            f"median(Standard)={np.median(std_vals):.4f}"
        ),
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"dist_{sanitize_metric(metric)}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_delta_scatter_plot(metric_df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    corpora = sorted(metric_df["corpus_name"].unique())
    cmap = plt.cm.get_cmap("tab20", max(len(corpora), 1))
    corpus_to_color = {c: cmap(i) for i, c in enumerate(corpora)}

    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h"]
    models = sorted(metric_df["model_group"].unique())
    model_to_marker = {m: markers[i % len(markers)] for i, m in enumerate(models)}

    for _, row in metric_df.iterrows():
        ax.scatter(
            row["standard_mean"],
            row["boe_mean"],
            color=corpus_to_color[row["corpus_name"]],
            marker=model_to_marker[row["model_group"]],
            s=70,
            alpha=0.85,
        )

    all_vals = np.concatenate(
        [metric_df["standard_mean"].to_numpy(), metric_df["boe_mean"].to_numpy()]
    )
    lo = np.min(all_vals)
    hi = np.max(all_vals)
    pad = (hi - lo) * 0.05 if hi > lo else 0.1
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        linestyle="--",
        color="black",
        linewidth=1.2,
    )

    top = metric_df.reindex(metric_df["delta"].abs().sort_values(ascending=False).index).head(
        TOP_K_LABELS
    )
    for _, row in top.iterrows():
        ax.annotate(
            f"{row['corpus_name']} | {row['model_group']} | k={int(row['num_topics'])}",
            (row["standard_mean"], row["boe_mean"]),
            fontsize=7,
            alpha=0.8,
        )

    ax.set_title(f"{metric}: BOE vs Standard Paired Scatter")
    ax.set_xlabel(f"Standard ({metric})")
    ax.set_ylabel(f"BOE ({metric})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"delta_scatter_{sanitize_metric(metric)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_delta_heatmap(metric_df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    heat = metric_df.pivot_table(
        index="corpus_name",
        columns="model_group",
        values="delta",
        aggfunc="mean",
    ).sort_index()
    if heat.empty:
        logger.warning("Skipping heatmap for %s: no data", metric)
        return

    values = heat.to_numpy(dtype=float)
    v = np.nanmax(np.abs(values))
    if not np.isfinite(v) or v == 0:
        v = 1e-8

    fig, ax = plt.subplots(
        figsize=(max(8, heat.shape[1] * 1.2), max(5, heat.shape[0] * 0.7))
    )
    im = ax.imshow(values, cmap="coolwarm", vmin=-v, vmax=v, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Delta (BOE - Standard)")

    ax.set_xticks(np.arange(heat.shape[1]), labels=heat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(heat.shape[0]), labels=heat.index)
    ax.set_title(f"{metric}: Mean Delta Heatmap")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        output_dir / f"delta_heatmap_{sanitize_metric(metric)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_legacy_plots(aligned: pd.DataFrame, min_pairs: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in sorted(aligned["metric_name"].unique()):
        metric_df = aligned[aligned["metric_name"] == metric].copy()
        if len(metric_df) < min_pairs:
            logger.warning(
                "Skipping legacy metric %s: %d aligned pairs < min_pairs=%d",
                metric,
                len(metric_df),
                min_pairs,
            )
            continue

        logger.info("Generating legacy BOE-vs-standard plots for %s (%d rows)", metric, len(metric_df))
        save_distribution_plot(metric_df, metric, output_dir)
        save_delta_scatter_plot(metric_df, metric, output_dir)
        save_delta_heatmap(metric_df, metric, output_dir)


def build_nested_metric_data(
    frame: pd.DataFrame,
    model_col: str,
    metrics: list[str],
    allowed_models: list[str] | None = None,
) -> dict[str, dict[str, dict[int, dict[str, list[float]]]]]:
    metric_set = set(metrics)
    subset = frame[frame["metric_name"].isin(metric_set)].copy()
    if allowed_models is not None:
        subset = subset[subset[model_col].isin(allowed_models)].copy()

    data: dict[str, dict[str, dict[int, dict[str, list[float]]]]] = {}
    for row in subset.itertuples(index=False):
        model = getattr(row, model_col)
        corpus = row.corpus_name
        num_topics = int(row.num_topics)
        metric = row.metric_name
        score = float(row.score)

        data.setdefault(model, {}).setdefault(corpus, {}).setdefault(num_topics, {}).setdefault(
            metric, []
        ).append(score)

    return data


def compute_win_matrix(
    data: dict[str, dict[str, dict[int, dict[str, list[float]]]]],
    models: list[str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    win_matrix = np.zeros((len(models), len(models)), dtype=float)
    draw_matrix = np.zeros((len(models), len(models)), dtype=float)
    total_matrix = np.zeros((len(models), len(models)), dtype=float)

    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i == j:
                continue

            wins = 0
            draws = 0
            total = 0
            for corpus in data.get(model_a, {}):
                if corpus not in data.get(model_b, {}):
                    continue

                for num_topics in data[model_a][corpus]:
                    if num_topics not in data[model_b][corpus]:
                        continue

                    values_a = data[model_a][corpus][num_topics].get(metric, [])
                    values_b = data[model_b][corpus][num_topics].get(metric, [])
                    if not values_a or not values_b:
                        continue

                    try:
                        _, p_value_ab = mannwhitneyu(values_a, values_b, alternative="greater")
                        _, p_value_ba = mannwhitneyu(values_b, values_a, alternative="greater")
                    except Exception:
                        continue

                    total += 1
                    a_wins = p_value_ab < 0.05
                    b_wins = p_value_ba < 0.05
                    if a_wins:
                        wins += 1
                    elif not b_wins:
                        draws += 1

            win_matrix[i, j] = wins
            draw_matrix[i, j] = draws
            total_matrix[i, j] = total

    return win_matrix, draw_matrix, total_matrix


def compute_win_matrix_rectangular(
    data: dict[str, dict[str, dict[int, dict[str, list[float]]]]],
    models_a: list[str],
    models_b: list[str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    win_matrix = np.zeros((len(models_a), len(models_b)), dtype=float)
    draw_matrix = np.zeros((len(models_a), len(models_b)), dtype=float)
    total_matrix = np.zeros((len(models_a), len(models_b)), dtype=float)

    for i, model_a in enumerate(models_a):
        for j, model_b in enumerate(models_b):
            if model_a == model_b:
                continue

            wins = 0
            draws = 0
            total = 0
            for corpus in data.get(model_a, {}):
                if corpus not in data.get(model_b, {}):
                    continue

                for num_topics in data[model_a][corpus]:
                    if num_topics not in data[model_b][corpus]:
                        continue

                    values_a = data[model_a][corpus][num_topics].get(metric, [])
                    values_b = data[model_b][corpus][num_topics].get(metric, [])
                    if not values_a or not values_b:
                        continue

                    try:
                        _, p_value_ab = mannwhitneyu(values_a, values_b, alternative="greater")
                        _, p_value_ba = mannwhitneyu(values_b, values_a, alternative="greater")
                    except Exception:
                        continue

                    total += 1
                    a_wins = p_value_ab < 0.05
                    b_wins = p_value_ba < 0.05
                    if a_wins:
                        wins += 1
                    elif not b_wins:
                        draws += 1

            win_matrix[i, j] = wins
            draw_matrix[i, j] = draws
            total_matrix[i, j] = total

    return win_matrix, draw_matrix, total_matrix


def draw_heatmap(
    win_matrix: np.ndarray,
    draw_matrix: np.ndarray,
    total_matrix: np.ndarray,
    models: list[str],
    metric: str,
    ax,
    vmin: float,
    vmax: float,
    show_xlabels: bool,
    show_ylabels: bool,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    im = ax.imshow(win_matrix, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    labels = [prettify_model_name(model) for model in models]

    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    if show_xlabels:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticklabels([])
    if show_ylabels:
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.set_yticklabels([])

    max_val = vmax if vmax > 0 else 1.0
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue
            value = win_matrix[i, j]
            text_color = "white" if value < max_val / 2 else "black"
            draws = int(draw_matrix[i, j])
            total = int(total_matrix[i, j])
            ax.text(
                j,
                i,
                f"{int(value)} ({draws}/{total})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.set_xlabel("Algorithm B" if show_xlabel else "", fontsize=9)
    ax.set_ylabel("Algorithm A" if show_ylabel else "", fontsize=9)
    im.colorbar = None


def draw_heatmap_rectangular(
    win_matrix: np.ndarray,
    draw_matrix: np.ndarray,
    total_matrix: np.ndarray,
    models_a: list[str],
    models_b: list[str],
    metric: str,
    ax,
    vmin: float,
    vmax: float,
    show_xlabels: bool,
    show_ylabels: bool,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    im = ax.imshow(win_matrix, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    labels_a = [prettify_model_name(model) for model in models_a]
    labels_b = [prettify_model_name(model) for model in models_b]

    ax.set_xticks(range(len(models_b)))
    ax.set_yticks(range(len(models_a)))
    if show_xlabels:
        ax.set_xticklabels(labels_b, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticklabels([])
    if show_ylabels:
        ax.set_yticklabels(labels_a, fontsize=8)
    else:
        ax.set_yticklabels([])

    max_val = vmax if vmax > 0 else 1.0
    for i in range(len(models_a)):
        for j in range(len(models_b)):
            if models_a[i] == models_b[j]:
                continue
            value = win_matrix[i, j]
            text_color = "white" if value < max_val / 2 else "black"
            draws = int(draw_matrix[i, j])
            total = int(total_matrix[i, j])
            ax.text(
                j,
                i,
                f"{int(value)} ({draws}/{total})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.set_xlabel("Algorithm B" if show_xlabel else "", fontsize=9)
    ax.set_ylabel("Algorithm A" if show_ylabel else "", fontsize=9)
    im.colorbar = None


def create_variant_algorithm_heatmaps(
    variant_df: pd.DataFrame,
    models: list[str],
    metrics: list[str],
    output_dir: Path,
    filename_suffix: str = "",
) -> None:
    data = build_nested_metric_data(variant_df, "variant_model_name", metrics, models)
    available_models = [model for model in models if model in data]
    missing_models = [model for model in models if model not in data]
    if missing_models:
        logger.warning("Missing variant models for heatmaps: %s", ", ".join(missing_models))

    if len(available_models) < 2:
        logger.warning("Skipping variant heatmaps: fewer than 2 models with data")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    matrices = {
        metric: compute_win_matrix(data, available_models, metric)
        for metric in metrics
    }
    global_min = min(np.min(matrix_tuple[0]) for matrix_tuple in matrices.values())
    global_max = max(np.max(matrix_tuple[0]) for matrix_tuple in matrices.values())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig.patch.set_facecolor("#e9ebe8")
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.set_facecolor("#e9ebe8")
        win_matrix, draw_matrix, total_matrix = matrices[metric]
        draw_heatmap(
            win_matrix,
            draw_matrix,
            total_matrix,
            available_models,
            metric,
            ax=ax,
            vmin=global_min,
            vmax=global_max,
            show_xlabels=idx in [2, 3],
            show_ylabels=idx in [0, 2],
            show_xlabel=idx in [2, 3],
            show_ylabel=idx in [0, 2],
        )

    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, wspace=0.05, hspace=0.05)
    fig.savefig(
        output_dir / f"algorithm_heatmaps_variants{filename_suffix}.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 8))
        win_matrix, draw_matrix, total_matrix = matrices[metric]
        draw_heatmap(
            win_matrix,
            draw_matrix,
            total_matrix,
            available_models,
            metric,
            ax=ax,
            vmin=global_min,
            vmax=global_max,
            show_xlabels=True,
            show_ylabels=True,
            show_xlabel=True,
            show_ylabel=True,
        )
        fig.tight_layout()
        filename = f"algorithm_heatmap_{sanitize_metric(metric)}_variants{filename_suffix}.png"
        fig.savefig(output_dir / filename, dpi=600, bbox_inches="tight")
        plt.close(fig)


def create_variant_algorithm_heatmaps_by_target_chunk_count(
    variant_df: pd.DataFrame,
    models: list[str],
    metrics: list[str],
    output_dir: Path,
) -> None:
    boe_rows = variant_df[variant_df["source"] == "BOE"].copy()
    if boe_rows.empty:
        logger.warning("Skipping BOE target-chunk heatmaps: no BOE rows in variant data")
        return

    target_chunk_counts = sorted(
        int(value) for value in boe_rows["target_chunk_count"].dropna().unique().tolist()
    )
    if not target_chunk_counts:
        logger.warning("Skipping BOE target-chunk heatmaps: no target_chunk_count values found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for target_chunk_count in target_chunk_counts:
        boe_tcc_rows = boe_rows[boe_rows["target_chunk_count"] == target_chunk_count].copy()
        if boe_tcc_rows.empty:
            logger.warning(
                "Skipping BOE target-chunk heatmaps for tcc=%d: no BOE rows",
                target_chunk_count,
            )
            continue

        key_frame = boe_tcc_rows[["corpus_name", "num_topics"]].drop_duplicates()
        if key_frame.empty:
            logger.warning(
                "Skipping BOE target-chunk heatmaps for tcc=%d: no corpus/topic keys",
                target_chunk_count,
            )
            continue

        filtered_df = variant_df.merge(
            key_frame,
            on=["corpus_name", "num_topics"],
            how="inner",
        )
        filename_suffix = f"_tcc_{target_chunk_count}"
        create_variant_algorithm_heatmaps(
            filtered_df,
            models,
            metrics,
            output_dir,
            filename_suffix=filename_suffix,
        )


def create_boe_padding_split_algorithm_heatmaps(
    variant_df: pd.DataFrame,
    models: list[str],
    metrics: list[str],
    output_dir: Path,
) -> None:
    boe_rows = variant_df[variant_df["source"] == "BOE"].copy()
    if boe_rows.empty:
        logger.warning("Skipping BOE padding split heatmaps: no BOE rows in variant data")
        return

    split_df = variant_df.copy()
    split_df["variant_model_name_split"] = split_df["variant_model_name"]
    knn_mask = (split_df["source"] == "BOE") & (split_df["padding_method"] == "knn_mean")
    noise_mask = (split_df["source"] == "BOE") & (split_df["padding_method"] == "noise_only")
    split_df.loc[knn_mask, "variant_model_name_split"] = (
        split_df.loc[knn_mask, "family_name"] + "_boe_knn"
    )
    split_df.loc[noise_mask, "variant_model_name_split"] = (
        split_df.loc[noise_mask, "family_name"] + "_boe_no"
    )

    base_boe_models = [model for model in models if model.endswith("_boe")]
    models_a: list[str] = []
    for model in base_boe_models:
        family = model[: -len("_boe")]
        models_a.extend([f"{family}_boe_knn", f"{family}_boe_no"])

    models_b: list[str] = []
    for model in models:
        if model.endswith("_boe"):
            family = model[: -len("_boe")]
            models_b.extend([f"{family}_boe_knn", f"{family}_boe_no"])
        else:
            models_b.append(model)

    data = build_nested_metric_data(split_df, "variant_model_name_split", metrics, models_b)
    models_a = [model for model in models_a if model in data]
    models_b = [model for model in models_b if model in data]
    if not models_a or len(models_b) < 2:
        logger.warning(
            "Skipping BOE padding split heatmaps: insufficient models (A=%d, B=%d)",
            len(models_a),
            len(models_b),
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    matrices = {
        metric: compute_win_matrix_rectangular(data, models_a, models_b, metric)
        for metric in metrics
    }
    global_min = min(np.min(matrix_tuple[0]) for matrix_tuple in matrices.values())
    global_max = max(np.max(matrix_tuple[0]) for matrix_tuple in matrices.values())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig.patch.set_facecolor("#e9ebe8")
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.set_facecolor("#e9ebe8")
        win_matrix, draw_matrix, total_matrix = matrices[metric]
        draw_heatmap_rectangular(
            win_matrix,
            draw_matrix,
            total_matrix,
            models_a,
            models_b,
            metric,
            ax=ax,
            vmin=global_min,
            vmax=global_max,
            show_xlabels=idx in [2, 3],
            show_ylabels=idx in [0, 2],
            show_xlabel=idx in [2, 3],
            show_ylabel=idx in [0, 2],
        )

    plt.subplots_adjust(
        left=0.08,
        right=0.95,
        top=0.95,
        bottom=0.08,
        wspace=0.05,
        hspace=0.05,
    )
    fig.savefig(
        output_dir / "algorithm_heatmaps_variants_boe_a_split_padding.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 8))
        win_matrix, draw_matrix, total_matrix = matrices[metric]
        draw_heatmap_rectangular(
            win_matrix,
            draw_matrix,
            total_matrix,
            models_a,
            models_b,
            metric,
            ax=ax,
            vmin=global_min,
            vmax=global_max,
            show_xlabels=True,
            show_ylabels=True,
            show_xlabel=True,
            show_ylabel=True,
        )
        fig.tight_layout()
        filename = f"algorithm_heatmap_{sanitize_metric(metric)}_variants_boe_a_split_padding.png"
        fig.savefig(output_dir / filename, dpi=600, bbox_inches="tight")
        plt.close(fig)


def create_boe_padding_split_algorithm_heatmaps_by_target_chunk_count(
    variant_df: pd.DataFrame,
    models: list[str],
    metrics: list[str],
    output_dir: Path,
) -> None:
    boe_rows = variant_df[variant_df["source"] == "BOE"].copy()
    if boe_rows.empty:
        logger.warning(
            "Skipping BOE padding split target-chunk heatmaps: no BOE rows in variant data"
        )
        return

    target_chunk_counts = sorted(
        int(value) for value in boe_rows["target_chunk_count"].dropna().unique().tolist()
    )
    if not target_chunk_counts:
        logger.warning(
            "Skipping BOE padding split target-chunk heatmaps: no target_chunk_count values found"
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for target_chunk_count in target_chunk_counts:
        boe_tcc_rows = boe_rows[boe_rows["target_chunk_count"] == target_chunk_count].copy()
        if boe_tcc_rows.empty:
            logger.warning(
                "Skipping BOE padding split target-chunk heatmaps for tcc=%d: no BOE rows",
                target_chunk_count,
            )
            continue

        key_frame = boe_tcc_rows[["corpus_name", "num_topics"]].drop_duplicates()
        if key_frame.empty:
            logger.warning(
                "Skipping BOE padding split target-chunk heatmaps for tcc=%d: no corpus/topic keys",
                target_chunk_count,
            )
            continue

        filtered_df = variant_df.merge(
            key_frame,
            on=["corpus_name", "num_topics"],
            how="inner",
        )
        create_boe_padding_split_algorithm_heatmaps(
            filtered_df,
            models,
            metrics,
            output_dir / f"tcc_{target_chunk_count}",
        )


def model_pair_means(
    data: dict[str, dict[str, dict[int, dict[str, list[float]]]]],
    models: list[str],
    metric1: str,
    metric2: str,
) -> dict[str, tuple[float, float]]:
    performance: dict[str, tuple[float, float]] = {}
    for model in models:
        values1: list[float] = []
        values2: list[float] = []
        for corpus in data.get(model, {}):
            for num_topics in data[model][corpus]:
                values1.extend(data[model][corpus][num_topics].get(metric1, []))
                values2.extend(data[model][corpus][num_topics].get(metric2, []))

        if values1 and values2:
            performance[model] = (float(np.mean(values1)), float(np.mean(values2)))
    return performance


def split_pareto(performance: dict[str, tuple[float, float]]) -> tuple[list[str], list[str]]:
    pareto: list[str] = []
    non_pareto: list[str] = []
    for model, (perf1, perf2) in performance.items():
        dominated = False
        for other_model, (other1, other2) in performance.items():
            if other_model == model:
                continue
            if other1 >= perf1 and other2 >= perf2 and (other1 > perf1 or other2 > perf2):
                dominated = True
                break

        if dominated:
            non_pareto.append(model)
        else:
            pareto.append(model)
    return pareto, non_pareto


def draw_single_pareto(
    ax,
    performance: dict[str, tuple[float, float]],
    metric1: str,
    metric2: str,
) -> None:
    if not performance:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{metric1} vs {metric2}")
        return

    pareto_models, non_pareto_models = split_pareto(performance)

    if non_pareto_models:
        x_vals = [performance[model][0] for model in non_pareto_models]
        y_vals = [performance[model][1] for model in non_pareto_models]
        ax.scatter(x_vals, y_vals, c="black", s=50, alpha=0.7, marker="x", label="Non-Pareto")

    if pareto_models:
        pareto_points = sorted(
            [(performance[model][0], performance[model][1], model) for model in pareto_models],
            key=lambda item: item[0],
            reverse=True,
        )
        x_vals = [point[0] for point in pareto_points]
        y_vals = [point[1] for point in pareto_points]
        labels = [point[2] for point in pareto_points]

        ax.scatter(x_vals, y_vals, c="black", s=80, alpha=0.8, marker="o", label="Pareto Optimal")
        ax.plot(x_vals, y_vals, "k--", linewidth=2, alpha=0.8, zorder=4)

        for idx, model in enumerate(labels):
            ax.annotate(
                prettify_model_name(model),
                (x_vals[idx], y_vals[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel(metric1, fontsize=11)
    ax.set_ylabel(metric2, fontsize=11)
    ax.set_title(f"{metric1} vs {metric2}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def create_variant_pareto_plots(
    variant_df: pd.DataFrame,
    models: list[str],
    metrics: list[str],
    output_dir: Path,
) -> None:
    metric_pairs = [pair for pair in PARETO_METRIC_PAIRS if pair[0] in metrics and pair[1] in metrics]
    if not metric_pairs:
        logger.warning("Skipping Pareto plots: no available metric pairs")
        return

    data = build_nested_metric_data(variant_df, "variant_model_name", metrics, models)
    available_models = [model for model in models if model in data]
    if len(available_models) < 2:
        logger.warning("Skipping variant Pareto plots: fewer than 2 models with data")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (metric1, metric2) in enumerate(metric_pairs):
        performance = model_pair_means(data, available_models, metric1, metric2)
        draw_single_pareto(axes[idx], performance, metric1, metric2)

    # Clear unused axes if fewer than 4 pairs
    for idx in range(len(metric_pairs), 4):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "pareto_plots_variants.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    for metric1, metric2 in metric_pairs:
        performance = model_pair_means(data, available_models, metric1, metric2)
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_single_pareto(ax, performance, metric1, metric2)
        fig.tight_layout()
        filename = f"pareto_plot_{sanitize_metric(metric1)}_vs_{sanitize_metric(metric2)}_variants.png"
        fig.savefig(output_dir / filename, dpi=600, bbox_inches="tight")
        plt.close(fig)


def create_variant_performance_summary_table(
    variant_df: pd.DataFrame,
    models: list[str],
    metrics: list[str],
    output_dir: Path,
) -> None:
    data = build_nested_metric_data(variant_df, "variant_model_name", metrics, models)
    available_models = [model for model in models if model in data]
    if not available_models:
        logger.warning("Skipping performance summary table: no variant model data")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    all_values: dict[str, dict[str, list[float]]] = {
        metric: {model: [] for model in available_models} for metric in metrics
    }
    for model in available_models:
        for corpus in data.get(model, {}):
            for num_topics in data[model][corpus]:
                for metric in metrics:
                    all_values[metric][model].extend(
                        data[model][corpus][num_topics].get(metric, [])
                    )

    results_df = pd.DataFrame(index=available_models)

    for metric in metrics:
        means: list[float] = []
        stds: list[float] = []
        for model in available_models:
            values = all_values[metric][model]
            if values:
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values)))
            else:
                means.append(np.nan)
                stds.append(np.nan)

        valid_means = [(i, value) for i, value in enumerate(means) if not np.isnan(value)]
        sorted_indices = sorted(valid_means, key=lambda item: item[1], reverse=True)
        model_ranks = [np.nan] * len(available_models)
        for rank, (idx, _) in enumerate(sorted_indices, 1):
            model_ranks[idx] = rank

        significance: list[str] = []
        valid_models = [idx for idx, model in enumerate(available_models) if all_values[metric][model]]

        if not valid_models:
            significance = ["" for _ in available_models]
        else:

            def is_significantly_better(model_a_idx: int, model_b_idx: int) -> bool:
                try:
                    _, p_value = mannwhitneyu(
                        all_values[metric][available_models[model_a_idx]],
                        all_values[metric][available_models[model_b_idx]],
                        alternative="greater",
                    )
                    return p_value < 0.05
                except Exception:
                    return False

            best_class: list[int] = []
            remaining_models = valid_models.copy()
            top_model_idx = min(remaining_models, key=lambda idx: model_ranks[idx])
            candidate_best = [top_model_idx]

            for model_idx in remaining_models:
                if model_idx == top_model_idx:
                    continue
                if not is_significantly_better(top_model_idx, model_idx) and not is_significantly_better(
                    model_idx, top_model_idx
                ):
                    candidate_best.append(model_idx)

            valid_best = True
            for i in range(len(candidate_best)):
                for j in range(i + 1, len(candidate_best)):
                    if is_significantly_better(candidate_best[i], candidate_best[j]) or is_significantly_better(
                        candidate_best[j], candidate_best[i]
                    ):
                        valid_best = False
                        break
                if not valid_best:
                    break

            if valid_best:
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

            second_best_class: list[int] = []
            remaining_models = [idx for idx in remaining_models if idx not in best_class]
            if remaining_models and best_class:
                top_remaining_idx = min(remaining_models, key=lambda idx: model_ranks[idx])
                candidate_second = [top_remaining_idx]

                for model_idx in remaining_models:
                    if model_idx == top_remaining_idx:
                        continue
                    if not is_significantly_better(top_remaining_idx, model_idx) and not is_significantly_better(
                        model_idx, top_remaining_idx
                    ):
                        candidate_second.append(model_idx)

                significantly_worse_than_best = True
                for second_idx in candidate_second:
                    for best_idx in best_class:
                        if not is_significantly_better(best_idx, second_idx):
                            significantly_worse_than_best = False
                            break
                    if not significantly_worse_than_best:
                        break

                other_models = [idx for idx in remaining_models if idx not in candidate_second]
                significantly_better_than_others = True
                if other_models:
                    for second_idx in candidate_second:
                        for other_idx in other_models:
                            if not is_significantly_better(second_idx, other_idx):
                                significantly_better_than_others = False
                                break
                        if not significantly_better_than_others:
                            break

                if significantly_worse_than_best and significantly_better_than_others:
                    second_best_class = candidate_second

            for i in range(len(available_models)):
                if i in best_class:
                    significance.append("**")
                elif i in second_best_class:
                    significance.append("*")
                else:
                    significance.append("")

        col_values: list[str] = []
        for idx in range(len(available_models)):
            if np.isnan(means[idx]):
                col_values.append("--")
                continue

            sig = significance[idx]
            if sig == "**":
                col_values.append(f"\\textbf{{{means[idx]:.3f} ± {stds[idx]:.3f}}}")
            elif sig == "*":
                col_values.append(f"\\underline{{{means[idx]:.3f} ± {stds[idx]:.3f}}}")
            else:
                col_values.append(f"{means[idx]:.3f} ± {stds[idx]:.3f}")

        results_df[metric] = col_values

    avg_ranks: list[float] = []
    for model in available_models:
        ranks: list[int] = []
        for metric in metrics:
            model_idx = available_models.index(model)
            metric_means = [
                float(np.mean(all_values[metric][item])) if all_values[metric][item] else np.nan
                for item in available_models
            ]
            valid_means = [(idx, val) for idx, val in enumerate(metric_means) if not np.isnan(val)]
            sorted_indices = sorted(valid_means, key=lambda item: item[1], reverse=True)
            for rank, (idx, _) in enumerate(sorted_indices, 1):
                if idx == model_idx:
                    ranks.append(rank)
                    break
        avg_ranks.append(float(np.mean(ranks)) if ranks else float("inf"))

    sort_order = np.argsort(avg_ranks)
    results_df = results_df.iloc[sort_order]
    results_df.index = [prettify_model_name(model) for model in results_df.index]

    latex_table = results_df.to_latex(
        escape=False,
        column_format="l" + "c" * len(metrics),
        caption=(
            "Variant performance summary across metrics. \\textbf{Bold} indicates "
            "statistically significant best performer, \\underline{Underlined} indicates "
            "statistically significant second-best performer (Mann-Whitney U, p < 0.05)."
        ),
        label="tab:boe_variant_performance_summary",
    )

    output_path = output_dir / "performance_summary_variants.tex"
    output_path.write_text(latex_table)
    logger.info("Wrote variant performance summary table: %s", output_path)


def create_family_distribution_plot(
    metric_df: pd.DataFrame,
    family: str,
    metric: str,
    variants: list[str],
    output_dir: Path,
) -> None:
    available_variants = [
        variant for variant in variants if not metric_df[metric_df["variant_model_name"] == variant].empty
    ]
    if len(available_variants) < 2:
        logger.warning(
            "Skipping family distribution for %s/%s: fewer than 2 variants with data",
            family,
            metric,
        )
        return

    positions = np.arange(1, len(available_variants) + 1)
    values = [
        metric_df[metric_df["variant_model_name"] == variant]["score"].to_numpy()
        for variant in available_variants
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    parts = ax.violinplot(values, positions=positions, showmeans=True)
    for body in parts["bodies"]:
        body.set_alpha(0.45)

    ax.boxplot(values, positions=positions, widths=0.2, showfliers=False)

    rng = np.random.default_rng(42)
    for idx, variant_values in enumerate(values):
        ax.scatter(
            positions[idx] + rng.normal(0, 0.03, size=len(variant_values)),
            variant_values,
            s=18,
            alpha=0.6,
        )

    ax.set_xticks(positions, labels=[prettify_model_name(item) for item in available_variants], rotation=15)
    ax.set_title(f"{family}: {metric} variant distribution")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    filename = f"family_{sanitize_metric(family)}_dist_{sanitize_metric(metric)}.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_family_delta_matrix_plot(
    metric_df: pd.DataFrame,
    family: str,
    metric: str,
    variants: list[str],
    output_dir: Path,
) -> None:
    if metric_df.empty:
        logger.warning("Skipping family delta matrix for %s/%s: no data", family, metric)
        return

    pivot = metric_df.pivot_table(
        index=["corpus_name", "num_topics"],
        columns="variant_model_name",
        values="score",
        aggfunc="mean",
    )
    available_variants = [variant for variant in variants if variant in pivot.columns]

    if len(available_variants) < 2:
        logger.warning(
            "Skipping family delta matrix for %s/%s: fewer than 2 variants with data",
            family,
            metric,
        )
        return

    matrix = np.zeros((len(available_variants), len(available_variants)), dtype=float)
    for i, variant_a in enumerate(available_variants):
        for j, variant_b in enumerate(available_variants):
            if i == j:
                continue
            paired = pivot[[variant_a, variant_b]].dropna()
            if paired.empty:
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = float((paired[variant_a] - paired[variant_b]).mean())

    vmax = np.nanmax(np.abs(matrix))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-8

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Delta (row - column)")

    labels = [prettify_model_name(item) for item in available_variants]
    ax.set_xticks(np.arange(len(available_variants)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(available_variants)), labels=labels)
    ax.set_title(f"{family}: {metric} pairwise mean deltas")

    for i in range(len(available_variants)):
        for j in range(len(available_variants)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center", fontsize=9)
            else:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    filename = f"family_{sanitize_metric(family)}_delta_matrix_{sanitize_metric(metric)}.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_family_variant_comparison_plots(
    variant_df: pd.DataFrame,
    families: list[str],
    metrics: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = (
        variant_df.groupby(
            [
                "corpus_name",
                "family_name",
                "num_topics",
                "metric_name",
                "variant_model_name",
            ],
            as_index=False,
        )["score"]
        .mean()
        .copy()
    )

    for family in families:
        family_variants = [
            f"{family}_boe",
            f"{family}_openai",
            f"{family}_sbert",
        ]
        family_frame = grouped[grouped["family_name"] == family].copy()

        if family_frame.empty:
            logger.warning("No data for family comparison: %s", family)
            continue

        available = sorted(family_frame["variant_model_name"].unique())
        missing = [item for item in family_variants if item not in available]
        if missing:
            logger.warning(
                "Family %s missing variants: %s",
                family,
                ", ".join(missing),
            )

        for metric in metrics:
            metric_frame = family_frame[family_frame["metric_name"] == metric].copy()
            if metric_frame.empty:
                continue

            create_family_distribution_plot(metric_frame, family, metric, family_variants, output_dir)
            create_family_delta_matrix_plot(metric_frame, family, metric, family_variants, output_dir)


def run() -> None:
    args = parse_args()

    requested_metrics = {m.strip().upper() for m in args.metrics.split(",") if m.strip()}
    query_metrics = normalize_query_metrics(requested_metrics)
    families = [family.strip() for family in args.families.split(",") if family.strip()]

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_from_env()
    with get_session(config.database) as session:
        boe_rows = fetch_boe_rows(session, query_metrics)
        std_rows = fetch_standard_rows(session, query_metrics)

    if not boe_rows or not std_rows:
        logger.warning(
            "Insufficient data: BOE rows=%d, Standard rows=%d",
            len(boe_rows),
            len(std_rows),
        )
        return

    all_df = rows_to_dataframe(boe_rows + std_rows)
    logger.info("Loaded total rows: %d", len(all_df))

    aligned = build_aligned_dataframe(all_df)
    logger.info("Aligned comparison rows (legacy): %d", len(aligned))

    if aligned.empty:
        logger.warning("No overlap found after alignment; no plots generated.")
        return

    available_metrics = [metric for metric in SUMMARY_METRICS if metric in set(all_df["metric_name"])]
    if not available_metrics:
        logger.warning("No metrics available after normalization; nothing to generate")
        return

    legacy_dir = output_dir / "legacy"
    run_legacy_plots(aligned, args.min_pairs, legacy_dir)

    variant_models = requested_variant_models(families)
    variant_df = all_df[all_df["variant_model_name"].isin(variant_models)].copy()
    if variant_df.empty:
        logger.warning("No rows available for requested families: %s", ", ".join(families))
        logger.info("Finished legacy outputs at %s", legacy_dir)
        return

    coverage = (
        variant_df.groupby(["family_name", "variant_model_name", "metric_name"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["family_name", "variant_model_name", "metric_name"])
    )
    logger.info("Variant coverage rows: %d", len(coverage))

    if args.generate_output_style_plots:
        create_variant_algorithm_heatmaps(
            variant_df,
            variant_models,
            available_metrics,
            output_dir / "algorithm_heatmaps",
        )
        create_boe_padding_split_algorithm_heatmaps(
            variant_df,
            variant_models,
            available_metrics,
            output_dir / "algorithm_heatmaps_boe_padding_split",
        )
        create_boe_padding_split_algorithm_heatmaps_by_target_chunk_count(
            variant_df,
            variant_models,
            available_metrics,
            output_dir / "algorithm_heatmaps_boe_padding_split_by_target_chunk_count",
        )
        create_variant_algorithm_heatmaps_by_target_chunk_count(
            variant_df,
            variant_models,
            available_metrics,
            output_dir / "algorithm_heatmaps_by_target_chunk_count",
        )
        create_variant_pareto_plots(
            variant_df,
            variant_models,
            available_metrics,
            output_dir / "pareto",
        )

    if args.generate_rank_table:
        create_variant_performance_summary_table(
            variant_df,
            variant_models,
            available_metrics,
            output_dir / "tables",
        )

    if args.generate_family_variant_comparisons:
        create_family_variant_comparison_plots(
            variant_df,
            families,
            available_metrics,
            output_dir / "family_comparisons",
        )

    logger.info("Finished. Plot and table files written to %s", output_dir)


if __name__ == "__main__":
    run()

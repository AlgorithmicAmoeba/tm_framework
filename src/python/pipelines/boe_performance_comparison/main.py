import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

from configuration import load_config_from_env
from database import get_session


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_METRICS = ("NPMI", "WEPS", "WECS", "ISH")
TOP_K_LABELS = 10


@dataclass(frozen=True)
class MetricRow:
    corpus_name: str
    model_name: str
    num_topics: int
    metric_name: str
    score: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BOE vs standard performance metrics with plots only."
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
    return parser.parse_args()


def metric_group_name(model_name: str) -> str:
    if model_name in {"BERTopic", "BERTopic_sbert"}:
        return "BERTopic"
    if model_name in {"ZeroShotTM", "ZeroShotTM_sbert"}:
        return "ZeroShotTM"
    return model_name


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
    return pd.DataFrame(
        [
            {
                "corpus_name": r.corpus_name,
                "model_name": r.model_name,
                "model_group": metric_group_name(r.model_name),
                "num_topics": r.num_topics,
                "metric_name": r.metric_name,
                "score": r.score,
                "source": r.source,
            }
            for r in rows
        ]
    )


def aggregate_side(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    subset = df[df["source"] == source_name].copy()
    grouped = (
        subset.groupby(
            ["corpus_name", "model_group", "num_topics", "metric_name"],
            as_index=False,
        )["score"]
        .agg(mean="mean", count="count")
        .rename(columns={"mean": f"{source_name.lower()}_mean", "count": f"n_{source_name.lower()}"})
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
        f"pairs={len(metric_df)} | median(BOE)={np.median(boe_vals):.4f} | median(Standard)={np.median(std_vals):.4f}",
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
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="black", linewidth=1.2)

    top = metric_df.reindex(metric_df["delta"].abs().sort_values(ascending=False).index).head(TOP_K_LABELS)
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

    fig, ax = plt.subplots(figsize=(max(8, heat.shape[1] * 1.2), max(5, heat.shape[0] * 0.7)))
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


def run() -> None:
    args = parse_args()
    metrics = {m.strip().upper() for m in args.metrics.split(",") if m.strip()}
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_from_env()
    with get_session(config.database) as session:
        boe_rows = fetch_boe_rows(session, metrics)
        std_rows = fetch_standard_rows(session, metrics)

    if not boe_rows or not std_rows:
        logger.warning("Insufficient data: BOE rows=%d, Standard rows=%d", len(boe_rows), len(std_rows))
        return

    all_df = rows_to_dataframe(boe_rows + std_rows)
    aligned = build_aligned_dataframe(all_df)
    logger.info("Aligned comparison rows: %d", len(aligned))

    if aligned.empty:
        logger.warning("No overlap found after alignment; no plots generated.")
        return

    for metric in sorted(aligned["metric_name"].unique()):
        metric_df = aligned[aligned["metric_name"] == metric].copy()
        if len(metric_df) < args.min_pairs:
            logger.warning(
                "Skipping metric %s: %d aligned pairs < min_pairs=%d",
                metric,
                len(metric_df),
                args.min_pairs,
            )
            continue

        logger.info("Generating plots for %s with %d aligned pairs", metric, len(metric_df))
        save_distribution_plot(metric_df, metric, output_dir)
        save_delta_scatter_plot(metric_df, metric, output_dir)
        save_delta_heatmap(metric_df, metric, output_dir)

    logger.info("Finished. Plot files written to %s", output_dir)


if __name__ == "__main__":
    run()

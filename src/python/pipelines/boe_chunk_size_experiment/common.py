"""
Shared configuration and helpers for the BOE chunk-size experiment.

The experiment varies ``target_chunk_count`` (the number of chunk slots each
document is padded/truncated to) for a fixed embedding slice and measures the
effect on downstream topic models. All stages write to dedicated
``pipeline.boe_cse_*`` tables (see schema.sql) so production is never touched.
"""
from typing import Any

import json

from sqlalchemy import text
from sqlalchemy.orm import Session

# --- Fixed embedding slice the experiment operates on -----------------------
SOURCE_MODEL_NAME = "all-MiniLM-L6-v2"
ALGORITHM = "none"          # unreduced embeddings
TARGET_DIMS = 0             # 0 == unreduced
PADDING_METHOD = "noise_only"

# Doc-embedding padding parameters (match the production boe_04 defaults).
KNN_K = 5
NOISE_SCALE = 0.01

# --- Chunk-size sweep -------------------------------------------------------
# Original sweep: only corpora where the chunk count meaningfully varies.
# BOE@1 extension (investigation 3 follow-up): chunk_count=1 added for ALL
# corpora — the degenerate single-chunk case approximates the existing
# (whole-document-embedding) models and lets us test whether the BOE-vs-existing
# complementarity survives without multi-chunk structure.
EXPERIMENT_CHUNK_COUNTS: dict[str, list[int]] = {
    "battery-abstracts": [1],
    "goodreads-bookgenres": [1],
    "imdb_reviews": [1, 2, 4],
    "newsgroups": [1],
    "patent-classification": [1, 2],
    "pubmed-multilabel": [1],
    "t2-ragbench-convfinqa": [1, 2, 4, 6],
    "trec_questions": [1],
    "twitter-financial-news": [1],
    "wikipedia_sample": [1, 3, 6, 9],
}

# --- Topic-model sweep ------------------------------------------------------
NUM_TOPICS_LIST = [10, 20, 50, 100, 200]
TARGET_RESULTS = 5          # number of result rows to accumulate per combo

BOE_TOPIC_MODELS = [
    "BERTopic",
    "ZeroShotTM",
    "CombinedTM",
    "KeyNMF",
    "SemanticSignalSeparation",
    "GMM",
]
MODELS_NEEDING_WORD_EMBEDDINGS = {"KeyNMF", "SemanticSignalSeparation"}

# --- Table names ------------------------------------------------------------
DOC_EMBEDDING_TABLE = "pipeline.boe_cse_document_embedding"
WORD_EMBEDDING_TABLE = "pipeline.boe_cse_word_embedding"
TOPIC_MODEL_TABLE = "pipeline.boe_cse_topic_model"
TOPIC_MODEL_RESULT_TABLE = "pipeline.boe_cse_topic_model_corpus_result"
TIMING_TABLE = "pipeline.boe_cse_timing_result"


def store_timing_result(
    session: Session,
    pipeline_stage: str,
    corpus_name: str,
    target_chunk_count: int,
    duration_seconds: float,
    model_name: str | None = None,
    num_topics: int | None = None,
    repeat_number: int = 1,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Insert one timing measurement into pipeline.boe_cse_timing_result."""
    query = text("""
        INSERT INTO pipeline.boe_cse_timing_result
        (pipeline_stage, corpus_name, model_name, num_topics, target_chunk_count,
         repeat_number, duration_seconds, metadata)
        VALUES (:pipeline_stage, :corpus_name, :model_name, :num_topics, :target_chunk_count,
                :repeat_number, :duration_seconds, :metadata)
    """)
    session.execute(query, {
        "pipeline_stage": pipeline_stage,
        "corpus_name": corpus_name,
        "model_name": model_name,
        "num_topics": num_topics,
        "target_chunk_count": target_chunk_count,
        "repeat_number": repeat_number,
        "duration_seconds": duration_seconds,
        "metadata": json.dumps(metadata or {}),
    })
    session.commit()

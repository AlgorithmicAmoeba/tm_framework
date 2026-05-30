"""
Timing pipeline for topic models and BOE embeddings.

Measures execution time for:
1. Standard topic models (LDA, BERTopic, etc.)
2. BOE topic models
3. BOE document embedding computation
4. BOE word embedding derivation

Usage:
    set -a && . .env && set +a
    uv run src/python/pipelines/timing/main.py
"""

import gc
import json
import logging
import time
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from configuration import load_config_from_env
from database import get_session
from pipelines.topic_models.data_handling import (
    get_tfidf_vectors,
    get_vocabulary,
    get_vocabulary_documents,
    get_chunk_embeddings,
    cleanup_model,
)
from pipelines.topic_models.LDA import LDA
from pipelines.topic_models.nmf import NMFModel
from pipelines.topic_models.BERTopic import BERTopicModel
from pipelines.topic_models.ZeroshotTM import AutoEncodingTopicModelWrapper
from pipelines.topic_models.KeyNMF import KeyNMFWrapper
from pipelines.topic_models.semantic_signal_separation import SemanticSignalSeparationWrapper
from pipelines.topic_models.gmm import GMMWrapper
from pipelines.boe_04_doc_embedding.main import BOEDocEmbeddingPipeline
from pipelines.boe_05_word_embedding.main import BOEWordEmbeddingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Configuration ────────────────────────────────────────────
CORPORA = ["battery-abstracts", "newsgroups", "wikipedia_sample"]
NUM_TOPICS_LIST = [50]
NUM_REPEATS = 1
DRY_RUN = False

# Standard topic models mapped to the embedding type they require
STANDARD_MODELS: dict[str, str] = {
    "LDA": "tfidf",
    "NMF": "tfidf",
    "BERTopic": "openai",
    "BERTopic_sbert": "sbert",
    "ZeroShotTM": "openai",
    "ZeroShotTM_sbert": "sbert",
    "CombinedTM": "openai",
    "CombinedTM_sbert": "sbert",
    "KeyNMF": "sbert",
    "SemanticSignalSeparation": "sbert",
    "GMM": "sbert",
}

BOE_TOPIC_MODELS = [
    "BERTopic", "ZeroShotTM", "CombinedTM",
    "KeyNMF", "SemanticSignalSeparation", "GMM",
]
MODELS_NEEDING_WORD_EMBEDDINGS = {"KeyNMF", "SemanticSignalSeparation"}

BOE_FILTERS = {
    "include": {
        "source_model_name": ["all-MiniLM-L6-v2"],
        "algorithm": [],
        "target_dims": [],
        "padding_method": ["noise_only"],
        "target_chunk_count": [],
    },
    "exclude": {
        "source_model_name": [],
        "algorithm": [],
        "target_dims": [],
        "padding_method": [],
        "target_chunk_count": [],
    },
}

BOE_CORPUS_CHUNK_COUNTS: dict[str, list[int]] = {
    "battery-abstracts": [2],
    "newsgroups": [2],
    "wikipedia_sample": [3, 6, 9],
}


# ── Helpers ──────────────────────────────────────────────────


def store_timing_result(
    session: Session,
    pipeline_stage: str,
    corpus_name: str,
    model_name: str | None,
    num_topics: int | None,
    repeat_number: int,
    duration_seconds: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Insert one timing measurement into the database."""
    query = text("""
        INSERT INTO pipeline.timing_result
        (pipeline_stage, corpus_name, model_name, num_topics,
         repeat_number, duration_seconds, metadata)
        VALUES (:pipeline_stage, :corpus_name, :model_name, :num_topics,
                :repeat_number, :duration_seconds, :metadata)
    """)
    session.execute(query, {
        "pipeline_stage": pipeline_stage,
        "corpus_name": corpus_name,
        "model_name": model_name,
        "num_topics": num_topics,
        "repeat_number": repeat_number,
        "duration_seconds": duration_seconds,
        "metadata": json.dumps(metadata or {}),
    })
    session.commit()


def count_existing_timing_results(
    session: Session,
    pipeline_stage: str,
    corpus_name: str,
    model_name: str | None,
    num_topics: int | None,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Count timing rows already recorded for this exact configuration."""
    conditions = [
        "pipeline_stage = :pipeline_stage",
        "corpus_name = :corpus_name",
    ]
    params: dict[str, Any] = {
        "pipeline_stage": pipeline_stage,
        "corpus_name": corpus_name,
    }
    if model_name is None:
        conditions.append("model_name IS NULL")
    else:
        conditions.append("model_name = :model_name")
        params["model_name"] = model_name
    if num_topics is None:
        conditions.append("num_topics IS NULL")
    else:
        conditions.append("num_topics = :num_topics")
        params["num_topics"] = num_topics
    for i, (key, value) in enumerate(sorted((metadata or {}).items())):
        param_key = f"meta_val_{i}"
        conditions.append(f"metadata ->> :meta_key_{i} = :{param_key}")
        params[f"meta_key_{i}"] = key
        params[param_key] = str(value)

    query = text(
        f"SELECT COUNT(*) FROM pipeline.timing_result WHERE {' AND '.join(conditions)}"
    )
    return session.execute(query, params).scalar() or 0


def build_standard_model(model_name: str, num_topics: int) -> Any:
    """Instantiate a standard topic model (no training yet)."""
    if model_name == "LDA":
        return LDA(num_topics=num_topics)
    if model_name == "NMF":
        return NMFModel(num_topics=num_topics)
    if model_name in ("BERTopic", "BERTopic_sbert"):
        return BERTopicModel(num_topics=num_topics)
    if model_name in ("ZeroShotTM", "ZeroShotTM_sbert"):
        return AutoEncodingTopicModelWrapper(num_topics=num_topics, combined=False)
    if model_name in ("CombinedTM", "CombinedTM_sbert"):
        return AutoEncodingTopicModelWrapper(num_topics=num_topics, combined=True)
    if model_name == "KeyNMF":
        return KeyNMFWrapper(num_topics=num_topics)
    if model_name == "SemanticSignalSeparation":
        return SemanticSignalSeparationWrapper(num_topics=num_topics)
    if model_name == "GMM":
        return GMMWrapper(num_topics=num_topics)
    raise ValueError(f"Unknown standard model: {model_name}")


def train_standard_model(
    model: Any,
    model_name: str,
    data: dict[str, Any],
) -> None:
    """Call the appropriate train method depending on model type."""
    if model_name in ("LDA", "NMF"):
        model.train(data["tfidf_vectors"], data["vocabulary"])
    else:
        model.train(data["documents"], data["embeddings"], data["vocabulary"])


def build_boe_model(model_name: str, num_topics: int) -> tuple[Any, dict[str, Any]]:
    """Instantiate a BOE topic model (mirrors boe_06 build_model)."""
    if model_name == "BERTopic":
        model = BERTopicModel(num_topics=num_topics)
        return model, {"n_reduce_to": num_topics}
    if model_name == "ZeroShotTM":
        model = AutoEncodingTopicModelWrapper(num_topics=num_topics, combined=False)
        return model, {"combined": False}
    if model_name == "CombinedTM":
        model = AutoEncodingTopicModelWrapper(num_topics=num_topics, combined=True)
        return model, {"combined": True}
    if model_name == "KeyNMF":
        model = KeyNMFWrapper(num_topics=num_topics)
        return model, {"seed_phrase": model.seed_phrase}
    if model_name == "SemanticSignalSeparation":
        model = SemanticSignalSeparationWrapper(num_topics=num_topics)
        return model, {"feature_importance": model.feature_importance}
    if model_name == "GMM":
        model = GMMWrapper(num_topics=num_topics)
        return model, {}
    raise ValueError(f"Unknown BOE model: {model_name}")


def filter_boe_combos(combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply BOE_FILTERS to embedding combinations."""
    def allowed(value: Any, include_list: list, exclude_list: list) -> bool:
        if include_list and value not in include_list:
            return False
        if value in exclude_list:
            return False
        return True

    filtered = []
    for combo in combos:
        ok = True
        for key in ["source_model_name", "algorithm", "target_dims",
                     "padding_method", "target_chunk_count"]:
            if not allowed(combo[key],
                           BOE_FILTERS["include"][key],
                           BOE_FILTERS["exclude"][key]):
                ok = False
                break
        if ok:
            filtered.append(combo)
    return filtered


# ── Standard topic models ────────────────────────────────────


def time_standard_topic_models(
    corpora: list[str],
    num_topics_list: list[int],
    num_repeats: int,
    config: Any,
) -> None:
    """Time all standard topic models on the given corpora."""
    logging.info("=" * 60)
    logging.info("TIMING STANDARD TOPIC MODELS")
    logging.info("=" * 60)

    for corpus_name in corpora:
        logging.info("Loading data for corpus: %s", corpus_name)

        vocabulary = get_vocabulary(corpus_name)
        if not vocabulary:
            logging.warning("No vocabulary for %s, skipping", corpus_name)
            continue

        vocabulary_docs = get_vocabulary_documents(corpus_name)
        documents = [content for _, content in vocabulary_docs]

        # Pre-load data by embedding type so we only hit the DB once each
        data_by_type: dict[str, dict[str, Any]] = {}

        try:
            _, tfidf_vectors = get_tfidf_vectors(corpus_name)
            if len(tfidf_vectors) > 0:
                data_by_type["tfidf"] = {
                    "tfidf_vectors": tfidf_vectors,
                    "vocabulary": vocabulary,
                }
        except Exception as exc:
            logging.warning("Could not load TF-IDF for %s: %s", corpus_name, exc)

        for emb_type in ("openai", "sbert"):
            try:
                _, embeddings = get_chunk_embeddings(corpus_name, emb_type)
                if len(embeddings) > 0:
                    data_by_type[emb_type] = {
                        "documents": documents,
                        "embeddings": embeddings,
                        "vocabulary": vocabulary,
                    }
            except Exception as exc:
                logging.warning("Could not load %s embeddings for %s: %s",
                                emb_type, corpus_name, exc)

        for model_name, emb_type in STANDARD_MODELS.items():
            if emb_type not in data_by_type:
                logging.warning("No %s data for %s, skipping %s",
                                emb_type, corpus_name, model_name)
                continue

            data = data_by_type[emb_type]

            for num_topics in num_topics_list:
                lookup_metadata = {"embedding_type": emb_type}
                with get_session(config.database) as session:
                    existing = count_existing_timing_results(
                        session,
                        pipeline_stage="standard_topic_model",
                        corpus_name=corpus_name,
                        model_name=model_name,
                        num_topics=num_topics,
                        metadata=lookup_metadata,
                    )
                if existing >= num_repeats:
                    logging.info(
                        "Skipping %s on %s (topics=%d): already have %d/%d repeats",
                        model_name, corpus_name, num_topics, existing, num_repeats,
                    )
                    continue

                for repeat in range(existing + 1, num_repeats + 1):
                    logging.info(
                        "Timing %s on %s, topics=%d, repeat=%d",
                        model_name, corpus_name, num_topics, repeat,
                    )

                    if DRY_RUN:
                        logging.info(
                            "[DRY RUN] Would time %s on %s, topics=%d, repeat=%d",
                            model_name, corpus_name, num_topics, repeat,
                        )
                        continue

                    model = None
                    try:
                        model = build_standard_model(model_name, num_topics)

                        start = time.perf_counter()
                        train_standard_model(model, model_name, data)
                        model.get_topics()
                        duration = time.perf_counter() - start

                        logging.info("  Duration: %.3fs", duration)

                        with get_session(config.database) as session:
                            store_timing_result(
                                session=session,
                                pipeline_stage="standard_topic_model",
                                corpus_name=corpus_name,
                                model_name=model_name,
                                num_topics=num_topics,
                                repeat_number=repeat,
                                duration_seconds=duration,
                                metadata={"embedding_type": emb_type},
                            )
                    except Exception:
                        logging.exception(
                            "Error timing %s on %s (topics=%d)",
                            model_name, corpus_name, num_topics,
                        )
                    finally:
                        if model is not None:
                            cleanup_model(model)
                        gc.collect()

        del data_by_type
        gc.collect()


# ── BOE topic models ─────────────────────────────────────────


def time_boe_topic_models(
    corpora: list[str],
    num_topics_list: list[int],
    num_repeats: int,
    config: Any,
) -> None:
    """Time BOE topic models on the given corpora."""
    from pipelines.boe_06_topic_models.experiment_runner import (
        get_embedding_combos,
        fetch_document_embeddings,
        fetch_boe_word_embeddings,
        align_documents_and_embeddings,
    )

    logging.info("=" * 60)
    logging.info("TIMING BOE TOPIC MODELS")
    logging.info("=" * 60)

    for corpus_name in corpora:
        with get_session(config.database) as session:
            combos = filter_boe_combos(get_embedding_combos(session, corpus_name))

        if not combos:
            logging.info("No BOE embedding combos for %s after filtering", corpus_name)
            continue

        vocabulary = get_vocabulary(corpus_name)
        if not vocabulary:
            logging.warning("No vocabulary for %s, skipping BOE models", corpus_name)
            continue

        for combo in combos:
            with get_session(config.database) as session:
                embedding_map, _ = fetch_document_embeddings(
                    session, corpus_name, combo,
                )
                word_embeddings = fetch_boe_word_embeddings(
                    session, corpus_name, combo,
                )

            documents, embeddings = align_documents_and_embeddings(
                corpus_name, embedding_map,
            )
            if len(documents) == 0:
                logging.warning("No aligned embeddings for %s combo %s", corpus_name, combo)
                continue

            for model_name in BOE_TOPIC_MODELS:
                for num_topics in num_topics_list:
                    if model_name == "GMM" and num_topics > embeddings.shape[1]:
                        logging.info(
                            "Skipping GMM: num_topics (%d) > embedding dim (%d)",
                            num_topics, embeddings.shape[1],
                        )
                        continue

                    lookup_metadata = {
                        "source_model_name": combo["source_model_name"],
                        "algorithm": combo["algorithm"],
                        "target_dims": combo["target_dims"],
                        "padding_method": combo["padding_method"],
                        "target_chunk_count": combo["target_chunk_count"],
                    }
                    with get_session(config.database) as session:
                        existing = count_existing_timing_results(
                            session,
                            pipeline_stage="boe_topic_model",
                            corpus_name=corpus_name,
                            model_name=model_name,
                            num_topics=num_topics,
                            metadata=lookup_metadata,
                        )
                    if existing >= num_repeats:
                        logging.info(
                            "Skipping BOE %s on %s (topics=%d): already have %d/%d repeats",
                            model_name, corpus_name, num_topics, existing, num_repeats,
                        )
                        continue

                    for repeat in range(existing + 1, num_repeats + 1):
                        logging.info(
                            "Timing BOE %s on %s (%s/%s/%d), topics=%d, repeat=%d",
                            model_name, corpus_name,
                            combo["algorithm"], combo["target_dims"],
                            combo["target_chunk_count"],
                            num_topics, repeat,
                        )

                        if DRY_RUN:
                            logging.info(
                                "[DRY RUN] Would time %s on %s, topics=%d, repeat=%d",
                                model_name, corpus_name, num_topics, repeat,
                            )
                            continue

                        model = None
                        try:
                            model, _ = build_boe_model(model_name, num_topics)

                            start = time.perf_counter()
                            if model_name in MODELS_NEEDING_WORD_EMBEDDINGS:
                                model.train(
                                    documents, embeddings, vocabulary,
                                    word_embeddings=word_embeddings,
                                )
                            else:
                                model.train(documents, embeddings, vocabulary)
                            model.get_topics()
                            duration = time.perf_counter() - start

                            logging.info("  Duration: %.3fs", duration)

                            with get_session(config.database) as session:
                                store_timing_result(
                                    session=session,
                                    pipeline_stage="boe_topic_model",
                                    corpus_name=corpus_name,
                                    model_name=model_name,
                                    num_topics=num_topics,
                                    repeat_number=repeat,
                                    duration_seconds=duration,
                                    metadata={
                                        "source_model_name": combo["source_model_name"],
                                        "algorithm": combo["algorithm"],
                                        "target_dims": combo["target_dims"],
                                        "padding_method": combo["padding_method"],
                                        "target_chunk_count": combo["target_chunk_count"],
                                    },
                                )
                        except Exception:
                            logging.exception(
                                "Error timing BOE %s on %s (topics=%d)",
                                model_name, corpus_name, num_topics,
                            )
                        finally:
                            if model is not None:
                                cleanup_model(model)
                            gc.collect()

            del embedding_map, word_embeddings, documents, embeddings
            gc.collect()


# ── BOE document embedding ───────────────────────────────────


def time_boe_doc_embedding(
    corpora: list[str],
    num_repeats: int,
    config: Any,
) -> None:
    """Time BOE document embedding computation."""
    logging.info("=" * 60)
    logging.info("TIMING BOE DOCUMENT EMBEDDING")
    logging.info("=" * 60)

    for corpus_name in corpora:
        chunk_counts = BOE_CORPUS_CHUNK_COUNTS.get(corpus_name)
        if not chunk_counts:
            logging.warning("No chunk counts configured for %s, skipping", corpus_name)
            continue

        # Discover available source models for unreduced embeddings
        # (matching BOE_FILTERS: algorithm=none, target_dims=0)
        with get_session(config.database) as session:
            probe = BOEDocEmbeddingPipeline(
                target_chunk_count=chunk_counts[0], padding_method="noise_only",
            )
            unreduced_models = probe.get_available_unreduced_models(session, corpus_name)

        source_models = [
            m for m in unreduced_models
            if not BOE_FILTERS["include"]["source_model_name"]
            or m in BOE_FILTERS["include"]["source_model_name"]
        ]

        for source_model_name in source_models:
            # Fetch chunk embeddings once (outside timing loop)
            with get_session(config.database) as session:
                fetcher = BOEDocEmbeddingPipeline(
                    target_chunk_count=chunk_counts[0], padding_method="noise_only",
                )
                chunk_hashes, raw_doc_hashes, chunk_starts, chunk_embeddings = \
                    fetcher.fetch_unreduced_chunk_embeddings(
                        session, corpus_name, source_model_name,
                    )

            if len(chunk_hashes) == 0:
                logging.warning("No chunks for %s / %s", corpus_name, source_model_name)
                continue

            for target_chunk_count in chunk_counts:
                lookup_metadata = {
                    "algorithm": "none",
                    "target_dims": 0,
                    "padding_method": "noise_only",
                    "target_chunk_count": target_chunk_count,
                }
                with get_session(config.database) as session:
                    existing = count_existing_timing_results(
                        session,
                        pipeline_stage="boe_doc_embedding",
                        corpus_name=corpus_name,
                        model_name=source_model_name,
                        num_topics=None,
                        metadata=lookup_metadata,
                    )
                if existing >= num_repeats:
                    logging.info(
                        "Skipping BOE doc embedding %s (%s, chunks=%d): already have %d/%d repeats",
                        corpus_name, source_model_name, target_chunk_count,
                        existing, num_repeats,
                    )
                    continue

                pipeline = BOEDocEmbeddingPipeline(
                    target_chunk_count=target_chunk_count,
                    padding_method="noise_only",
                )

                doc_groups = pipeline.group_chunks_by_document(
                    chunk_hashes, raw_doc_hashes, chunk_starts, chunk_embeddings,
                )

                for repeat in range(existing + 1, num_repeats + 1):
                    logging.info(
                        "Timing BOE doc embedding %s (%s, chunks=%d), repeat=%d",
                        corpus_name, source_model_name,
                        target_chunk_count, repeat,
                    )

                    if DRY_RUN:
                        logging.info(
                            "[DRY RUN] Would time %s on %s, topics=None, repeat=%d",
                            source_model_name, corpus_name, repeat,
                        )
                        continue

                    start = time.perf_counter()
                    pipeline.compute_document_embeddings(doc_groups, chunk_embeddings)
                    duration = time.perf_counter() - start

                    logging.info("  Duration: %.3fs", duration)

                    with get_session(config.database) as session:
                        store_timing_result(
                            session=session,
                            pipeline_stage="boe_doc_embedding",
                            corpus_name=corpus_name,
                            model_name=source_model_name,
                            num_topics=None,
                            repeat_number=repeat,
                            duration_seconds=duration,
                            metadata={
                                **lookup_metadata,
                                "num_documents": len(doc_groups),
                                "num_chunks": len(chunk_hashes),
                            },
                        )


# ── BOE word embedding ───────────────────────────────────────


def time_boe_word_embedding(
    corpora: list[str],
    num_repeats: int,
    config: Any,
) -> None:
    """Time BOE word embedding derivation."""
    logging.info("=" * 60)
    logging.info("TIMING BOE WORD EMBEDDING")
    logging.info("=" * 60)

    word_pipeline = BOEWordEmbeddingPipeline()

    for corpus_name in corpora:
        with get_session(config.database) as session:
            reductions = word_pipeline.get_available_reductions(session, corpus_name)

        for red in reductions:
            smn = red["source_model_name"]
            alg = red["algorithm"]
            td = red["target_dims"]
            pm = red["padding_method"]
            tcc = red["target_chunk_count"]

            # Apply BOE_FILTERS
            inc = BOE_FILTERS["include"]
            if inc["source_model_name"] and smn not in inc["source_model_name"]:
                continue
            if inc["algorithm"] and alg not in inc["algorithm"]:
                continue
            if inc["target_dims"] and td not in inc["target_dims"]:
                continue
            if inc["padding_method"] and pm not in inc["padding_method"]:
                continue
            if inc["target_chunk_count"] and tcc not in inc["target_chunk_count"]:
                continue

            lookup_metadata = {
                "algorithm": alg,
                "target_dims": td,
                "padding_method": pm,
                "target_chunk_count": tcc,
            }
            with get_session(config.database) as session:
                existing = count_existing_timing_results(
                    session,
                    pipeline_stage="boe_word_embedding",
                    corpus_name=corpus_name,
                    model_name=smn,
                    num_topics=None,
                    metadata=lookup_metadata,
                )
            if existing >= num_repeats:
                logging.info(
                    "Skipping BOE word embedding %s (%s/%s/%d/%s/chunks=%d): "
                    "already have %d/%d repeats",
                    corpus_name, smn, alg, td, pm, tcc, existing, num_repeats,
                )
                continue

            # Fetch data once (outside timing loop)
            with get_session(config.database) as session:
                doc_hashes, vocab_docs, doc_embeddings = \
                    word_pipeline.fetch_document_data(
                        session, corpus_name, smn, alg, td, pm, tcc,
                    )

            if len(doc_hashes) == 0:
                logging.warning("No documents for %s %s/%s/%d/%s/chunks=%d",
                                corpus_name, smn, alg, td, pm, tcc)
                continue

            for repeat in range(existing + 1, num_repeats + 1):
                logging.info(
                    "Timing BOE word embedding %s (%s/%s/%d/%s/chunks=%d), repeat=%d",
                    corpus_name, smn, alg, td, pm, tcc, repeat,
                )

                if DRY_RUN:
                    logging.info(
                        "[DRY RUN] Would time %s on %s, topics=None, repeat=%d",
                        smn, corpus_name, repeat,
                    )
                    continue

                start = time.perf_counter()
                word_pipeline.derive_word_embeddings(vocab_docs, doc_embeddings)
                duration = time.perf_counter() - start

                logging.info("  Duration: %.3fs", duration)

                with get_session(config.database) as session:
                    store_timing_result(
                        session=session,
                        pipeline_stage="boe_word_embedding",
                        corpus_name=corpus_name,
                        model_name=smn,
                        num_topics=None,
                        repeat_number=repeat,
                        duration_seconds=duration,
                        metadata={
                            **lookup_metadata,
                            "num_documents": len(doc_hashes),
                        },
                    )


# ── Entry point ──────────────────────────────────────────────


def main() -> None:
    config = load_config_from_env()

    logging.info("Timing Pipeline")
    logging.info("=" * 60)
    logging.info("DRY_RUN: %s", DRY_RUN)
    logging.info("Corpora: %s", CORPORA)
    logging.info("Num topics: %s", NUM_TOPICS_LIST)
    logging.info("Num repeats: %d", NUM_REPEATS)
    logging.info("=" * 60)

    time_standard_topic_models(CORPORA, NUM_TOPICS_LIST, NUM_REPEATS, config)
    time_boe_topic_models(CORPORA, NUM_TOPICS_LIST, NUM_REPEATS, config)
    time_boe_doc_embedding(CORPORA, NUM_REPEATS, config)
    time_boe_word_embedding(CORPORA, NUM_REPEATS, config)

    logging.info("=" * 60)
    logging.info("TIMING COMPLETE")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()

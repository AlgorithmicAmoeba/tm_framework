import json
import logging
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm

from configuration import load_config_from_env
from database import get_session
from pipelines.topic_models.BERTopic import BERTopicModel
from pipelines.topic_models.ZeroshotTM import AutoEncodingTopicModelWrapper
from pipelines.topic_models.KeyNMF import KeyNMFWrapper
from pipelines.topic_models.semantic_signal_separation import SemanticSignalSeparationWrapper
from pipelines.topic_models.gmm import GMMWrapper
from pipelines.topic_models.data_handling import get_vocabulary_documents, get_vocabulary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


TARGET_RESULTS = 3
NUM_TOPICS_LIST = [10, 20, 50, 100, 200]

FILTERS = {
    "include": {
        "source_model_name": [],
        "algorithm": [],
        "target_dims": [],
        "padding_method": [],
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

BOE_TOPIC_MODELS = [
    "BERTopic",
    "ZeroShotTM",
    "CombinedTM",
    "KeyNMF",
    "SemanticSignalSeparation",
    "GMM",
]


def get_available_corpora(session: Session) -> list[str]:
    query = text("""
        SELECT DISTINCT corpus_name
        FROM pipeline.boe_document_embedding
        ORDER BY corpus_name
    """)
    return [row[0] for row in session.execute(query)]


def get_embedding_combos(session: Session, corpus_name: str) -> list[dict[str, Any]]:
    query = text("""
        SELECT DISTINCT source_model_name, algorithm, target_dims, padding_method, target_chunk_count
        FROM pipeline.boe_document_embedding
        WHERE corpus_name = :corpus_name
        ORDER BY source_model_name, algorithm, target_dims, padding_method, target_chunk_count
    """)
    result = session.execute(query, {"corpus_name": corpus_name})
    return [
        {
            "source_model_name": row[0],
            "algorithm": row[1],
            "target_dims": row[2],
            "padding_method": row[3],
            "target_chunk_count": row[4],
        }
        for row in result
    ]


def filter_combos(combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def allowed(value: Any, include_list: list[Any], exclude_list: list[Any]) -> bool:
        if include_list and value not in include_list:
            return False
        if value in exclude_list:
            return False
        return True

    filtered = []
    for combo in combos:
        ok = True
        for key in [
            "source_model_name",
            "algorithm",
            "target_dims",
            "padding_method",
            "target_chunk_count",
        ]:
            if not allowed(combo[key], FILTERS["include"][key], FILTERS["exclude"][key]):
                ok = False
                break
        if ok:
            filtered.append(combo)
    return filtered


def get_boe_topic_model_id(session: Session, model_name: str) -> int:
    query = text("""
        SELECT id FROM pipeline.boe_topic_model
        WHERE name = :model_name
    """)
    model_id = session.execute(query, {"model_name": model_name}).scalar()
    if not model_id:
        raise ValueError(f"BOE topic model not found in database: {model_name}")
    return model_id


def count_existing_results(
    session: Session,
    corpus_name: str,
    topic_model_id: int,
    num_topics: int,
    combo: dict[str, Any],
) -> int:
    query = text("""
        SELECT COUNT(*)
        FROM pipeline.boe_topic_model_corpus_result
        WHERE corpus_name = :corpus_name
        AND topic_model_id = :topic_model_id
        AND num_topics = :num_topics
        AND source_model_name = :source_model_name
        AND algorithm = :algorithm
        AND target_dims = :target_dims
        AND padding_method = :padding_method
        AND target_chunk_count = :target_chunk_count
        AND soft_delete = FALSE
    """)
    return session.execute(query, {
        "corpus_name": corpus_name,
        "topic_model_id": topic_model_id,
        "num_topics": num_topics,
        "source_model_name": combo["source_model_name"],
        "algorithm": combo["algorithm"],
        "target_dims": combo["target_dims"],
        "padding_method": combo["padding_method"],
        "target_chunk_count": combo["target_chunk_count"],
    }).scalar() or 0


def fetch_document_embeddings(
    session: Session,
    corpus_name: str,
    combo: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[int]]:
    query = text("""
        SELECT raw_document_hash, vector, padded_to
        FROM pipeline.boe_document_embedding
        WHERE corpus_name = :corpus_name
        AND source_model_name = :source_model_name
        AND algorithm = :algorithm
        AND target_dims = :target_dims
        AND padding_method = :padding_method
        AND target_chunk_count = :target_chunk_count
        ORDER BY raw_document_hash
    """)
    result = session.execute(query, {
        "corpus_name": corpus_name,
        "source_model_name": combo["source_model_name"],
        "algorithm": combo["algorithm"],
        "target_dims": combo["target_dims"],
        "padding_method": combo["padding_method"],
        "target_chunk_count": combo["target_chunk_count"],
    })

    embedding_map: dict[str, np.ndarray] = {}
    padded_to_values: set[int] = set()

    for row in result:
        embedding_map[row[0]] = np.array(row[1], dtype=np.float32)
        padded_to_values.add(int(row[2]))

    return embedding_map, sorted(padded_to_values)


def align_documents_and_embeddings(
    corpus_name: str,
    embedding_map: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    vocab_docs = get_vocabulary_documents(corpus_name)
    if not vocab_docs:
        return [], np.array([])

    documents: list[str] = []
    embeddings: list[np.ndarray] = []
    missing = 0

    for doc_hash, content in vocab_docs:
        vector = embedding_map.get(doc_hash)
        if vector is None:
            missing += 1
            continue
        documents.append(content)
        embeddings.append(vector)

    if missing:
        logging.warning("Missing embeddings for %s documents in corpus %s", missing, corpus_name)

    if not embeddings:
        return [], np.array([])

    return documents, np.vstack(embeddings)


def build_model(model_name: str, num_topics: int) -> tuple[Any, dict[str, Any]]:
    if model_name == "BERTopic":
        model = BERTopicModel(num_topics=num_topics)
        hparams = {"n_reduce_to": num_topics}
        return model, hparams

    if model_name == "ZeroShotTM":
        model = AutoEncodingTopicModelWrapper(num_topics=num_topics, combined=False)
        hparams = {"combined": False, "batch_size": model._batch_size}
        return model, hparams

    if model_name == "CombinedTM":
        model = AutoEncodingTopicModelWrapper(num_topics=num_topics, combined=True)
        hparams = {"combined": True, "batch_size": model._batch_size}
        return model, hparams

    if model_name == "KeyNMF":
        model = KeyNMFWrapper(num_topics=num_topics)
        hparams = {"seed_phrase": model.seed_phrase}
        return model, hparams

    if model_name == "SemanticSignalSeparation":
        model = SemanticSignalSeparationWrapper(num_topics=num_topics)
        hparams = {"feature_importance": model.feature_importance}
        return model, hparams

    if model_name == "GMM":
        model = GMMWrapper(num_topics=num_topics)
        hparams = {
            "weight_prior": model.weight_prior,
            "gamma": model.gamma,
            "use_dimensionality_reduction": model.use_dimensionality_reduction,
        }
        return model, hparams

    raise ValueError(f"Unknown BOE topic model: {model_name}")


def run_experiments(target_results: int = TARGET_RESULTS) -> None:
    config = load_config_from_env()

    with get_session(config.database) as session:
        corpora = get_available_corpora(session)

    logging.info("Corpora: %s", corpora)
    logging.info("BOE topic models: %s", BOE_TOPIC_MODELS)

    total_steps = 0
    with get_session(config.database) as session:
        for corpus_name in corpora:
            combos = filter_combos(get_embedding_combos(session, corpus_name))
            total_steps += len(combos) * len(BOE_TOPIC_MODELS) * len(NUM_TOPICS_LIST)

    pbar = tqdm(total=total_steps, desc="Running BOE experiments")

    for corpus_name in corpora:
        with get_session(config.database) as session:
            combos = filter_combos(get_embedding_combos(session, corpus_name))

        if not combos:
            logging.info("No embedding combos after filtering for corpus: %s", corpus_name)
            continue

        vocabulary = get_vocabulary(corpus_name)
        if not vocabulary:
            logging.warning("No vocabulary found for corpus: %s", corpus_name)
            pbar.update(len(combos) * len(BOE_TOPIC_MODELS) * len(NUM_TOPICS_LIST))
            continue

        for combo in combos:
            with get_session(config.database) as session:
                embedding_map, padded_to_values = fetch_document_embeddings(session, corpus_name, combo)

            documents, embeddings = align_documents_and_embeddings(corpus_name, embedding_map)
            if len(documents) == 0:
                logging.warning("No aligned embeddings for corpus %s with combo %s", corpus_name, combo)
                pbar.update(len(BOE_TOPIC_MODELS) * len(NUM_TOPICS_LIST))
                continue

            for num_topics in NUM_TOPICS_LIST:
                for model_name in BOE_TOPIC_MODELS:
                    with get_session(config.database) as session:
                        model_id = get_boe_topic_model_id(session, model_name)
                        existing = count_existing_results(
                            session,
                            corpus_name,
                            model_id,
                            num_topics,
                            combo,
                        )

                    iterations_to_run = target_results - existing
                    if iterations_to_run <= 0:
                        pbar.update(1)
                        continue

                    pbar.set_description(
                        f"{model_name} on {corpus_name} ({combo['algorithm']}/{combo['target_dims']}), "
                        f"topics={num_topics}, iters={iterations_to_run}"
                    )

                    for _ in range(iterations_to_run):
                        try:
                            model, model_hparams = build_model(model_name, num_topics)
                            model.train(documents, embeddings, vocabulary)
                            topics = model.get_topics()

                            hyperparameters = {
                                "boe_embedding": {
                                    "source_model_name": combo["source_model_name"],
                                    "algorithm": combo["algorithm"],
                                    "target_dims": combo["target_dims"],
                                    "padding_method": combo["padding_method"],
                                    "target_chunk_count": combo["target_chunk_count"],
                                    "padded_to": padded_to_values,
                                },
                                "model": model_hparams,
                            }

                            with get_session(config.database) as session:
                                insert_query = text("""
                                    INSERT INTO pipeline.boe_topic_model_corpus_result
                                    (topic_model_id, corpus_name, source_model_name, algorithm,
                                     target_dims, padding_method, target_chunk_count, topics,
                                     num_topics, hyperparameters)
                                    VALUES (:topic_model_id, :corpus_name, :source_model_name, :algorithm,
                                            :target_dims, :padding_method, :target_chunk_count, :topics,
                                            :num_topics, :hyperparameters)
                                """)

                                session.execute(insert_query, {
                                    "topic_model_id": model_id,
                                    "corpus_name": corpus_name,
                                    "source_model_name": combo["source_model_name"],
                                    "algorithm": combo["algorithm"],
                                    "target_dims": combo["target_dims"],
                                    "padding_method": combo["padding_method"],
                                    "target_chunk_count": combo["target_chunk_count"],
                                    "topics": json.dumps(topics),
                                    "num_topics": num_topics,
                                    "hyperparameters": json.dumps(hyperparameters),
                                })
                                session.commit()
                        except Exception:
                            logging.exception(
                                "Error running %s on %s with combo %s",
                                model_name,
                                corpus_name,
                                combo,
                            )

                    pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    run_experiments()

"""
BOE (Bag of Embeddings) Document Embedding pipeline.
Aggregates chunk embeddings into document embeddings by padding/truncating
to a fixed chunk count and then stacking chunk vectors. Supports multiple
padding methods tracked in the database.
"""
import logging
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg


class BOEDocEmbeddingPipeline:
    """
    Pipeline for aggregating chunk embeddings into document embeddings.

    Uses content-similarity KNN padding to fixed length with inverse-distance weighting:
    1. Group chunks by raw_document_hash (ordered by chunk position)
    2. For documents with fewer chunks than target, pad using KNN-weighted mean
    3. Stack all chunk embeddings (original + padding) -> document embedding
    """

    def __init__(
        self,
        target_chunk_count: int = 3,
        knn_k: int = 5,
        noise_scale: float = 0.01,
        padding_method: str = "knn_mean"
    ):
        """
        Initialize the document embedding pipeline.

        Args:
            target_chunk_count: Fixed number of chunks to pad to
            knn_k: Number of neighbors for weighted mean calculation
            noise_scale: Multiplier for corpus std dev (for Gaussian noise)
            padding_method: "knn_mean" or "noise_only"
        """
        if padding_method not in {"knn_mean", "noise_only"}:
            raise ValueError(f"Unknown padding_method: {padding_method}")
        self.target_chunk_count = target_chunk_count
        self.knn_k = knn_k
        self.noise_scale = noise_scale
        self.padding_method = padding_method
        logging.info(
            f"Initialized BOE document embedding pipeline "
            f"(target_chunks={target_chunk_count}, knn_k={knn_k}, "
            f"noise_scale={noise_scale}, padding_method={padding_method})"
        )

    def get_available_reductions(
        self,
        session: Session,
        corpus_name: str
    ) -> list[dict[str, Any]]:
        """
        Get available reduced embeddings for a corpus.

        Args:
            session: Database session
            corpus_name: Name of the corpus

        Returns:
            List of dicts with source_model_name, algorithm, target_dims combinations
        """
        query = text("""
            SELECT DISTINCT source_model_name, algorithm, target_dims
            FROM pipeline.boe_embedding_reduced
            WHERE corpus_name = :corpus_name
            ORDER BY source_model_name, algorithm, target_dims
        """)
        result = session.execute(query, {"corpus_name": corpus_name})
        return [
            {
                "source_model_name": row[0],
                "algorithm": row[1],
                "target_dims": row[2]
            }
            for row in result
        ]

    def count_expected_documents(
        self,
        session: Session,
        corpus_name: str,
        source_model_name: str,
        algorithm: str,
        target_dims: int
    ) -> int:
        """
        Count expected documents for a given reduction based on available reduced embeddings.

        This mirrors the fetch query to avoid counting documents that have no reduced chunks.
        """
        query = text("""
            SELECT COUNT(DISTINCT c.raw_document_hash)
            FROM pipeline.boe_chunked_document c
            JOIN pipeline.boe_embedding_reduced r
                ON c.chunk_hash = r.boe_chunked_document_hash
            WHERE c.corpus_name = :corpus_name
            AND r.source_model_name = :source_model_name
            AND r.algorithm = :algorithm
            AND r.target_dims = :target_dims
        """)
        result = session.execute(query, {
            "corpus_name": corpus_name,
            "source_model_name": source_model_name,
            "algorithm": algorithm,
            "target_dims": target_dims
        })
        return int(result.scalar() or 0)

    def count_existing_documents(
        self,
        session: Session,
        corpus_name: str,
        source_model_name: str,
        algorithm: str,
        target_dims: int,
        padding_method: str,
        target_chunk_count: int
    ) -> int:
        """
        Count existing document embeddings for a given combination.

        Args:
            session: Database session
            corpus_name: Corpus name
            source_model_name: Source embedding model name
            algorithm: Reduction algorithm
            target_dims: Target dimensions

        Returns:
            Number of document embeddings stored
        """
        query = text("""
            SELECT COUNT(*) FROM pipeline.boe_document_embedding
            WHERE corpus_name = :corpus_name
            AND source_model_name = :source_model_name
            AND algorithm = :algorithm
            AND target_dims = :target_dims
            AND padding_method = :padding_method
            AND target_chunk_count = :target_chunk_count
            LIMIT 1
        """)
        result = session.execute(query, {
            "corpus_name": corpus_name,
            "source_model_name": source_model_name,
            "algorithm": algorithm,
            "target_dims": target_dims,
            "padding_method": padding_method,
            "target_chunk_count": target_chunk_count
        })
        return int(result.scalar() or 0)

    def fetch_chunk_embeddings(
        self,
        session: Session,
        corpus_name: str,
        source_model_name: str,
        algorithm: str,
        target_dims: int
    ) -> tuple[list[str], list[str], list[int], np.ndarray]:
        """
        Fetch chunk embeddings grouped by document.

        Args:
            session: Database session
            corpus_name: Name of the corpus
            source_model_name: Source embedding model name
            algorithm: Reduction algorithm
            target_dims: Target dimensions

        Returns:
            Tuple of (chunk_hashes, raw_document_hashes, chunk_starts, embeddings_matrix)
            ordered by raw_document_hash and chunk_start for proper chunk ordering
        """
        logging.info(f"Fetching chunk embeddings for {source_model_name}/{algorithm}/{target_dims}")

        query = text("""
            SELECT c.chunk_hash, c.raw_document_hash, c.chunk_start, r.vector
            FROM pipeline.boe_chunked_document c
            JOIN pipeline.boe_embedding_reduced r
                ON c.chunk_hash = r.boe_chunked_document_hash
            WHERE c.corpus_name = :corpus_name
            AND r.source_model_name = :source_model_name
            AND r.algorithm = :algorithm
            AND r.target_dims = :target_dims
            ORDER BY c.raw_document_hash, c.chunk_start
        """)

        result = session.execute(query, {
            "corpus_name": corpus_name,
            "source_model_name": source_model_name,
            "algorithm": algorithm,
            "target_dims": target_dims
        })

        chunk_hashes = []
        raw_document_hashes = []
        chunk_starts = []
        vectors = []

        for row in result:
            chunk_hashes.append(row[0])
            raw_document_hashes.append(row[1])
            chunk_starts.append(row[2])
            vectors.append(row[3])

        if not vectors:
            logging.warning(f"No chunk embeddings found for {source_model_name}/{algorithm}/{target_dims}")
            return [], [], [], np.array([])

        embeddings_matrix = np.array(vectors, dtype=np.float32)
        logging.info(f"Fetched {len(chunk_hashes)} chunks with embeddings shape {embeddings_matrix.shape}")

        return chunk_hashes, raw_document_hashes, chunk_starts, embeddings_matrix

    def group_chunks_by_document(
        self,
        chunk_hashes: list[str],
        raw_document_hashes: list[str],
        chunk_starts: list[int],
        embeddings: np.ndarray
    ) -> dict[str, dict[str, Any]]:
        """
        Group chunks by document.

        Args:
            chunk_hashes: List of chunk hashes
            raw_document_hashes: List of document hashes
            chunk_starts: List of chunk start positions
            embeddings: Matrix of chunk embeddings

        Returns:
            Dictionary mapping raw_document_hash to {indices, chunk_count}
            where indices are positions in the original arrays
        """
        doc_groups: dict[str, dict[str, Any]] = {}

        for i, doc_hash in enumerate(raw_document_hashes):
            if doc_hash not in doc_groups:
                doc_groups[doc_hash] = {
                    "indices": [],
                    "chunk_count": 0
                }
            doc_groups[doc_hash]["indices"].append(i)
            doc_groups[doc_hash]["chunk_count"] += 1

        logging.info(f"Grouped chunks into {len(doc_groups)} documents")
        return doc_groups

    def compute_corpus_noise_std(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute corpus-wide noise standard deviation.

        Args:
            embeddings: All chunk embeddings in the corpus

        Returns:
            Noise standard deviation per embedding dimension
        """
        corpus_std = np.std(embeddings, axis=0)
        noise_std = self.noise_scale * corpus_std
        logging.info(
            "Corpus std (per-dim): min=%.6f max=%.6f mean=%.6f; "
            "Noise std (per-dim): min=%.6f max=%.6f mean=%.6f",
            float(np.min(corpus_std)),
            float(np.max(corpus_std)),
            float(np.mean(corpus_std)),
            float(np.min(noise_std)),
            float(np.max(noise_std)),
            float(np.mean(noise_std)),
        )
        return noise_std.astype(np.float32)

    def create_padding_vectors(
        self,
        doc_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        knn_model: NearestNeighbors,
        noise_std: np.ndarray,
        num_padding: int
    ) -> np.ndarray:
        """
        Create padding vectors using KNN-weighted mean with Gaussian noise.

        For each padding slot i:
        - Use source chunk = doc_embeddings[i % num_chunks]
        - Find K nearest neighbors to the source chunk
        - Compute weighted mean with inverse square distance weights
        - Add Gaussian noise

        Args:
            doc_embeddings: Embeddings of the document's chunks
            all_embeddings: All chunk embeddings in the corpus (for KNN lookup)
            knn_model: Fitted NearestNeighbors model
            noise_std: Per-dimension standard deviation for Gaussian noise
            num_padding: Number of padding vectors to create

        Returns:
            Array of padding vectors (num_padding × n_dims)
        """
        n_dims = doc_embeddings.shape[1]
        num_chunks = len(doc_embeddings)
        padding_vectors = np.zeros((num_padding, n_dims), dtype=np.float32)

        epsilon = 1e-8  # To avoid division by zero

        for i in range(num_padding):
            # Use cyclic source chunk selection
            source_idx = i % num_chunks
            source_chunk = doc_embeddings[source_idx:source_idx + 1]  # Keep 2D shape

            # Find K nearest neighbors
            distances, neighbor_indices = knn_model.kneighbors(source_chunk)
            distances = distances[0]  # Flatten
            neighbor_indices = neighbor_indices[0]  # Flatten

            # Get neighbor embeddings
            neighbor_embeddings = all_embeddings[neighbor_indices]

            # Compute inverse square distance weights
            weights = 1.0 / (distances ** 2 + epsilon)
            weights = weights / weights.sum()  # Normalize

            # Compute weighted mean
            weighted_mean = np.sum(weights[:, np.newaxis] * neighbor_embeddings, axis=0)

            # Add Gaussian noise
            noise = np.random.normal(0, noise_std, n_dims)
            padding_vectors[i] = weighted_mean + noise

        return padding_vectors.astype(np.float32)

    def create_padding_vectors_precomputed(
        self,
        doc_indices: list[int],
        doc_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
        noise_std: np.ndarray,
        num_padding: int
    ) -> np.ndarray:
        """
        Create padding vectors using precomputed KNN neighbors and weights.

        Args:
            doc_indices: Indices of the document's chunks in the full corpus array
            doc_embeddings: Embeddings of the document's chunks
            all_embeddings: All chunk embeddings in the corpus
            neighbor_indices: Precomputed neighbor indices for each corpus embedding
            neighbor_weights: Precomputed neighbor weights for each corpus embedding
            noise_std: Per-dimension standard deviation for Gaussian noise
            num_padding: Number of padding vectors to create

        Returns:
            Array of padding vectors (num_padding × n_dims)
        """
        n_dims = doc_embeddings.shape[1]
        num_chunks = len(doc_embeddings)
        padding_vectors = np.zeros((num_padding, n_dims), dtype=np.float32)
        weighted_mean_cache: dict[int, np.ndarray] = {}

        for i in range(num_padding):
            source_local_idx = i % num_chunks
            source_global_idx = doc_indices[source_local_idx]

            weighted_mean = weighted_mean_cache.get(source_global_idx)
            if weighted_mean is None:
                neighbors = neighbor_indices[source_global_idx]
                weights = neighbor_weights[source_global_idx]
                weighted_mean = np.sum(
                    weights[:, np.newaxis] * all_embeddings[neighbors],
                    axis=0
                )
                weighted_mean_cache[source_global_idx] = weighted_mean

            noise = np.random.normal(0, noise_std, n_dims)
            padding_vectors[i] = weighted_mean + noise

        return padding_vectors.astype(np.float32)

    def create_noise_only_padding_vectors(
        self,
        doc_embeddings: np.ndarray,
        noise_std: np.ndarray,
        num_padding: int
    ) -> np.ndarray:
        """
        Create padding vectors by adding Gaussian noise to existing chunks.

        For each padding slot i:
        - Use source chunk = doc_embeddings[i % num_chunks]
        - Add Gaussian noise with per-dimension std

        Args:
            doc_embeddings: Embeddings of the document's chunks
            noise_std: Per-dimension standard deviation for Gaussian noise
            num_padding: Number of padding vectors to create

        Returns:
            Array of padding vectors (num_padding × n_dims)
        """
        n_dims = doc_embeddings.shape[1]
        num_chunks = len(doc_embeddings)
        padding_vectors = np.zeros((num_padding, n_dims), dtype=np.float32)

        for i in range(num_padding):
            source_idx = i % num_chunks
            source_chunk = doc_embeddings[source_idx]
            noise = np.random.normal(0, noise_std, n_dims)
            padding_vectors[i] = source_chunk + noise

        return padding_vectors.astype(np.float32)
    def compute_document_embeddings(
        self,
        doc_groups: dict[str, dict[str, Any]],
        all_embeddings: np.ndarray
    ) -> dict[str, dict[str, Any]]:
        """
        Compute document embeddings with padding and stacking.

        Args:
            doc_groups: Document groups from group_chunks_by_document
            all_embeddings: All chunk embeddings in the corpus

        Returns:
            Dictionary mapping raw_document_hash to {
                vector: document embedding,
                chunk_count: original chunk count,
                padded_to: target chunk count
            }
        """
        logging.info(
            "Computing document embeddings with target_chunks=%d (stacking, padding_method=%s)",
            self.target_chunk_count,
            self.padding_method
        )

        # Compute corpus-wide noise std
        noise_std = self.compute_corpus_noise_std(all_embeddings)

        # Build KNN model only if needed
        knn_model = None
        neighbor_indices = None
        neighbor_weights = None
        if self.padding_method == "knn_mean":
            needs_padding = any(
                group["chunk_count"] < self.target_chunk_count
                for group in doc_groups.values()
            )
            if needs_padding:
                logging.info(f"Building KNN model with k={self.knn_k}")
                knn_model = NearestNeighbors(
                    n_neighbors=self.knn_k,
                    metric="cosine",
                    n_jobs=-1
                )
                knn_model.fit(all_embeddings)
                logging.info("Precomputing KNN neighbors for all embeddings")
                distances, neighbor_indices = knn_model.kneighbors(
                    all_embeddings,
                    n_neighbors=self.knn_k
                )
                distances = distances.astype(np.float32)
                neighbor_indices = neighbor_indices.astype(np.int32)
                epsilon = 1e-8
                neighbor_weights = 1.0 / (distances ** 2 + epsilon)
                neighbor_weights = neighbor_weights / neighbor_weights.sum(axis=1, keepdims=True)
                neighbor_weights = neighbor_weights.astype(np.float32)

        doc_embeddings_result: dict[str, dict[str, Any]] = {}

        for doc_hash, group in doc_groups.items():
            indices = group["indices"]
            chunk_count = group["chunk_count"]

            # Get document's chunk embeddings
            doc_chunk_embeddings = all_embeddings[indices]

            # Pad or truncate to target chunk count, then stack
            if chunk_count >= self.target_chunk_count:
                combined = doc_chunk_embeddings[:self.target_chunk_count]
                padded_to = self.target_chunk_count
            else:
                num_padding = self.target_chunk_count - chunk_count
                if self.padding_method == "knn_mean":
                    if neighbor_indices is None or neighbor_weights is None:
                        padding_vectors = self.create_padding_vectors(
                            doc_embeddings=doc_chunk_embeddings,
                            all_embeddings=all_embeddings,
                            knn_model=knn_model,
                            noise_std=noise_std,
                            num_padding=num_padding
                        )
                    else:
                        padding_vectors = self.create_padding_vectors_precomputed(
                            doc_indices=indices,
                            doc_embeddings=doc_chunk_embeddings,
                            all_embeddings=all_embeddings,
                            neighbor_indices=neighbor_indices,
                            neighbor_weights=neighbor_weights,
                            noise_std=noise_std,
                            num_padding=num_padding
                        )
                else:
                    padding_vectors = self.create_noise_only_padding_vectors(
                        doc_embeddings=doc_chunk_embeddings,
                        noise_std=noise_std,
                        num_padding=num_padding
                    )
                combined = np.vstack([doc_chunk_embeddings, padding_vectors])
                padded_to = self.target_chunk_count

            doc_embedding = combined.reshape(-1)

            doc_embeddings_result[doc_hash] = {
                "vector": doc_embedding.astype(np.float32),
                "chunk_count": chunk_count,
                "padded_to": padded_to
            }

        logging.info(f"Computed {len(doc_embeddings_result)} document embeddings")
        return doc_embeddings_result

    def store_document_embeddings(
        self,
        session: Session,
        corpus_name: str,
        source_model_name: str,
        algorithm: str,
        target_dims: int,
        padding_method: str,
        target_chunk_count: int,
        doc_embeddings: dict[str, dict[str, Any]]
    ) -> tuple[int, int]:
        """
        Store document embeddings in the database.

        Args:
            session: Database session
            corpus_name: Name of the corpus
            source_model_name: Source embedding model name
            algorithm: Reduction algorithm
            target_dims: Target dimensions
            doc_embeddings: Dictionary of document embeddings

        Returns:
            Tuple of (inserted_count, existing_count)
        """
        total = len(doc_embeddings)
        logging.info(f"Storing {total} document embeddings")

        insert_query = text("""
            INSERT INTO pipeline.boe_document_embedding
            (corpus_name, raw_document_hash, source_model_name, algorithm,
             target_dims, padding_method, target_chunk_count, vector, chunk_count, padded_to)
            VALUES (:corpus_name, :raw_document_hash, :source_model_name, :algorithm,
                    :target_dims, :padding_method, :target_chunk_count, :vector, :chunk_count, :padded_to)
            ON CONFLICT (corpus_name, raw_document_hash, source_model_name, algorithm, target_dims, padding_method, target_chunk_count)
            DO NOTHING
        """)

        batch_size = 5000
        rows: list[dict[str, Any]] = []
        inserted_count = 0

        def flush_batch(batch: list[dict[str, Any]]) -> int:
            if not batch:
                return 0
            result = session.execute(insert_query, batch)
            return int(result.rowcount or 0)

        for doc_hash, embed_data in doc_embeddings.items():
            rows.append({
                "corpus_name": corpus_name,
                "raw_document_hash": doc_hash,
                "source_model_name": source_model_name,
                "algorithm": algorithm,
                "target_dims": target_dims,
                "padding_method": padding_method,
                "target_chunk_count": target_chunk_count,
                "vector": embed_data["vector"].tolist(),
                "chunk_count": embed_data["chunk_count"],
                "padded_to": embed_data["padded_to"]
            })

            if len(rows) >= batch_size:
                inserted_count += flush_batch(rows)
                logging.info(f"Flushed {len(rows)} rows to the database. Total inserted: {inserted_count}")
                rows.clear()

        if rows:
            inserted_count += flush_batch(rows)


        session.commit()
        existing_count = total - inserted_count
        logging.info(f"Document embeddings stored: {inserted_count} inserted, {existing_count} existing")
        return inserted_count, existing_count

    def process_reduction(
        self,
        session: Session,
        corpus_name: str,
        source_model_name: str,
        algorithm: str,
        target_dims: int
    ) -> dict[str, Any]:
        """
        Process a single reduction combination to compute document embeddings.

        Args:
            session: Database session
            corpus_name: Name of the corpus
            source_model_name: Source embedding model name
            algorithm: Reduction algorithm
            target_dims: Target dimensions

        Returns:
            Dictionary with processing statistics
        """
        logging.info(f"Processing {source_model_name}/{algorithm}/{target_dims}")

        expected_docs = self.count_expected_documents(
            session=session,
            corpus_name=corpus_name,
            source_model_name=source_model_name,
            algorithm=algorithm,
            target_dims=target_dims
        )

        if expected_docs == 0:
            logging.info(f"No reducible documents found for {source_model_name}/{algorithm}/{target_dims}, skipping")
            return {
                "source_model_name": source_model_name,
                "algorithm": algorithm,
                "target_dims": target_dims,
                "skipped": True,
                "num_documents": 0,
                "inserted": 0,
                "existing": 0
            }

        existing_docs = self.count_existing_documents(
            session=session,
            corpus_name=corpus_name,
            source_model_name=source_model_name,
            algorithm=algorithm,
            target_dims=target_dims,
            padding_method=self.padding_method,
            target_chunk_count=self.target_chunk_count
        )

        if existing_docs >= expected_docs:
            logging.info(
                "Document embeddings already complete for %s/%s/%s (%d/%d), skipping",
                source_model_name,
                algorithm,
                target_dims,
                existing_docs,
                expected_docs
            )
            return {
                "source_model_name": source_model_name,
                "algorithm": algorithm,
                "target_dims": target_dims,
                "skipped": True,
                "num_documents": 0,
                "inserted": 0,
                "existing": 0
            }

        # Fetch chunk embeddings
        chunk_hashes, raw_doc_hashes, chunk_starts, embeddings = self.fetch_chunk_embeddings(
            session, corpus_name, source_model_name, algorithm, target_dims
        )

        if len(chunk_hashes) == 0:
            logging.warning(f"No chunks found for {source_model_name}/{algorithm}/{target_dims}")
            return {
                "source_model_name": source_model_name,
                "algorithm": algorithm,
                "target_dims": target_dims,
                "skipped": True,
                "num_documents": 0,
                "inserted": 0,
                "existing": 0
            }

        # Group chunks by document
        doc_groups = self.group_chunks_by_document(
            chunk_hashes, raw_doc_hashes, chunk_starts, embeddings
        )

        # Compute document embeddings
        doc_embeddings = self.compute_document_embeddings(doc_groups, embeddings)

        # Store document embeddings
        inserted, existing = self.store_document_embeddings(
            session,
            corpus_name,
            source_model_name,
            algorithm,
            target_dims,
            self.padding_method,
            self.target_chunk_count,
            doc_embeddings
        )

        return {
            "source_model_name": source_model_name,
            "algorithm": algorithm,
            "target_dims": target_dims,
            "skipped": False,
            "num_documents": len(doc_embeddings),
            "inserted": inserted,
            "existing": existing
        }

    def process_corpus(
        self,
        session: Session,
        corpus_name: str
    ) -> dict[str, Any]:
        """
        Process all available reductions for a corpus to compute document embeddings.

        Args:
            session: Database session
            corpus_name: Name of the corpus to process

        Returns:
            Dictionary containing processing statistics
        """
        logging.info(f"Starting document embedding computation for corpus: {corpus_name}")

        # Get available reductions
        available_reductions = self.get_available_reductions(session, corpus_name)
        logging.info(f"Found {len(available_reductions)} reduction combinations to process")
        for reduction in available_reductions:
            logging.info(f"Reduction: {reduction['source_model_name']}/{reduction['algorithm']}/{reduction['target_dims']}")

        if not available_reductions:
            logging.warning(f"No reduced embeddings found for corpus: {corpus_name}")
            return {
                "corpus_name": corpus_name,
                "reductions_processed": []
            }

        reductions_processed = []

        for reduction in available_reductions:
            logging.info("=" * 60)
            logging.info(f"Processing: {reduction['source_model_name']}/{reduction['algorithm']}/{reduction['target_dims']}")
            logging.info("=" * 60)

            stats = self.process_reduction(
                session=session,
                corpus_name=corpus_name,
                source_model_name=reduction["source_model_name"],
                algorithm=reduction["algorithm"],
                target_dims=reduction["target_dims"]
            )
            reductions_processed.append(stats)

        return {
            "corpus_name": corpus_name,
            "reductions_processed": reductions_processed
        }


def get_available_corpora(session: Session) -> list[str]:
    """
    Get list of available corpora from the chunking pipeline.

    Returns:
        List of corpus names
    """
    query = text("""
        SELECT DISTINCT corpus_name FROM pipeline.boe_chunked_document
    """)
    result = session.execute(query)
    return [row[0] for row in result]


def main():
    """Main execution function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Direct logging to file
    logging.getLogger().addHandler(logging.FileHandler("boe_doc_embedding.log"))

    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database

    # Configuration parameters
    knn_k = 5
    noise_scale = 0.01
    padding_methods = ["knn_mean", "noise_only"]

    # Create database session
    with get_session(db_config) as session:
        print("BOE Document Embedding Pipeline")
        print("=" * 60)
        print("Method: KNN Padding with Inverse-Distance Weighting + Stacking")
        print(f"KNN neighbors: {knn_k}")
        print(f"Noise scale: {noise_scale}")
        print(f"Padding methods: {padding_methods}")
        print(f"Available corpora: {get_available_corpora(session)}")
        print("=" * 60)

        corpus_chunk_counts: dict[str, int | list[int]] = {
            'battery-abstracts': 2,
            'goodreads-bookgenres': 2,
            'imdb_reviews': 4,
            'newsgroups': 2,
            'patent-classification': 1,
            'pubmed-multilabel': 2,
            't2-ragbench-convfinqa': 6,
            'trec_questions': 1,
            'twitter-financial-news': 1,
            'wikipedia_sample': 9
        }


        # Process each corpus
        for corpus_name in get_available_corpora(session):
            target_counts = corpus_chunk_counts.get(corpus_name)
            if target_counts is None:
                logging.warning("No target chunk count configured for corpus: %s", corpus_name)
                continue

            if isinstance(target_counts, int):
                target_counts = [target_counts]

            for target_chunk_count in target_counts:
                # Initialize pipeline
                for padding_method in padding_methods:
                    pipeline = BOEDocEmbeddingPipeline(
                        target_chunk_count=target_chunk_count,
                        knn_k=knn_k,
                        noise_scale=noise_scale,
                        padding_method=padding_method
                    )
                    print(
                        f"\nProcessing corpus: {corpus_name} "
                        f"(padding_method={padding_method}, target_chunk_count={target_chunk_count})"
                    )
                    print("-" * 40)

                    try:
                        stats = pipeline.process_corpus(session, corpus_name)

                        # Print statistics
                        print(
                            f"\nResults for {corpus_name} "
                            f"(padding_method={padding_method}, target_chunk_count={target_chunk_count}):"
                        )

                        for red_stats in stats["reductions_processed"]:
                            if red_stats.get("skipped"):
                                status = "skipped"
                            else:
                                status = f"{red_stats['inserted']} docs inserted, {red_stats['num_documents']} total docs"
                            print(
                                f"  {red_stats['source_model_name']}/"
                                f"{red_stats['algorithm']}/"
                                f"{red_stats['target_dims']}: {status}"
                            )

                    except Exception as e:
                        logging.error(
                            "Error processing corpus %s (padding_method=%s, target_chunk_count=%s): %s",
                            corpus_name,
                            padding_method,
                            target_chunk_count,
                            str(e)
                        )
                        raise

    print("\n" + "=" * 60)
    print("DOCUMENT EMBEDDING COMPUTATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

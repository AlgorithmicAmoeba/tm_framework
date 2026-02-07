"""
BOE (Bag of Embeddings) Word Embedding pipeline.
Derives word embeddings from chunk embeddings using TF-IDF matrix factorization
with Ridge regression.
"""
import logging
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg


class BOEWordEmbeddingPipeline:
    """
    Pipeline for deriving word embeddings from chunk embeddings.

    Uses TF-IDF matrix factorization with Ridge regression:
    X: TF-IDF matrix (chunks × vocab_words)
    C: chunk embeddings (chunks × embedding_dim)
    W = Ridge.fit(X, C).coef_.T (word embeddings)
    """

    TARGET_DIMS = [20, 50, 100]
    ALGORITHMS = ['umap', 'pca']

    def __init__(self, ridge_alpha: float = 1.0):
        """
        Initialize the word embedding pipeline.

        Args:
            ridge_alpha: Regularization strength for Ridge regression
        """
        self.ridge_alpha = ridge_alpha
        logging.info(f"Initialized BOE word embedding pipeline (ridge_alpha={ridge_alpha})")

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

    def check_word_embeddings_exist(
        self,
        session: Session,
        corpus_name: str,
        source_model_name: str,
        algorithm: str,
        target_dims: int
    ) -> bool:
        """
        Check if word embeddings already exist for a given combination.

        Args:
            session: Database session
            corpus_name: Corpus name
            source_model_name: Source embedding model name
            algorithm: Reduction algorithm
            target_dims: Target dimensions

        Returns:
            True if word embeddings exist, False otherwise
        """
        query = text("""
            SELECT 1 FROM pipeline.boe_word_embedding
            WHERE corpus_name = :corpus_name
            AND source_model_name = :source_model_name
            AND algorithm = :algorithm
            AND target_dims = :target_dims
            LIMIT 1
        """)
        result = session.execute(query, {
            "corpus_name": corpus_name,
            "source_model_name": source_model_name,
            "algorithm": algorithm,
            "target_dims": target_dims
        })
        return result.fetchone() is not None

    def fetch_chunk_data(
        self,
        session: Session,
        corpus_name: str,
        source_model_name: str,
        algorithm: str,
        target_dims: int
    ) -> tuple[list[str], list[str], np.ndarray]:
        """
        Fetch chunk vocabulary words and reduced embeddings.

        Args:
            session: Database session
            corpus_name: Name of the corpus
            source_model_name: Source embedding model name
            algorithm: Reduction algorithm
            target_dims: Target dimensions

        Returns:
            Tuple of (chunk_hashes, vocabulary_words_list, embeddings_matrix)
            where vocabulary_words_list contains space-separated vocabulary words per chunk
        """
        logging.info(f"Fetching chunk data for {source_model_name}/{algorithm}/{target_dims}")

        query = text("""
            SELECT c.chunk_hash, c.chunk_vocabulary_words, r.vector
            FROM pipeline.boe_chunked_document c
            JOIN pipeline.boe_embedding_reduced r
                ON c.chunk_hash = r.boe_chunked_document_hash
            WHERE c.corpus_name = :corpus_name
            AND r.source_model_name = :source_model_name
            AND r.algorithm = :algorithm
            AND r.target_dims = :target_dims
            ORDER BY c.chunk_hash
        """)

        result = session.execute(query, {
            "corpus_name": corpus_name,
            "source_model_name": source_model_name,
            "algorithm": algorithm,
            "target_dims": target_dims
        })

        chunk_hashes = []
        vocab_words_list = []
        vectors = []

        for row in result:
            chunk_hashes.append(row[0])
            vocab_words_list.append(row[1])
            vectors.append(row[2])

        if not vectors:
            logging.warning(f"No chunk data found for {source_model_name}/{algorithm}/{target_dims}")
            return [], [], np.array([])

        embeddings_matrix = np.array(vectors, dtype=np.float32)
        logging.info(f"Fetched {len(chunk_hashes)} chunks with embeddings shape {embeddings_matrix.shape}")

        return chunk_hashes, vocab_words_list, embeddings_matrix

    def derive_word_embeddings(
        self,
        vocab_words_list: list[str],
        chunk_embeddings: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        """
        Derive word embeddings from chunk embeddings using TF-IDF + Ridge regression.

        Args:
            vocab_words_list: List of space-separated vocabulary words per chunk
            chunk_embeddings: Matrix of chunk embeddings (n_chunks × n_dims)

        Returns:
            Tuple of (vocabulary_words, word_embeddings) where word_embeddings
            is (n_words × n_dims)
        """
        logging.info("Building TF-IDF matrix from chunk vocabulary words")

        # Build TF-IDF matrix from chunk vocabulary words
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b'
        )
        tfidf_matrix = vectorizer.fit_transform(vocab_words_list)
        vocabulary_words = vectorizer.get_feature_names_out().tolist()

        logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        logging.info(f"Vocabulary size: {len(vocabulary_words)}")

        if len(vocabulary_words) == 0:
            logging.warning("Empty vocabulary, cannot derive word embeddings")
            return [], np.array([])

        # Solve for word embeddings using Ridge regression
        # X: TF-IDF matrix (chunks × vocab_words)
        # C: chunk embeddings (chunks × embedding_dim)
        # W = Ridge.fit(X, C).coef_.T (word_embeddings)
        logging.info(f"Fitting Ridge regression (alpha={self.ridge_alpha})")

        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        ridge.fit(tfidf_matrix, chunk_embeddings)

        # coef_ is (n_dims × n_words) after fitting, so transpose to get (n_words × n_dims)
        word_embeddings = ridge.coef_.T.astype(np.float32)

        logging.info(f"Derived word embeddings shape: {word_embeddings.shape}")

        return vocabulary_words, word_embeddings

    def store_word_embeddings(
        self,
        session: Session,
        corpus_name: str,
        source_model_name: str,
        algorithm: str,
        target_dims: int,
        vocabulary_words: list[str],
        word_embeddings: np.ndarray
    ) -> tuple[int, int]:
        """
        Store word embeddings in the database.

        Args:
            session: Database session
            corpus_name: Name of the corpus
            source_model_name: Source embedding model name
            algorithm: Reduction algorithm
            target_dims: Target dimensions
            vocabulary_words: List of vocabulary words
            word_embeddings: Word embedding vectors (n_words × n_dims)

        Returns:
            Tuple of (inserted_count, existing_count)
        """
        logging.info(f"Storing {len(vocabulary_words)} word embeddings")

        inserted_count = 0
        existing_count = 0

        for i, word in enumerate(vocabulary_words):
            # Check if word embedding already exists
            check_query = text("""
                SELECT id FROM pipeline.boe_word_embedding
                WHERE corpus_name = :corpus_name
                AND word = :word
                AND source_model_name = :source_model_name
                AND algorithm = :algorithm
                AND target_dims = :target_dims
            """)

            existing = session.execute(check_query, {
                "corpus_name": corpus_name,
                "word": word,
                "source_model_name": source_model_name,
                "algorithm": algorithm,
                "target_dims": target_dims
            }).fetchone()

            if existing:
                existing_count += 1
                continue

            # Insert new word embedding
            insert_query = text("""
                INSERT INTO pipeline.boe_word_embedding
                (corpus_name, word, source_model_name, algorithm, target_dims, vector)
                VALUES (:corpus_name, :word, :source_model_name, :algorithm, :target_dims, :vector)
            """)

            session.execute(insert_query, {
                "corpus_name": corpus_name,
                "word": word,
                "source_model_name": source_model_name,
                "algorithm": algorithm,
                "target_dims": target_dims,
                "vector": word_embeddings[i].tolist()
            })
            inserted_count += 1

        session.commit()
        logging.info(f"Word embeddings stored: {inserted_count} inserted, {existing_count} existing")
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
        Process a single reduction combination to derive word embeddings.

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

        # Check if word embeddings already exist
        if self.check_word_embeddings_exist(
            session, corpus_name, source_model_name, algorithm, target_dims
        ):
            logging.info(f"Word embeddings already exist for {source_model_name}/{algorithm}/{target_dims}, skipping")
            return {
                "source_model_name": source_model_name,
                "algorithm": algorithm,
                "target_dims": target_dims,
                "skipped": True,
                "vocabulary_size": 0,
                "inserted": 0,
                "existing": 0
            }

        # Fetch chunk data
        chunk_hashes, vocab_words_list, chunk_embeddings = self.fetch_chunk_data(
            session, corpus_name, source_model_name, algorithm, target_dims
        )

        if len(chunk_hashes) == 0:
            logging.warning(f"No chunks found for {source_model_name}/{algorithm}/{target_dims}")
            return {
                "source_model_name": source_model_name,
                "algorithm": algorithm,
                "target_dims": target_dims,
                "skipped": True,
                "vocabulary_size": 0,
                "inserted": 0,
                "existing": 0
            }

        # Derive word embeddings
        vocabulary_words, word_embeddings = self.derive_word_embeddings(
            vocab_words_list, chunk_embeddings
        )

        if len(vocabulary_words) == 0:
            logging.warning(f"Empty vocabulary for {source_model_name}/{algorithm}/{target_dims}")
            return {
                "source_model_name": source_model_name,
                "algorithm": algorithm,
                "target_dims": target_dims,
                "skipped": True,
                "vocabulary_size": 0,
                "inserted": 0,
                "existing": 0
            }

        # Store word embeddings
        inserted, existing = self.store_word_embeddings(
            session, corpus_name, source_model_name, algorithm, target_dims,
            vocabulary_words, word_embeddings
        )

        return {
            "source_model_name": source_model_name,
            "algorithm": algorithm,
            "target_dims": target_dims,
            "skipped": False,
            "vocabulary_size": len(vocabulary_words),
            "inserted": inserted,
            "existing": existing
        }

    def process_corpus(
        self,
        session: Session,
        corpus_name: str
    ) -> dict[str, Any]:
        """
        Process all available reductions for a corpus to derive word embeddings.

        Args:
            session: Database session
            corpus_name: Name of the corpus to process

        Returns:
            Dictionary containing processing statistics
        """
        logging.info(f"Starting word embedding derivation for corpus: {corpus_name}")

        # Get available reductions
        available_reductions = self.get_available_reductions(session, corpus_name)
        logging.info(f"Found {len(available_reductions)} reduction combinations to process")

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
    logging.getLogger().addHandler(logging.FileHandler("boe_word_embedding.log"))

    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database

    # Create database session
    with get_session(db_config) as session:
        print("BOE Word Embedding Pipeline")
        print("=" * 60)
        print("Method: TF-IDF + Ridge Regression")
        print(f"Available corpora: {get_available_corpora(session)}")
        print("=" * 60)

        # Initialize pipeline
        pipeline = BOEWordEmbeddingPipeline()

        # Process each corpus
        for corpus_name in get_available_corpora(session):
            print(f"\nProcessing corpus: {corpus_name}")
            print("-" * 40)

            try:
                stats = pipeline.process_corpus(session, corpus_name)

                # Print statistics
                print(f"\nResults for {corpus_name}:")

                for red_stats in stats["reductions_processed"]:
                    if red_stats.get("skipped"):
                        status = "skipped"
                    else:
                        status = f"{red_stats['inserted']} words inserted, vocab size: {red_stats['vocabulary_size']}"
                    print(f"  {red_stats['source_model_name']}/{red_stats['algorithm']}/{red_stats['target_dims']}: {status}")

            except Exception as e:
                logging.error(f"Error processing corpus {corpus_name}: {str(e)}")
                raise

    print("\n" + "=" * 60)
    print("WORD EMBEDDING DERIVATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

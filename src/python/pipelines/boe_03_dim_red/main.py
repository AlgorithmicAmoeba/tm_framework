"""
BOE (Bag of Embeddings) Dimensionality Reduction pipeline.
This pipeline reads embeddings from the database, applies centering and scaling,
then reduces dimensionality using UMAP and PCA to 10 and 20 dimensions.
"""
import logging
import json
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg


class BOEDimRedPipeline:
    """Main class for dimensionality reduction of BOE embeddings."""
    
    # Configuration for dimensionality reduction
    TARGET_DIMS = [10, 20]
    ALGORITHMS = ['umap', 'pca']
    MAX_DIM_BY_SPARSE_MODEL = {
        'naver/splade-v3': 30522,
    }
    
    def __init__(self):
        """Initialize the dimensionality reduction pipeline."""
        logging.info("Initialized BOE dimensionality reduction pipeline")
    
    def get_available_models(self, session: Session, corpus_name: str) -> dict[str, list[str]]:
        """
        Get available embedding models for a corpus.
        
        Args:
            session: Database session
            corpus_name: Name of the corpus
            
        Returns:
            Dictionary with 'dense' and 'sparse' keys containing lists of model names
        """
        # Get dense models
        dense_query = text("""
            SELECT DISTINCT e.model_name
            FROM pipeline.boe_embedding e
            JOIN pipeline.boe_chunked_document c ON e.boe_chunked_document_hash = c.chunk_hash
            WHERE c.corpus_name = :corpus_name
        """)
        dense_result = session.execute(dense_query, {"corpus_name": corpus_name})
        dense_models = [row[0] for row in dense_result]
        
        # Get sparse models
        sparse_query = text("""
            SELECT DISTINCT e.model_name
            FROM pipeline.boe_embedding_sparse e
            JOIN pipeline.boe_chunked_document c ON e.boe_chunked_document_hash = c.chunk_hash
            WHERE c.corpus_name = :corpus_name
        """)
        sparse_result = session.execute(sparse_query, {"corpus_name": corpus_name})
        sparse_models = [row[0] for row in sparse_result]
        
        return {
            'dense': dense_models,
            'sparse': sparse_models
        }
    
    def fetch_dense_embeddings(
        self, 
        session: Session, 
        corpus_name: str, 
        model_name: str
    ) -> tuple[list[str], np.ndarray]:
        """
        Fetch dense embeddings from the database for a corpus and model.
        
        Args:
            session: Database session
            corpus_name: Name of the corpus
            model_name: Name of the embedding model
            
        Returns:
            Tuple of (chunk_hashes, embeddings_matrix)
        """
        logging.info(f"Fetching dense embeddings for corpus '{corpus_name}', model '{model_name}'")
        
        query = text("""
            SELECT e.boe_chunked_document_hash, e.vector
            FROM pipeline.boe_embedding e
            JOIN pipeline.boe_chunked_document c ON e.boe_chunked_document_hash = c.chunk_hash
            WHERE c.corpus_name = :corpus_name
            AND e.model_name = :model_name
            ORDER BY e.boe_chunked_document_hash
        """)
        
        result = session.execute(query, {"corpus_name": corpus_name, "model_name": model_name})
        
        chunk_hashes = []
        vectors = []
        
        for row in result:
            chunk_hashes.append(row[0])
            vectors.append(row[1])
        
        if not vectors:
            logging.warning(f"No dense embeddings found for corpus '{corpus_name}', model '{model_name}'")
            return [], np.array([])
        
        embeddings_matrix = np.array(vectors, dtype=np.float32)
        logging.info(f"Fetched {len(chunk_hashes)} dense embeddings with shape {embeddings_matrix.shape}")
        
        return chunk_hashes, embeddings_matrix
    
    def fetch_sparse_embeddings(
        self, 
        session: Session, 
        corpus_name: str, 
        model_name: str
    ) -> tuple[list[str], sp.csr_matrix, int]:
        """
        Fetch sparse embeddings from the database for a corpus and model.
        
        Args:
            session: Database session
            corpus_name: Name of the corpus
            model_name: Name of the embedding model
            
        Returns:
            Tuple of (chunk_hashes, sparse_matrix, max_dim)
        """
        logging.info(f"Fetching sparse embeddings for corpus '{corpus_name}', model '{model_name}'")
        
        query = text("""
            SELECT e.boe_chunked_document_hash, e.sparse_vector
            FROM pipeline.boe_embedding_sparse e
            JOIN pipeline.boe_chunked_document c ON e.boe_chunked_document_hash = c.chunk_hash
            WHERE c.corpus_name = :corpus_name
            AND e.model_name = :model_name
            ORDER BY e.boe_chunked_document_hash
        """)
        
        result = session.execute(query, {"corpus_name": corpus_name, "model_name": model_name})
        
        chunk_hashes = []
        sparse_data = []
        max_dim = self.MAX_DIM_BY_SPARSE_MODEL[model_name]
        
        for row in result:
            chunk_hashes.append(row[0])
            sparse_vec = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            sparse_data.append(sparse_vec)
        
        if not sparse_data:
            logging.warning(f"No sparse embeddings found for corpus '{corpus_name}', model '{model_name}'")
            return [], sp.csr_matrix((0, 0)), max_dim
        
        # Build sparse matrix
        n_samples = len(sparse_data)
        
        # Prepare data for CSR matrix construction
        row_indices = []
        col_indices = []
        values = []
        
        for i, sparse_vec in enumerate(sparse_data):
            indices = sparse_vec['indices']
            vals = sparse_vec['values']
            
            row_indices.extend([i] * len(indices))
            col_indices.extend(indices)
            values.extend(vals)
        
        sparse_matrix = sp.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_samples, max_dim),
            dtype=np.float32
        )
        
        logging.info(f"Fetched {len(chunk_hashes)} sparse embeddings with shape {sparse_matrix.shape}")
        
        return chunk_hashes, sparse_matrix, max_dim
    
    def scale_embeddings(
        self, 
        embeddings: np.ndarray | sp.csr_matrix, 
        is_sparse: bool
    ) -> np.ndarray | sp.csr_matrix:
        """
        Center and scale embeddings using StandardScaler.
        
        For sparse data, only scaling is applied (with_mean=False) to preserve sparsity.
        For dense data, both centering and scaling are applied.
        
        Args:
            embeddings: Input embeddings (dense array or sparse matrix)
            is_sparse: Whether the input is sparse
            
        Returns:
            Scaled embeddings
        """
        if is_sparse:
            # For sparse data, don't center (would destroy sparsity)
            scaler = StandardScaler(with_mean=False, with_std=True)
            scaled = scaler.fit_transform(embeddings)
            logging.info("Scaled sparse embeddings (with_mean=False)")
        else:
            # For dense data, center and scale
            scaler = StandardScaler(with_mean=True, with_std=True)
            scaled = scaler.fit_transform(embeddings)
            logging.info("Scaled dense embeddings (with_mean=True)")
        
        return scaled
    
    def reduce_dimensions(
        self, 
        embeddings: np.ndarray | sp.csr_matrix, 
        algorithm: str, 
        n_components: int,
        is_sparse: bool
    ) -> np.ndarray:
        """
        Apply dimensionality reduction to embeddings.
        
        Args:
            embeddings: Input embeddings (can be sparse for both UMAP and PCA)
            algorithm: 'umap' or 'pca'
            n_components: Target number of dimensions
            is_sparse: Whether the input is sparse
            
        Returns:
            Reduced embeddings as dense numpy array
        """
        logging.info(f"Reducing dimensions using {algorithm} to {n_components} components")
        
        if algorithm == 'umap':
            # UMAP can handle sparse matrices directly
            reducer = umap.UMAP(
                n_components=n_components,
                metric='cosine',
                random_state=42,
                low_memory=True,
                n_jobs=1,
            )
            reduced = reducer.fit_transform(embeddings)
            
        elif algorithm == 'pca':
            # PCA can handle sparse matrices (uses arpack solver automatically for sparse)
            reducer = PCA(
                n_components=n_components,
                random_state=42
            )
            reduced = reducer.fit_transform(embeddings)
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        logging.info(f"Reduced embeddings shape: {reduced.shape}")
        return reduced.astype(np.float32)
    
    def check_existing_embeddings(
        self, 
        session: Session, 
        corpus_name: str, 
        source_model_name: str, 
        algorithm: str, 
        target_dims: int
    ) -> int:
        """
        Check how many reduced embeddings already exist for given parameters.
        
        Args:
            session: Database session
            corpus_name: Corpus name
            source_model_name: Source embedding model name
            algorithm: Reduction algorithm
            target_dims: Target dimensions
            
        Returns:
            Count of existing embeddings
        """
        query = text("""
            SELECT COUNT(*) 
            FROM pipeline.boe_embedding_reduced
            WHERE corpus_name = :corpus_name
            AND source_model_name = :source_model_name
            AND algorithm = :algorithm
            AND target_dims = :target_dims
        """)
        
        result = session.execute(query, {
            "corpus_name": corpus_name,
            "source_model_name": source_model_name,
            "algorithm": algorithm,
            "target_dims": target_dims
        })
        
        count = result.scalar()
        return count if count is not None else 0
    
    def store_reduced_embeddings(
        self, 
        session: Session,
        chunk_hashes: list[str],
        reduced_embeddings: np.ndarray,
        source_model_name: str,
        source_type: str,
        algorithm: str,
        target_dims: int,
        corpus_name: str
    ) -> tuple[int, int]:
        """
        Store reduced embeddings in the database.
        
        Args:
            session: Database session
            chunk_hashes: List of chunk hashes
            reduced_embeddings: Reduced embedding vectors
            source_model_name: Name of the source embedding model
            source_type: 'dense' or 'sparse'
            algorithm: 'umap' or 'pca'
            target_dims: Number of dimensions
            corpus_name: Name of the corpus
            
        Returns:
            Tuple of (inserted_count, existing_count)
        """
        logging.info(f"Storing {len(chunk_hashes)} reduced embeddings")
        
        inserted_count = 0
        existing_count = 0
        
        for i, chunk_hash in enumerate(chunk_hashes):
            # Check if embedding already exists
            check_query = text("""
                SELECT id FROM pipeline.boe_embedding_reduced
                WHERE boe_chunked_document_hash = :chunk_hash 
                AND source_model_name = :source_model_name
                AND algorithm = :algorithm
                AND target_dims = :target_dims
            """)
            
            existing = session.execute(check_query, {
                "chunk_hash": chunk_hash,
                "source_model_name": source_model_name,
                "algorithm": algorithm,
                "target_dims": target_dims
            }).fetchone()
            
            if existing:
                existing_count += 1
                continue
            
            # Insert new embedding
            insert_query = text("""
                INSERT INTO pipeline.boe_embedding_reduced 
                (boe_chunked_document_hash, source_model_name, source_type, 
                 algorithm, target_dims, corpus_name, vector)
                VALUES (:chunk_hash, :source_model_name, :source_type,
                        :algorithm, :target_dims, :corpus_name, :vector)
            """)
            
            session.execute(insert_query, {
                "chunk_hash": chunk_hash,
                "source_model_name": source_model_name,
                "source_type": source_type,
                "algorithm": algorithm,
                "target_dims": target_dims,
                "corpus_name": corpus_name,
                "vector": reduced_embeddings[i].tolist()
            })
            inserted_count += 1
        
        session.commit()
        logging.info(f"Reduced embeddings stored: {inserted_count} inserted, {existing_count} existing")
        return inserted_count, existing_count
    
    def process_embeddings(
        self,
        session: Session,
        chunk_hashes: list[str],
        embeddings: np.ndarray | sp.csr_matrix,
        source_model_name: str,
        source_type: str,
        corpus_name: str,
        is_sparse: bool
    ) -> dict[str, Any]:
        """
        Process embeddings through scaling and all reduction algorithms/dimensions.
        
        Args:
            session: Database session
            chunk_hashes: List of chunk hashes
            embeddings: Original embeddings
            source_model_name: Name of the source embedding model
            source_type: 'dense' or 'sparse'
            corpus_name: Name of the corpus
            is_sparse: Whether embeddings are sparse
            
        Returns:
            Dictionary with processing statistics
        """
        reductions: list[dict[str, Any]] = []
        n_embeddings = len(chunk_hashes)
        
        # First pass: check what work needs to be done for all algorithm/dimension combinations
        work_needed: dict[tuple[str, int], int] = {}
        for algorithm in self.ALGORITHMS:
            for n_dims in self.TARGET_DIMS:
                existing_count = self.check_existing_embeddings(
                    session, corpus_name, source_model_name, algorithm, n_dims
                )
                if existing_count < n_embeddings:
                    work_needed[(algorithm, n_dims)] = existing_count
        
        # Early exit if all reductions already exist
        if not work_needed:
            logging.info(f"All reductions already exist for {source_model_name} ({source_type}), skipping scaling and reduction")
            for algorithm in self.ALGORITHMS:
                for n_dims in self.TARGET_DIMS:
                    reductions.append({
                        "algorithm": algorithm,
                        "target_dims": n_dims,
                        "inserted": 0,
                        "existing": n_embeddings,
                        "skipped": True
                    })
            return {
                "source_model": source_model_name,
                "source_type": source_type,
                "n_embeddings": n_embeddings,
                "reductions": reductions
            }
        
        # Only scale embeddings if there's work to do
        logging.info(f"Work needed for {len(work_needed)} algorithm/dimension combinations")
        scaled_embeddings = self.scale_embeddings(embeddings, is_sparse)
        
        # Apply each algorithm and dimension combination
        for algorithm in self.ALGORITHMS:
            for n_dims in self.TARGET_DIMS:
                logging.info(f"Processing {source_model_name} ({source_type}) with {algorithm}/{n_dims}")
                
                # Check if this combination needs work
                if (algorithm, n_dims) not in work_needed:
                    logging.info(f"Skipping {algorithm}/{n_dims} - all {n_embeddings} embeddings already exist")
                    reductions.append({
                        "algorithm": algorithm,
                        "target_dims": n_dims,
                        "inserted": 0,
                        "existing": n_embeddings,
                        "skipped": True
                    })
                    continue
                
                # Reduce dimensions
                reduced = self.reduce_dimensions(scaled_embeddings, algorithm, n_dims, is_sparse)
                
                # Store reduced embeddings
                inserted, existing = self.store_reduced_embeddings(
                    session=session,
                    chunk_hashes=chunk_hashes,
                    reduced_embeddings=reduced,
                    source_model_name=source_model_name,
                    source_type=source_type,
                    algorithm=algorithm,
                    target_dims=n_dims,
                    corpus_name=corpus_name
                )
                
                reductions.append({
                    "algorithm": algorithm,
                    "target_dims": n_dims,
                    "inserted": inserted,
                    "existing": existing,
                    "skipped": False
                })
        
        return {
            "source_model": source_model_name,
            "source_type": source_type,
            "n_embeddings": n_embeddings,
            "reductions": reductions
        }
    
    def process_corpus(
        self, 
        session: Session, 
        corpus_name: str
    ) -> dict[str, Any]:
        """
        Process all embeddings for a corpus.
        
        Args:
            session: Database session
            corpus_name: Name of the corpus to process
            
        Returns:
            Dictionary containing processing statistics
        """
        logging.info(f"Starting dimensionality reduction for corpus: {corpus_name}")
        
        # Get available models
        available_models = self.get_available_models(session, corpus_name)
        logging.info(f"Available models - Dense: {available_models['dense']}, Sparse: {available_models['sparse']}")
        
        dense_model_stats: list[dict[str, Any]] = []
        sparse_model_stats: list[dict[str, Any]] = []
        
        # Process dense embeddings
        for model_name in available_models['dense']:
            logging.info("="*60)
            logging.info(f"Processing dense model: {model_name}")
            logging.info("="*60)
            
            chunk_hashes, embeddings = self.fetch_dense_embeddings(session, corpus_name, model_name)
            
            if len(chunk_hashes) == 0:
                logging.warning(f"No embeddings found for dense model: {model_name}")
                continue
            
            stats = self.process_embeddings(
                session=session,
                chunk_hashes=chunk_hashes,
                embeddings=embeddings,
                source_model_name=model_name,
                source_type='dense',
                corpus_name=corpus_name,
                is_sparse=False
            )
            dense_model_stats.append(stats)
        
        # Process sparse embeddings
        for model_name in available_models['sparse']:
            logging.info("="*60)
            logging.info(f"Processing sparse model: {model_name}")
            logging.info("="*60)
            
            chunk_hashes, embeddings, _ = self.fetch_sparse_embeddings(session, corpus_name, model_name)
            
            if len(chunk_hashes) == 0:
                logging.warning(f"No embeddings found for sparse model: {model_name}")
                continue
            
            stats = self.process_embeddings(
                session=session,
                chunk_hashes=chunk_hashes,
                embeddings=embeddings,
                source_model_name=model_name,
                source_type='sparse',
                corpus_name=corpus_name,
                is_sparse=True
            )
            sparse_model_stats.append(stats)
        
        return {
            "corpus_name": corpus_name,
            "dense_models": dense_model_stats,
            "sparse_models": sparse_model_stats
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
    logging.getLogger().addHandler(logging.FileHandler("boe_dim_red.log"))

    
    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database
    
    # Create database session
    with get_session(db_config) as session:
        print("BOE Dimensionality Reduction Pipeline")
        print("="*60)
        print("Algorithms: UMAP, PCA")
        print("Target dimensions: 10, 20")
        print(f"Available corpora: {get_available_corpora(session)}")
        print("="*60)

        
        # Initialize pipeline
        pipeline = BOEDimRedPipeline()
        
        # Process each corpus
        for corpus_name in get_available_corpora(session):
            print(f"\nProcessing corpus: {corpus_name}")
            print("-"*40)
            
            try:
                stats = pipeline.process_corpus(session, corpus_name)
                
                # Print statistics
                print(f"\nResults for {corpus_name}:")
                
                for model_stats in stats["dense_models"]:
                    print(f"\n  Dense model: {model_stats['source_model']}")
                    print(f"    Embeddings: {model_stats['n_embeddings']}")
                    for red_stats in model_stats["reductions"]:
                        status = "skipped" if red_stats.get("skipped") else f"{red_stats['inserted']} new, {red_stats['existing']} existing"
                        print(f"    {red_stats['algorithm']}/{red_stats['target_dims']}: {status}")
                
                for model_stats in stats["sparse_models"]:
                    print(f"\n  Sparse model: {model_stats['source_model']}")
                    print(f"    Embeddings: {model_stats['n_embeddings']}")
                    for red_stats in model_stats["reductions"]:
                        status = "skipped" if red_stats.get("skipped") else f"{red_stats['inserted']} new, {red_stats['existing']} existing"
                        print(f"    {red_stats['algorithm']}/{red_stats['target_dims']}: {status}")
                
            except Exception as e:
                logging.error(f"Error processing corpus {corpus_name}: {str(e)}")
                raise
    
    print("\n" + "="*60)
    print("DIMENSIONALITY REDUCTION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

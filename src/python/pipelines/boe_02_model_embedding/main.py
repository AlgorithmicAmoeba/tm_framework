"""
BOE (Bag of Embeddings) Model Embedding pipeline for chunked documents.
This pipeline generates embeddings for chunked documents using various embedding models
and stores them in the database for downstream topic modeling tasks.
"""
import logging
import json
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import SparseEncoder
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg


@dataclass
class EmbeddingModel:
    """Configuration for an embedding model."""
    name: str
    model_type: str  # 'dense' or 'sparse'
    model_instance: Union[SentenceTransformer, SparseEncoder]


class BOEEmbeddingPipeline:
    """Main class for generating BOE embeddings from chunked documents."""
    
    def __init__(self, model_configs: List[Dict[str, str]]):
        """
        Initialize the BOE embedding pipeline.
        
        Args:
            model_configs: List of model configurations with 'name' and 'type' keys
        """
        self.model_configs = model_configs
        self.models = self._create_models()
        logging.info(f"Initialized BOE embedding pipeline with {len(self.models)} models")
    
    def _create_models(self) -> Dict[str, EmbeddingModel]:
        """
        Create model instances from the provided configurations.
        
        Returns:
            Dictionary mapping model names to EmbeddingModel objects
        """
        models = {}
        for config in self.model_configs:
            name = config['name']
            model_type = config['type']
            
            try:
                if model_type == 'dense':
                    model_instance = SentenceTransformer(name)
                    embedding_model = EmbeddingModel(
                        name=name,
                        model_type='dense',
                        model_instance=model_instance
                    )
                elif model_type == 'sparse':
                    model_instance = SparseEncoder(name, model_kwargs={"dtype": "float16"})
                    model_instance.max_seq_length = 256
                    embedding_model = EmbeddingModel(
                        name=name,
                        model_type='sparse',
                        model_instance=model_instance
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                models[name] = embedding_model
                logging.info(f"Successfully created model: {name} ({model_type})")
            except Exception as e:
                logging.error(f"Failed to create model '{name}' ({model_type}): {e}")
                raise
        
        return models
    
    def fetch_chunked_documents(
        self, 
        session: Session, 
        corpus_name: str, 
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch chunked documents from the database for a specified corpus.
        
        Args:
            session: Database session
            corpus_name: Name of the corpus to fetch chunks from
            limit: Maximum number of chunks to fetch (None for all)
            
        Returns:
            List of chunk dictionaries
        """
        logging.info(f"Fetching chunked documents from corpus: {corpus_name}")
        
        query = """
            SELECT 
                chunk_hash,
                content,
                raw_document_hash
            FROM pipeline.boe_chunked_document
            WHERE corpus_name = :corpus_name
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = session.execute(text(query), {"corpus_name": corpus_name})
        
        chunks = []
        for row in result:
            chunks.append({
                'chunk_hash': row[0],
                'content': row[1],
                'raw_document_hash': row[2]
            })
        
        logging.info(f"Fetched {len(chunks)} chunked documents from corpus: {corpus_name}")
        return chunks
    
    
    def generate_dense_embeddings_batch(
        self, 
        chunks: List[Dict[str, Any]], 
        batch_size: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate dense embeddings for chunks in batches.
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping chunk_hash to dense embeddings
        """
        # Get dense models only
        dense_models = {name: model for name, model in self.models.items() if model.model_type == 'dense'}
        
        if not dense_models:
            logging.info("No dense models configured, skipping dense embedding generation")
            return {}
        
        logging.info(f"Generating dense embeddings for {len(chunks)} chunks using {len(dense_models)} models")
        
        results: dict[str, Any] = {}
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            logging.info(f"Processing dense embedding batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            # Prepare texts for this batch
            batch_texts = []
            batch_chunk_hashes = []
            
            for chunk in batch_chunks:
                chunk_hash = chunk['chunk_hash']
                content = chunk['content']
                
                if not content.strip():
                    logging.warning(f"Skipping empty chunk: {chunk_hash}")
                    continue
                
                batch_texts.append(content)
                batch_chunk_hashes.append(chunk_hash)
            
            if not batch_texts:
                continue
            
            # Generate embeddings for each dense model
            for model_name, model in dense_models.items():
                try:
                    # Generate embeddings for the entire batch
                    batch_embeddings = model.model_instance.encode(batch_texts, convert_to_tensor=False)
                    assert isinstance(batch_embeddings, np.ndarray), "batch_embeddings must be a numpy array"
                    
                    # Store embeddings for each chunk in the batch
                    for j, chunk_hash in enumerate(batch_chunk_hashes):
                        if chunk_hash not in results:
                            results[chunk_hash] = {}
                        
                        assert hasattr(batch_embeddings[j], 'tolist'), "batch_embeddings[j] must have a tolist method"
                        results[chunk_hash][model_name] = {
                            'type': 'dense',
                            'vector': batch_embeddings[j].tolist()
                        }
                        
                except Exception as e:
                    logging.error(f"Failed to generate dense embeddings for batch with model {model_name}: {e}")
                    continue
        
        logging.info(f"Generated dense embeddings for {len(results)} chunks")
        return results
    
    def generate_sparse_embeddings_batch(
        self, 
        chunks: List[Dict[str, Any]], 
        batch_size: int = 256
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate sparse embeddings for chunks in batches.
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping chunk_hash to sparse embeddings
        """
        # Get sparse models only
        sparse_models = {name: model for name, model in self.models.items() if model.model_type == 'sparse'}
        
        if not sparse_models:
            logging.info("No sparse models configured, skipping sparse embedding generation")
            return {}
        
        logging.info(f"Generating sparse embeddings for {len(chunks)} chunks using {len(sparse_models)} models")
        
        results: dict[str, Any] = {}
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            logging.info(f"Processing sparse embedding batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            # Prepare texts for this batch
            batch_texts = []
            batch_chunk_hashes = []
            
            for chunk in batch_chunks:
                chunk_hash = chunk['chunk_hash']
                content = chunk['content']
                
                # Use vocabulary-filtered content for embedding generation
                text_to_embed = content
                
                if not text_to_embed.strip():
                    logging.warning(f"Skipping empty chunk: {chunk_hash}")
                    continue
                
                batch_texts.append(text_to_embed)
                batch_chunk_hashes.append(chunk_hash)
            
            if not batch_texts:
                continue
            
            # Generate embeddings for each sparse model
            for model_name, model in sparse_models.items():
                try:
                    # Generate embeddings for the entire batch
                    batch_embeddings = model.model_instance.encode(batch_texts)
                    assert isinstance(batch_embeddings, np.ndarray), "batch_embeddings must be a numpy array"
                    
                    # Store embeddings for each chunk in the batch
                    for j, chunk_hash in enumerate(batch_chunk_hashes):
                        if chunk_hash not in results:
                            results[chunk_hash] = {}
                        
                        # Convert sparse embedding to dictionary format
                        sparse_embedding = batch_embeddings[j].coalesce()
                        if hasattr(sparse_embedding, 'indices') and hasattr(sparse_embedding, 'values'):
                            sparse_dict = {
                                'indices': sparse_embedding.indices().tolist(),
                                'values': sparse_embedding.values().tolist()
                            }
                        else:
                            # Fallback: convert to dense and then to sparse representation
                            dense_embedding = sparse_embedding.toarray()[0]
                            indices = np.where(dense_embedding != 0)[0].tolist()
                            values = dense_embedding[indices].tolist()
                            sparse_dict = {
                                'indices': indices,
                                'values': values
                            }
                        
                        results[chunk_hash][model_name] = {
                            'type': 'sparse',
                            'vector': sparse_dict
                        }
                        
                except Exception as e:
                    logging.error(f"Failed to generate sparse embeddings for batch with model {model_name}: {e}")
                    continue
        
        logging.info(f"Generated sparse embeddings for {len(results)} chunks")
        return results
    
    def store_dense_embeddings(
        self, 
        session: Session, 
        embeddings_data: Dict[str, Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Store dense embeddings in the database.
        
        Args:
            session: Database session
            embeddings_data: Dictionary of embeddings data
            
        Returns:
            Tuple of (inserted_count, existing_count)
        """
        logging.info("Storing dense embeddings in database")
        
        inserted_count = 0
        existing_count = 0
        
        for chunk_hash, chunk_embeddings in embeddings_data.items():
            for model_name, embedding_info in chunk_embeddings.items():
                if embedding_info['type'] != 'dense':
                    continue
                
                # Check if embedding already exists
                check_query = text("""
                    SELECT id FROM pipeline.boe_embedding
                    WHERE boe_chunked_document_hash = :chunk_hash 
                    AND model_name = :model_name
                """)
                
                existing = session.execute(
                    check_query, 
                    {"chunk_hash": chunk_hash, "model_name": model_name}
                ).fetchone()
                
                if existing:
                    existing_count += 1
                    continue
                
                # Insert new embedding
                insert_query = text("""
                    INSERT INTO pipeline.boe_embedding 
                    (boe_chunked_document_hash, model_name, vector)
                    VALUES (:chunk_hash, :model_name, :vector)
                """)
                
                session.execute(insert_query, {
                    "chunk_hash": chunk_hash,
                    "model_name": model_name,
                    "vector": embedding_info['vector']
                })
                inserted_count += 1
        
        session.commit()
        logging.info(f"Dense embeddings stored: {inserted_count} inserted, {existing_count} existing")
        return inserted_count, existing_count
    
    def store_sparse_embeddings(
        self, 
        session: Session, 
        embeddings_data: Dict[str, Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Store sparse embeddings in the database.
        
        Args:
            session: Database session
            embeddings_data: Dictionary of embeddings data
            
        Returns:
            Tuple of (inserted_count, existing_count)
        """
        logging.info("Storing sparse embeddings in database")
        
        inserted_count = 0
        existing_count = 0
        
        for chunk_hash, chunk_embeddings in embeddings_data.items():
            for model_name, embedding_info in chunk_embeddings.items():
                if embedding_info['type'] != 'sparse':
                    continue
                
                # Check if embedding already exists
                check_query = text("""
                    SELECT id FROM pipeline.boe_embedding_sparse
                    WHERE boe_chunked_document_hash = :chunk_hash 
                    AND model_name = :model_name
                """)
                
                existing = session.execute(
                    check_query, 
                    {"chunk_hash": chunk_hash, "model_name": model_name}
                ).fetchone()
                
                if existing:
                    existing_count += 1
                    continue
                
                # Insert new embedding
                insert_query = text("""
                    INSERT INTO pipeline.boe_embedding_sparse 
                    (boe_chunked_document_hash, model_name, sparse_vector)
                    VALUES (:chunk_hash, :model_name, :sparse_vector)
                """)
                
                session.execute(insert_query, {
                    "chunk_hash": chunk_hash,
                    "model_name": model_name,
                    "sparse_vector": json.dumps(embedding_info['vector'])
                })
                inserted_count += 1
        
        session.commit()
        logging.info(f"Sparse embeddings stored: {inserted_count} inserted, {existing_count} existing")
        return inserted_count, existing_count
    
    def process_corpus_embeddings(
        self, 
        session: Session, 
        corpus_name: str, 
        batch_size: int,
        limit: int,
    ) -> Dict[str, Any]:
        """
        Process embeddings for all chunks in a corpus.
        First processes all dense embeddings, then all sparse embeddings.
        
        Args:
            session: Database session
            corpus_name: Name of the corpus to process
            limit: Maximum number of chunks to process (None for all)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing processing statistics
        """
        logging.info(f"Starting embedding generation for corpus: {corpus_name}")
        
        # Fetch chunked documents
        chunks = self.fetch_chunked_documents(session, corpus_name, limit)
        
        if not chunks:
            logging.warning(f"No chunked documents found for corpus: {corpus_name}")
            return {"total_chunks": 0, "embeddings_generated": 0}
        
        # Step 1: Generate and store dense embeddings
        logging.info("="*60)
        logging.info("STEP 1: Processing dense embeddings")
        logging.info("="*60)
        
        dense_embeddings_data = self.generate_dense_embeddings_batch(chunks, batch_size)
        dense_inserted, dense_existing = 0, 0
        
        if dense_embeddings_data:
            dense_inserted, dense_existing = self.store_dense_embeddings(session, dense_embeddings_data)
            logging.info(f"Dense embeddings complete: {dense_inserted} inserted, {dense_existing} existing")
        else:
            logging.info("No dense embeddings generated")
        
        # Step 2: Generate and store sparse embeddings
        logging.info("="*60)
        logging.info("STEP 2: Processing sparse embeddings")
        logging.info("="*60)
        
        sparse_embeddings_data = self.generate_sparse_embeddings_batch(chunks, batch_size)
        sparse_inserted, sparse_existing = 0, 0
        
        if sparse_embeddings_data:
            sparse_inserted, sparse_existing = self.store_sparse_embeddings(session, sparse_embeddings_data)
            logging.info(f"Sparse embeddings complete: {sparse_inserted} inserted, {sparse_existing} existing")
        else:
            logging.info("No sparse embeddings generated")
        
        # Calculate total statistics
        total_inserted = dense_inserted + sparse_inserted
        total_existing = dense_existing + sparse_existing
        chunks_with_embeddings = len(set(list(dense_embeddings_data.keys()) + list(sparse_embeddings_data.keys())))
        
        stats = {
            "total_chunks": len(chunks),
            "chunks_with_embeddings": chunks_with_embeddings,
            "dense_embeddings_inserted": dense_inserted,
            "dense_embeddings_existing": dense_existing,
            "sparse_embeddings_inserted": sparse_inserted,
            "sparse_embeddings_existing": sparse_existing,
            "total_embeddings_inserted": total_inserted,
            "total_embeddings_existing": total_existing
        }
        
        logging.info("="*60)
        logging.info("EMBEDDING GENERATION COMPLETE")
        logging.info("="*60)
        logging.info(f"Corpus: {corpus_name}")
        logging.info(f"Total chunks: {stats['total_chunks']}")
        logging.info(f"Chunks with embeddings: {stats['chunks_with_embeddings']}")
        logging.info(f"Dense embeddings: {dense_inserted} inserted, {dense_existing} existing")
        logging.info(f"Sparse embeddings: {sparse_inserted} inserted, {sparse_existing} existing")
        logging.info(f"Total embeddings: {total_inserted} inserted, {total_existing} existing")
        
        return stats


def get_available_corpora() -> List[str]:
    """
    Get list of available corpora from the chunking pipeline.
    
    Returns:
        List of corpus names
    """
    return [
        # "newsgroups",
        # "wikipedia_sample", 
        "imdb_reviews",
        "trec_questions",
        "twitter-financial-news",
        # "pubmed-multilabel",
        # "patent-classification",
        # "goodreads-bookgenres",
        # "battery-abstracts",
        # "t2-ragbench-convfinqa"
    ]


def main():
    """Main execution function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Direct logging to file
    logging.getLogger().addHandler(logging.FileHandler("boe_embedding.log"))
    
    # Model configurations - using the same models from tokenizer tests
    model_configs = [
        {
            'name': 'naver/splade-v3',
            'type': 'sparse'
        },
        {
            'name': 'all-MiniLM-L6-v2',
            'type': 'dense'
        },
    ]
    
    # Configuration parameters
    for corpus_name in get_available_corpora():
        chunk_limit = None  # Set to a number to limit chunks processed, None for all
        batch_size = 4096
        
        print("BOE Embedding Pipeline Configuration:")
        print("  Models:")
        for config in model_configs:
            print(f"    - {config['name']} ({config['type']})")
        print(f"  Corpus: {corpus_name}")
        print(f"  Chunk limit: {chunk_limit if chunk_limit else 'All chunks'}")
        print(f"  Batch size: {batch_size}")
        print(f"  Available corpora: {get_available_corpora()}")
        
        # Load configuration
        config = cfg.load_config_from_env()
        db_config = config.database
        
        # Create database session
        with get_session(db_config) as session:
            # Initialize embedding pipeline
            pipeline = BOEEmbeddingPipeline(model_configs)
            
            # Process embeddings for the corpus
            try:
                stats = pipeline.process_corpus_embeddings(
                    session, 
                    corpus_name, 
                    limit=chunk_limit,
                    batch_size=batch_size
                )
                
                # Print final statistics
                print("\n" + "="*60)
                print("EMBEDDING GENERATION COMPLETE")
                print("="*60)
                print(f"Corpus: {corpus_name}")
                print(f"Total chunks processed: {stats['total_chunks']}")
                print(f"Chunks with embeddings: {stats['chunks_with_embeddings']}")
                print(f"Dense embeddings: {stats['dense_embeddings_inserted']} inserted, {stats['dense_embeddings_existing']} existing")
                print(f"Sparse embeddings: {stats['sparse_embeddings_inserted']} inserted, {stats['sparse_embeddings_existing']} existing")
                print(f"Total embeddings: {stats['total_embeddings_inserted']} inserted, {stats['total_embeddings_existing']} existing")
                print("="*60)
                
            except Exception as e:
                logging.error(f"Error processing corpus {corpus_name}: {str(e)}")
                raise


if __name__ == '__main__':
    main()

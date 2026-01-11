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
    
    def fetch_unembedded_chunks_for_model(
        self,
        session: Session,
        corpus_name: str,
        model_name: str,
        model_type: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch chunks that don't have embeddings for a specific model.
        
        Args:
            session: Database session
            corpus_name: Name of the corpus to fetch chunks from
            model_name: Name of the embedding model
            model_type: Type of model ('dense' or 'sparse')
            limit: Maximum number of chunks to fetch (None for all)
            
        Returns:
            List of chunk dictionaries that need embedding for this model
        """
        # Choose the appropriate embedding table based on model type
        embedding_table = "pipeline.boe_embedding" if model_type == "dense" else "pipeline.boe_embedding_sparse"
        
        query = f"""
            SELECT 
                bcd.chunk_hash,
                bcd.content,
                bcd.raw_document_hash
            FROM pipeline.boe_chunked_document bcd
            WHERE bcd.corpus_name = :corpus_name
            AND NOT EXISTS (
                SELECT 1 FROM {embedding_table} be
                WHERE be.boe_chunked_document_hash = bcd.chunk_hash
                AND be.model_name = :model_name
            )
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = session.execute(text(query), {"corpus_name": corpus_name, "model_name": model_name})
        
        chunks = []
        for row in result:
            chunks.append({
                'chunk_hash': row[0],
                'content': row[1],
                'raw_document_hash': row[2]
            })
        
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
                    
                    # Store embeddings for each chunk in the batch
                    for j, chunk_hash in enumerate(batch_chunk_hashes):
                        if chunk_hash not in results:
                            results[chunk_hash] = {}
                        
                        # Convert sparse embedding to dictionary format
                        sparse_embedding = batch_embeddings[j].coalesce()
                        if hasattr(sparse_embedding, 'indices') and hasattr(sparse_embedding, 'values'):
                            sparse_dict = {
                                'indices': sparse_embedding.indices().tolist()[0],
                                'values': sparse_embedding.values().tolist()
                            }

                            assert len(sparse_dict['indices']) == len(sparse_dict['values']), "indices and values must have the same length"
                        else:
                            # Fallback: convert to dense and then to sparse representation
                            dense_embedding = sparse_embedding.toarray()[0]
                            indices = np.where(dense_embedding != 0)[0].tolist()[0]
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
        Processes each model separately, skipping models where all chunks are already embedded.
        
        Args:
            session: Database session
            corpus_name: Name of the corpus to process
            limit: Maximum number of chunks to process (None for all)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing processing statistics
        """
        logging.info(f"Starting embedding generation for corpus: {corpus_name}")
        
        # Track statistics per model
        stats = {
            "corpus_name": corpus_name,
            "models_processed": [],
            "models_skipped": [],
            "dense_embeddings_inserted": 0,
            "sparse_embeddings_inserted": 0,
            "total_embeddings_inserted": 0,
        }
        
        # Process each model separately
        for model_name, model in self.models.items():
            logging.info("="*60)
            logging.info(f"Processing model: {model_name} ({model.model_type})")
            logging.info("="*60)
            
            # Fetch only chunks that need embedding for this model
            unembedded_chunks = self.fetch_unembedded_chunks_for_model(
                session, corpus_name, model_name, model.model_type, limit
            )
            
            if not unembedded_chunks:
                logging.info(f"Model '{model_name}' already has embeddings for all chunks, skipping")
                stats["models_skipped"].append(model_name)
                continue
            
            logging.info(f"Found {len(unembedded_chunks)} chunks needing embeddings for model '{model_name}'")
            stats["models_processed"].append(model_name)
            
            if model.model_type == 'dense':
                # Generate dense embeddings for unembedded chunks
                embeddings_data = self._generate_dense_embeddings_for_model(
                    unembedded_chunks, model_name, model, batch_size
                )
                if embeddings_data:
                    inserted, _ = self.store_dense_embeddings(session, embeddings_data)
                    stats["dense_embeddings_inserted"] += inserted
                    stats["total_embeddings_inserted"] += inserted
                    logging.info(f"Dense embeddings for '{model_name}': {inserted} inserted")
            
            elif model.model_type == 'sparse':
                # Generate sparse embeddings for unembedded chunks
                embeddings_data = self._generate_sparse_embeddings_for_model(
                    unembedded_chunks, model_name, model, batch_size
                )
                if embeddings_data:
                    inserted, _ = self.store_sparse_embeddings(session, embeddings_data)
                    stats["sparse_embeddings_inserted"] += inserted
                    stats["total_embeddings_inserted"] += inserted
                    logging.info(f"Sparse embeddings for '{model_name}': {inserted} inserted")
        
        logging.info("="*60)
        logging.info("EMBEDDING GENERATION COMPLETE")
        logging.info("="*60)
        logging.info(f"Corpus: {corpus_name}")
        logging.info(f"Models processed: {stats['models_processed']}")
        logging.info(f"Models skipped (already complete): {stats['models_skipped']}")
        logging.info(f"Dense embeddings inserted: {stats['dense_embeddings_inserted']}")
        logging.info(f"Sparse embeddings inserted: {stats['sparse_embeddings_inserted']}")
        logging.info(f"Total embeddings inserted: {stats['total_embeddings_inserted']}")
        
        return stats
    
    def _generate_dense_embeddings_for_model(
        self,
        chunks: List[Dict[str, Any]],
        model_name: str,
        model: EmbeddingModel,
        batch_size: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate dense embeddings for a specific model.
        
        Args:
            chunks: List of chunk dictionaries
            model_name: Name of the model
            model: EmbeddingModel instance
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping chunk_hash to embedding data
        """
        logging.info(f"Generating dense embeddings for {len(chunks)} chunks using model '{model_name}'")
        
        results: dict[str, Any] = {}
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
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
            
            try:
                batch_embeddings = model.model_instance.encode(batch_texts, convert_to_tensor=False)
                assert isinstance(batch_embeddings, np.ndarray), "batch_embeddings must be a numpy array"
                
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
    
    def _generate_sparse_embeddings_for_model(
        self,
        chunks: List[Dict[str, Any]],
        model_name: str,
        model: EmbeddingModel,
        batch_size: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate sparse embeddings for a specific model.
        
        Args:
            chunks: List of chunk dictionaries
            model_name: Name of the model
            model: EmbeddingModel instance
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping chunk_hash to embedding data
        """
        logging.info(f"Generating sparse embeddings for {len(chunks)} chunks using model '{model_name}'")
        
        results: dict[str, Any] = {}
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
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
            
            try:
                batch_embeddings = model.model_instance.encode(batch_texts)
                
                for j, chunk_hash in enumerate(batch_chunk_hashes):
                    if chunk_hash not in results:
                        results[chunk_hash] = {}
                    
                    sparse_embedding = batch_embeddings[j].coalesce()
                    sparse_dict = {
                        'indices': sparse_embedding.indices().tolist()[0],
                        'values': sparse_embedding.values().tolist()
                    }
                    assert len(sparse_dict['indices']) == len(sparse_dict['values']), "indices and values must have the same length"
                    
                    results[chunk_hash][model_name] = {
                        'type': 'sparse',
                        'vector': sparse_dict
                    }
            except Exception as e:
                logging.error(f"Failed to generate sparse embeddings for batch with model {model_name}: {e}")
                continue
        
        logging.info(f"Generated sparse embeddings for {len(results)} chunks")
        return results


def get_available_corpora(session: Session) -> List[str]:
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
    
    # Load configuration
    config = cfg.load_config_from_env()
    db_config = config.database
    
    # Create database session
    with get_session(db_config) as session:
        # Initialize embedding pipeline
        pipeline = BOEEmbeddingPipeline(model_configs)
        
        # Get available corpora
        corpora = get_available_corpora(session)
        
        # Configuration parameters
        chunk_limit = None  # Set to a number to limit chunks processed, None for all
        batch_size = 4096
        
        print("BOE Embedding Pipeline Configuration:")
        print("  Models:")
        for model_config in model_configs:
            print(f"    - {model_config['name']} ({model_config['type']})")
        print(f"  Chunk limit: {chunk_limit if chunk_limit else 'All chunks'}")
        print(f"  Batch size: {batch_size}")
        print(f"  Available corpora: {corpora}")
        
        for corpus_name in corpora:
            print(f"\nProcessing corpus: {corpus_name}")
            
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
                print(f"Models processed: {stats['models_processed']}")
                print(f"Models skipped (already complete): {stats['models_skipped']}")
                print(f"Dense embeddings inserted: {stats['dense_embeddings_inserted']}")
                print(f"Sparse embeddings inserted: {stats['sparse_embeddings_inserted']}")
                print(f"Total embeddings inserted: {stats['total_embeddings_inserted']}")
                print("="*60)
                
            except Exception as e:
                logging.error(f"Error processing corpus {corpus_name}: {str(e)}")
                raise


if __name__ == '__main__':
    main()

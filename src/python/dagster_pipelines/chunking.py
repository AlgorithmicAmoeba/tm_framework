"""
Chunking assets for Dagster pipelines.
"""
import logging
from typing import Any, Dict

from dagster import asset, Config, AssetExecutionContext, MaterializeResult, MetadataValue, Definitions
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg
import pipelines.chunking.main as chunking


class ChunkingConfig(Config):
    """Configuration for chunking assets."""
    corpus_name: str
    max_tokens: int
    source_table: str = "pipeline.used_raw_document"


def build_chunking_asset(
    corpus_name: str,
    max_tokens: int, 
    deps: list[str] = None
) -> Definitions:
    """
    Factory function to create chunking assets.
    
    Args:
        corpus_name: Name of the corpus to chunk
        max_tokens: Maximum number of tokens per chunk
        deps: List of asset dependencies
        
    Returns:
        Dagster Definitions object containing the asset
    """
    asset_key = f"chunked_{corpus_name}"
    
    deps_list = deps or [f"preprocessed_{corpus_name}"]
    
    @asset(
        name=asset_key,
        deps=deps_list,
    )
    def chunking_asset(context: AssetExecutionContext) -> MaterializeResult:
        """Chunk corpus into smaller pieces."""
        tm_config = cfg.load_config_from_env()
        
        context.log.info(f"Chunking {corpus_name} corpus with {max_tokens} max tokens...")
        
        with get_session(tm_config.database) as session:
            orig_doc_count, chunk_count, avg_chunks_per_doc = chunking.chunk_corpus(
                session, corpus_name, max_tokens
            )
        
        return MaterializeResult(
            metadata={
                "corpus_name": MetadataValue.text(corpus_name),
                "max_tokens": MetadataValue.int(max_tokens),
                "original_document_count": MetadataValue.int(orig_doc_count),
                "chunk_count": MetadataValue.int(chunk_count), 
                "avg_chunks_per_document": MetadataValue.float(avg_chunks_per_doc),
                "chunking_ratio": MetadataValue.float(chunk_count / orig_doc_count if orig_doc_count > 0 else 0)
            }
        )
    
    return Definitions(
        assets=[chunking_asset],
    )


# Create the chunking assets
newsgroups_chunking_def = build_chunking_asset(
    corpus_name="newsgroups",
    max_tokens=8190
)

wikipedia_chunking_def = build_chunking_asset(
    corpus_name="wikipedia_sample",
    max_tokens=8190
)

imdb_chunking_def = build_chunking_asset(
    corpus_name="imdb_reviews",
    max_tokens=8190
)

trec_chunking_def = build_chunking_asset(
    corpus_name="trec_questions",
    max_tokens=8190
)

twitter_chunking_def = build_chunking_asset(
    corpus_name="twitter-financial-news",
    max_tokens=8190
)

chunking_definitions = [
    newsgroups_chunking_def,
    wikipedia_chunking_def,
    imdb_chunking_def,
    trec_chunking_def,
    twitter_chunking_def
]
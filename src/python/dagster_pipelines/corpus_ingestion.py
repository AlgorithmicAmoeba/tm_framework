"""
Corpus ingestion assets for Dagster pipelines.
"""
import logging
from typing import Any, Callable

from dagster import asset, Config, AssetExecutionContext, MaterializeResult, MetadataValue, Definitions
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg
import pipelines.corpus_ingestion.main as corpus_ingestion


class CorpusConfig(Config):
    """Configuration for corpus ingestion assets."""
    corpus_name: str
    subset: int | None = None
    description: str | None = None
    file_path: str | None = None


def build_corpus_ingestion_asset(
    corpus_name: str,
    description: str | None = None,
    file_path: str | None = None,
    subset: int | None = None,
    ingestion_function: Callable = None,
) -> Definitions:
    """
    Factory function to create corpus ingestion assets.
    
    Args:
        corpus_name: Name of the corpus to ingest
        description: Description of the corpus
        file_path: Path to the corpus file (if applicable)
        subset: Number of documents to ingest (if subset desired)
        ingestion_function: The function to call for ingestion
        
    Returns:
        Dagster Definitions object containing the asset
    """
    asset_key = f"ingested_{corpus_name}"
    
    @asset(name=asset_key)
    def corpus_ingestion_asset(context: AssetExecutionContext) -> MaterializeResult:
        """Ingest corpus into the database."""
        tm_config = cfg.load_config_from_env()
        
        context.log.info(f"Ingesting {corpus_name} corpus...")
        
        doc_count = 0
        with get_session(tm_config.database) as session:
            if corpus_name == "wikipedia_sample" and file_path:
                doc_count = ingestion_function(
                    session, 
                    file_path=file_path,
                    subset=subset, 
                    description=description
                )
            else:
                doc_count = ingestion_function(
                    session, 
                    subset=subset, 
                    description=description
                )
        
        return MaterializeResult(
            metadata={
                "corpus_name": MetadataValue.text(corpus_name),
                "document_count": MetadataValue.int(doc_count),
                "subset_size": MetadataValue.int(subset) if subset else MetadataValue.text("Full corpus"),
                "description": MetadataValue.text(description) if description else MetadataValue.text("No description")
            }
        )
    
    return Definitions(
        assets=[corpus_ingestion_asset],
    )


# Create the corpus ingestion assets
newsgroups_asset_def = build_corpus_ingestion_asset(
    corpus_name="newsgroups",
    description="20 Newsgroups dataset - collection of newsgroup documents",
    ingestion_function=corpus_ingestion.ingest_newsgroups,
)

wikipedia_asset_def = build_corpus_ingestion_asset(
    corpus_name="wikipedia_sample",
    description="Wikipedia sample articles",
    file_path=str(cfg.load_config_from_env().get_data_path() / 'raw_data/wikipedia_20k_sample.jsonl'),
    ingestion_function=corpus_ingestion.ingest_wikipedia,
)

imdb_asset_def = build_corpus_ingestion_asset(
    corpus_name="imdb_reviews",
    description="IMDB movie reviews dataset for sentiment analysis",
    ingestion_function=corpus_ingestion.ingest_imdb,
)

trec_asset_def = build_corpus_ingestion_asset(
    corpus_name="trec_questions",
    description="TREC question classification dataset",
    ingestion_function=corpus_ingestion.ingest_trec,
)

twitter_asset_def = build_corpus_ingestion_asset(
    corpus_name="twitter-financial-news",
    description="Twitter Financial News Topic dataset",
    ingestion_function=corpus_ingestion.ingest_twitter_financial,
)

ingestion_definitions = [
    newsgroups_asset_def,
    wikipedia_asset_def,
    imdb_asset_def,
    trec_asset_def,
    twitter_asset_def,
]
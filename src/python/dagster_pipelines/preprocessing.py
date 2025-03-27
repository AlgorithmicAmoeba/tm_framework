"""
Preprocessing assets for Dagster pipelines.
"""
import logging
from typing import Any, Callable, Dict

from dagster import asset, Config, AssetExecutionContext, AssetIn, MaterializeResult, MetadataValue, Definitions
from sqlalchemy.orm import Session

from database import get_session
import configuration as cfg
import pipelines.preprocessing.main as preprocessing


class PreprocessingConfig(Config):
    """Configuration for preprocessing assets."""
    corpus_name: str
    top_n: int | None = None
    min_words_per_document: int | None = None
    min_df: float = 0.001
    max_df: float = 0.7
    min_chars: int = 3
    remove_stopwords: bool = True
    lemmatize: bool = True
    remove_numbers: bool = True
    remove_urls: bool = True


def build_preprocessing_asset(
    corpus_name: str,
    preprocessing_params: Dict[str, Any], 
    deps: list[str] = None
) -> Definitions:
    """
    Factory function to create preprocessing assets.
    
    Args:
        corpus_name: Name of the corpus to preprocess
        preprocessing_params: Dictionary of preprocessing parameters
        deps: List of asset dependencies
        
    Returns:
        Dagster Definitions object containing the asset
    """
    asset_key = f"preprocessed_{corpus_name}"
    
    deps_list = deps or [f"ingested_{corpus_name}"]
    
    @asset(
        name=asset_key,
        deps=deps_list,
    )
    def preprocessing_asset(context: AssetExecutionContext) -> MaterializeResult:
        """Preprocess corpus."""
        tm_config = cfg.load_config_from_env()
        
        context.log.info(f"Preprocessing {corpus_name} corpus...")
        
        with get_session(tm_config.database) as session:
            vocab_size, doc_count = preprocessing.preprocess_corpus(
                session, corpus_name, preprocessing_params
            )
        
        return MaterializeResult(
            metadata={
                "corpus_name": MetadataValue.text(corpus_name),
                "document_count": MetadataValue.int(doc_count),
                "vocabulary_size": MetadataValue.int(vocab_size),
                "min_df": MetadataValue.float(preprocessing_params.get('min_df', 0.001)),
                "max_df": MetadataValue.float(preprocessing_params.get('max_df', 0.7)),
                "top_n": MetadataValue.int(preprocessing_params.get('top_n', 0)),
                "removed_stopwords": MetadataValue.bool(preprocessing_params.get('remove_stopwords', True)),
                "lemmatized": MetadataValue.bool(preprocessing_params.get('lemmatize', True)),
            }
        )
    
    return Definitions(
        assets=[preprocessing_asset],
    )


# Create the preprocessing assets
newsgroups_preprocessing_def = build_preprocessing_asset(
    corpus_name="newsgroups",
    preprocessing_params={
        'top_n': 10000,
        'min_words_per_document': 5,
        'min_df': 0.005,
        'max_df': 0.8,
        'min_chars': 3,
        'remove_stopwords': True,
        'lemmatize': True,
        'remove_numbers': True,
        'remove_urls': True
    }
)

wikipedia_preprocessing_def = build_preprocessing_asset(
    corpus_name="wikipedia_sample",
    preprocessing_params={
        'top_n': 20000,
        'min_words_per_document': 10,
        'min_df': 0.02,
        'max_df': 0.7,
        'min_chars': 3,
        'remove_stopwords': True,
        'lemmatize': True,
        'remove_numbers': True,
        'remove_urls': True
    }
)

imdb_preprocessing_def = build_preprocessing_asset(
    corpus_name="imdb_reviews",
    preprocessing_params={
        'top_n': 15000,
        'min_words_per_document': 10,
        'min_df': 0.003,
        'max_df': 0.7,
        'min_chars': 3,
        'remove_stopwords': True,
        'lemmatize': True,
        'remove_numbers': True,
        'remove_urls': True
    }
)

trec_preprocessing_def = build_preprocessing_asset(
    corpus_name="trec_questions",
    preprocessing_params={
        'top_n': 5000,
        'min_words_per_document': 2,
        'min_df': 0.0005,
        'max_df': 0.9,
        'min_chars': 3,
        'remove_stopwords': True,
        'lemmatize': True,
        'remove_numbers': True,
        'remove_urls': True
    }
)

twitter_preprocessing_def = build_preprocessing_asset(
    corpus_name="twitter-financial-news",
    preprocessing_params={
        'top_n': 8000,
        'min_words_per_document': 3,
        'min_df': 0.001,
        'max_df': 0.8,
        'min_chars': 3,
        'remove_stopwords': True,
        'lemmatize': True,
        'remove_numbers': True,
        'remove_urls': True
    }
)

preprocessing_definitions = [
    newsgroups_preprocessing_def,
    wikipedia_preprocessing_def,
    imdb_preprocessing_def,
    trec_preprocessing_def,
    twitter_preprocessing_def,
]
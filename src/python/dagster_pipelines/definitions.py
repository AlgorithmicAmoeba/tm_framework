"""
Dagster pipeline definition for the topic modeling framework.
"""
from dagster import Definitions, AssetSelection, define_asset_job, ScheduleDefinition, EnvVar

from dagster_pipelines.corpus_ingestion import ingestion_definitions

from dagster_pipelines.preprocessing import preprocessing_definitions

from dagster_pipelines.chunking import chunking_definitions

# Define jobs for each corpus
newsgroups_job = define_asset_job(
    name="newsgroups_pipeline",
    selection=AssetSelection.assets(
        "ingested_newsgroups",
        "preprocessed_newsgroups",
        "chunked_newsgroups"
    ),
    description="Process 20 Newsgroups corpus through ingestion, preprocessing, and chunking"
)

wikipedia_job = define_asset_job(
    name="wikipedia_pipeline",
    selection=AssetSelection.assets(
        "ingested_wikipedia_sample",
        "preprocessed_wikipedia_sample",
        "chunked_wikipedia_sample"
    ),
    description="Process Wikipedia corpus through ingestion, preprocessing, and chunking"
)

imdb_job = define_asset_job(
    name="imdb_pipeline",
    selection=AssetSelection.assets(
        "ingested_imdb_reviews",
        "preprocessed_imdb_reviews",
        "chunked_imdb_reviews"
    ),
    description="Process IMDB reviews corpus through ingestion, preprocessing, and chunking"
)

trec_job = define_asset_job(
    name="trec_pipeline",
    selection=AssetSelection.assets(
        "ingested_trec_questions",
        "preprocessed_trec_questions",
        "chunked_trec_questions"
    ),
    description="Process TREC questions corpus through ingestion, preprocessing, and chunking"
)

twitter_job = define_asset_job(
    name="twitter_pipeline",
    selection=AssetSelection.assets(
        "ingested_twitter-financial-news",
        "preprocessed_twitter-financial-news",
        "chunked_twitter-financial-news"
    ),
    description="Process Twitter financial news corpus through ingestion, preprocessing, and chunking"
)

# Full pipeline job
full_pipeline_job = define_asset_job(
    name="full_pipeline",
    selection=AssetSelection.all(),
    description="Process all corpora through the full pipeline"
)

# Create the Definitions for all corpora pipelines
defs = Definitions.merge(
    *ingestion_definitions,
    *preprocessing_definitions,
    *chunking_definitions,
    Definitions(
        jobs=[
            newsgroups_job, 
            wikipedia_job,
            imdb_job,
            trec_job,
            twitter_job,
            full_pipeline_job
        ],
    )
)
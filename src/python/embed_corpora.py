import warnings
import tqdm
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.sql import insert

import database
import configuration as cfg
from models import Corpus, Document, Embedding, DocumentType
import openai
import tiktoken

def embed_corpora():
    import configuration as cfg
    import openai
    from batch_processor import BatchProcessor

    config = cfg.load_config_from_env()
    client = openai.Client(api_key=config.openai.api_key)

    processor = BatchProcessor(client, max_batch_size=49_990)

    # Process the large batch file
    # batches = processor.process_large_batch(
    #     "ignore/processed_data/batch_embedding.jsonl",
    #     "ignore/processed_data/batch_splits"
    # )

    # Save batch tracking information
    # processor.save_batch_info("ignore/processed_data/batch_tracking.json")

    # load batch tracking information
    processor.load_batch_info("ignore/processed_data/batch_tracking.json")

    # Check status of batches
    # status = processor.check_batch_statuses()
    # print("Batch processing status:", status)

    results = processor.combine_batch_results(
        "ignore/processed_data/batch_results",
        "ignore/processed_data/combined_embeddings.json"
    )
    print("Results summary:", results)

if __name__ == '__main__':
    embed_corpora()

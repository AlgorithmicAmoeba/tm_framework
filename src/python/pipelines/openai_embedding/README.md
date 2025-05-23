# OpenAI Embedding Pipeline

This directory contains the components for managing document embeddings using OpenAI's API. The pipeline consists of four main processes:

## Processes

1. **new_jobs**: Identifies new document chunks that need embeddings and creates jobs for them in the database. It checks both the cache and existing jobs to avoid duplicates.

2. **synchronous_embedding**: Processes embedding jobs from the database queue, generates embeddings using OpenAI's API, and stores the results in the cache.

3. **batch_runner**: Manages batch processing of embedding jobs, allowing for efficient processing of large numbers of documents.

4. **ingest_from_cache**: Handles importing and managing embeddings from the cache, providing access to previously generated embeddings.

## Cache

The pipeline uses a DuckDB-based cache to store embeddings, which helps avoid regenerating embeddings for previously processed documents. The cache is stored in `data/embeddings/cache.db`. 
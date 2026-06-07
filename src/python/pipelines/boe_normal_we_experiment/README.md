# BOE normal word-embedding ablation

This experiment is a **word-embedding ablation** against the BOE@1 condition. It
reuses the BOE@1 **document** representation (first-chunk `all-MiniLM-L6-v2`
embeddings already stored in `pipeline.boe_cse_document_embedding` at
`target_chunk_count=1`; filter `algorithm='none'`, `target_dims=0`,
`padding_method='noise_only'`) but replaces the BOE-derived (TF-IDF + Ridge)
word vectors with **"normal" word embeddings**: each vocabulary word is encoded
*directly* with `SentenceTransformer('all-MiniLM-L6-v2')` into the same 384-d
space as the documents. Only the two word-embedding-driven topic-model families,
**KeyNMF** and **SemanticSignalSeparation (S³)**, are trained, over
`num_topics ∈ {10,20,50,100,200}`, 5 repeats each, across all 10 corpora.
Everything lives in dedicated `pipeline.boe_nwe_*` tables.

## How to run (in order)

```bash
# 0. apply schema
. .env && psql $DB_URI -f src/python/pipelines/boe_normal_we_experiment/schema.sql

# 1. normal word embeddings for all corpora
set -a && . .env && set +a
uv run src/python/pipelines/boe_normal_we_experiment/word_embedding.py

# 2. full KeyNMF + S³ training sweep
uv run src/python/pipelines/boe_normal_we_experiment/experiment_runner.py

# 3. coherence metrics (NPMI / WEPS / WECS / ISH)
uv run src/python/pipelines/boe_normal_we_experiment/metrics.py
```

## Comparison logic

- **vs BOE@1** (`boe_cse_*` at `target_chunk_count=1`): same doc representation,
  same model families — isolates the effect of *derived* (BOE) vs *normal*
  (directly-encoded) word embeddings.
- **vs existing KeyNMF / S³** (production `topic_model_corpus_result`): isolates
  the effect of the embedder / document representation, since those models use
  their normal in-model encoding rather than the BOE@1 first-chunk vectors.
